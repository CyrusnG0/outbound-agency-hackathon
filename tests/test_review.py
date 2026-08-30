"""Tests for the human review gate (ticket B4b): app/review.py's five
decisions, the kill-switch asymmetry, the refusal paths, the audit trail,
and write_kill_switch's round-trip.

The service is a plain module — every test here drives it directly against
a fresh SQLite database, with the kill switch pointed at a tmp file via
OUTBOUND_KILL_SWITCH_PATH (the committed config/kill_switch.json stays
enabled=false — the B4a convention).  Seeding goes through write_gate.commit
on purpose: fixtures are normal pipeline writes, and the audit-trail tests
rely on every seeded core-table row having a write_log row.
"""

import json  # parsing critique_json / payload_json in the audit-trail assertions

import pytest  # fixtures, tmp_path, monkeypatch

from app.agents_registry import seed_agent_registry  # the five principals — the write gate refuses unregistered writers
from app.db import apply_schema, connect, normalize_email  # F1b: the shared suppression key helper, used to seed pre-existing rows exactly as the writers do
from app.ids import new_id  # fresh ids for seeded rows
from app.kill_switch import read_kill_switch, write_kill_switch  # the reader/writer the gate and the toggle tests exercise
from app.review import (  # the gate under test
    ReviewDecisionRequest,
    VALID_DECISIONS,
    record_review_decision,
)
from app.write_gate import commit  # every seeded core-table row goes through the gate, never a raw INSERT

# The five decisions and the state each must land in (the ticket's table,
# copied so a drift between code and test is a failure, not a surprise).
DECISION_STATES = {
    "approve": "approved",
    "approve_with_edits": "approved",
    "reject": "not_target",
    "reject_and_suppress": "suppressed",
    "escalate": "researched",
}


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def switch_path(tmp_path, monkeypatch):
    """A tmp kill-switch file, written DISENGAGED, and the env var pointing
    the gate's reader at it — so no test here reads (or writes) the
    committed config/kill_switch.json, and engaging the switch is just a
    rewrite of this file."""
    path = tmp_path / "kill_switch.json"
    write_kill_switch(engaged=False, updated_by="fixture", path=str(path))
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(path))
    return path


@pytest.fixture
def conn(scratch_db_target, switch_path):
    """Fresh SQLite DB with schema, the five seeded principals, one offer /
    account / contact (with email) / contact (no email), and four targets:

    - tgt_1  awaiting_review, contact with email, TWO draft revisions
      (the happy-path subject of most tests)
    - tgt_2  awaiting_review, contact WITHOUT email, one revision (the
      no-email suppression case)
    - tgt_3  scored, one revision (the wrong-state case)
    - tgt_4  awaiting_review, contact with an ALREADY-SUPPRESSED email
      (the idempotent-suppression case)
    """
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    seed_agent_registry(c, run_id="r0", step_id="s0")
    commit(
        c, action="insert_offer", table_name="offers", record_id="off_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "acme", 1),
    )
    commit(
        c, action="insert_account", table_name="accounts", record_id="acc_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               industry, estimated_size, geo, company_summary, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_1", "Acme", "acme.test", "acme.test", "Logistics", "11-50", "HK",
                "Acme coordinates logistics bookings."),
    )
    # A contact WITH an email (tgt_1 / tgt_3), one WITHOUT (tgt_2), and one
    # whose email is ALREADY suppressed (tgt_4's con_3 — its own address so
    # the pre-seed never collides with tgt_1's suppression writes).
    commit(
        c, action="insert_contact", table_name="contacts", record_id="con_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
               email_verified, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("con_1", "acc_1", "Jane Doe", "jane@acme.test", 1),
    )
    commit(
        c, action="insert_contact", table_name="contacts", record_id="con_2",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
               email_verified, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("con_2", "acc_1", "No Email Person", None, 0),
    )
    commit(
        c, action="insert_contact", table_name="contacts", record_id="con_3",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
               email_verified, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("con_3", "acc_1", "Already Suppressed", "already@acme.test", 1),
    )
    for target_id, contact_id, state in (
        ("tgt_1", "con_1", "awaiting_review"),
        ("tgt_2", "con_2", "awaiting_review"),
        ("tgt_3", "con_1", "scored"),
        ("tgt_4", "con_3", "awaiting_review"),
    ):
        commit(
            c, action="insert_target", table_name="targets", record_id=target_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id,
                   source, state, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(target_id, "acc_1", contact_id, "off_1", "csv", state),
        )
    # tgt_1: two revisions so the edit tests have a base to increment from.
    _insert_draft_version(c, "tgt_1", revision=1, subject="Cold subject v1", body="Body version one.")
    _insert_draft_version(c, "tgt_1", revision=2, subject="Cold subject v2", body="Body version two.")
    _insert_draft_version(c, "tgt_2", revision=1, subject="S", body="B" * 80)
    _insert_draft_version(c, "tgt_3", revision=1, subject="S", body="B" * 80)
    _insert_draft_version(c, "tgt_4", revision=1, subject="S", body="B" * 80)
    # tgt_4's email is already suppressed (the idempotent case).
    commit(
        c, action="insert_suppression", table_name="suppressions", record_id="already@acme.test",
        payload={"reason": "manual", "added_by": "operator"}, run_id="r0", step_id="s0",
        actor="operator", agent_id="operator",
        sql="""INSERT INTO suppressions (email, email_normalized, domain, reason, added_at, added_by, notes)
               VALUES (?,?,?,?,datetime('now'),?,?)""",
        params=("already@acme.test", normalize_email("already@acme.test"), None, "manual", "operator", "pre-seeded"),
    )
    yield c
    c.close()


def _insert_draft_version(c, target_id: str, *, revision: int, subject: str, body: str) -> None:
    """Insert one message_draft_versions row through the write gate, the way
    B3's persist node does (agent-authored: edited_by=draft_writer, gate
    columns NULL — the B3-Z3 invariant)."""
    commit(
        c, action="insert_message_draft_version", table_name="message_draft_versions",
        record_id=new_id("dv"), payload={"revision_number": revision},
        run_id="r0", step_id="s0", actor="system", agent_id="draft_writer",
        sql="""INSERT INTO message_draft_versions
               (draft_version_id, target_id, message_id, revision_number, subject,
                body, footer, edited_by, policy_check_passed, injection_scan_passed,
                send_gate_passed, critique_passed, critique_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=(new_id("dv"), target_id, None, revision, subject, body,
                "[unsubscribe: {UNSUBSCRIBE_URL}]", "draft_writer",
                None, None, None, None, None),
    )


def _decide(conn, *, target_id, decision, **kwargs) -> object:
    """Drive one decision through the gate and return the outcome."""
    return record_review_decision(
        conn,
        request=ReviewDecisionRequest(target_id=target_id, decision=decision, **kwargs),
        run_id="r1",
    )


def _review_rows(conn, target_id: str):
    return conn.execute(
        "SELECT * FROM review_decisions WHERE target_id=? ORDER BY created_at;",
        (target_id,),
    ).fetchall()


def _draft_versions(conn, target_id: str):
    return conn.execute(
        "SELECT * FROM message_draft_versions WHERE target_id=? "
        "ORDER BY revision_number, created_at;",
        (target_id,),
    ).fetchall()


# ── 1. The five decisions ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "decision,extra,expected_state",
    [
        ("approve", {}, "approved"),
        ("approve_with_edits",
         {"edited_subject": "Edited subject", "edited_body": "Edited body text."},
         "approved"),
        ("reject", {}, "not_target"),
        ("reject_and_suppress", {}, "suppressed"),
        ("escalate", {"research_note": "Look into the funding round again."}, "researched"),
    ],
)
def test_each_decision_transitions_and_writes_review_row(
    conn, decision, extra, expected_state
):
    """Every one of the five decisions must (a) transition the target to its
    mapped state, (b) write exactly one review_decisions row carrying the
    decision verbatim, and (c) return an un-refused outcome."""
    outcome = _decide(conn, target_id="tgt_1", decision=decision, **extra)
    assert outcome.refused is False
    assert outcome.new_state == expected_state
    assert outcome.review_decision_id is not None
    # The target's state actually moved.
    state = conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_1';"
    ).fetchone()["state"]
    assert state == expected_state
    # Exactly one decision row, with the decision string verbatim.
    rows = _review_rows(conn, "tgt_1")
    assert len(rows) == 1
    assert rows[0]["decision"] == decision
    assert rows[0]["actor"] == "operator"
    assert rows[0]["edited"] == (1 if decision == "approve_with_edits" else 0)


# ── 2. approve_with_edits ────────────────────────────────────────────────────


def test_approve_with_edits_creates_new_revision_and_preserves_prior_byte_identical(conn):
    """The §2.1 rule most likely to be simplified by accident: editing NEVER
    overwrites a draft in place.  The edit is a NEW revision with an
    incremented revision_number and edited_by=operator, and the prior
    revisions are byte-identical afterwards."""
    before = [dict(row) for row in _draft_versions(conn, "tgt_1")]
    assert len(before) == 2
    outcome = _decide(
        conn, target_id="tgt_1", decision="approve_with_edits",
        edited_subject="EDITED subject", edited_body="EDITED body.",
    )
    assert outcome.refused is False
    after = [dict(row) for row in _draft_versions(conn, "tgt_1")]
    assert len(after) == 3  # the two originals plus the new revision
    # The prior revisions are byte-identical (every column, not just a spot
    # check) — the originals survive the edit untouched.
    assert after[:2] == before
    # The new revision: incremented number, operator-authored, the edited
    # text, and the deterministic footer carried over from the prior
    # revision unchanged (the operator edits subject/body only — B3-Z1).
    new_rev = after[2]
    assert new_rev["revision_number"] == 3
    assert new_rev["edited_by"] == "operator"
    assert new_rev["subject"] == "EDITED subject"
    assert new_rev["body"] == "EDITED body."
    assert new_rev["footer"] == before[1]["footer"]
    assert new_rev["message_id"] is None  # no messages row until B5 sends


def test_edited_revision_gate_columns_are_written_by_the_runner(conn):
    """human-review.md §5: the edited revision must independently re-pass its
    gates.  B4b still writes the columns NULL (B3-Z3 — the review gate does
    not judge the text); the G2 runner then fires on the edited revision and
    writes policy_check_passed / injection_scan_passed — while
    send_gate_passed stays NULL (the send gate's own) and the critique columns
    stay NULL (no critic ran on an operator edit)."""
    _decide(
        conn, target_id="tgt_1", decision="approve_with_edits",
        edited_subject="Edited subject",
        edited_body=(
            "Hello, this is the operator's edited body with enough length to "
            "pass the content-policy gate and no injected instruction text."
        ),
    )
    new_rev = _draft_versions(conn, "tgt_1")[-1]
    assert new_rev["policy_check_passed"] == 1
    assert new_rev["injection_scan_passed"] == 1
    assert new_rev["send_gate_passed"] is None
    assert new_rev["critique_passed"] is None
    assert new_rev["critique_json"] is None


def test_approve_with_edits_without_edited_text_is_refused(conn):
    """An approve_with_edits with no usable edited text is not an edit — it
    must be refused (with a pointer at plain approve), never silently
    approved as the un-edited draft."""
    outcome = _decide(conn, target_id="tgt_1", decision="approve_with_edits")
    assert outcome.refused is True
    assert "edited" in outcome.refusal_reason
    # Nothing moved, nothing written.
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_1';"
    ).fetchone()["state"] == "awaiting_review"
    assert _review_rows(conn, "tgt_1") == []
    assert len(_draft_versions(conn, "tgt_1")) == 2


# ── 3. reject_and_suppress ───────────────────────────────────────────────────


def test_reject_and_suppress_writes_suppression_row(conn):
    """The suppression extra effect: a suppressions row with the CHECK-pinned
    vocabulary reason='manual', added_by='operator', keyed by the contact's
    email, plus the awaiting_review -> suppressed transition."""
    outcome = _decide(conn, target_id="tgt_1", decision="reject_and_suppress")
    assert outcome.refused is False
    assert outcome.new_state == "suppressed"
    row = conn.execute(
        "SELECT * FROM suppressions WHERE email='jane@acme.test';"
    ).fetchone()
    assert row is not None
    assert row["reason"] == "manual"
    assert row["added_by"] == "operator"


def test_reject_and_suppress_without_contact_email_is_refused(conn):
    """The no-email case (§2.2 — my call, documented): there is nothing to
    suppress, so the WHOLE decision is refused with a logged reason pointing
    at reject.  It is never silently downgraded to reject — that would be
    changing the operator's decision — and never left to raise an
    IntegrityError."""
    outcome = _decide(conn, target_id="tgt_2", decision="reject_and_suppress")
    assert outcome.refused is True
    assert "use reject instead" in outcome.refusal_reason
    # The target stayed put, and no suppression or decision row appeared.
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_2';"
    ).fetchone()["state"] == "awaiting_review"
    assert conn.execute("SELECT COUNT(*) AS n FROM suppressions;").fetchone()["n"] == 1  # only tgt_4's pre-seeded row
    assert _review_rows(conn, "tgt_2") == []
    # The refusal is logged (never a silent no-op).
    step = conn.execute(
        "SELECT * FROM steps WHERE tool_name='review_decision' AND target_id='tgt_2';"
    ).fetchone()
    assert step is not None
    assert step["status"] == "failed"
    assert "use reject instead" in json.loads(step["output_json"])["refusal_reason"]


def test_reject_and_suppress_already_suppressed_is_idempotent(conn):
    """The already-suppressed case (§2.2 — my call, documented): the
    operator's goal (this email can never be mailed) is already true, so the
    decision proceeds and the INSERT is skipped — idempotent, never an
    IntegrityError on the primary key."""
    outcome = _decide(conn, target_id="tgt_4", decision="reject_and_suppress")
    assert outcome.refused is False
    assert outcome.new_state == "suppressed"
    # Exactly ONE suppression row for the email (the pre-seeded one) — no
    # duplicate insert, no crash.
    rows = conn.execute(
        "SELECT * FROM suppressions WHERE email='already@acme.test';"
    ).fetchall()
    assert len(rows) == 1
    # The review decision row still records the decision.
    assert len(_review_rows(conn, "tgt_4")) == 1


# ── 4. escalate ──────────────────────────────────────────────────────────────


def test_escalate_uses_research_escalation_reason_and_lands_in_researched(conn):
    """The escalate mapping is PINNED (resolved open-question 14,
    docs/state-machine.md §7): awaiting_review -> researched with
    reason=research_escalation verbatim, and the operator's note travels in
    review_decisions.reason."""
    note = "Re-check the funding round with the new search tool."
    outcome = _decide(conn, target_id="tgt_1", decision="escalate", research_note=note)
    assert outcome.refused is False
    assert outcome.new_state == "researched"
    transition_row = conn.execute(
        "SELECT * FROM state_transitions WHERE target_id='tgt_1' "
        "AND new_state='researched';"
    ).fetchone()
    assert transition_row is not None
    assert transition_row["reason"] == "research_escalation"  # verbatim, pinned by the docs
    # The note is stored on the decision row — the escalation "carries the
    # operator's research_note forward" (state-machine.md §7).
    row = _review_rows(conn, "tgt_1")[0]
    assert row["reason"] == note


# ── 5. The kill-switch asymmetry (the §2.3 test) ─────────────────────────────


@pytest.mark.parametrize("decision", ["approve", "approve_with_edits"])
def test_kill_switch_engaged_refuses_approve_and_approve_with_edits(conn, switch_path, decision):
    """§2.3: an engaged switch refuses the two outbound-AUTHORIZING
    decisions — both approve and approve_with_edits authorize a send, and
    P6 denies all outbound actions unconditionally."""
    # Engage the switch by rewriting the tmp file the env var points at.
    write_kill_switch(engaged=True, updated_by="test", path=str(switch_path))
    extra = (
        {"edited_subject": "Edited subject", "edited_body": "Edited body text."}
        if decision == "approve_with_edits"
        else {}
    )
    outcome = _decide(conn, target_id="tgt_1", decision=decision, **extra)
    assert outcome.refused is True
    assert "kill switch" in outcome.refusal_reason  # the switch's reason is surfaced to the operator
    # Nothing moved, nothing written — the refusal is total.
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_1';"
    ).fetchone()["state"] == "awaiting_review"
    assert _review_rows(conn, "tgt_1") == []
    assert len(_draft_versions(conn, "tgt_1")) == 2  # no edited revision either
    # And the refusal is a logged, observable outcome.
    step = conn.execute(
        "SELECT * FROM steps WHERE tool_name='review_decision' AND target_id='tgt_1';"
    ).fetchone()
    assert step is not None
    assert step["status"] == "failed"


@pytest.mark.parametrize(
    "decision,extra",
    [
        ("reject", {}),
        ("reject_and_suppress", {}),
        ("escalate", {"research_note": "check again"}),
    ],
)
def test_kill_switch_engaged_still_allows_deescalating_decisions(conn, switch_path, decision, extra):
    """§2.3's asymmetry: the three DE-ESCALATING decisions must still work
    while the switch is engaged — an operator who just hit the emergency
    stop must not be locked out of rejecting and suppressing the things
    that caused it.  A kill switch that blocks the brakes as well as the
    accelerator is a bug.  (The decision row records kill_switch_active=1:
    the switch state AT DECISION TIME.)"""
    write_kill_switch(engaged=True, updated_by="test", path=str(switch_path))
    outcome = _decide(conn, target_id="tgt_1", decision=decision, **extra)
    assert outcome.refused is False
    row = _review_rows(conn, "tgt_1")[0]
    assert row["kill_switch_active"] == 1  # the engaged state is recorded, not lost


def test_review_row_records_switch_state_when_disengaged(conn):
    """The kill_switch_active column's other half: a decision made while the
    switch is disengaged records 0 — so the column always answers "was the
    switch on when the operator decided?", never "maybe"."""
    outcome = _decide(conn, target_id="tgt_1", decision="approve")
    assert outcome.refused is False
    assert _review_rows(conn, "tgt_1")[0]["kill_switch_active"] == 0


# ── 6. Refusal paths ─────────────────────────────────────────────────────────


def test_target_not_in_awaiting_review_is_refused(conn):
    """A target in any state but awaiting_review must be refused with a
    logged step and NO transition — tgt_3 is in scored."""
    outcome = _decide(conn, target_id="tgt_3", decision="approve")
    assert outcome.refused is True
    assert "not awaiting_review" in outcome.refusal_reason
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_3';"
    ).fetchone()["state"] == "scored"
    assert _review_rows(conn, "tgt_3") == []
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_3';"
    ).fetchone()["n"] == 0


def test_double_submit_is_refused(conn):
    """A double-submitted form must not double-decide: the first approve
    moves the target to approved, and the second attempt finds it no longer
    awaiting_review and is refused — one decision row, one transition, and
    a logged refusal for the second submit."""
    first = _decide(conn, target_id="tgt_1", decision="approve")
    assert first.refused is False
    second = _decide(conn, target_id="tgt_1", decision="approve")
    assert second.refused is True
    assert "not awaiting_review" in second.refusal_reason
    # Exactly one decision row and one state_transitions row for the target.
    assert len(_review_rows(conn, "tgt_1")) == 1
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()["n"] == 1


def test_unknown_decision_string_is_refused(conn):
    """An unknown decision string must be refused — never a fall-through to
    a default action, never a guess at the nearest known decision."""
    outcome = _decide(conn, target_id="tgt_1", decision="send_it_now")
    assert outcome.refused is True
    assert "unknown decision" in outcome.refusal_reason
    assert VALID_DECISIONS == (
        "approve", "approve_with_edits", "reject", "reject_and_suppress", "escalate",
    )  # the vocabulary is exactly five — no sixth was invented
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_1';"
    ).fetchone()["state"] == "awaiting_review"
    assert _review_rows(conn, "tgt_1") == []


def test_unknown_target_is_refused(conn):
    """A decision for a phantom target is refused and logged, not a crash."""
    outcome = _decide(conn, target_id="tgt_ghost", decision="approve")
    assert outcome.refused is True
    assert "unknown target" in outcome.refusal_reason


# ── 7. Audit trail ───────────────────────────────────────────────────────────


def test_every_review_row_and_suppression_row_is_gated(conn):
    """The audit-trail guarantee: every review_decisions row and every
    suppressions row must have a matching write_log row with the matching
    action and record_id — a raw conn.execute replacing the gated write
    would leave data rows with no audit row and this test fails."""
    # tgt_1's email is NOT suppressed before this decision, so
    # reject_and_suppress exercises a REAL new suppression insert (not the
    # idempotent skip).
    _decide(conn, target_id="tgt_1", decision="reject_and_suppress")
    # The decision row was written THROUGH the gate, attributed to the
    # operator.
    decision_row = _review_rows(conn, "tgt_1")[0]
    log_row = conn.execute(
        "SELECT * FROM write_log WHERE record_id=? AND action='insert_review_decision';",
        (decision_row["review_decision_id"],),
    ).fetchone()
    assert log_row is not None
    assert log_row["actor"] == "operator"
    assert log_row["agent_id"] == "operator"
    # Every suppression row (the new one AND the pre-seeded one) has an
    # audit row with action=insert_suppression.
    for sup_row in conn.execute("SELECT email FROM suppressions;").fetchall():
        audit = conn.execute(
            "SELECT 1 FROM write_log WHERE record_id=? AND action='insert_suppression';",
            (sup_row["email"],),
        ).fetchone()
        assert audit is not None, f"suppression {sup_row['email']} has no write_log row"
    # The transition is gated too (the targets UPDATE write_log row — the
    # state machine's own guarantee).
    transition_audits = conn.execute(
        "SELECT * FROM write_log WHERE action='state_transition' "
        "AND record_id='tgt_1';"
    ).fetchall()
    assert len(transition_audits) == 1


# ── 8. write_kill_switch ─────────────────────────────────────────────────────


def test_write_kill_switch_round_trips_through_read(switch_path):
    """The writer and the reader must agree: what write_kill_switch writes
    is exactly what read_kill_switch reads back — engaged and disengaged,
    with the updated_by carried through."""
    state = write_kill_switch(engaged=True, updated_by="test-roundtrip", path=str(switch_path))
    assert state.engaged is True  # the returned state is the reader's view of the file
    assert state.updated_by == "test-roundtrip"
    assert state.updated_at  # the write stamped a UTC timestamp
    assert read_kill_switch(str(switch_path)).engaged is True  # and a fresh read agrees
    # Flip it back — the file remains the single source of truth.
    state = write_kill_switch(engaged=False, updated_by="test-roundtrip", path=str(switch_path))
    assert state.engaged is False
    assert read_kill_switch(str(switch_path)).engaged is False


def test_write_kill_switch_preserves_documented_shape(switch_path):
    """The writer emits EXACTLY the three-field shape runbook.md §1
    documents — enabled, updated_at, updated_by — nothing else."""
    write_kill_switch(engaged=True, updated_by="operator", path=str(switch_path))
    doc = json.loads(switch_path.read_text(encoding="utf-8"))
    assert set(doc.keys()) == {"enabled", "updated_at", "updated_by"}
    assert doc["enabled"] is True
    assert doc["updated_by"] == "operator"
