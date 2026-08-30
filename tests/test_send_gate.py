"""Tests for the send gate (ticket B5): app/send_gate.py's §2.2 preflight,
one test per checklist item so a broken rule names itself, plus the
STRUCTURAL test that makes DRY_RUN a property of the code rather than a
configuration — no app/ module may import any mail transport, and
pyproject.toml may declare no mail-transport dependency.

The §2 finding is asserted as CORRECT behaviour: a real unverified contact
(email_verified=0) IS refused — get_targets hardcodes 0 on import and no
verification path exists, so the gate refuses every real target today,
DRY_RUN included (docs/gates.md §2.3a: the test must prove the gate works).
The allow path is exercised by seeding email_verified=1 (and the gate
columns) directly in the fixture — normal DB seeding, not a weakening.

Seeding here goes through write_gate.commit on purpose (the fixture is a
normal pipeline write), and the audit-trail tests rely on every seeded
core-table row having a write_log row.
"""

import ast  # the structural transport test: parse every app/ module and inspect its imports
import json  # parsing reasons_json / missing_requirements_json in the decision-row assertions
import tomllib  # stdlib TOML parser — the pyproject dependency check (Python 3.11+)
from pathlib import Path  # locating the app/ package and pyproject.toml from the test file
from typing import Any  # the docstring-skip helper's container annotation

import pytest  # fixtures, tmp_path, monkeypatch

from app.agents_registry import seed_agent_registry  # the five principals — the write gate refuses unregistered writers
from app.db import apply_schema, connect, normalize_email  # F1b: the shared suppression key helper the fixtures use to seed rows exactly as the writers do
from app.ids import new_id  # fresh ids for seeded rows
from app.kill_switch import write_kill_switch  # the switch writer — tests flip the tmp switch file the env var points at
from app.send_gate import (  # the gate under test, plus the vocabulary the assertions read
    DRY_RUN_STATUS,
    SendGateDecision,
    evaluate_send_gate,
)
from app.write_gate import commit  # every seeded core-table row goes through the gate, never a raw INSERT

# The unsubscribe token B3's deterministic footer carries — the seeded
# revisions use the real token so the draft-content check passes honestly.
UNSUBSCRIBE_FOOTER = "[unsubscribe: {UNSUBSCRIBE_URL}]"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def switch_path(tmp_path, monkeypatch):
    """A tmp kill-switch file, written DISENGAGED, and the env var pointing
    the gate's reader at it — so no test here reads the committed
    config/kill_switch.json, and engaging the switch is just a rewrite of
    this file (the B4a convention)."""
    path = tmp_path / "kill_switch.json"
    write_kill_switch(engaged=False, updated_by="fixture", path=str(path))
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(path))
    return path


@pytest.fixture
def conn(scratch_db_target, switch_path):
    """Fresh SQLite DB with schema + the seeded principals + one shared
    offer.  Targets are added per test via _seed_target, so each test's
    data is exactly what it needs and nothing else."""
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
    yield c
    c.close()


def _insert_draft_version(
    c, *, target_id: str, draft_version_id: str, revision_number: int,
    policy_check_passed=None, injection_scan_passed=None, footer=UNSUBSCRIBE_FOOTER,
) -> None:
    """Insert one message_draft_versions row through the write gate, the way
    B3's persist node does (agent-authored: edited_by=draft_writer; the
    three gate columns default to NULL — the B3-Z3 invariant).  insert_seq
    is written with the same scalar-subquery form the production writers
    use, so the fixture rows order exactly like real rows (ticket B5's
    determinism fix)."""
    commit(
        c, action="insert_message_draft_version", table_name="message_draft_versions",
        record_id=draft_version_id, payload={"revision_number": revision_number},
        run_id="r0", step_id="s0", actor="system", agent_id="draft_writer",
        sql="""INSERT INTO message_draft_versions
               (draft_version_id, target_id, message_id, revision_number, subject,
                body, footer, edited_by, policy_check_passed, injection_scan_passed,
                send_gate_passed, critique_passed, critique_json, insert_seq, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,
                       (SELECT COALESCE(MAX(insert_seq),0)+1 FROM message_draft_versions),
                       datetime('now'))""",
        params=(draft_version_id, target_id, None, revision_number,
                "Cold subject", "Cold body text.", footer, "draft_writer",
                policy_check_passed, injection_scan_passed, None, None, None),
    )


def _seed_target(c, *, target_id: str, **overrides) -> None:
    """Seed one target with EVERY §2.2 condition satisfied, plus its full
    FK chain (account, contact, signal, policy row, revision, review row) —
    all through the write gate.  Each override key breaks exactly one
    checklist item, so a per-rule test is `_seed_target(..., key=bad)` and
    the rest of the chain stays green.

    Override keys: email (contact address), email_verified (1/0),
    with_contact (False → contact_id NULL), state, fit_label, fit_score,
    signal_strength (None → no signal row), footer, policy_decision
    (None → no policy row), review_decision (None → no review row),
    review_edited (0/1), policy_check_passed, injection_scan_passed,
    draft_ref (the review row's draft reference).
    """
    account_id = f"acc_{target_id}"
    contact_id = f"con_{target_id}"
    email = overrides.get("email", f"jane@{target_id.split('_', 1)[-1]}.test")
    # The account: domain derived from the email's domain half so the
    # domain-suppression and domain-limit tests have a real domain to hit.
    domain = email.split("@", 1)[-1] if email else "noemail.test"
    commit(
        c, action="insert_account", table_name="accounts", record_id=account_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               industry, estimated_size, geo, company_summary, icp_fit_label, icp_fit_score,
               created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(account_id, "Acme", domain, domain, "Healthcare", "11-50", "HK",
                "A healthcare company.", overrides.get("fit_label", "strong_fit"),
                overrides.get("fit_score", 88)),
    )
    with_contact = overrides.get("with_contact", True)
    contact_id_seeded = contact_id if with_contact else None
    if with_contact:
        commit(
            c, action="insert_contact", table_name="contacts", record_id=contact_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
                   email_verified, created_at, updated_at)
                   VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(contact_id, account_id, "Jane Doe", email,
                    overrides.get("email_verified", 1)),
        )
    commit(
        c, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id,
               source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, account_id, contact_id_seeded, "off_1", "csv",
                overrides.get("state", "approved")),
    )
    # The signal: one strong entry on a run_id of its own (the gate reads
    # the LATEST run's signals).  strength=None → no signal row at all.
    signal_strength = overrides.get("signal_strength", 1.0)
    if signal_strength is not None:
        commit(
            c, action="insert_signal", table_name="signals", record_id=f"sig_{target_id}",
            payload={}, run_id="r0", step_id="s1", actor="system", agent_id="system",
            sql="""INSERT INTO signals (signal_id, run_id, target_id, signal_type,
                   signal_value, signal_strength, source_url, source_confidence,
                   evidence_quote, evidence_verified, evidence_tier, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
            params=(f"sig_{target_id}", "r0", target_id, "hiring_relevant_role",
                    "Hiring 3 SDRs", signal_strength, "https://acme.test/careers", 0.9,
                    "The careers page lists three open SDR roles.", 0, "findings"),
        )
    # The policy row: decision override, or no row at all (fail-closed case).
    # insert_seq uses the production writers' scalar-subquery form (B5).
    policy_decision = overrides.get("policy_decision", "allow")
    if policy_decision is not None:
        commit(
            c, action="insert_policy_decision", table_name="policy_decisions",
            record_id=f"pd_{target_id}", payload={}, run_id="r0", step_id="s1",
            actor="system", agent_id="system",
            sql="""INSERT INTO policy_decisions (policy_decision_id, run_id, step_id,
                   target_id, action, decision, risk_level, reasons_json,
                   matched_rules_json, missing_fields_json, insert_seq, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,
                           (SELECT COALESCE(MAX(insert_seq),0)+1 FROM policy_decisions),
                           datetime('now'))""",
            params=(f"pd_{target_id}", "r0", "s1", target_id, "score_lead",
                    policy_decision, "low", '["all clear"]', '["P3a"]', '[]'),
        )
    # The draft revision (gate columns default 1 = passed; the NULL tests
    # override them to None) and the review row referencing it.
    draft_ref = overrides.get("draft_ref", f"dv_{target_id}")
    _insert_draft_version(
        c, target_id=target_id, draft_version_id=f"dv_{target_id}", revision_number=1,
        policy_check_passed=overrides.get("policy_check_passed", 1),
        injection_scan_passed=overrides.get("injection_scan_passed", 1),
        footer=overrides.get("footer", UNSUBSCRIBE_FOOTER),
    )
    review_decision = overrides.get("review_decision", "approve")
    if review_decision is not None:
        # insert_seq uses the production writers' scalar-subquery form (B5):
        # rows seeded in this fixture order 1, 2, ... exactly like real rows,
        # so the gate's insert_seq DESC read is exercised, not bypassed.
        commit(
            c, action="insert_review_decision", table_name="review_decisions",
            record_id=f"rev_{target_id}", payload={}, run_id="r0", step_id="s0",
            actor="operator", agent_id="operator",
            sql="""INSERT INTO review_decisions (review_decision_id, run_id, target_id,
                   draft_message_id, decision, edited, reason, actor,
                   kill_switch_active, insert_seq, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,
                           (SELECT COALESCE(MAX(insert_seq),0)+1 FROM review_decisions),
                           datetime('now'))""",
            params=(f"rev_{target_id}", "r0", target_id, draft_ref, review_decision,
                    overrides.get("review_edited", 0), "", "operator", 0),
        )


def _seed_real_send(c, *, offer_id: str, email: str, thread_id=None,
                    created_at="datetime('now')", status="sent",
                    contact_id=None, target_id=None) -> str:
    """Seed one PRIOR REAL send — an outbound messages row with status
    'sent' (the status the rate-limit counters count).  Used by the §2.2a
    tests.  By default it creates its own account+contact+target chain
    (each call gets a unique domain from its email, so a mailbox-limit test
    can seed 20 sends without tripping the domain or cooldown checks of the
    target under evaluation); passing contact_id/target_id attaches the
    send to an EXISTING contact/target instead — what the cooldown and
    thread tests need, since those rules are per-contact."""
    if contact_id is None:
        # A fresh FK chain so this send is its own conversation: the
        # normalized_domain (UNIQUE) gets the random stem prefix, because a
        # domain-limit test legitimately seeds SEVERAL sends sharing one
        # email domain and only the email column may repeat.
        stem = new_id("seed")
        account_id = f"acc_{stem}"
        contact_id = f"con_{stem}"
        target_id = f"tgt_{stem}"
        domain = email.split("@", 1)[-1]
        commit(
            c, action="insert_account", table_name="accounts", record_id=account_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
                   created_at, updated_at) VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
            params=(account_id, "Seed Co", domain, f"{stem}.{domain}"),
        )
        commit(
            c, action="insert_contact", table_name="contacts", record_id=contact_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
                   email_verified, created_at, updated_at)
                   VALUES (?,?,?,?,1,datetime('now'),datetime('now'))""",
            params=(contact_id, account_id, "Seed Person", email),
        )
        commit(
            c, action="insert_target", table_name="targets", record_id=target_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id,
                   source, state, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(target_id, account_id, contact_id, offer_id, "csv", "sent"),
        )
    message_id = new_id("msg")
    # created_at is a parameter so a test can backdate a send outside the
    # 21-day cooldown window (the thread-rule tests need exactly that).
    commit(
        c, action="insert_message", table_name="messages", record_id=message_id,
        payload={"status": status}, run_id="r0", step_id="s0",
        actor="system", agent_id="system",
        sql=f"""INSERT INTO messages (message_id, target_id, contact_id, direction,
                 provider_message_id, thread_id, subject, body, body_redacted,
                 status, sent_at, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,{created_at})""",
        params=(message_id, target_id, contact_id, "outbound", None, thread_id,
                "Prior subject", "Prior body.", None, status,
                "2026-01-01 00:00:00",),  # sent_at: a real send has one
    )
    return contact_id


def _reasons_text(decision: SendGateDecision) -> str:
    """Join the decision's reasons into one string for substring assertions."""
    return " | ".join(decision.reasons)


def _decision_rows(c, target_id: str):
    return c.execute(
        "SELECT * FROM send_gate_decisions WHERE target_id=?;", (target_id,)
    ).fetchall()


# ── The allow path ───────────────────────────────────────────────────────────


def test_all_conditions_satisfied_is_allowed(conn):
    """Every §2.2 item satisfied → allowed=True, and the decision row
    records allowed=1 with simulated: true INSIDE reasons_json (never in
    the allowed column — it is INTEGER NOT NULL, gates.md §2.3a)."""
    _seed_target(conn, target_id="tgt_ok")
    decision = evaluate_send_gate(conn, target_id="tgt_ok", run_id="r1", step_id="s9")
    assert decision.allowed is True
    assert decision.suppression_hit is False
    assert decision.approval_verified is True
    assert decision.kill_switch_active is False
    rows = _decision_rows(conn, "tgt_ok")
    assert len(rows) == 1  # exactly one row per evaluation
    row = rows[0]
    assert row["allowed"] == 1
    assert row["suppression_hit"] == 0
    assert row["approval_verified"] == 1
    assert row["kill_switch_active"] == 0
    parsed = json.loads(row["reasons_json"])
    assert parsed["simulated"] is True  # the §2.3a marker, inside the JSON
    assert parsed["reasons"][0].startswith("all §2.2 preflight checks passed")


# ── One test per §2.2 checklist item, refused individually ───────────────────


def test_unknown_target_is_refused_without_decision_row(conn):
    """Checklist item 1: a phantom target is refused, and — the documented
    schema-forced exception — NO decision row is written, because
    send_gate_decisions.contact_id is NOT NULL and a phantom target has no
    contact to record.  The refusal still names the target."""
    decision = evaluate_send_gate(conn, target_id="tgt_ghost", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "unknown target" in _reasons_text(decision)
    assert _decision_rows(conn, "tgt_ghost") == []


def test_target_without_contact_is_refused_without_decision_row(conn):
    """Checklist item 2's sibling (company-only lead, contact_id NULL —
    allowed by CSV import): refused, and the second schema-forced exception
    — no decision row (no contact_id to write)."""
    _seed_target(conn, target_id="tgt_nocontact", with_contact=False)
    decision = evaluate_send_gate(conn, target_id="tgt_nocontact", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "no contact" in _reasons_text(decision)
    assert _decision_rows(conn, "tgt_nocontact") == []


def test_missing_contact_email_is_refused(conn):
    """Checklist item 2: a contact row with no email is refused, with the
    decision row written (the contact_id exists to record)."""
    _seed_target(conn, target_id="tgt_noemail", email=None)
    decision = evaluate_send_gate(conn, target_id="tgt_noemail", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "no email address" in _reasons_text(decision)
    assert len(_decision_rows(conn, "tgt_noemail")) == 1


def test_unverified_email_is_refused(conn):
    """Checklist item 3 — THE §2 FINDING, asserted as CORRECT behaviour:
    a real unverified contact (email_verified=0, which get_targets writes
    for every CSV import) IS refused, DRY_RUN included.  The gate must not
    wave unverified addresses through just because nothing is really sent
    — that would prove nothing about the gate (gates.md §2.3a)."""
    _seed_target(conn, target_id="tgt_unverified", email_verified=0)
    decision = evaluate_send_gate(conn, target_id="tgt_unverified", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "email_verified" in _reasons_text(decision)
    assert "contact.email_verified == true" in decision.missing_requirements


def test_suppressed_email_is_refused(conn):
    """Checklist item 4: an email on the suppression list is a hard refusal
    and sets suppression_hit on the decision row."""
    _seed_target(conn, target_id="tgt_supemail")
    commit(
        conn, action="insert_suppression", table_name="suppressions",
        record_id="jane@supemail.test", payload={"reason": "manual", "added_by": "operator"},
        run_id="r0", step_id="s0", actor="operator", agent_id="operator",
        sql="""INSERT INTO suppressions (email, email_normalized, domain, reason, added_at, added_by, notes)
               VALUES (?,?,?,?,datetime('now'),?,?)""",
        params=("jane@supemail.test", normalize_email("jane@supemail.test"), None, "manual", "operator", None),
    )
    decision = evaluate_send_gate(conn, target_id="tgt_supemail", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert decision.suppression_hit is True
    assert "suppression list" in _reasons_text(decision)
    assert "contact.email not in suppressions" in decision.missing_requirements


def test_suppressed_domain_is_refused(conn):
    """Checklist item 5: a suppressed domain refuses every address under it
    and sets suppression_hit."""
    _seed_target(conn, target_id="tgt_supdomain")
    commit(
        conn, action="insert_suppression", table_name="suppressions",
        record_id="domain:supdomain.test", payload={"reason": "manual", "added_by": "operator"},
        run_id="r0", step_id="s0", actor="operator", agent_id="operator",
        sql="""INSERT INTO suppressions (email, domain, reason, added_at, added_by, notes)
               VALUES (?,?,?,datetime('now'),?,?)""",
        params=(None, "supdomain.test", "manual", "operator", None),
    )
    decision = evaluate_send_gate(conn, target_id="tgt_supdomain", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert decision.suppression_hit is True
    assert "domain" in _reasons_text(decision)
    assert "contact.domain not in suppressions" in decision.missing_requirements


# ── F1b: suppression matching is normalised ─────────────────────────────────


def _seed_suppression(c, *, email: str | None = None, domain: str | None = None) -> None:
    """Seed one suppression row the way the F1b writers now do: the email is
    preserved AS WRITTEN and email_normalized is computed by the ONE shared
    helper — so the fixtures cannot drift from production write shape."""
    record_id = email if email is not None else f"domain:{domain}"
    commit(
        c, action="insert_suppression", table_name="suppressions",
        record_id=record_id, payload={"reason": "manual", "added_by": "operator"},
        run_id="r0", step_id="s0", actor="operator", agent_id="operator",
        sql="""INSERT INTO suppressions (email, email_normalized, domain, reason, added_at, added_by, notes)
               VALUES (?,?,?,?,datetime('now'),?,?)""",
        params=(email, normalize_email(email) if email else None, domain,
                "manual", "operator", None),
    )


@pytest.mark.parametrize("stored_email,probe_email", [
    # Probe variants against a canonical stored address (read-side folding).
    ("dr.chan@serenity-clinic.test", "Dr.Chan@serenity-clinic.test"),
    ("dr.chan@serenity-clinic.test", "dr.chan+alias@serenity-clinic.test"),
    ("dr.chan@serenity-clinic.test", "DR.CHAN@SERENITY-CLINIC.TEST"),
    # Stored variants against a canonical probe (write-side folding).
    ("Dr.Chan@serenity-clinic.test", "dr.chan@serenity-clinic.test"),
    ("DR.CHAN@SERENITY-CLINIC.TEST", "dr.chan@serenity-clinic.test"),
    ("dr.chan+alias@serenity-clinic.test", "dr.chan@serenity-clinic.test"),
])
def test_suppression_blocks_address_variants_in_both_directions(conn, stored_email, probe_email):
    """F1b §4: each variant class from the threat-model table, in BOTH
    directions — a suppression stored in any casing blocks a probe in any
    casing, and a plus-tag alias of a suppressed mailbox is blocked."""
    _seed_suppression(conn, email=stored_email)
    _seed_target(conn, target_id="tgt_variant", email=probe_email)
    decision = evaluate_send_gate(conn, target_id="tgt_variant", run_id="r1", step_id="s9")
    assert decision.suppression_hit is True
    assert decision.allowed is False
    assert "contact.email not in suppressions" in decision.missing_requirements


@pytest.mark.parametrize("probe_email", [
    "dr.chan@other-clinic.test",        # same local part, DIFFERENT domain
    "dr.chang@serenity-clinic.test",    # different local part, same domain
    # Lead-added after an F1b sabotage: an over-broad normaliser that ALSO
    # stripped dots from the local part was caught by only one incidental
    # test, because neither case above differs by a dot.  Gmail folds dots;
    # almost no other provider does, so folding them here would silently
    # suppress a different person on every non-Gmail domain.  This pins
    # "dots are NOT folded" so that change has to be deliberate.
    "drchan@serenity-clinic.test",      # same letters, dots removed — a DIFFERENT mailbox
])
def test_suppression_does_not_block_different_people(conn, probe_email):
    """F1b §4 regression that matters most: the normaliser must NOT
    over-broaden into blocking different people.  A suppression on
    dr.chan@serenity-clinic.test leaves these two fully-sendable."""
    _seed_suppression(conn, email="dr.chan@serenity-clinic.test")
    _seed_target(conn, target_id="tgt_other", email=probe_email)
    decision = evaluate_send_gate(conn, target_id="tgt_other", run_id="r1", step_id="s9")
    assert decision.suppression_hit is False
    assert decision.allowed is True  # still a fully-green target


def test_suppressed_domain_matches_probe_with_mixed_case(conn):
    """F1b §4 domain half: a stored lowercase domain suppresses a probe whose
    domain half is mixed-case (the gate folds the probe via normalize_domain)."""
    _seed_suppression(conn, domain="supdomain.test")
    _seed_target(conn, target_id="tgt_mixed_domain", email="jane@SUPDOMAIN.TEST")
    decision = evaluate_send_gate(conn, target_id="tgt_mixed_domain", run_id="r1", step_id="s9")
    assert decision.suppression_hit is True
    assert "contact.domain not in suppressions" in decision.missing_requirements


def test_non_fit_label_is_refused(conn):
    """Checklist item 6: the non-fit label (the repo's vocabulary names the
    checklist's 'not_fit' tier 'not_target') refuses the send."""
    _seed_target(conn, target_id="tgt_notfit", fit_label="not_target")
    decision = evaluate_send_gate(conn, target_id="tgt_notfit", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "not_target" in _reasons_text(decision)


def test_fit_score_below_floor_is_refused(conn):
    """Checklist item 7: fit_score below 60 is a hard refusal (59 refuses;
    60 would pass — the floor is >= 60)."""
    _seed_target(conn, target_id="tgt_lowscore", fit_score=59)
    decision = evaluate_send_gate(conn, target_id="tgt_lowscore", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "below the 60" in _reasons_text(decision)


def test_no_strong_signal_is_refused(conn):
    """Checklist item 8: the latest run's signals must include at least one
    with strength >= 0.6 — weak-only and absent signals both refuse."""
    _seed_target(conn, target_id="tgt_weaksignal", signal_strength=0.3)
    decision = evaluate_send_gate(conn, target_id="tgt_weaksignal", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "strength" in _reasons_text(decision)
    _seed_target(conn, target_id="tgt_nosignal", signal_strength=None)
    decision = evaluate_send_gate(conn, target_id="tgt_nosignal", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "no signals" in _reasons_text(decision)


def test_footer_without_unsubscribe_token_is_refused(conn):
    """Checklist item 9: the deterministic footer must carry the
    unsubscribe token — a footer without it (a mangled or missing
    compliance footer) refuses the send."""
    _seed_target(conn, target_id="tgt_badfooter", footer="No unsubscribe here.")
    decision = evaluate_send_gate(conn, target_id="tgt_badfooter", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "unsubscribe token" in _reasons_text(decision)


def test_policy_check_passed_null_is_refused(conn):
    """Checklist item 11: policy_check_passed NULL means 'no check has run',
    which is NOT 'passed' — fail closed.  (Also a structural gap: no
    draft-content policy runner exists in the repo, so every real revision
    is NULL here and correctly refused.)"""
    _seed_target(conn, target_id="tgt_nullpolicy", policy_check_passed=None)
    decision = evaluate_send_gate(conn, target_id="tgt_nullpolicy", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "policy_check_passed" in _reasons_text(decision)
    assert "draft passed length and content policy" in decision.missing_requirements


def test_injection_scan_passed_null_is_refused(conn):
    """Checklist item 10: injection_scan_passed NULL — the Guardrails AI
    scanner (open-questions.md item 8) is not implemented — fails closed
    as a missing requirement.  No fake scanner was written, so 'scan not
    run' can never masquerade as a pass."""
    _seed_target(conn, target_id="tgt_nullinjection", injection_scan_passed=None)
    decision = evaluate_send_gate(conn, target_id="tgt_nullinjection", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "injection_scan_passed" in _reasons_text(decision)
    assert "draft passed the prompt-injection scan" in decision.missing_requirements


def test_no_review_approval_is_refused(conn):
    """Checklist item 12 — the CLAUDE.md §9 test: NO review_decisions row
    means no send, even though the target sits in state 'approved'.  The
    approval is a recorded operator decision, never implied by state."""
    _seed_target(conn, target_id="tgt_noreview", review_decision=None)
    decision = evaluate_send_gate(conn, target_id="tgt_noreview", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert decision.approval_verified is False
    assert "review_decisions" in _reasons_text(decision)


def test_review_rejection_is_refused(conn):
    """Checklist item 12's other half: a recorded NON-approval decision
    (reject) refuses the send even if the target's state says approved."""
    _seed_target(conn, target_id="tgt_rejected", review_decision="reject")
    decision = evaluate_send_gate(conn, target_id="tgt_rejected", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "not approve/approve_with_edits" in _reasons_text(decision)


def test_target_state_not_approved_is_refused(conn):
    """Checklist item 12's third half: an approval row that has NOT taken
    effect (the target never moved to approved) refuses — the recorded
    decision and the live state must agree."""
    _seed_target(conn, target_id="tgt_wrongstate", state="awaiting_review")
    decision = evaluate_send_gate(conn, target_id="tgt_wrongstate", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "not approved" in _reasons_text(decision)


def test_no_policy_decision_is_refused(conn):
    """Checklist item 13: no policy_decisions row at all → fail closed
    (policy-matrix.md: an unmapped action resolves to deny)."""
    _seed_target(conn, target_id="tgt_nopolicy", policy_decision=None)
    decision = evaluate_send_gate(conn, target_id="tgt_nopolicy", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "no policy_decisions" in _reasons_text(decision)


def test_policy_deny_is_refused(conn):
    """Checklist item 13's other half: a current decision that is NOT allow
    (deny) refuses the send."""
    _seed_target(conn, target_id="tgt_policy_deny", policy_decision="deny")
    decision = evaluate_send_gate(conn, target_id="tgt_policy_deny", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "not allow" in _reasons_text(decision)


def test_kill_switch_engaged_is_refused(conn, switch_path):
    """Checklist item 14: an engaged switch refuses unconditionally (P6
    dominates — the other checks are not even evaluated, mirroring
    policy_check_phase1), and the decision row records
    kill_switch_active=1."""
    _seed_target(conn, target_id="tgt_killswitch")
    write_kill_switch(engaged=True, updated_by="test", path=str(switch_path))
    decision = evaluate_send_gate(conn, target_id="tgt_killswitch", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert decision.kill_switch_active is True
    assert "kill switch engaged" in _reasons_text(decision)
    row = _decision_rows(conn, "tgt_killswitch")[0]
    assert row["kill_switch_active"] == 1
    assert row["allowed"] == 0


# ── §2.2a rate limits, one test per rule ─────────────────────────────────────


def test_mailbox_daily_limit_is_refused(conn):
    """§2.2a per-mailbox daily limit: 20 real sends on the target's offer in
    the last 24h refuse the next send.  The seeds use distinct contacts and
    domains so ONLY the mailbox rule fires (no cooldown, no domain limit)."""
    _seed_target(conn, target_id="tgt_mailbox_daily")
    for i in range(20):
        _seed_real_send(conn, offer_id="off_1", email=f"s{i}@seed{i}.test")
    decision = evaluate_send_gate(conn, target_id="tgt_mailbox_daily", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "per-mailbox daily" in _reasons_text(decision)


def test_mailbox_hourly_limit_is_refused(conn):
    """§2.2a per-mailbox hourly limit: 5 real sends in the last hour refuse
    (5 is under the 20/day limit, so the hourly rule is what fires)."""
    _seed_target(conn, target_id="tgt_mailbox_hourly")
    for i in range(5):
        _seed_real_send(conn, offer_id="off_1", email=f"s{i}@seed{i}.test")
    decision = evaluate_send_gate(conn, target_id="tgt_mailbox_hourly", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "per-mailbox hourly" in _reasons_text(decision)


def test_domain_daily_limit_is_refused(conn):
    """§2.2a per-domain daily limit: 2 real sends to the target's domain in
    the last 24h refuse (2 is under the mailbox limits, so the domain rule
    is what fires)."""
    _seed_target(conn, target_id="tgt_domain_limit")  # contact: jane@domain_limit.test
    _seed_real_send(conn, offer_id="off_1", email="a@domain_limit.test")
    _seed_real_send(conn, offer_id="off_1", email="b@domain_limit.test")
    decision = evaluate_send_gate(conn, target_id="tgt_domain_limit", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "per-domain daily" in _reasons_text(decision)


def test_contact_cooldown_is_refused(conn):
    """§2.2a per-contact cooldown: one real send to the SAME contact in the
    last 21 days refuses (a single send is under every other limit, so the
    cooldown is what fires).  The seeded send attaches to the target's own
    contact — the cooldown is per contact, not per email address."""
    _seed_target(conn, target_id="tgt_cooldown")  # contact: con_tgt_cooldown / jane@cooldown.test
    _seed_real_send(
        conn, offer_id="off_1", email="jane@cooldown.test",
        contact_id="con_tgt_cooldown", target_id="tgt_cooldown",
    )
    decision = evaluate_send_gate(conn, target_id="tgt_cooldown", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "cooldown" in _reasons_text(decision)


def test_unanswered_thread_blocks_second_send(conn):
    """§2.2a per-thread rule: an outbound on a thread that has received NO
    reply blocks the next unprompted send — with no time window (the seed
    is backdated 30 days, outside every OTHER limit, so the thread rule is
    what fires).  The seeded send attaches to the target's own contact: a
    thread belongs to a conversation, not to the whole mailbox."""
    _seed_target(conn, target_id="tgt_thread")  # contact: con_tgt_thread
    _seed_real_send(
        conn, offer_id="off_1", email="jane@thread.test", thread_id="thr_1",
        created_at="'2026-07-01 00:00:00'",  # 30+ days back: cooldown/mailbox/domain windows all pass
        contact_id="con_tgt_thread", target_id="tgt_thread",
    )
    decision = evaluate_send_gate(conn, target_id="tgt_thread", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "unanswered outbound" in _reasons_text(decision)


def test_replied_thread_allows_send(conn):
    """§2.2a per-thread rule's other half: once a reply lands on the
    thread, the block clears (the backdated outbound also sits outside
    every other window, so the target is otherwise fully allowed)."""
    _seed_target(conn, target_id="tgt_thread_replied")  # contact: con_tgt_thread_replied
    _seed_real_send(
        conn, offer_id="off_1", email="jane@thread_replied.test", thread_id="thr_1",
        created_at="'2026-07-01 00:00:00'",
        contact_id="con_tgt_thread_replied", target_id="tgt_thread_replied",
    )
    # The inbound reply on the same thread — direction inbound, so it never
    # counts against any outbound limit.
    commit(
        conn, action="insert_message", table_name="messages", record_id="msg_reply",
        payload={"direction": "inbound"}, run_id="r0", step_id="s0",
        actor="system", agent_id="system",
        sql="""INSERT INTO messages (message_id, target_id, contact_id, direction,
               provider_message_id, thread_id, subject, body, body_redacted,
               status, sent_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("msg_reply", "tgt_thread_replied", "con_tgt_thread_replied",
                "inbound", None, "thr_1", "Re: Cold subject", "Reply text.",
                None, "received", None),
    )
    decision = evaluate_send_gate(conn, target_id="tgt_thread_replied", run_id="r1", step_id="s9")
    assert decision.allowed is True


# ── approve_with_edits re-check (§3.1 / human-review.md §5) ──────────────────


def test_approve_with_edits_with_null_gate_columns_is_refused(conn):
    """The edited revision must independently re-pass policy + injection
    (human-review.md §5).  B4b writes the edit's gate columns NULL, so a
    NULL edit refuses even though the ORIGINAL revision (dv_1) has
    policy/injection = 1 — the original's passes never transfer to the
    edit (NULL is not 'passed')."""
    _seed_target(conn, target_id="tgt_edited_null")
    _insert_draft_version(
        conn, target_id="tgt_edited_null", draft_version_id="dv_edited",
        revision_number=2, policy_check_passed=None, injection_scan_passed=None,
    )
    commit(
        conn, action="insert_review_decision", table_name="review_decisions",
        record_id="rev_edited", payload={}, run_id="r0", step_id="s0",
        actor="operator", agent_id="operator",
        sql="""INSERT INTO review_decisions (review_decision_id, run_id, target_id,
               draft_message_id, decision, edited, reason, actor,
               kill_switch_active, insert_seq, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,
                       (SELECT COALESCE(MAX(insert_seq),0)+1 FROM review_decisions),
                       datetime('now'))""",
        params=("rev_edited", "r0", "tgt_edited_null", "dv_edited",
                "approve_with_edits", 1, "", "operator", 0),
    )
    decision = evaluate_send_gate(conn, target_id="tgt_edited_null", run_id="r1", step_id="s9")
    assert decision.allowed is False
    reasons = _reasons_text(decision)
    assert "policy_check_passed" in reasons
    assert "injection_scan_passed" in reasons


def test_approve_with_edits_with_repassed_columns_is_allowed(conn):
    """The edited revision that HAS independently passed policy + injection
    (its own columns = 1) is allowed — the re-pass is read from the EDITED
    revision's columns, never inherited from the original."""
    _seed_target(conn, target_id="tgt_edited_ok")
    _insert_draft_version(
        conn, target_id="tgt_edited_ok", draft_version_id="dv_edited",
        revision_number=2, policy_check_passed=1, injection_scan_passed=1,
    )
    commit(
        conn, action="insert_review_decision", table_name="review_decisions",
        record_id="rev_edited", payload={}, run_id="r0", step_id="s0",
        actor="operator", agent_id="operator",
        sql="""INSERT INTO review_decisions (review_decision_id, run_id, target_id,
               draft_message_id, decision, edited, reason, actor,
               kill_switch_active, insert_seq, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,
                       (SELECT COALESCE(MAX(insert_seq),0)+1 FROM review_decisions),
                       datetime('now'))""",
        params=("rev_edited", "r0", "tgt_edited_ok", "dv_edited",
                "approve_with_edits", 1, "", "operator", 0),
    )
    decision = evaluate_send_gate(conn, target_id="tgt_edited_ok", run_id="r1", step_id="s9")
    assert decision.allowed is True


def test_approved_revision_not_latest_is_refused(conn):
    """The operator approved a specific text: if a NEWER revision exists,
    the approved text is not what the send would deliver — refused, even
    though the referenced revision itself passes every gate."""
    _seed_target(conn, target_id="tgt_stale_ref")
    # A newer revision (dv_2) that the review row does NOT reference.
    _insert_draft_version(
        conn, target_id="tgt_stale_ref", draft_version_id="dv_2", revision_number=2,
        policy_check_passed=1, injection_scan_passed=1,
    )
    decision = evaluate_send_gate(conn, target_id="tgt_stale_ref", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert "did not approve" in _reasons_text(decision)


# ── The refusal side effects and the audit trail ─────────────────────────────


def test_refusal_writes_decision_row_but_nothing_else(conn):
    """A refusal writes exactly one send_gate_decisions row (allowed=0) and
    NOTHING else: no messages row, no state change, no transition — the
    target stays in approved so a fixed condition can be retried."""
    _seed_target(conn, target_id="tgt_refused_sideeffects", email_verified=0)
    decision = evaluate_send_gate(conn, target_id="tgt_refused_sideeffects", run_id="r1", step_id="s9")
    assert decision.allowed is False
    rows = _decision_rows(conn, "tgt_refused_sideeffects")
    assert len(rows) == 1
    assert rows[0]["allowed"] == 0
    # The specific reasons are recorded — not a bare refusal.
    parsed = json.loads(rows[0]["reasons_json"])
    assert parsed["simulated"] is False
    assert any("email_verified" in r for r in parsed["reasons"])
    # No messages row, no transition, state untouched.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE target_id='tgt_refused_sideeffects';"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_refused_sideeffects';"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_refused_sideeffects';"
    ).fetchone()["state"] == "approved"


def test_every_decision_row_is_gated(conn):
    """The audit-trail guarantee: every send_gate_decisions row (allow AND
    refuse) has a matching write_log row with action
    insert_send_gate_decision and the same record_id — a raw conn.execute
    replacing the gated write would leave a decision row with no audit row
    and this test fails."""
    _seed_target(conn, target_id="tgt_audit_ok")
    _seed_target(conn, target_id="tgt_audit_refused", email_verified=0)
    evaluate_send_gate(conn, target_id="tgt_audit_ok", run_id="r1", step_id="s9")
    evaluate_send_gate(conn, target_id="tgt_audit_refused", run_id="r1", step_id="s9")
    for row in conn.execute("SELECT send_gate_id FROM send_gate_decisions;").fetchall():
        audit = conn.execute(
            "SELECT 1 FROM write_log WHERE record_id=? AND action='insert_send_gate_decision';",
            (row["send_gate_id"],),
        ).fetchone()
        assert audit is not None, f"decision row {row['send_gate_id']} has no write_log row"


# ── H6: policy rule IDs are recorded on the decision row ────────────────────
# Ticket H6 makes send_gate_decisions carry the policy-matrix rule ID behind
# each refusal in matched_rules_json (same shape as policy_decisions), so an
# audit query can answer "every send P2 refused" without parsing prose.  Each
# test breaks exactly one checklist item and asserts the decision row's
# matched_rules_json names that item's rule — plus the multi-rule and allow
# cases.  These are ADDITIONS to the suite; the pre-H6 assertions above are
# untouched, which is the proof that H6 is attribution only.


def test_suppression_refusal_records_p2(conn):
    """H6: a suppression refusal records P2 in matched_rules_json (the rule
    that 'any target present in suppressions → deny')."""
    _seed_target(conn, target_id="tgt_h6_p2")  # contact: jane@h6_p2.test
    _seed_suppression(conn, email="jane@h6_p2.test")
    decision = evaluate_send_gate(conn, target_id="tgt_h6_p2", run_id="r1", step_id="s9")
    assert decision.allowed is False
    row = _decision_rows(conn, "tgt_h6_p2")[0]
    assert json.loads(row["matched_rules_json"]) == ["P2"]


def test_suppression_refusal_returns_p2_on_the_object(conn):
    """H6 (review gate): the RETURNED decision object carries matched_rules —
    not just the DB row — so a caller handling the refusal (e.g.
    app/tools/send_email.py, which builds its refusal payload from the object
    and never re-queries the table) sees the rule without its own SQL.  This
    is the PolicyGateDecision.matched_rules parity H6 claims."""
    _seed_target(conn, target_id="tgt_h6_obj_p2")  # contact: jane@h6_obj_p2.test
    _seed_suppression(conn, email="jane@h6_obj_p2.test")
    decision = evaluate_send_gate(conn, target_id="tgt_h6_obj_p2", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert decision.matched_rules == ["P2"]


def test_kill_switch_refusal_records_p6(conn, switch_path):
    """H6: a kill-switch refusal records P6 in matched_rules_json (the rule
    that dominates — engaged switch denies unconditionally)."""
    _seed_target(conn, target_id="tgt_h6_p6")
    write_kill_switch(engaged=True, updated_by="test", path=str(switch_path))
    decision = evaluate_send_gate(conn, target_id="tgt_h6_p6", run_id="r1", step_id="s9")
    assert decision.allowed is False
    row = _decision_rows(conn, "tgt_h6_p6")[0]
    assert json.loads(row["matched_rules_json"]) == ["P6"]


def test_kill_switch_refusal_returns_p6_on_the_object(conn, switch_path):
    """H6 (review gate): the kill-switch early-return path returns P6 on the
    RETURNED object, not just in the DB row — the object is what the caller
    handles, so the attribution must be visible without a re-query."""
    _seed_target(conn, target_id="tgt_h6_obj_p6")
    write_kill_switch(engaged=True, updated_by="test", path=str(switch_path))
    decision = evaluate_send_gate(conn, target_id="tgt_h6_obj_p6", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert decision.matched_rules == ["P6"]


def test_missing_approval_refusal_records_p1(conn):
    """H6: a missing-approval refusal records P1 (the rule that a send_email
    without an operator approved state → deny)."""
    _seed_target(conn, target_id="tgt_h6_p1", review_decision=None)
    decision = evaluate_send_gate(conn, target_id="tgt_h6_p1", run_id="r1", step_id="s9")
    assert decision.allowed is False
    assert decision.approval_verified is False
    row = _decision_rows(conn, "tgt_h6_p1")[0]
    assert json.loads(row["matched_rules_json"]) == ["P1"]


def test_fit_score_refusal_records_p4(conn):
    """H6: a below-floor fit_score refusal records P4 (the rule that
    fit_score < 60 → deny)."""
    _seed_target(conn, target_id="tgt_h6_p4", fit_score=59)
    decision = evaluate_send_gate(conn, target_id="tgt_h6_p4", run_id="r1", step_id="s9")
    assert decision.allowed is False
    row = _decision_rows(conn, "tgt_h6_p4")[0]
    assert json.loads(row["matched_rules_json"]) == ["P4"]


def test_rate_limit_refusal_records_p7(conn):
    """H6: a §2.2a rate-limit refusal records P7 (the send-side analogue of
    the rule that the same target cannot be auto-sent/resend more than N times
    per rolling window)."""
    _seed_target(conn, target_id="tgt_h6_p7")  # offer off_1
    for i in range(20):  # 20 real sends on off_1 in the last 24h → mailbox-daily fires
        _seed_real_send(conn, offer_id="off_1", email=f"s{i}@seed{i}.test")
    decision = evaluate_send_gate(conn, target_id="tgt_h6_p7", run_id="r1", step_id="s9")
    assert decision.allowed is False
    row = _decision_rows(conn, "tgt_h6_p7")[0]
    assert json.loads(row["matched_rules_json"]) == ["P7"]


def test_multi_rule_refusal_records_all_rules(conn):
    """H6: a refusal that trips several rules records ALL of them, not just
    the first.  The seeded target fails P3 (email_verified=0), P4 (fit_score
    59) and P1 (no review row) — the decision row must name all three."""
    _seed_target(
        conn, target_id="tgt_h6_multi",
        email_verified=0, fit_score=59, review_decision=None,
    )
    decision = evaluate_send_gate(conn, target_id="tgt_h6_multi", run_id="r1", step_id="s9")
    assert decision.allowed is False
    row = _decision_rows(conn, "tgt_h6_multi")[0]
    rules = json.loads(row["matched_rules_json"])
    assert sorted(rules) == ["P1", "P3", "P4"]


def test_allow_records_empty_rule_list(conn):
    """H6: an allow records an empty rule list in matched_rules_json —
    matching the policy_decisions allow precedent (app/policy.py appends
    nothing to matched_rules when no rule fires)."""
    _seed_target(conn, target_id="tgt_h6_allow")
    decision = evaluate_send_gate(conn, target_id="tgt_h6_allow", run_id="r1", step_id="s9")
    assert decision.allowed is True
    row = _decision_rows(conn, "tgt_h6_allow")[0]
    assert json.loads(row["matched_rules_json"]) == []


# ── THE STRUCTURAL TEST — DRY_RUN by construction, not by configuration ──────
# This is what makes the no-real-email rule enforceable: adding a transport
# becomes a deliberate edit to this test, never a config flip.  Modelled on
# test_console_cannot_import_any_write_path (denylist with specific
# messages) plus the docstring-skipping helper so prose may discuss the
# transports by name while only code-level strings are scanned.


def _app_python_files() -> list[Path]:
    # Every .py file under app/ — the whole package is the enforcement
    # surface, not just the send modules (a transport hidden in an unrelated
    # module is still a transport).
    app_dir = Path(__file__).resolve().parent.parent / "app"
    assert app_dir.is_dir(), "app/ missing — did the package move?"
    files = sorted(app_dir.rglob("*.py"))
    assert files, "no .py files found under app/"
    return files


def _docstring_constant_ids(tree: ast.Module) -> set[int]:
    # A module/class/function's first statement, when it is a bare string,
    # is its docstring.  Collect those Constant nodes' ids so the
    # string-literal scan below skips prose: only strings that could
    # actually execute (e.g. the argument of a dynamic
    # importlib.import_module(...) call) are checked, while docstrings stay
    # free to discuss the forbidden transports by name.
    containers: list[Any] = [tree]
    containers.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    ids: set[int] = set()
    for container in containers:
        body = getattr(container, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


# The mail-transport module roots no app/ module may import.  Each entry is
# a deliberate enforcement decision:
# - smtplib / aiosmtplib / poplib / imaplib / smtpd — the stdlib and async
#   SMTP/IMAP/POP transports
# - googleapiclient / google_auth_oauthlib / google.oauth2 — the Gmail API
#   client stack (discovery client + OAuth flow + credentials)
# - yagmail / redmail / sendgrid / mailgun / exchangelib / imapclient /
#   imbox — the common third-party mail SDKs
# google.auth itself is deliberately NOT listed: it is generic Google auth,
# not a mail client, and banning it would be scope creep.  Adding any entry
# here (or importing any of these) is a deliberate test edit.
_FORBIDDEN_TRANSPORT_MODULES = (
    "smtplib",
    "aiosmtplib",
    "poplib",
    "imaplib",
    "smtpd",
    "googleapiclient",
    "google_auth_oauthlib",
    "google.oauth2",
    "yagmail",
    "redmail",
    "sendgrid",
    "mailgun",
    "exchangelib",
    "imapclient",
    "imbox",
)


def test_app_imports_no_mail_transport():
    """Structural guarantee, not behavioural: walk every Import/ImportFrom
    node in every app/ module and fail on any mail-transport module.  A
    behavioural test could pass a module that imports a transport and
    simply has not been asked to send yet; this test fails the moment the
    import appears, so the package is UNABLE to send even in principle.
    Two scans:
    1. Import/ImportFrom nodes for the banned module roots (both spellings
       — `import smtplib` and `from smtplib import SMTP` — are caught).
    2. Non-docstring string literals containing a banned module name —
       this catches the dynamic-import bypass (`importlib.import_module`,
       `getattr(importlib, ...)`) where the module name never appears as an
       Import node.  Docstrings are skipped (see _docstring_constant_ids)
       so prose may name the transports; executable strings may not.
    """
    for path in _app_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        skip = _docstring_constant_ids(tree)
        for node in ast.walk(tree):
            # ── Scan 1: static imports ─────────────────────────────────
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                imported = []
            for name in imported:
                for forbidden in _FORBIDDEN_TRANSPORT_MODULES:
                    assert not (
                        name == forbidden or name.startswith(forbidden + ".")
                    ), (
                        f"{path.name} imports {name!r} — a mail transport. "
                        f"No real email may ever leave this repository; the "
                        f"only send is the DRY_RUN .eml write.  Removing this "
                        f"import is the fix, not adding it to an allowlist."
                    )
            # ── Scan 2: dynamic-import strings ─────────────────────────
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in skip
            ):
                for forbidden in _FORBIDDEN_TRANSPORT_MODULES:
                    assert forbidden not in node.value, (
                        f"{path.name} contains the string literal {node.value!r} "
                        f"which names transport module {forbidden!r} — a dynamic "
                        f"import (e.g. importlib.import_module) would bypass the "
                        f"static scan.  Executable strings must not name mail "
                        f"transports."
                    )


def test_pyproject_declares_no_mail_transport_dependency():
    """The second half of the structural guarantee: pyproject.toml declares
    no mail-transport dependency — adding a transport SDK is a deliberate
    edit to this denylist, and the dependency list is the surface where
    such a decision would land."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    # Dep strings carry version specifiers ("google-adk==2.7.1") — split on
    # the first specifier character to get the bare distribution name.
    import re

    for dep in deps:
        name = re.split(r"[<>=!~\[;]", dep.strip(), maxsplit=1)[0].strip().lower()
        assert name not in {
            "aiosmtplib", "yagmail", "redmail", "sendgrid", "mailgun",
            "mailjet", "google-api-python-client", "google-auth-oauthlib",
            "exchangelib", "imapclient", "imbox", "emails",
        }, (
            f"pyproject.toml declares mail-transport dependency {name!r} — "
            f"no real email may ever leave this repository; the only send is "
            f"the DRY_RUN .eml write"
        )


def test_dry_run_status_is_the_rate_limit_exclusion(conn):
    """The §2.3a exemption is the counter filter itself: the DRY_RUN status
    is a module constant the messages row and the rate-limit queries share,
    so a dry-run row can never drift into the counters.  (Guarded here so a
    rename that split the two would fail loudly.)"""
    assert DRY_RUN_STATUS == "dry_run_sent"
