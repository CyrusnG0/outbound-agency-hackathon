"""Tests for the DRY_RUN send (ticket B5): app/tools/send_email.py — the
ONLY send that exists in this repository, and the one place where an
approved draft becomes an outbox artifact, a messages row, and a state hop.

WHAT IS PROVEN HERE (docs/gates.md §2.3a, ticket §3.1):
1.  An allowed send writes exactly one .eml, one messages row with
    sent_at NULL, and transitions approved → dry_run_sent.
2.  The .eml is well-formed RFC-5322 and carries the subject, the body, and
    the deterministic footer's unsubscribe token.
3.  A refused send writes NO file, NO messages row, and NO transition.
4.  A DRY_RUN send consumes NO rate limit — proven behaviourally: after the
    dry-run send, the per-mailbox, per-domain and per-contact-cooldown
    counters all still allow what they would allow without it.
5.  No OutcomeRecord is initialized for a dry_run_sent target (no outcome
    table exists in the DDL at all — §2.3a is structurally true).
6.  A filesystem failure leaves no inconsistent state: the file-first write
    order guarantees a dry_run_sent target always has its artifact, and a
    DB failure after the file write leaves only the documented, harmless
    orphan artifact plus an honestly-approved target.
7.  The audit trail: every messages row has a matching write_log row
    (action insert_message, same record_id).
8.  A target not in approved is refused and logged.

Seeding here goes through write_gate.commit on purpose (fixtures are normal
pipeline writes), and every test points the outbox at a tmp dir so no
artifact ever lands in the repo's data/.
"""

import json  # parsing write_log payloads and steps output_json in the audit-trail assertions
from datetime import datetime, timedelta, timezone  # backdating seeded REAL sends so only ONE rate-limit window fires at a time
from email import policy as email_policy  # the parsing policy for the .eml read-back (the stdlib's non-deprecated parser entry point)
from email.parser import BytesParser  # RFC-5322 parsing of the written .eml — proves it is a real, well-formed message artifact
from pathlib import Path  # tmp outbox handling and .eml existence checks

import pytest  # fixtures, tmp_path, monkeypatch, raises

from app.agents_registry import seed_agent_registry  # the principals — the write gate refuses unregistered writers
from app.db import apply_schema, connect  # fresh per-test SQLite database
from app.ids import new_id  # fresh ids for seeded rows
from app.kill_switch import write_kill_switch  # the switch writer — tests flip the tmp switch file the env var points at
from app.send_gate import evaluate_send_gate  # the gate's read of the rate-limit counters is what the §2.3a test exercises
from app.tools.send_email import send_email  # THE module under test — the DRY_RUN send
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
    offer (slug "acme", matching the offers_dir fixture's YAML).  Targets
    are added per test via _seed_target."""
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


@pytest.fixture
def offers_dir(tmp_path):
    """A tmp offers directory with one offer yaml carrying from_address —
    send_email resolves the From header from here (config-as-code, the same
    source the draft stage used), so the .eml gets a real configured sender
    instead of the .invalid fallback."""
    d = tmp_path / "offers"
    d.mkdir()
    (d / "acme.yaml").write_text(
        "pitch: We cut intake admin time in half.\n"
        "from_address: outreach@acme.test\n"
    )
    return d


# ── Seeding helpers (mirroring tests/test_send_gate.py's, per-file by the
#    repo's precedent — test files do not import each other's privates) ───────


def _insert_draft_version(
    c, *, target_id: str, draft_version_id: str, revision_number: int,
    policy_check_passed=1, injection_scan_passed=1, footer=UNSUBSCRIBE_FOOTER,
) -> None:
    """Insert one message_draft_versions row through the write gate with the
    gate columns passed (1) so the seeded revision is sendable.  insert_seq
    uses the production writers' scalar-subquery form (ticket B5)."""
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
    """Seed one target with EVERY send-gate condition satisfied, plus its
    full FK chain (account, contact, signal, policy row, revision, review
    row) — all through the write gate.  Each override key breaks exactly one
    condition, so a per-rule test is `_seed_target(..., key=bad)`.

    Override keys: email (contact address), email_verified (1/0),
    state, contact_id / account_id (REUSE an existing contact/account —
    the cooldown probe needs two targets on one contact).
    """
    account_id = overrides.get("account_id", f"acc_{target_id}")
    contact_id = overrides.get("contact_id", f"con_{target_id}")
    email = overrides.get("email", f"jane@{target_id.split('_', 1)[-1]}.test")
    domain = email.split("@", 1)[-1] if email else "noemail.test"
    if "account_id" not in overrides:
        # The account carries the ICP verdict the gate reads (fit label and
        # the P4 floor score) — seeded strong so only the overridden
        # condition fails.  normalized_domain gets the account-id prefix
        # because it is UNIQUE in the DDL and several probes legitimately
        # share one email domain (the domain rate-limit check derives the
        # domain from the EMAIL, never from this column).
        commit(
            c, action="insert_account", table_name="accounts", record_id=account_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
                   industry, estimated_size, geo, company_summary, icp_fit_label, icp_fit_score,
                   created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(account_id, "Acme", domain, f"{account_id}.{domain}", "Healthcare",
                    "11-50", "HK", "A healthcare company.", "strong_fit", 88),
        )
    if "contact_id" not in overrides:
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
        params=(target_id, account_id, contact_id, "off_1", "csv",
                overrides.get("state", "approved")),
    )
    # The signal: one strong entry on a run_id of its own (the gate reads
    # the LATEST run's signals).
    commit(
        c, action="insert_signal", table_name="signals", record_id=f"sig_{target_id}",
        payload={}, run_id="r0", step_id="s1", actor="system", agent_id="system",
        sql="""INSERT INTO signals (signal_id, run_id, target_id, signal_type,
               signal_value, signal_strength, source_url, source_confidence,
               evidence_quote, evidence_verified, evidence_tier, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=(f"sig_{target_id}", "r0", target_id, "hiring_relevant_role",
                "Hiring 3 SDRs", 1.0, "https://acme.test/careers", 0.9,
                "The careers page lists three open SDR roles.", 0, "findings"),
    )
    # The policy row (allow) — insert_seq via the production subquery form.
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
                "allow", "low", '["all clear"]', '["P3a"]', '[]'),
    )
    # The draft revision and the operator's approve decision referencing it.
    draft_ref = overrides.get("draft_ref", f"dv_{target_id}")
    _insert_draft_version(
        c, target_id=target_id, draft_version_id=f"dv_{target_id}", revision_number=1,
    )
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
        params=(f"rev_{target_id}", "r0", target_id, draft_ref, "approve",
                0, "", "operator", 0),
    )


def _seed_real_send(c, *, email: str, created_at: str, contact_id=None,
                    target_id=None) -> None:
    """Seed one PRIOR REAL send — an outbound messages row with status
    'sent' (the status the rate-limit counters count).  By default it
    builds its own account+contact+target chain keyed by the email's domain
    stem, so a probe can seed several sends without tripping the domain or
    cooldown checks of the target under evaluation; passing
    contact_id/target_id attaches the send to an EXISTING contact/target
    instead — what the cooldown control needs, since that rule is
    per-contact.  created_at is a parameter so a probe can backdate a send
    out of the 1-hour window while keeping it inside the 24-hour one."""
    stem = new_id("seed")
    domain = email.split("@", 1)[-1]
    if contact_id is None:
        # A fresh FK chain so this send is its own conversation.
        contact_id = f"con_{stem}"
        target_id = f"tgt_{stem}"
        commit(
            c, action="insert_account", table_name="accounts", record_id=f"acc_{stem}",
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
                   created_at, updated_at) VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
            params=(f"acc_{stem}", "Seed Co", domain, f"{stem}.{domain}"),
        )
        commit(
            c, action="insert_contact", table_name="contacts", record_id=contact_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
                   email_verified, created_at, updated_at)
                   VALUES (?,?,?,?,1,datetime('now'),datetime('now'))""",
            params=(contact_id, f"acc_{stem}", "Seed Person", email),
        )
        commit(
            c, action="insert_target", table_name="targets", record_id=target_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id,
                   source, state, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(target_id, f"acc_{stem}", contact_id, "off_1", "csv", "sent"),
        )
    # The messages row: status 'sent' (a REAL send — the only status the
    # counters include), created_at from the caller's window choice.
    commit(
        c, action="insert_message", table_name="messages", record_id=f"msg_{stem}",
        payload={"status": "sent"}, run_id="r0", step_id="s0",
        actor="system", agent_id="system",
        sql="""INSERT INTO messages (message_id, target_id, contact_id, direction,
               provider_message_id, thread_id, subject, body, body_redacted,
               status, sent_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        params=(f"msg_{stem}", target_id, contact_id, "outbound", None, None,
                "Prior subject", "Prior body.", None, "sent",
                "2026-01-01 00:00:00", created_at),
    )


def _hours_ago(hours: float) -> str:
    """A DB-format UTC timestamp N hours in the past — used to backdate
    seeded REAL sends so they sit inside the 24-hour daily window but
    outside the 1-hour hourly window (the probes must trip exactly one
    counter)."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _send(conn, target_id: str, outbox_dir: Path, offers_dir: Path):
    """Drive one send through the DRY_RUN tool with tmp outbox and offers."""
    return send_email(
        conn, target_id=target_id, run_id="r1",
        outbox_dir=str(outbox_dir), offers_dir=str(offers_dir),
    )


def _eml_files(outbox_dir: Path) -> list[Path]:
    """Every .eml file currently in the outbox (empty list when it does not
    exist — a refused send never creates the directory)."""
    if not outbox_dir.exists():
        return []
    return sorted(outbox_dir.glob("*.eml"))


# ── 1 + 2 + 7: the allowed send — artifact, row, transition, audit trail ────


def test_allowed_send_writes_eml_row_and_transition(conn, tmp_path, offers_dir):
    """A fully-approved target that passes every gate check must produce
    exactly: one .eml in the outbox, one messages row (direction outbound,
    status dry_run_sent, sent_at NULL — nothing was sent), and one
    approved → dry_run_sent transition.  The .eml must parse as a
    well-formed RFC-5322 message carrying the subject, the body, and the
    footer's unsubscribe token."""
    _seed_target(conn, target_id="tgt_ok")
    outbox = tmp_path / "outbox"
    result = _send(conn, "tgt_ok", outbox, offers_dir)

    # The outcome model: un-refused, artifact named, state hopped.
    assert result.refused is False
    assert result.message_id is not None
    assert result.new_state == "dry_run_sent"

    # Exactly ONE .eml, named {message_id}.eml.
    files = _eml_files(outbox)
    assert len(files) == 1, "an allowed send writes exactly one .eml"
    assert files[0].name == f"{result.message_id}.eml"
    assert Path(result.outbox_path) == files[0]

    # The messages row: the audit ledger entry for the simulated send.
    rows = conn.execute(
        "SELECT * FROM messages WHERE target_id='tgt_ok';"
    ).fetchall()
    assert len(rows) == 1, "an allowed send writes exactly one messages row"
    row = rows[0]
    assert row["message_id"] == result.message_id
    assert row["direction"] == "outbound"
    assert row["status"] == "dry_run_sent"
    assert row["sent_at"] is None, "sent_at must stay NULL — NOTHING was sent"
    assert row["body"] == "Cold body text.\n\n" + UNSUBSCRIBE_FOOTER

    # The transition: approved → dry_run_sent, through the state machine.
    tr = conn.execute(
        "SELECT previous_state, new_state, reason FROM state_transitions "
        "WHERE target_id='tgt_ok';"
    ).fetchall()
    assert len(tr) == 1, "an allowed send writes exactly one transition"
    assert (tr[0]["previous_state"], tr[0]["new_state"]) == ("approved", "dry_run_sent")
    assert tr[0]["reason"] == "send_gate_success_dry_run"
    target = conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_ok';"
    ).fetchone()
    assert target["state"] == "dry_run_sent"

    # ── 2: the .eml is a well-formed RFC-5322 message ─────────────────────
    # Parsing with the stdlib's BytesParser proves the artifact is a real
    # message: headers present, body decodable, wire-format CRLF.
    raw = files[0].read_bytes()
    assert b"\r\n" in raw, "the SMTP policy must produce CRLF wire format"
    msg = BytesParser(policy=email_policy.default).parsebytes(raw)
    assert msg["From"] == "outreach@acme.test"  # the offer config's sender
    assert msg["To"] == "jane@ok.test"  # the contact's address
    assert msg["Subject"] == "Cold subject"  # the approved subject, verbatim
    assert msg["Date"] is not None, "RFC 5322 requires a Date header"
    assert msg["Message-ID"] is not None, "the artifact must carry its Message-ID"
    payload = msg.get_body(preferencelist=("plain",)).get_content()
    assert "Cold body text." in payload, "the .eml must carry the approved body"
    assert UNSUBSCRIBE_FOOTER in payload, (
        "the .eml must carry the deterministic footer's unsubscribe token"
    )

    # ── 7: the audit trail — every messages row has its write_log row ─────
    audit = conn.execute(
        "SELECT 1 FROM write_log WHERE record_id=? AND action='insert_message';",
        (result.message_id,),
    ).fetchone()
    assert audit is not None, (
        f"messages row {result.message_id} has no write_log row — the row "
        "must be written through the write gate, never a raw INSERT"
    )


# ── 3: the refused send — the default path writes nothing ───────────────────


def test_refused_send_writes_nothing(conn, tmp_path, offers_dir):
    """A target the gate refuses (unverified email — the §2 finding) must
    produce NO .eml, NO messages row, and NO transition; the target stays
    in approved so a fixed condition can be retried.  The refusal is a
    first-class result, not an exception."""
    _seed_target(conn, target_id="tgt_unverified", email_verified=0)
    outbox = tmp_path / "outbox"
    result = _send(conn, "tgt_unverified", outbox, offers_dir)

    assert result.refused is True
    assert "email_verified" in result.refusal_reason
    assert result.message_id is None and result.outbox_path is None
    assert result.new_state is None
    # No artifact: the outbox directory was never even created.
    assert _eml_files(outbox) == []
    # No messages row, no transition, state untouched.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE target_id='tgt_unverified';"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_unverified';"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_unverified';"
    ).fetchone()["state"] == "approved"
    # The refusal IS a logged step (never skip logs) — the failed trace row
    # carries the gate's reasons.
    step = conn.execute(
        "SELECT status, output_json FROM steps WHERE target_id='tgt_unverified' "
        "AND tool_name='send_email';"
    ).fetchone()
    assert step is not None and step["status"] == "failed"
    reasons = json.loads(step["output_json"])["refusal_reasons"]
    assert any("email_verified" in r for r in reasons)


# ── 4: DRY_RUN consumes no rate limit (§2.3a) — proven behaviourally ────────


def test_dry_run_send_consumes_no_rate_limit(conn, tmp_path, offers_dir):
    """§2.3a's exemption, proven by counter probes rather than by reading
    the counters' SQL: after ONE dry-run send, each §2.2a counter must
    behave exactly as if the send had never happened.  Each probe seeds
    real sends up to the boundary MINUS ONE and then evaluates a fresh
    target — if the dry-run row counted, the probe would trip the limit
    and be refused; it is allowed, so it did not.  (The counters'
    sensitivity — that each limit DOES refuse at the boundary — is proven
    by tests/test_send_gate.py's §2.2a tests.)
    """
    outbox = tmp_path / "outbox"
    # The dry-run send: tgt_a, fully allowed.
    _seed_target(conn, target_id="tgt_a", email="jane@dryrun_a.test")
    result = _send(conn, "tgt_a", outbox, offers_dir)
    assert result.refused is False

    # ── Cooldown probe: tgt_b SHARES tgt_a's contact.  One real send to a
    # contact trips the 21-day cooldown (test_send_gate.py proves it), so
    # tgt_b is allowed iff the dry-run row was excluded.
    _seed_target(
        conn, target_id="tgt_b", email="jane@dryrun_a.test",
        contact_id="con_tgt_a", account_id="acc_tgt_a",
    )
    cooldown_probe = evaluate_send_gate(conn, target_id="tgt_b", run_id="r1", step_id="s2")
    assert cooldown_probe.allowed is True, (
        "the dry-run row must not count toward the per-contact cooldown — "
        "tgt_b shares tgt_a's contact and was refused anyway"
    )

    # ── Domain probe: one REAL send to dryrun_a.test (backdated 2h so it
    # stays inside the 24h domain window without tripping the hourly
    # mailbox counter), then tgt_d — a fresh contact on the SAME domain.
    # The domain limit is 2/day: 1 real + the dry-run-if-counted = 2 →
    # refuse; excluded → 1 → allow.
    _seed_real_send(conn, email="seed@dryrun_a.test", created_at=_hours_ago(2))
    _seed_target(conn, target_id="tgt_d", email="x@dryrun_a.test")
    domain_probe = evaluate_send_gate(conn, target_id="tgt_d", run_id="r1", step_id="s2")
    assert domain_probe.allowed is True, (
        "the dry-run row must not count toward the per-domain daily limit"
    )

    # ── Hourly mailbox probe: 4 REAL recent sends on the offer, then
    # tgt_f.  The hourly limit is 5: 4 real + the dry-run-if-counted = 5 →
    # refuse; excluded → 4 → allow.
    for i in range(4):
        _seed_real_send(conn, email=f"s{i}@recent{i}.test", created_at=_hours_ago(0.01))
    _seed_target(conn, target_id="tgt_f", email="jane@hourly_probe.test")
    hourly_probe = evaluate_send_gate(conn, target_id="tgt_f", run_id="r1", step_id="s2")
    assert hourly_probe.allowed is True, (
        "the dry-run row must not count toward the per-mailbox hourly limit"
    )

    # ── Daily mailbox probe: 14 more REAL sends backdated 2h (inside the
    # 24h daily window, outside the hourly one).  Real sends in 24h on the
    # offer are now 4 + 1 + 14 = 19 — one under the daily limit of 20.
    # tgt_e is allowed iff the dry-run row was excluded (19 + 1 = 20 →
    # refuse).
    for i in range(14):
        _seed_real_send(conn, email=f"s{i}@daily{i}.test", created_at=_hours_ago(2))
    _seed_target(conn, target_id="tgt_e", email="jane@daily_probe.test")
    daily_probe = evaluate_send_gate(conn, target_id="tgt_e", run_id="r1", step_id="s2")
    assert daily_probe.allowed is True, (
        "the dry-run row must not count toward the per-mailbox daily limit"
    )

    # ── Control: prove the cooldown probe above was SENSITIVE — one REAL
    # send ATTACHED TO tgt_a's contact trips the cooldown for a target
    # sharing that contact (the counters fire at 1 real send; the probe's
    # allow therefore genuinely proves the dry-run row was excluded).
    _seed_real_send(
        conn, email="jane@dryrun_a.test", created_at=_hours_ago(0.01),
        contact_id="con_tgt_a", target_id="tgt_a",
    )
    _seed_target(
        conn, target_id="tgt_ctrl", email="jane@dryrun_a.test",
        contact_id="con_tgt_a", account_id="acc_tgt_a",
    )
    control = evaluate_send_gate(conn, target_id="tgt_ctrl", run_id="r1", step_id="s2")
    assert control.allowed is False
    assert "cooldown" in " | ".join(control.reasons)


# ── 5: no OutcomeRecord for a dry_run_sent target ───────────────────────────


def test_dry_run_sent_target_initializes_no_outcome(conn, tmp_path, offers_dir):
    """§2.3a: a dry_run_sent target gets NO OutcomeRecord — there is no real
    outcome to track.  This is structurally true: the DDL contains no
    outcome table at all (the only outcome-shaped table is
    signal_outcome_link, the feedback-loop link written by the learning
    job, and nothing in the send path touches it).  The test asserts the
    send wrote ONLY its documented rows: one messages, one
    send_gate_decisions, one state_transitions, one steps."""
    _seed_target(conn, target_id="tgt_outcome")
    result = _send(conn, "tgt_outcome", tmp_path / "outbox", offers_dir)
    assert result.refused is False
    # No outcome-shaped row exists for the target anywhere the send path
    # could have written one.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM signal_outcome_link;"
    ).fetchone()["n"] == 0
    # The documented row inventory, and nothing more: the send's only
    # writes are the gate's decision row, the messages row, the transition
    # (plus their write_log rows), and the steps row.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE target_id='tgt_outcome';"
    ).fetchone()["n"] == 1
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM send_gate_decisions WHERE target_id='tgt_outcome';"
    ).fetchone()["n"] == 1
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_outcome';"
    ).fetchone()["n"] == 1
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM steps WHERE target_id='tgt_outcome' "
        "AND tool_name='send_email' AND status='success';"
    ).fetchone()["n"] == 1


# ── 6: filesystem failure leaves no inconsistent state ──────────────────────


def test_unwritable_outbox_leaves_no_partial_state(conn, tmp_path, offers_dir):
    """The outbox path is a FILE, so mkdir raises before any DB write: the
    send must raise, and NOTHING may be written — no messages row, no
    transition, target still approved.  This is the file-first WRITE ORDER
    invariant: a dry_run_sent target can never exist without its artifact
    because the file write precedes every DB write."""
    _seed_target(conn, target_id="tgt_badoutbox")
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("a file in the way")
    with pytest.raises(OSError):
        _send(conn, "tgt_badoutbox", blocking_file, offers_dir)
    # No artifact anywhere, no row, no transition, honest state.
    assert not blocking_file.is_dir()
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE target_id='tgt_badoutbox';"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_badoutbox';"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_badoutbox';"
    ).fetchone()["state"] == "approved"


def test_db_failure_after_file_write_leaves_honest_approved_target(
    conn, tmp_path, offers_dir, monkeypatch
):
    """The WRITE ORDER's crash-window half: if the DB write fails AFTER the
    .eml landed, the invariant still holds — the target must never be
    dry_run_sent without an artifact.  The documented outcome is an orphan
    .eml (harmless: it was never sent anywhere) plus a target still
    honestly in approved, so the operator simply retries (a fresh
    message_id gives the retry a fresh filename)."""
    _seed_target(conn, target_id="tgt_dbfail")
    outbox = tmp_path / "outbox"
    # The gate itself must still run for real (it uses app.send_gate's own
    # write_gate import, untouched) — only send_email's OWN messages-row
    # write fails, exactly as a DB outage after the file write would.
    import app.tools.send_email as send_email_module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated DB outage after the file write")

    monkeypatch.setattr(send_email_module, "write_gate_commit", _boom)
    with pytest.raises(RuntimeError):
        _send(conn, "tgt_dbfail", outbox, offers_dir)
    # The orphan artifact exists — but the target's state is honest.
    assert len(_eml_files(outbox)) == 1
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE target_id='tgt_dbfail';"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_dbfail';"
    ).fetchone()["state"] == "approved", (
        "a target whose messages row failed to write must stay approved — "
        "never dry_run_sent without its artifact"
    )


# ── 8: a target not in approved is refused and logged ───────────────────────


def test_target_not_in_approved_is_refused_and_logged(conn, tmp_path, offers_dir):
    """The state machine's only inbound edge to dry_run_sent is
    approved → dry_run_sent: a target in awaiting_review (an approval that
    has not taken effect) is refused by the gate, with a logged step and a
    recorded decision row — never silently skipped."""
    _seed_target(conn, target_id="tgt_notapproved", state="awaiting_review")
    outbox = tmp_path / "outbox"
    result = _send(conn, "tgt_notapproved", outbox, offers_dir)
    assert result.refused is True
    assert "not approved" in result.refusal_reason
    assert _eml_files(outbox) == []
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE target_id='tgt_notapproved';"
    ).fetchone()["n"] == 0
    # The refusal is recorded: the gate's decision row (allowed=0) AND the
    # failed steps row both exist.
    assert conn.execute(
        "SELECT allowed FROM send_gate_decisions WHERE target_id='tgt_notapproved';"
    ).fetchone()["allowed"] == 0
    assert conn.execute(
        "SELECT 1 FROM steps WHERE target_id='tgt_notapproved' "
        "AND tool_name='send_email' AND status='failed';"
    ).fetchone() is not None
