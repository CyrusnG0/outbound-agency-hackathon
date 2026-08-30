"""Tests for the simulated inbox (ticket C1): app/tools/fetch_inbox.py —
the ONLY inbox that exists in this repository.

WHAT IS PROVEN HERE (ticket §3.1, docs/reply-routing.md §4, item 18):
1.  Header threading (In-Reply-To/References) matches the right messages
    row; the fallback (sender email + normalized subject + 14-day window)
    works when headers are absent; the window excludes an older message.
2.  An .eml matching nothing is skipped without a row and without an
    exception.
3.  A malformed .eml is logged, skipped, and the batch continues.
4.  Redaction: an inbound message containing an email address and a phone
    number produces a redacted_text without them, and NO steps row
    anywhere contains the raw values (asserted against the steps table,
    not just the return value — item 18 is about the trace log).
5.  A matched reply transitions dry_run_sent → replied (the C1 edge);
    a second reply on an already-replied (or terminal) target records its
    row WITHOUT a transition (§5).
6.  The audit trail: every replies row has a write_log row.

Seeding here goes through write_gate.commit on purpose (fixtures are
normal pipeline writes), and every test points the inbox at a tmp dir so
the committed data/inbox/ fixtures are never consumed by the suite.
"""

import json  # parsing steps output_json in the redaction-leak assertions
from pathlib import Path  # tmp inbox handling and .eml writes

import pytest  # fixtures, tmp_path

from app.agents_registry import seed_agent_registry  # the principals — the write gate refuses unregistered writers
from app.db import apply_schema, connect  # fresh per-test SQLite database
from app.tools.fetch_inbox import fetch_inbox, redact_text  # the module under test, plus the pure redactor for a direct unit check
from app.write_gate import commit  # every seeded core-table row goes through the gate, never a raw INSERT


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def conn(scratch_db_target):
    """Fresh SQLite DB with schema + the seeded principals + one shared
    offer.  Targets/messages are added per test via _seed_outbound_send,
    so each test's data is exactly what it needs and nothing else."""
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


def _seed_outbound_send(
    c, *, target_id: str, email: str, subject: str,
    message_id: str, created_at: str = "2026-08-20 09:00:00",
    state: str = "dry_run_sent",
) -> str:
    """Seed one target's full FK chain (account, contact, target) plus one
    outbound messages row shaped exactly like B5's DRY_RUN send writes it
    (direction outbound, status dry_run_sent, sent_at NULL, thread_id
    NULL).  Returns the message_id.  ``created_at`` is a parameter so a
    test can backdate a send outside the 14-day fallback window."""
    account_id = f"acc_{target_id}"
    contact_id = f"con_{target_id}"
    domain = email.split("@", 1)[-1]
    # normalized_domain is UNIQUE — two targets sharing one email domain
    # (the header-threading tests seed exactly that) get a synthetic
    # per-target normalized_domain, the test_send_gate.py precedent.
    normalized_domain = f"{target_id}.{domain}"
    commit(
        c, action="insert_account", table_name="accounts", record_id=account_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               created_at, updated_at) VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=(account_id, "Seed Clinic", domain, normalized_domain),
    )
    commit(
        c, action="insert_contact", table_name="contacts", record_id=contact_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
               email_verified, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(contact_id, account_id, "Seed Person", email, 1),
    )
    commit(
        c, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id,
               source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, account_id, contact_id, "off_1", "csv", state),
    )
    commit(
        c, action="insert_message", table_name="messages", record_id=message_id,
        payload={"status": "dry_run_sent"}, run_id="r0", step_id="s0",
        actor="system", agent_id="system",
        # created_at is interpolated as a literal here — the helper's own
        # backdating knob (a bound parameter would also work; the literal
        # keeps the SQL readable for the window tests).
        sql=f"""INSERT INTO messages (message_id, target_id, contact_id, direction,
                 provider_message_id, thread_id, subject, body, body_redacted,
                 status, sent_at, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        params=(message_id, target_id, contact_id, "outbound", None, None,
                subject, "Cold body text.", None, "dry_run_sent", None, created_at),
    )
    return message_id


def _write_eml(inbox: Path, name: str, *, from_addr: str, subject: str, body: str,
               in_reply_to: str | None = None, date: str = "Fri, 22 Aug 2026 09:14:00 +0800") -> Path:
    """Write one .eml file into the tmp inbox with realistic headers.

    ``in_reply_to`` is the raw In-Reply-To header value; when None the
    header is omitted entirely (the fallback-threading tests' shape).
    The Date header is fixed so the 14-day window arithmetic is
    deterministic (the seeded created_at values are written to match).
    """
    headers = [
        f"From: Test Sender <{from_addr}>",
        "To: outreach@outbound-agency.invalid",
        f"Subject: {subject}",
        f"Date: {date}",
        "Message-ID: <demo.test@example.test>",
    ]
    if in_reply_to is not None:
        headers.append(f"In-Reply-To: {in_reply_to}")
    path = inbox / name
    path.write_text("\n".join(headers) + "\n\n" + body)
    return path


# ── Threading: header path and fallback path ─────────────────────────────────


def test_header_threading_matches_the_right_message_row(conn, tmp_path):
    """Primary threading: the reply's In-Reply-To token embeds the
    message_id B5's make_msgid produced, and the fetch links the replies
    row to THAT message — not to another outbound to the same contact."""
    # Two outbound sends to the same contact; the reply answers the SECOND.
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_aaaa00000001")
    _seed_outbound_send(conn, target_id="tgt_b", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_bbbb00000002")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_eml(
        inbox, "reply.eml", from_addr="jane@clinic.test",
        subject="Re: Cold subject", body="Sounds interesting, tell me more.",
        # The token embeds msg_bbbb00000002 the way make_msgid does.
        in_reply_to="<1234567890.12345.1234567890123456789.msg_bbbb00000002@outbound-agency.invalid>",
    )
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert result.files_seen == 1
    assert len(result.replies_created) == 1
    assert result.skipped == [] and result.errors == []
    row = conn.execute(
        "SELECT message_id, thread_id, from_email FROM replies WHERE reply_id=?;",
        (result.replies_created[0],),
    ).fetchone()
    # The link is to the message the header named — never the other one.
    assert row["message_id"] == "msg_bbbb00000002"
    # thread_id is the matched message id on the header path (§4: the RFC
    # thread identity, normalized to our id vocabulary).
    assert row["thread_id"] == "msg_bbbb00000002"
    assert row["from_email"] == "jane@clinic.test"


def test_fallback_threading_without_headers(conn, tmp_path):
    """Fallback threading: no In-Reply-To header at all — the match is
    sender email + normalized subject + the most recent outbound within
    the 14-day window, and the thread_id is the deterministic
    fallback:{message_id} key."""
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_cccc00000003")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_eml(
        inbox, "reply.eml", from_addr="JANE@clinic.test",  # upper-case sender: the match lowercases both sides
        subject="RE:  cold subject  ",  # prefix + stray whitespace: normalization strips both
        body="Tell me more please.",
    )
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert len(result.replies_created) == 1
    row = conn.execute(
        "SELECT message_id, thread_id FROM replies WHERE reply_id=?;",
        (result.replies_created[0],),
    ).fetchone()
    assert row["message_id"] == "msg_cccc00000003"
    # The deterministic generated key, naming the fallback method (§4).
    assert row["thread_id"] == "fallback:msg_cccc00000003"


def test_fallback_window_excludes_an_older_message(conn, tmp_path):
    """The 14-day window: an identical-sender/subject outbound from 30
    days before the reply's Date is OUTSIDE the window and must not
    match — only the recent one links.  (Both messages share the sender
    and the subject, so the window is the only discriminator.)"""
    _seed_outbound_send(conn, target_id="tgt_old", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_old000000004",
                        created_at="2026-07-01 09:00:00")  # 52 days before the reply's 2026-08-22 date
    _seed_outbound_send(conn, target_id="tgt_new", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_new000000005",
                        created_at="2026-08-20 09:00:00")  # 2 days before: inside the window
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_eml(
        inbox, "reply.eml", from_addr="jane@clinic.test",
        subject="Re: Cold subject", body="Following up.",
    )
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert len(result.replies_created) == 1
    row = conn.execute(
        "SELECT message_id FROM replies WHERE reply_id=?;",
        (result.replies_created[0],),
    ).fetchone()
    # Only the in-window message may be linked — the stale one is excluded.
    assert row["message_id"] == "msg_new000000005"


# ── Skips and failures ───────────────────────────────────────────────────────


def test_unmatched_message_is_skipped_without_row_or_exception(conn, tmp_path):
    """An .eml that matches no known message is NOT an error: logged,
    skipped, no row, and the fetch returns normally — never a guess."""
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_dddd00000006")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    # No In-Reply-To AND a wrong subject AND a different sender — all
    # three threading inputs disagree, so nothing can match.
    _write_eml(inbox, "stranger.eml", from_addr="unknown@elsewhere.test",
               subject="Completely different topic", body="Hello?")
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert result.replies_created == []
    assert result.errors == []
    assert any("matches no known outbound message" in s for s in result.skipped)
    assert conn.execute("SELECT COUNT(*) AS n FROM replies;").fetchone()["n"] == 0
    # The skip IS logged — a trace row names the file and the reason.
    steps = conn.execute(
        "SELECT output_json FROM steps WHERE tool_name='fetch_inbox';"
    ).fetchall()
    assert any(json.loads(s["output_json"]).get("outcome") == "no_known_message" for s in steps)


def test_malformed_eml_is_logged_skipped_and_batch_continues(conn, tmp_path):
    """Per-file isolation (the B1f rule): a malformed first file must not
    stop the second, valid file from being processed."""
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_eeee00000007")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    # Not an RFC-5322 message at all: no headers, no blank line separator.
    (inbox / "01_broken.eml").write_bytes(b"\x00\x01\x02garbage\xff\xfe")
    _write_eml(inbox, "02_good.eml", from_addr="jane@clinic.test",
               subject="Re: Cold subject", body="A valid reply.")
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert len(result.replies_created) == 1  # the valid file was still processed
    assert any("malformed" in s for s in result.skipped)
    # The parse failure is in the trace with the exception type.
    steps = conn.execute(
        "SELECT output_json FROM steps WHERE tool_name='fetch_inbox' AND status='failed';"
    ).fetchall()
    assert any(json.loads(s["output_json"]).get("outcome") == "malformed_file" for s in steps)


# ── Redaction (item 18 — the trace must never leak) ──────────────────────────


def test_redaction_removes_email_and_phone_from_redacted_text_and_steps(conn, tmp_path):
    """The item-18 test, asserted against the STEPS table — not just the
    return value: an inbound message carrying an email address and a
    phone number must produce a redacted_text without them, and NO steps
    row anywhere (input_json or output_json) may contain the raw values.
    raw_text itself is the master table and may hold them — that is the
    documented split, not a leak."""
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_ffff00000008")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    # The body carries BOTH a raw email and a raw phone (fictional values,
    # but the point is the mechanism — real replies will carry real ones).
    body = (
        "Please contact jane.doe@example.com or call +852 9123 4567 "
        "regarding this matter. Our office is at 12 Nathan Road, Central."
    )
    _write_eml(inbox, "reply.eml", from_addr="jane@clinic.test",
               subject="Re: Cold subject", body=body)
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert len(result.replies_created) == 1
    row = conn.execute(
        "SELECT raw_text, redacted_text FROM replies WHERE reply_id=?;",
        (result.replies_created[0],),
    ).fetchone()
    # The master table keeps the real data (the documented split)...
    assert "jane.doe@example.com" in row["raw_text"]
    assert "+852 9123 4567" in row["raw_text"]
    # ...while the redacted copy leaks neither value (nor the street
    # address, fully removed per the threat-model standard).
    assert "jane.doe@example.com" not in row["redacted_text"]
    assert "+852 9123 4567" not in row["redacted_text"]
    assert "Nathan Road" not in row["redacted_text"]
    # The threat model's partial forms ARE present (useful context, no
    # full value): first-two-chars email and last-two-digits phone.
    assert "ja***@example.com" in row["redacted_text"]
    assert "67" in row["redacted_text"]
    # ── THE TRACE ASSERTION (the point of item 18) ──────────────────────
    # Every steps row written by this run — input_json AND output_json —
    # must be free of the raw values.  A single leaked phone number in
    # the trace is a policy violation, not a cosmetic issue.
    for step in conn.execute("SELECT input_json, output_json FROM steps;").fetchall():
        for payload in (step["input_json"], step["output_json"]):
            if payload is None:
                continue
            assert "jane.doe@example.com" not in payload
            assert "+852 9123 4567" not in payload
            assert "Nathan Road" not in payload
    # And the write_log payloads too — the audit row must not carry raw
    # PII either (only the replies master row may).
    for wl in conn.execute("SELECT payload_json FROM write_log;").fetchall():
        assert "jane.doe@example.com" not in wl["payload_json"]
        assert "+852 9123 4567" not in wl["payload_json"]


def test_redact_text_follows_the_threat_model_standard():
    """A direct unit check on the pure redactor: secrets and meeting-link
    query strings are fully removed, dates survive (the phone-regex
    guard), and partial forms follow docs/threat-model.md exactly."""
    redacted = redact_text(
        "Meeting at https://zoom.example.com/j/12345?pwd=sekrit&x=1 "
        "api_key=ABCDEF1234567890 on 2026-08-23. Call 555-1234."
    )
    # The meeting-link query string is fully removed; the link itself stays.
    assert "https://zoom.example.com/j/12345" in redacted
    assert "pwd=sekrit" not in redacted
    # The secret assignment is fully removed.
    assert "ABCDEF1234567890" not in redacted
    assert "[SECRET]" in redacted
    # The separator date is NOT a phone number and survives intact.
    assert "2026-08-23" in redacted
    # The phone keeps only its last two digits.
    assert "555-1234" not in redacted
    assert "34" in redacted


# ── The state hop and its guards ─────────────────────────────────────────────


def test_matched_reply_transitions_dry_run_sent_to_replied(conn, tmp_path):
    """The C1 edge: a matched inbound message moves its target
    dry_run_sent → replied, through the state machine (a
    state_transitions row exists with the documented reason), and the
    transition is attributed to the system principal."""
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_aaaa11110001")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_eml(
        inbox, "reply.eml", from_addr="jane@clinic.test",
        subject="Re: Cold subject", body="Interested.",
        in_reply_to="<1.2.3.msg_aaaa11110001@outbound-agency.invalid>",
    )
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert len(result.replies_created) == 1
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_a';"
    ).fetchone()["state"] == "replied"
    tr = conn.execute(
        "SELECT previous_state, new_state, reason FROM state_transitions "
        "WHERE target_id='tgt_a';"
    ).fetchone()
    assert (tr["previous_state"], tr["new_state"]) == ("dry_run_sent", "replied")
    assert tr["reason"] == "inbound_message_linked"


def test_reply_for_already_replied_target_records_without_transition(conn, tmp_path):
    """A second reply on the same thread: the row is written (every
    inbound message gets its own record, §5), but the target — already in
    replied — is NOT transitioned again (the edge requires dry_run_sent)."""
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_aaaa11110002",
                        state="replied")  # the first reply already moved it
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_eml(
        inbox, "reply.eml", from_addr="jane@clinic.test",
        subject="Re: Cold subject", body="Also, one more question.",
        in_reply_to="<1.2.3.msg_aaaa11110002@outbound-agency.invalid>",
    )
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert len(result.replies_created) == 1  # the row was recorded
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_a';"
    ).fetchone()["state"] == "replied"
    # No new transition row for this fetch (the state was not dry_run_sent).
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_a';"
    ).fetchone()["n"] == 0


def test_terminal_target_records_without_transition(conn, tmp_path):
    """§5's terminal guard at the fetch: a reply for a SUPPRESSED target
    is recorded (the row is written) but never transitioned — no reply
    ever overrides a terminal state."""
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_aaaa11110003",
                        state="suppressed")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_eml(
        inbox, "reply.eml", from_addr="jane@clinic.test",
        subject="Re: Cold subject", body="I already asked you to stop.",
        in_reply_to="<1.2.3.msg_aaaa11110003@outbound-agency.invalid>",
    )
    result = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert len(result.replies_created) == 1
    assert conn.execute(
        "SELECT state FROM targets WHERE target_id='tgt_a';"
    ).fetchone()["state"] == "suppressed"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_a';"
    ).fetchone()["n"] == 0


# ── The audit trail ──────────────────────────────────────────────────────────


def test_every_reply_row_is_gated(conn, tmp_path):
    """The audit-trail guarantee: every replies row has a matching
    write_log row with action insert_reply and the same record_id — a raw
    conn.execute replacing the gated write would leave a reply row with
    no audit row and this test fails."""
    _seed_outbound_send(conn, target_id="tgt_a", email="jane@clinic.test",
                        subject="Cold subject", message_id="msg_aaaa11110004")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write_eml(
        inbox, "reply.eml", from_addr="jane@clinic.test",
        subject="Re: Cold subject", body="Interested.",
        in_reply_to="<1.2.3.msg_aaaa11110004@outbound-agency.invalid>",
    )
    fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    for row in conn.execute("SELECT reply_id FROM replies;").fetchall():
        audit = conn.execute(
            "SELECT 1 FROM write_log WHERE record_id=? AND action='insert_reply';",
            (row["reply_id"],),
        ).fetchone()
        assert audit is not None, f"reply row {row['reply_id']} has no write_log row"
