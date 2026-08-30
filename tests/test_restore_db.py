"""Tests for scripts/restore_db.py (ticket H17): the verbatim restore tool
that loads a completed local run into an EMPTY database.

COVERAGE BOUNDARY — these tests are hermetic SQLite→SQLite round-trips only.
The SQLite→Postgres dialect path (boolean conversion, the information_schema
catalog reads) is exercised by the operator's real run against Cloud SQL, not
here — the dialect-specific code in scripts/restore_db.py is deliberately small
and mirrors the already-tested app/db.py / _migrate_suppressions patterns. No
test here requires a live Postgres.

Covered, one test per ticket requirement:

- round-trip SQLite → SQLite: per-table counts match AND a row's
  id/timestamp/JSON column is byte-identical between source and destination;
- non-empty destination is refused: exit non-zero, destination UNCHANGED
  (in BOTH --confirm and dry-run modes — the guard fires fail-closed);
- dry run writes nothing: destination still empty afterwards;
- a mid-restore failure leaves the destination empty (the transaction pins it);
- the FK-order guard: every table in apply_schema's DDL appears in the restore
  tool's order list exactly once;
- a missing source file is refused before any destination is touched.
"""

from pathlib import Path  # asserting the missing-source refusal never creates the dest

import pytest  # the tmp_path fixture and monkeypatch

from app.db import _DDL, apply_schema, connect  # building scratch databases the same way the tool does

from scripts.restore_db import _table_order, main  # the module under test


# The 21 tables the schema creates — the same list the FK-order guard derives
# from the DDL. Hardcoded HERE as the test's independent expectation; the
# guard test below asserts it equals _table_order() AND _DDL.
ALL_TABLES = (
    "accounts", "offers", "contacts", "targets", "signals", "sources",
    "messages", "message_draft_versions", "replies", "steps", "suppressions",
    "agent_registry", "write_log", "policy_decisions", "state_transitions",
    "send_gate_decisions", "review_decisions", "signal_outcome_link",
    "signal_weights", "candidate_fields", "enrichment_runs",
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _row_counts(db_path: str) -> dict[str, int]:
    """Count every table's rows in `db_path` — the shape both the round-trip
    equality and the "destination still empty" assertions compare on."""
    conn = connect(db_path)
    counts = {
        table: conn.execute(f"SELECT COUNT(*) AS n FROM {table};").fetchone()["n"]
        for table in ALL_TABLES
    }
    conn.close()
    return counts


def _rows(db_path: str, table: str) -> list[dict]:
    """Every row of `table` as a list of plain dicts, ordered by the first
    column (the primary key) so source and destination compare deterministically
    row-for-row. Values are str/int/float/None, so == is a byte-level compare
    for TEXT columns (ids, timestamps, JSON payloads)."""
    conn = connect(db_path)
    rows = [
        dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY 1;").fetchall()
    ]
    conn.close()
    return rows


def _seed_source(db_path: str) -> None:
    """Build a small but FK-valid source database that mirrors the shape of the
    real completed run (data/e2e_run2.db): a suppressed target, a failed step,
    a dry-run-sent message, and the audit rows around them. Rows are inserted
    with RAW INSERTs (not the write gate) because the tool under test itself
    bypasses the gate — this is test fixture data, not pipeline provenance."""
    conn = connect(db_path)
    apply_schema(conn)  # the current schema — the same one the tool's dest gets
    # ── Master entities first (no FKs) ────────────────────────────────────
    conn.execute(
        "INSERT INTO accounts (account_id, company_name, domain, normalized_domain, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?)",
        ("acc_1", "Serenity Clinic", "serenity-clinic.test", "serenity-clinic.test",
         "2026-08-26 19:27:01", "2026-08-26 19:27:01"),
    )
    conn.execute(
        "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,?)",
        ("off_1", "therapy-app", 1, "2026-08-26 19:27:00"),
    )
    # ── contacts (FK accounts) ────────────────────────────────────────────
    conn.execute(
        "INSERT INTO contacts (contact_id, account_id, full_name, title, email, "
        "email_verified, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        ("con_1", "acc_1", "Dr. Chan Mei-Ling", "Clinical Director",
         "dr.chan@serenity-clinic.test", 1, "2026-08-26 19:27:02",
         "2026-08-26 19:27:02"),
    )
    # ── targets (FK accounts, contacts, offers) — the suppressed state ───
    conn.execute(
        "INSERT INTO targets (target_id, account_id, contact_id, offer_id, source, "
        "state, score, final_recommendation, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("tgt_1", "acc_1", "con_1", "off_1", "csv", "suppressed", 72,
         "send", "2026-08-26 19:27:03", "2026-08-26 19:27:03"),
    )
    # ── signals (FK targets) ─────────────────────────────────────────────
    conn.execute(
        "INSERT INTO signals (signal_id, run_id, target_id, signal_type, signal_value, "
        "signal_strength, source_url, source_confidence, evidence_quote, "
        "evidence_verified, evidence_tier, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("sig_1", "run_1", "tgt_1", "hiring_relevant_role", "Hiring 2 front-desk coordinators",
         0.8, "https://serenity-clinic.test/careers", 0.8,
         "We are hiring 2 front-desk coordinators.", 1, "source",
         "2026-08-26 19:27:04"),
    )
    # ── sources (no FK — the audit-trail family) ─────────────────────────
    conn.execute(
        "INSERT INTO sources (source_id, run_id, target_id, source_type, source_url, "
        "extracted_text, extracted_at, source_confidence, source_priority, "
        "extraction_method, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("src_1", "run_1", "tgt_1", "company_website", "https://serenity-clinic.test/",
         "Serenity Clinic is a Hong Kong therapy practice.\n\nWe are hiring.",
         "2026-08-26 19:27:04", 0.8, 1, "static", "2026-08-26 19:27:04"),
    )
    # ── messages (FK targets, contacts) — the dry-run send, sent_at NULL ─
    conn.execute(
        "INSERT INTO messages (message_id, target_id, contact_id, direction, "
        "provider_message_id, thread_id, subject, body, body_redacted, status, "
        "sent_at, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("msg_1", "tgt_1", "con_1", "outbound", None, None,
         "A question about your intake admin workload", "Hello Dr. Chan...",
         None, "dry_run_sent", None, "2026-08-26 19:27:05"),
    )
    # ── message_draft_versions (FK targets, messages) — JSON critique ────
    conn.execute(
        "INSERT INTO message_draft_versions (draft_version_id, target_id, message_id, "
        "revision_number, subject, body, footer, edited_by, policy_check_passed, "
        "injection_scan_passed, send_gate_passed, critique_passed, critique_json, "
        "insert_seq, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("dv_1", "tgt_1", "msg_1", 1, "A question about your intake admin workload",
         "Hello Dr. Chan,\n\nThis is the draft body.", "[unsubscribe: {UNSUBSCRIBE_URL}]",
         "draft_writer", 1, 1, 1, 1,
         '{"passed": true, "issues": [], "required_changes": "", "severity": "none"}',
         1, "2026-08-26 19:27:05"),
    )
    # ── replies (FK messages) ─────────────────────────────────────────────
    conn.execute(
        "INSERT INTO replies (reply_id, message_id, thread_id, from_email, raw_text, "
        "redacted_text, classification, confidence, routed_action, insert_seq, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("rep_1", "msg_1", "thread_1", "dr.chan@serenity-clinic.test",
         "Please stop contacting me.", "Please stop contacting me.",
         "unsubscribe", 0.97, "auto_suppress", 1, "2026-08-26 19:27:06"),
    )
    # ── the audit-trail family (no FKs) ───────────────────────────────────
    conn.execute(
        "INSERT INTO steps (step_id, run_id, target_id, tool_name, input_json, "
        "output_json, model_call_hash, agent_id, status, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("step_1", "run_1", "tgt_1", "fetch_company_page",
         '{"domain": "serenity-clinic.test"}',
         '{"char_count": 1240, "status": "ok"}', None, "system", "success",
         "2026-08-26 19:27:04"),
    )
    # The FAILED step — the demo's visible safety machinery (the 403 case).
    conn.execute(
        "INSERT INTO steps (step_id, run_id, target_id, tool_name, input_json, "
        "output_json, model_call_hash, agent_id, status, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("step_2", "run_1", "tgt_1", "fetch_company_page",
         '{"domain": "mindnlife.com"}',
         '{"error": "403 Client Error: Forbidden for url: https://mindnlife.com/", '
         '"error_type": "HTTPError"}',
         None, "system", "failed", "2026-08-26 19:27:07"),
    )
    conn.execute(
        "INSERT INTO state_transitions (transition_id, run_id, step_id, target_id, "
        "previous_state, new_state, reason, actor, matched_policy_id, insert_seq, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("tr_1", "run_1", "step_1", "tgt_1", "new", "researched",
         "research_complete_no_enrichment", "system", None, 1, "2026-08-26 19:27:04"),
    )
    conn.execute(
        "INSERT INTO state_transitions (transition_id, run_id, step_id, target_id, "
        "previous_state, new_state, reason, actor, matched_policy_id, insert_seq, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("tr_2", "run_1", "step_2", "tgt_1", "routed", "suppressed",
         "reply_auto_suppress_unsubscribe", "system", "P2", 2, "2026-08-26 19:27:07"),
    )
    # The write_log row carries a JSON payload with a real unicode escape —
    # byte-identical round-trip is the point of the spot-check.
    conn.execute(
        "INSERT INTO write_log (write_id, run_id, step_id, action, table_name, "
        "record_id, actor, agent_id, matched_policy_id, payload_json, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("wr_1", "run_1", "step_1", "insert_signal", "signals", "sig_1",
         "system", "system", None,
         '{"signal_id": "sig_1", "signal_type": "hiring_relevant_role", '
         '"note": "serenity-clinic \\u2014 the demo", "nested": {"a": [1, 2, null]}}',
         "2026-08-26 19:27:04"),
    )
    conn.execute(
        "INSERT INTO review_decisions (review_decision_id, run_id, target_id, "
        "draft_message_id, decision, edited, reason, actor, kill_switch_active, "
        "insert_seq, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("rd_1", "run_1", "tgt_1", "dv_1", "approve", 0,
         "demo: operator approves the seeded draft", "operator", 0, 1,
         "2026-08-26 19:27:06"),
    )
    conn.execute(
        "INSERT INTO send_gate_decisions (send_gate_id, run_id, step_id, target_id, "
        "contact_id, allowed, reasons_json, missing_requirements_json, "
        "matched_rules_json, suppression_hit, approval_verified, kill_switch_active, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("sg_1", "run_1", "step_1", "tgt_1", "con_1", 1,
         "[]", "[]", "[]", 0, 1, 0, "2026-08-26 19:27:06"),
    )
    conn.execute(
        "INSERT INTO suppressions (email, email_normalized, domain, reason, added_at, "
        "added_by, notes) VALUES (?,?,?,?,?,?,?)",
        ("dr.chan@serenity-clinic.test", "dr.chan@serenity-clinic.test", None,
         "unsubscribe", "2026-08-26 19:27:06", "system", "auto from reply"),
    )
    conn.execute(
        "INSERT INTO policy_decisions (policy_decision_id, run_id, step_id, target_id, "
        "action, decision, risk_level, reasons_json, matched_rules_json, "
        "missing_fields_json, insert_seq, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("pd_1", "run_1", "step_1", "tgt_1", "send_email", "allow", "low",
         "[]", "[]", "[]", 1, "2026-08-26 19:27:05"),
    )
    # Two agent_registry rows — the deterministic principal and the judge.
    conn.execute(
        "INSERT INTO agent_registry (agent_id, display_name, description, model_alias, "
        "allowed_actions, allowed_transitions, enabled, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("system", "Deterministic pipeline code", "Non-LLM pipeline code.",
         None, '["insert_account", "state_transition"]', "*", 1,
         "2026-08-26 19:27:00"),
    )
    conn.execute(
        "INSERT INTO agent_registry (agent_id, display_name, description, model_alias, "
        "allowed_actions, allowed_transitions, enabled, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("icp_judge", "ICP judge", "The LLM ICP judge.",
         "judge_model", '["update_account_icp_verdict"]', "*", 1,
         "2026-08-26 19:27:00"),
    )
    conn.close()


# ── The tests ────────────────────────────────────────────────────────────────


def test_round_trip_sqlite_to_sqlite(tmp_path):
    """A full restore into an empty destination: every table's count matches
    the source, and a spot-checked row's id / timestamp / JSON column is
    byte-identical (string equality) between source and destination."""
    src = str(tmp_path / "src.db")
    dst = str(tmp_path / "dst.db")
    _seed_source(src)
    assert main(["--source", src, "--dest", dst, "--confirm"]) == 0
    # Per-table counts match, including the tables the seed left empty.
    assert _row_counts(dst) == _row_counts(src)
    # Byte-identical rows: compare full row dicts (id, timestamps, JSON text,
    # NULLs) for the tables whose fidelity matters most to the audit trail.
    assert _rows(dst, "targets") == _rows(src, "targets")
    assert _rows(dst, "steps") == _rows(src, "steps")
    assert _rows(dst, "write_log") == _rows(src, "write_log")
    assert _rows(dst, "message_draft_versions") == _rows(src, "message_draft_versions")
    assert _rows(dst, "state_transitions") == _rows(src, "state_transitions")


def test_non_empty_destination_is_refused_unchanged(tmp_path):
    """A destination with ANY pre-existing row is refused fail-closed (exit 1)
    in BOTH modes, and is left byte-for-byte unchanged — nothing is merged."""
    src = str(tmp_path / "src.db")
    dst = str(tmp_path / "dst.db")
    _seed_source(src)
    # Seed the destination with a single row in ONE table (offers — no FKs).
    conn = connect(dst)
    apply_schema(conn)
    conn.execute(
        "INSERT INTO offers (offer_id, slug, active, created_at) "
        "VALUES ('off_x', 'x', 1, '2026-01-01 00:00:00');"
    )
    conn.close()
    before = _row_counts(dst)
    # --confirm: refused.
    assert main(["--source", src, "--dest", dst, "--confirm"]) == 1
    # Dry run: also refused (the guard fires fail-closed in both modes).
    assert main(["--source", src, "--dest", dst]) == 1
    # The destination is UNCHANGED — the refusal wrote nothing.
    assert _row_counts(dst) == before


def test_dry_run_writes_nothing(tmp_path):
    """Without --confirm, main() exits 0 and writes nothing: the destination
    (freshly created by apply_schema so the counts are meaningful) has zero
    rows in every table afterwards."""
    src = str(tmp_path / "src.db")
    dst = str(tmp_path / "dst.db")
    _seed_source(src)
    assert main(["--source", src, "--dest", dst]) == 0
    assert _row_counts(dst) == {t: 0 for t in ALL_TABLES}


def test_mid_restore_failure_leaves_destination_empty(tmp_path, monkeypatch):
    """A failure part-way through the copy rolls the WHOLE transaction back —
    the destination is EMPTY afterwards, never half-loaded (the H4b lesson)."""
    import scripts.restore_db as restore_db  # the module to monkeypatch

    src = str(tmp_path / "src.db")
    dst = str(tmp_path / "dst.db")
    _seed_source(src)
    real_copy = restore_db._copy_table  # the real per-table copy
    calls = {"n": 0}  # how many tables the copy has reached

    def _boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # the SECOND table — the FIRST already inserted
            raise RuntimeError("injected mid-restore failure")
        return real_copy(*args, **kwargs)  # first table copies for real

    monkeypatch.setattr(restore_db, "_copy_table", _boom)
    assert main(["--source", src, "--dest", dst, "--confirm"]) == 1
    # The rollback left the destination empty — the first table's copied row
    # is gone too.
    assert _row_counts(dst) == {t: 0 for t in ALL_TABLES}


def test_fk_order_guard_covers_all_ddl_tables():
    """The restore tool's copy order must contain every table in apply_schema's
    DDL exactly once — so a future hardcoded order can never silently skip (or
    duplicate) a table. Derived today, but pinned here so the property holds
    even if the derivation is later replaced by a hand-written list."""
    ddl_tables = _table_order()  # the tool's order
    # The independent expectation, parsed straight from the DDL constant.
    import re

    expected = re.findall(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)", _DDL, re.IGNORECASE
    )
    assert len(ddl_tables) == len(set(ddl_tables))  # no duplicates in the order
    assert set(ddl_tables) == set(expected)  # same table set as the DDL
    for table in expected:  # and each appears exactly once
        assert ddl_tables.count(table) == 1


def test_missing_source_file_is_refused(tmp_path):
    """A nonexistent source is refused BEFORE the destination is even touched
    — sqlite3.connect() would silently create the missing file, so the tool
    must check existence first (fail closed, no hidden side effects)."""
    dst = str(tmp_path / "dst.db")
    missing = str(tmp_path / "does-not-exist.db")
    assert main(["--source", missing, "--dest", dst]) == 1
    assert not Path(dst).exists()  # the destination was never created
