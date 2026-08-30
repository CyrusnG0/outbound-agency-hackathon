"""
Tests for app.db — the database connection and schema layer.

These tests verify the three hard requirements for the storage layer:
1. WAL mode is enabled on every connection (prevents corruption under concurrent writers).
2. All core tables from docs/db-schema.md exist after apply_schema().
3. BEGIN IMMEDIATE semantics prevent silent lock upgrades — two concurrent writers
   cannot both proceed; the second one fails with an OperationalError rather than
   quietly upgrading a read lock to a write lock (which could interleave writes).

Every later task in the pipeline (write_gate, state transitions, log_step, etc.)
depends on this layer being correct. If these tests break, nothing else is safe.
"""

import os
import sqlite3
import pytest

from app.db import connect, apply_schema


def test_connect_enables_wal_mode(tmp_path):
    """connect() must set journal_mode=WAL so concurrent readers don't block writers."""
    db_path = str(tmp_path / "test.db")
    conn = connect(db_path)
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal"
    conn.close()


def test_apply_schema_creates_all_core_tables(tmp_path):
    """apply_schema() must create every table listed in docs/db-schema.md.

    If a table is missing here, later tasks that INSERT into it will fail with
    'no such table' — this test catches schema drift early.
    """
    db_path = str(tmp_path / "test.db")
    conn = connect(db_path)
    apply_schema(conn)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }
    required = {
        "accounts", "signals", "contacts", "offers", "targets", "messages",
        "message_draft_versions", "replies", "steps", "suppressions",
        "write_log", "policy_decisions", "state_transitions",
        "send_gate_decisions", "review_decisions", "signal_outcome_link",
        "signal_weights", "candidate_fields", "enrichment_runs",
        "agent_registry",  # Added in plan task A3 (write gate capability table).
        "sources",  # Added in ticket B2b (persisted evidence: raw fetched pages + research findings).
        "meetings",  # Added in the 2026-08-30 real-scheduling demo (app/tools/schedule_meeting.py).
    }
    assert required.issubset(tables)
    conn.close()


def test_write_transaction_uses_begin_immediate_semantics(tmp_path):
    """A write-mode transaction must not silently upgrade from a read lock.

    Two connections both try BEGIN IMMEDIATE. The first one takes the write lock.
    The second must raise OperationalError — it must NOT silently wait or upgrade.
    This is the core correctness property that write_gate.py (Task 4) relies on:
    if two concurrent writers could both proceed, they'd interleave writes and
    corrupt the audit trail.
    """
    db_path = str(tmp_path / "test.db")
    conn_a = connect(db_path)
    apply_schema(conn_a)
    conn_b = connect(db_path)

    conn_a.execute("BEGIN IMMEDIATE")
    conn_a.execute(
        "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,?)",
        ("off_1", "test-offer", 1, "2026-08-04T00:00:00"),
    )
    with pytest.raises(sqlite3.OperationalError):
        conn_b.execute("BEGIN IMMEDIATE")
        conn_b.execute(
            "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,?)",
            ("off_2", "other-offer", 1, "2026-08-04T00:00:00"),
        )
    conn_a.commit()
    conn_a.close()
    conn_b.close()


def test_connect_succeeds_on_readonly_sqlite_database(tmp_path):
    """connect() must open a read-only database instead of dying on the WAL pragma.

    The operator console (plan task A5b) mounts the operator's database
    read-only; before this regression was fixed, connect() ran
    PRAGMA journal_mode=WAL unconditionally, which writes the database file's
    header and raises OperationalError("attempt to write a readonly
    database") on a read-only file — so every console request 500'd before a
    single query could run. The fix skips ONLY that pragma when it raises and
    keeps the connection usable for reads.
    """
    db_path = str(tmp_path / "readonly.db")
    conn = connect(db_path)
    apply_schema(conn)
    # The fixture must leave the file in the default DELETE journal mode (not
    # WAL) before the permissions drop. That is a SQLite constraint, not a
    # connect() concern: SQLite cannot read a WAL-mode database without its
    # -shm/-wal sidecar files, and creating them requires a WRITABLE parent
    # directory — so a WAL-header database in a read-only directory would
    # fail the SELECT below for a reason this fix can never address (verified
    # against sqlite3 directly). DELETE mode needs no sidecars, so it is the
    # mode a read-only copy (e.g. made via VACUUM INTO, which produces a
    # delete-mode file) will actually be in.
    conn.execute("PRAGMA journal_mode=DELETE;")
    conn.close()

    os.chmod(tmp_path, 0o555)  # parent directory read-only
    os.chmod(db_path, 0o444)   # database file read-only
    try:
        ro_conn = connect(db_path)
        try:
            # A read must succeed — this is the core regression: connect()
            # must not raise, and the resulting connection must be usable.
            tables = ro_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
            assert any(row[0] == "offers" for row in tables)
            # The fallback skipped ONLY the WAL pragma — foreign_keys is a
            # per-connection runtime setting (no file write), so it must
            # still be ON even though the file is read-only.
            fk = ro_conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            assert fk == 1
            # And the file must still be in its existing journal mode: the
            # skipped pragma must not have half-applied WAL.
            mode = ro_conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert mode.lower() == "delete"
        finally:
            ro_conn.close()
    finally:
        # Restore permissions so pytest can clean up tmp_path — a read-only
        # directory breaks tmp_path teardown on some systems.
        os.chmod(db_path, 0o644)
        os.chmod(tmp_path, 0o755)


def test_apply_schema_adds_b2a_signal_columns_to_existing_database(tmp_path):
    """The migration path (plan tasks B2a/B2b): a pre-existing signals table
    without evidence_quote / evidence_verified / evidence_tier must gain all
    three columns when apply_schema() runs, because CREATE TABLE IF NOT
    EXISTS never adds columns to a table that already exists — without the
    _MIGRATION_COLUMNS pass, the operator's already-provisioned database
    would silently keep running without them.  And the migration must be
    idempotent: SQLite's ALTER TABLE has no "ADD COLUMN IF NOT EXISTS", so a
    naive second run would crash with "duplicate column name"."""
    db_path = str(tmp_path / "pre_b2a.db")
    # Build the pre-B2a database by hand: the signals table exactly as it
    # existed before plan task B2a — every column except the three new ones.
    raw = sqlite3.connect(db_path)
    raw.executescript(
        """
        CREATE TABLE signals (
          signal_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          signal_type TEXT NOT NULL,
          signal_value TEXT NOT NULL,
          signal_strength REAL NOT NULL,
          source_url TEXT,
          source_confidence REAL,
          created_at TEXT NOT NULL,
          UNIQUE (target_id, run_id, signal_type, signal_value)
        );
        """
    )
    raw.close()

    conn = connect(db_path)
    # First apply_schema: the CREATE IF NOT EXISTS no-ops on the existing
    # table, then _MIGRATION_COLUMNS adds the B2a/B2b columns.  The NEW
    # `sources` table (B2b) is created outright by the CREATE pass — new
    # tables need no column migration.
    apply_schema(conn)
    cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(signals);").fetchall()
    }
    assert {"evidence_quote", "evidence_verified", "evidence_tier"}.issubset(cols)
    # The B2b sources table must also exist on the provisioned database —
    # persistence of evidence is the ticket's whole point.
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    }
    assert "sources" in tables
    # Second apply_schema: must not raise and must not change the column set
    # — _ensure_column's existence check makes the rerun a no-op.
    apply_schema(conn)
    cols_again = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(signals);").fetchall()
    }
    assert cols_again == cols
    conn.close()


def test_apply_schema_adds_b4b_review_column_to_existing_database(tmp_path):
    """The migration path (ticket B4b): a pre-existing review_decisions
    table without kill_switch_active must gain the column when
    apply_schema() runs — CREATE TABLE IF NOT EXISTS never adds columns to
    a table that already exists, so without the _MIGRATION_COLUMNS entry
    the operator's already-provisioned database would silently keep
    running without it (the same failure mode as the B2a/B2b test above).
    The migration must also be idempotent."""
    db_path = str(tmp_path / "pre_b4b.db")
    # Build the pre-B4b table by hand: exactly the columns the table had
    # before ticket B4b (no kill_switch_active).
    raw = sqlite3.connect(db_path)
    raw.executescript(
        """
        CREATE TABLE review_decisions (
          review_decision_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          draft_message_id TEXT NOT NULL,
          decision TEXT NOT NULL,
          edited INTEGER NOT NULL DEFAULT 0,
          reason TEXT,
          actor TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )
    raw.close()

    conn = connect(db_path)
    # First apply_schema: the CREATE IF NOT EXISTS no-ops on the existing
    # table, then _MIGRATION_COLUMNS adds the B4b column.
    apply_schema(conn)
    cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(review_decisions);").fetchall()
    }
    assert "kill_switch_active" in cols
    # Second apply_schema: must not raise and must not change the column set.
    apply_schema(conn)
    cols_again = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(review_decisions);").fetchall()
    }
    assert cols_again == cols
    conn.close()


def test_apply_schema_adds_h6_matched_rules_column_to_existing_database(tmp_path):
    """The migration path (ticket H6): a pre-existing send_gate_decisions
    table WITHOUT matched_rules_json must gain the column when apply_schema()
    runs — CREATE TABLE IF NOT EXISTS never adds columns to a table that
    already exists, so without the _MIGRATION_COLUMNS entry the operator's
    already-provisioned databases would keep writing send-gate rows without
    the H6 rule-ID attribution (the same failure mode as the B2a/B4b tests
    above).  The migration must also be idempotent."""
    db_path = str(tmp_path / "pre_h6.db")
    # Build the pre-H6 table by hand: exactly the columns the table had
    # before ticket H6 (no matched_rules_json).
    raw = sqlite3.connect(db_path)
    raw.executescript(
        """
        CREATE TABLE send_gate_decisions (
          send_gate_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          step_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          contact_id TEXT NOT NULL,
          allowed INTEGER NOT NULL,
          reasons_json TEXT NOT NULL,
          missing_requirements_json TEXT NOT NULL,
          suppression_hit INTEGER NOT NULL,
          approval_verified INTEGER NOT NULL,
          kill_switch_active INTEGER NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )
    raw.close()

    conn = connect(db_path)
    # First apply_schema: the CREATE IF NOT EXISTS no-ops on the existing
    # table, then _MIGRATION_COLUMNS adds the H6 column.
    apply_schema(conn)
    cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(send_gate_decisions);").fetchall()
    }
    assert "matched_rules_json" in cols
    # Second apply_schema: must not raise and must not change the column set.
    apply_schema(conn)
    cols_again = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(send_gate_decisions);").fetchall()
    }
    assert cols_again == cols
    conn.close()


# The four tables that carry B5/C1's monotonic insert_seq column — one
# list so the migration assertions below can never drift apart from the
# migration DDL they verify.
_INSERT_SEQ_TABLES = (
    "review_decisions",
    "policy_decisions",
    "message_draft_versions",
    "state_transitions",
    # ticket E1: replies — the follow-up path resolves "which reply is the
    # LATEST?" by insert_seq, so the replies table joins the list.
    "replies",
)


def test_apply_schema_adds_insert_seq_columns_to_existing_database(tmp_path):
    """The migration path (tickets B5 + C1 + E1): pre-existing
    review_decisions / policy_decisions / message_draft_versions (B5),
    state_transitions (C1 — B5's fix extended to the state machine's own
    audit log), and replies (E1 — extended again, for the follow-up
    path's "latest reply" read) tables WITHOUT insert_seq must gain the
    column when apply_schema() runs — CREATE TABLE IF NOT EXISTS never
    adds columns, so without the _MIGRATION_COLUMNS entries the
    operator's provisioned databases would keep resolving row order
    arbitrarily (the same-second created_at tie the column exists to
    fix).  The migration must also be idempotent."""
    db_path = str(tmp_path / "pre_insert_seq.db")
    # Build the pre-insert_seq tables by hand: exactly the columns the
    # five tables had before their insert_seq was added (no insert_seq
    # anywhere).
    raw = sqlite3.connect(db_path)
    raw.executescript(
        """
        CREATE TABLE review_decisions (
          review_decision_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          draft_message_id TEXT NOT NULL,
          decision TEXT NOT NULL,
          edited INTEGER NOT NULL DEFAULT 0,
          reason TEXT,
          actor TEXT NOT NULL,
          kill_switch_active INTEGER,
          created_at TEXT NOT NULL
        );
        CREATE TABLE policy_decisions (
          policy_decision_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          step_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          action TEXT NOT NULL,
          decision TEXT NOT NULL,
          risk_level TEXT NOT NULL,
          reasons_json TEXT NOT NULL,
          matched_rules_json TEXT NOT NULL,
          missing_fields_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE message_draft_versions (
          draft_version_id TEXT PRIMARY KEY,
          target_id TEXT NOT NULL,
          message_id TEXT,
          revision_number INTEGER NOT NULL,
          subject TEXT NOT NULL,
          body TEXT NOT NULL,
          footer TEXT NOT NULL,
          edited_by TEXT NOT NULL,
          policy_check_passed INTEGER,
          injection_scan_passed INTEGER,
          send_gate_passed INTEGER,
          critique_passed INTEGER,
          critique_json TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE state_transitions (
          transition_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          step_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          previous_state TEXT NOT NULL,
          new_state TEXT NOT NULL,
          reason TEXT NOT NULL,
          actor TEXT NOT NULL,
          matched_policy_id TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE replies (
          reply_id TEXT PRIMARY KEY,
          message_id TEXT NOT NULL,
          thread_id TEXT,
          from_email TEXT NOT NULL,
          raw_text TEXT NOT NULL,
          redacted_text TEXT NOT NULL,
          classification TEXT,
          confidence REAL,
          routed_action TEXT,
          created_at TEXT NOT NULL
        );
        """
    )
    raw.close()

    conn = connect(db_path)
    # First apply_schema: the CREATE IF NOT EXISTS no-ops on the existing
    # tables, then _MIGRATION_COLUMNS adds the insert_seq columns to all
    # five.
    apply_schema(conn)
    for table in _INSERT_SEQ_TABLES:
        cols = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
        }
        assert "insert_seq" in cols, f"{table} is missing insert_seq after migration"
    # Second apply_schema: must not raise and must not change the column set.
    first_cols = {
        table: {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
        }
        for table in _INSERT_SEQ_TABLES
    }
    apply_schema(conn)
    for table, cols_first in first_cols.items():
        cols_again = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
        }
        assert cols_again == cols_first
    conn.close()


def test_apply_schema_backfills_suppression_normalized_keys(tmp_path):
    """F1b migration: a pre-existing suppressions row with a mixed-case
    address (and a mixed-case domain) must match a lowercase probe after
    apply_schema() runs.  CREATE TABLE IF NOT EXISTS never adds columns to a
    table that already exists, so without the _MIGRATION_COLUMNS entry AND
    the data backfill the operator's provisioned database would keep its old
    suppression and still miss the normalised probe — the exact C2 breach
    this ticket closes."""
    db_path = str(tmp_path / "pre_f1b.db")
    # Build the pre-F1b table by hand: every column except email_normalized,
    # plus one legacy row written with mixed case (the audit record of what
    # arrived) and a mixed-case domain row.
    raw = sqlite3.connect(db_path)
    raw.executescript(
        """
        CREATE TABLE suppressions (
          email TEXT PRIMARY KEY,
          domain TEXT,
          reason TEXT NOT NULL CHECK (reason IN ('unsubscribe','bounce','complaint','manual','legal','risky_reply')),
          added_at TEXT NOT NULL,
          added_by TEXT NOT NULL CHECK (added_by IN ('system','operator')),
          notes TEXT
        );
        INSERT INTO suppressions (email, domain, reason, added_at, added_by, notes)
        VALUES ('Dr.Chan@serenity-clinic.test', 'SERENITY-CLINIC.TEST', 'unsubscribe',
                '2026-08-24 00:00:00', 'system', NULL);
        """
    )
    raw.close()

    conn = connect(db_path)
    # First apply_schema: CREATE IF NOT EXISTS no-ops on the existing table,
    # _MIGRATION_COLUMNS adds email_normalized, then _backfill_suppressions
    # fills it and lowercases the legacy domain.
    apply_schema(conn)
    cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(suppressions);").fetchall()
    }
    assert "email_normalized" in cols
    row = conn.execute(
        "SELECT email, email_normalized, domain FROM suppressions "
        "WHERE email='Dr.Chan@serenity-clinic.test';"
    ).fetchone()
    # The written address is preserved byte-for-byte (the audit record)...
    assert row["email"] == "Dr.Chan@serenity-clinic.test"
    # ...while the matching key is folded, and the legacy domain lowercased.
    assert row["email_normalized"] == "dr.chan@serenity-clinic.test"
    assert row["domain"] == "serenity-clinic.test"
    # The specific §2 requirement: a lowercase probe now matches the legacy row.
    hit = conn.execute(
        "SELECT 1 FROM suppressions WHERE email_normalized=?;",
        ("dr.chan@serenity-clinic.test",),
    ).fetchone()
    assert hit is not None
    # Second apply_schema: idempotent — no error, no change to the values.
    apply_schema(conn)
    row_again = conn.execute(
        "SELECT email_normalized, domain FROM suppressions "
        "WHERE email='Dr.Chan@serenity-clinic.test';"
    ).fetchone()
    assert row_again["email_normalized"] == "dr.chan@serenity-clinic.test"
    assert row_again["domain"] == "serenity-clinic.test"
    conn.close()
