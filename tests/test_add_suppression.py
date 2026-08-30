"""Tests for scripts/add_suppression.py (ticket H4b) and the H4b
suppressions-schema migration in app/db.py.

The script is the operator's ONLY permitted manual suppression add/remove path
(suppression-policy.md §2 / runbook.md §4).  These tests prove:

1.  An address row is stored AS WRITTEN with the shared normaliser's canonical
    key beside it, and both the write_log audit row and the steps trace row are
    written (Golden Rule: never skip logs).
2.  A domain-only row is stored lowercased with email NULL — legal on BOTH
    dialects since H4b (before H4b, `email TEXT PRIMARY KEY` made a NULL email
    a Postgres not-null violation).
3.  Refusals: neither --email nor --domain, both --email and --domain, and a
    --reason outside the CHECK vocabulary.
4.  Idempotency: re-adding an existing key is a no-op.
5.  Multiple domain-only rows coexist — the property that makes the new
    schema's `email_normalized TEXT UNIQUE` legal for domain rows (SQL treats
    NULLs as distinct in a UNIQUE column on both SQLite and Postgres).
6.  End to end: a domain row added through the script makes the REAL send gate
    refuse a send to an address under that domain — the read path now has a
    writer.
7.  The H4b migration rebuilds a pre-H4b table (email PRIMARY KEY) into the
    H4b shape, preserves every row, and is idempotent.

Every test honours OUTBOUND_TEST_DB_TARGET via scratch_db_target, so the same
file runs against SQLite (plain pytest) and Postgres (with the env var set).
"""

import sqlite3  # building the pre-H4b SQLite shape in the migration test

import pytest  # fixtures, raises

from app.agents_registry import seed_agent_registry  # the write gate refuses unregistered agents
from app.db import (  # the DB layer under test
    apply_schema,
    connect,
    normalize_domain,  # the shared domain fold the script must reuse
    normalize_email,  # the shared address fold the script must reuse
)
from app.ids import new_id  # fresh ids for seeded rows
from app.kill_switch import write_kill_switch  # the switch writer — tests point the reader at a tmp file
from app.send_gate import evaluate_send_gate  # the REAL send gate — the end-to-end refusal proof
from app.write_gate import commit  # every seeded core-table row goes through the gate, never a raw INSERT

from scripts.add_suppression import main  # the CLI under test — main() returns the exit code


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def switch_path(tmp_path, monkeypatch):
    """A tmp kill-switch file, written DISENGAGED, and the env var pointing
    the gate's reader at it — so the end-to-end test never depends on the
    committed config/kill_switch.json (the B4a convention test_send_gate.py
    uses)."""
    path = tmp_path / "kill_switch.json"
    write_kill_switch(engaged=False, updated_by="fixture", path=str(path))
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(path))
    return path


def _seed_target_for_send_gate(c, *, target_id: str, email: str) -> None:
    """Seed the minimal FK chain (account + contact + target) the send gate
    needs to reach the domain-suppression check and record a decision row:
    the domain check fires regardless of the other (unseeded) checklist
    items, and the decision row needs a contact_id (NOT NULL there)."""
    account_id = f"acc_{target_id}"
    contact_id = f"con_{target_id}"
    commit(
        c, action="insert_account", table_name="accounts", record_id=account_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               industry, estimated_size, geo, company_summary, icp_fit_label, icp_fit_score,
               created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(account_id, "Acme", email.split("@", 1)[-1], email.split("@", 1)[-1],
                "Healthcare", "11-50", "HK", "A company.", "strong_fit", 88),
    )
    commit(
        c, action="insert_contact", table_name="contacts", record_id=contact_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
               email_verified, created_at, updated_at)
               VALUES (?,?,?,?,1,datetime('now'),datetime('now'))""",
        params=(contact_id, account_id, "Jane Doe", email),
    )
    commit(
        c, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id,
               source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, account_id, contact_id, "off_1", "csv", "approved"),
    )


def _conn_for(target: str):
    """Open a fresh connection to ``target`` (the scratch fixture's value)."""
    c = connect(target)
    # Seed the offer the target FK references (off_1 above) and the agent
    # registry, so gated writes and the send-gate decision row work.  The
    # registry seed is an idempotent upsert; the offer insert is guarded by
    # its slug UNIQUE so re-running is safe.
    seed_agent_registry(c, run_id="r0", step_id="s0")
    commit(
        c, action="insert_offer", table_name="offers", record_id="off_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "acme", 1),
    )
    return c


# ── The script: add paths ────────────────────────────────────────────────────


def test_adds_email_row_as_written_with_canonical_key(scratch_db_target):
    """An address row is stored AS WRITTEN (the audit record) with the shared
    normaliser's canonical key beside it, through the write gate, and both
    the write_log audit row and the steps trace row are written."""
    rc = main(["--db", scratch_db_target, "--email", "Jane@Example.com", "--notes", "op note"])
    assert rc == 0
    c = _conn_for(scratch_db_target)
    row = c.execute(
        "SELECT email, email_normalized, domain, reason, added_by, notes "
        "FROM suppressions WHERE email_normalized=?;",
        (normalize_email("Jane@Example.com"),),
    ).fetchone()
    assert row is not None
    assert row["email"] == "Jane@Example.com"  # the address AS WRITTEN, preserved
    assert row["email_normalized"] == "jane@example.com"  # the canonical key
    assert row["domain"] is None  # an address row suppresses the address, not the domain
    assert row["reason"] == "manual"  # the default reason
    assert row["added_by"] == "operator"  # the CHECK-constrained added_by
    assert row["notes"] == "op note"
    # The write_log audit row: attributed to the operator, action insert_suppression.
    audit = c.execute(
        "SELECT action, actor, agent_id, table_name FROM write_log "
        "WHERE table_name='suppressions' AND action='insert_suppression';",
    ).fetchone()
    assert audit is not None
    assert audit["actor"] == "operator"
    assert audit["agent_id"] == "operator"
    # The steps trace row: never skip logs.
    step = c.execute(
        "SELECT tool_name, status, agent_id FROM steps WHERE tool_name='add_suppression';",
    ).fetchone()
    assert step is not None
    assert step["status"] == "success"
    assert step["agent_id"] == "operator"
    c.close()


def test_adds_domain_only_row_lowercased(scratch_db_target):
    """A domain-only row is stored lowercased with email NULL — the row shape
    that was impossible on Postgres before H4b (email TEXT PRIMARY KEY rejected
    a NULL email)."""
    rc = main(["--db", scratch_db_target, "--domain", "SUP.TEST", "--reason", "legal"])
    assert rc == 0
    c = _conn_for(scratch_db_target)
    row = c.execute(
        "SELECT email, email_normalized, domain, reason FROM suppressions WHERE domain=?;",
        (normalize_domain("SUP.TEST"),),
    ).fetchone()
    assert row is not None
    assert row["email"] is None  # no address — a domain-only suppression
    assert row["email_normalized"] is None  # no canonical address key either
    assert row["domain"] == "sup.test"  # lowercased by the shared normaliser
    assert row["reason"] == "legal"  # the CHECK-constrained reason honored
    c.close()


def test_two_domain_only_rows_coexist(scratch_db_target):
    """The property that makes the new schema legal: SQL treats NULLs as
    distinct in a UNIQUE column on BOTH dialects, so any number of domain-only
    rows (email_normalized = NULL) can coexist.  Before H4b the email PK made
    a second domain-only row impossible on Postgres."""
    assert main(["--db", scratch_db_target, "--domain", "one.test"]) == 0
    assert main(["--db", scratch_db_target, "--domain", "two.test"]) == 0
    c = _conn_for(scratch_db_target)
    rows = c.execute("SELECT domain FROM suppressions ORDER BY domain;").fetchall()
    assert [r["domain"] for r in rows] == ["one.test", "two.test"]
    c.close()


def test_add_is_idempotent_on_readd(scratch_db_target):
    """Re-adding an existing key is a logged no-op (the check-then-insert
    precedent from app/review.py / app/agents/reply.py) — not a constraint
    error, and not a second row."""
    assert main(["--db", scratch_db_target, "--email", "Dr.Chan@serenity-clinic.test"]) == 0
    rc = main(["--db", scratch_db_target, "--email", "dr.chan+alias@serenity-clinic.test"])
    assert rc == 0  # idempotent: same exit code, no error
    c = _conn_for(scratch_db_target)
    rows = c.execute(
        "SELECT email FROM suppressions WHERE email_normalized=?;",
        (normalize_email("Dr.Chan@serenity-clinic.test"),),
    ).fetchall()
    assert len(rows) == 1  # exactly one row, however many spellings were tried
    c.close()


# ── The script: refusals ─────────────────────────────────────────────────────


def test_refuses_with_neither_email_nor_domain(scratch_db_target):
    """A suppression must suppress SOMETHING (the table-level CHECK): with
    neither --email nor --domain the script refuses before any DB I/O."""
    rc = main(["--db", scratch_db_target])
    assert rc == 1


def test_refuses_both_email_and_domain(scratch_db_target):
    """One canonical key per invocation: giving both --email and --domain is
    refused (the check-then-insert idempotency contract is keyed by one), and
    the operator runs the script twice for two independent suppressions."""
    rc = main(["--db", scratch_db_target, "--email", "a@b.test", "--domain", "b.test"])
    assert rc == 1


def test_refuses_bad_reason(scratch_db_target):
    """--reason must be one of the CHECK vocabulary; argparse refuses anything
    else with SystemExit(2) before any DB I/O."""
    with pytest.raises(SystemExit) as exc:
        main(["--db", scratch_db_target, "--email", "a@b.test", "--reason", "bogus"])
    assert exc.value.code == 2


def test_remove_without_remove_flag_is_an_add(scratch_db_target):
    """--remove is the operator flag: WITHOUT it the same --email invocation
    ADDS rather than removes (so a removal can never happen by accident)."""
    assert main(["--db", scratch_db_target, "--email", "a@b.test"]) == 0
    c = _conn_for(scratch_db_target)
    assert c.execute(
        "SELECT 1 FROM suppressions WHERE email_normalized=?;",
        (normalize_email("a@b.test"),),
    ).fetchone() is not None  # the row was added, not removed
    c.close()


# ── The script: remove paths ─────────────────────────────────────────────────


def test_removes_email_row_by_canonical_key(scratch_db_target):
    """--remove --email deletes the row whose canonical key matches, whatever
    spelling the operator uses (any casing/plus-tag of a suppressed mailbox
    removes it), through the write gate with action delete_suppression."""
    assert main(["--db", scratch_db_target, "--email", "Dr.Chan@serenity-clinic.test"]) == 0
    rc = main(["--db", scratch_db_target, "--email", "dr.chan+alias@serenity-clinic.test", "--remove"])
    assert rc == 0
    c = _conn_for(scratch_db_target)
    assert c.execute(
        "SELECT 1 FROM suppressions WHERE email_normalized=?;",
        (normalize_email("Dr.Chan@serenity-clinic.test"),),
    ).fetchone() is None  # the row is gone
    audit = c.execute(
        "SELECT action, actor FROM write_log WHERE action='delete_suppression';",
    ).fetchone()
    assert audit is not None
    assert audit["actor"] == "operator"
    step = c.execute(
        "SELECT tool_name, status FROM steps WHERE tool_name='remove_suppression';",
    ).fetchone()
    assert step is not None
    assert step["status"] == "success"
    c.close()


def test_removes_domain_row(scratch_db_target):
    """--remove --domain deletes the domain-only row by its lowercased key."""
    assert main(["--db", scratch_db_target, "--domain", "Sup.Test"]) == 0
    assert main(["--db", scratch_db_target, "--domain", "sup.test", "--remove"]) == 0
    c = _conn_for(scratch_db_target)
    assert c.execute(
        "SELECT 1 FROM suppressions WHERE domain='sup.test';",
    ).fetchone() is None
    c.close()


def test_remove_missing_is_a_noop(scratch_db_target):
    """Removing a key that is not suppressed is a logged no-op (idempotent),
    not an error — the operator's goal (this suppression is gone) is already
    true."""
    rc = main(["--db", scratch_db_target, "--email", "nobody@missing.test", "--remove"])
    assert rc == 0
    c = _conn_for(scratch_db_target)
    assert c.execute(
        "SELECT COUNT(*) AS n FROM suppressions;",
    ).fetchone()["n"] == 0
    c.close()


# ── End to end: the read path now has a writer ───────────────────────────────


def test_domain_row_added_by_script_blocks_send(scratch_db_target, switch_path):
    """The end-to-end proof: a domain suppression added through the script
    makes the REAL send gate refuse a send to an address under that domain
    (the read path in app/send_gate.py that previously had no writer)."""
    assert main(["--db", scratch_db_target, "--domain", "supdomain.test"]) == 0
    c = _conn_for(scratch_db_target)
    # Seed a target whose contact is under the suppressed domain, mixed-case
    # on purpose — the gate folds the probe via normalize_domain().
    _seed_target_for_send_gate(c, target_id="tgt_e2e", email="jane@SUPDOMAIN.TEST")
    decision = evaluate_send_gate(c, target_id="tgt_e2e", run_id="r1", step_id="s9")
    assert decision.suppression_hit is True  # the domain check fired
    assert decision.allowed is False  # the send is refused
    assert "contact.domain not in suppressions" in decision.missing_requirements
    c.close()


def test_email_row_added_by_script_blocks_send(scratch_db_target, switch_path):
    """The address-side end-to-end proof: an address suppression added through
    the script makes the send gate refuse a send to that exact mailbox."""
    assert main(["--db", scratch_db_target, "--email", "Jane@Supdomain.test"]) == 0
    c = _conn_for(scratch_db_target)
    _seed_target_for_send_gate(c, target_id="tgt_e2e2", email="jane+alias@supdomain.test")
    decision = evaluate_send_gate(c, target_id="tgt_e2e2", run_id="r1", step_id="s9")
    assert decision.suppression_hit is True
    assert decision.allowed is False
    assert "contact.email not in suppressions" in decision.missing_requirements
    c.close()


# ── The H4b migration (app/db.py) ────────────────────────────────────────────


def _assert_no_suppressions_primary_key(c) -> None:
    """Assert the suppressions table has no PRIMARY KEY — the H4b shape.
    Dialect-branched: PRAGMA is SQLite-only, information_schema is postgres."""
    if c.dialect == "sqlite":
        for row in c.execute("PRAGMA table_info(suppressions);").fetchall():
            assert row["pk"] == 0, f"column {row['name']} is still part of a PK"
        return
    row = c.execute(
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE constraint_type='PRIMARY KEY' "
        "AND table_schema='public' AND table_name='suppressions';",
    ).fetchone()
    assert row is None, "suppressions still has a PRIMARY KEY"


def _assert_suppressions_is_old_shape(c) -> None:
    """Assert the suppressions table is STILL the pre-H4b shape (email
    PRIMARY KEY).  Used by the crash-regression test to prove the rebuild
    rolled back completely rather than half-applying.  Dialect-branched:
    PRAGMA is SQLite-only, information_schema is postgres."""
    if c.dialect == "sqlite":
        email_pk = [
            row
            for row in c.execute("PRAGMA table_info(suppressions);").fetchall()
            if row["name"] == "email" and row["pk"]
        ]
        assert email_pk, "suppressions is not the pre-H4b shape (email PRIMARY KEY)"
        return
    row = c.execute(
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE constraint_type='PRIMARY KEY' "
        "AND table_schema='public' AND table_name='suppressions';",
    ).fetchone()
    assert row is not None, "suppressions is not the pre-H4b shape (email PRIMARY KEY)"


def test_h4b_migration_preserves_rows_and_is_idempotent(scratch_db_target):
    """Build a database with the OLD suppressions shape (email PRIMARY KEY,
    pre-F1b — no email_normalized column), insert a row, run apply_schema:
    the row survives, the H4b shape is in place, and a second apply_schema is
    a no-op.  Runs against both dialects via scratch_db_target (the fixture
    resets the target first, so this builds the old shape on an empty DB)."""
    c = connect(scratch_db_target)
    # The pre-H4b / pre-F1b shape: email is the PRIMARY KEY, and
    # email_normalized does not exist yet (the F1b _MIGRATION_COLUMNS pass
    # adds it, the backfill fills it, and the H4b rebuild then copies it).
    c.execute(
        """
        CREATE TABLE suppressions (
          email TEXT PRIMARY KEY,
          domain TEXT,
          reason TEXT NOT NULL CHECK (reason IN ('unsubscribe','bounce','complaint','manual','legal','risky_reply')),
          added_at TEXT NOT NULL,
          added_by TEXT NOT NULL CHECK (added_by IN ('system','operator')),
          notes TEXT
        )
        """
    )
    c.execute(
        """
        INSERT INTO suppressions (email, domain, reason, added_at, added_by, notes)
        VALUES ('Dr.Chan@serenity-clinic.test', 'SERENITY-CLINIC.TEST', 'unsubscribe',
                '2026-08-24 00:00:00', 'system', NULL)
        """
    )
    c.close()

    # First apply_schema: _ensure_column adds email_normalized,
    # _backfill_suppressions fills it and lowercases the domain, and the H4b
    # rebuild replaces the email-PK table with the new shape — preserving the
    # row through the copy.
    c = connect(scratch_db_target)
    apply_schema(c)
    row = c.execute(
        "SELECT email, email_normalized, domain FROM suppressions "
        "WHERE email='Dr.Chan@serenity-clinic.test';",
    ).fetchone()
    assert row is not None, "the legacy row must survive the rebuild"
    assert row["email"] == "Dr.Chan@serenity-clinic.test"  # as-written preserved
    assert row["email_normalized"] == "dr.chan@serenity-clinic.test"  # backfilled key
    assert row["domain"] == "serenity-clinic.test"  # lowercased in place
    # The matching key now answers a lowercase probe (the F1b guarantee).
    hit = c.execute(
        "SELECT 1 FROM suppressions WHERE email_normalized=?;",
        ("dr.chan@serenity-clinic.test",),
    ).fetchone()
    assert hit is not None
    # A domain-only row is now insertable — the whole point of H4b.
    c.execute(
        "INSERT INTO suppressions (email, email_normalized, domain, reason, added_at, added_by, notes) "
        "VALUES (NULL, NULL, 'newdomain.test', 'manual', '2026-08-26 00:00:00', 'operator', NULL);"
    )
    _assert_no_suppressions_primary_key(c)
    before = c.execute("SELECT email, email_normalized, domain FROM suppressions ORDER BY domain;").fetchall()
    c.close()

    # Second apply_schema: idempotent — no error, no change to rows or shape.
    c = connect(scratch_db_target)
    apply_schema(c)
    after = c.execute("SELECT email, email_normalized, domain FROM suppressions ORDER BY domain;").fetchall()
    assert [tuple(r) for r in after] == [tuple(r) for r in before]
    _assert_no_suppressions_primary_key(c)
    c.close()


def test_h4b_migration_detects_already_migrated_table(scratch_db_target):
    """A fresh database (new shape from the DDL) is a no-op for the migration:
    apply_schema twice on a table that was never the old shape must not
    rebuild or error."""
    c = connect(scratch_db_target)
    apply_schema(c)  # creates the NEW shape from _DDL
    _assert_no_suppressions_primary_key(c)
    apply_schema(c)  # second run: must not raise, must not change the shape
    _assert_no_suppressions_primary_key(c)
    c.close()


def test_h4b_migration_crash_between_drop_and_rename_preserves_table(
    scratch_db_target, monkeypatch
):
    """Regression for the H4b BLOCKER: a crash in the middle of the rebuild
    must NOT destroy the suppression list.

    The pre-fix migration ran five bare conn.execute() calls in autocommit
    (sqlite isolation_level=None, Postgres autocommit=True), so a failure
    between ``DROP TABLE suppressions`` and the RENAME unrecoverably lost
    every suppression — and the next apply_schema silently recreated an
    EMPTY table (CLAUDE.md §9: unsubscribe always suppresses, so that loss
    means mailing people who opted out).  The rebuild must be ONE
    transaction: when the RENAME fails, the whole rebuild rolls back and
    the old table survives with every row.

    Fault injection: monkeypatch this connection's execute() so the RENAME
    statement raises — a deterministic, dialect-agnostic stand-in for the
    process dying (kill -9 / connection drop) at the exact worst point.
    """
    # Seed a pre-H4b table (email PRIMARY KEY, pre-F1b — no email_normalized)
    # with one row on the empty scratch target.
    c = connect(scratch_db_target)
    c.execute(
        """
        CREATE TABLE suppressions (
          email TEXT PRIMARY KEY,
          domain TEXT,
          reason TEXT NOT NULL CHECK (reason IN ('unsubscribe','bounce','complaint','manual','legal','risky_reply')),
          added_at TEXT NOT NULL,
          added_by TEXT NOT NULL CHECK (added_by IN ('system','operator')),
          notes TEXT
        )
        """
    )
    c.execute(
        """
        INSERT INTO suppressions (email, domain, reason, added_at, added_by, notes)
        VALUES ('Dr.Chan@serenity-clinic.test', 'SERENITY-CLINIC.TEST', 'unsubscribe',
                '2026-08-24 00:00:00', 'system', NULL)
        """
    )
    c.close()

    # Run apply_schema on a connection whose RENAME statement raises.  The
    # DDL pass (executescript on SQLite, split executes on Postgres), the
    # _ensure_column pass, and _backfill_suppressions all execute normally;
    # only the migration's RENAME is crashed — exactly the DROP->RENAME gap.
    c = connect(scratch_db_target)
    real_execute = c.execute  # the connection's original bound method

    def crash_on_rename(sql, params=()):
        if "RENAME TO suppressions" in str(sql):
            raise RuntimeError("simulated crash between DROP and RENAME")
        return real_execute(sql, params)

    monkeypatch.setattr(c, "execute", crash_on_rename)
    with pytest.raises(RuntimeError, match="simulated crash between DROP and RENAME"):
        apply_schema(c)
    c.close()  # drop the crashed connection; the assertion reads a fresh one

    # The suppression list must have survived the failed rebuild: the table
    # still exists (a fresh apply_schema will NOT recreate an empty one) and
    # the row is intact.  The temp table must be gone too — the rebuild left
    # no half-created artifact behind.
    c = connect(scratch_db_target)
    try:
        row = c.execute(
            "SELECT email, email_normalized, domain FROM suppressions "
            "WHERE email='Dr.Chan@serenity-clinic.test';",
        ).fetchone()
    except Exception as exc:  # the table is gone — the catastrophic failure
        pytest.fail(f"suppressions table destroyed by the failed rebuild: {exc}")
    assert row is not None, "the legacy row must survive a mid-rebuild crash"
    assert row["email"] == "Dr.Chan@serenity-clinic.test"  # as-written preserved
    # The F1b backfill runs BEFORE the crashed rebuild in apply_schema, so
    # its per-row changes are already committed; only the rebuild rolled back.
    assert row["email_normalized"] == "dr.chan@serenity-clinic.test"
    assert row["domain"] == "serenity-clinic.test"
    # The table is STILL the old shape — the rebuild's DROP/RENAME fully
    # rolled back, so a re-run of apply_schema will rebuild it from scratch.
    _assert_suppressions_is_old_shape(c)
    temp = c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='suppressions_h4b_mig';"
        if c.dialect == "sqlite" else
        "SELECT 1 FROM information_schema.tables WHERE table_schema='public' "
        "AND table_name='suppressions_h4b_mig';"
    ).fetchone()
    assert temp is None, "the temp table must be rolled back with the rebuild"
    c.close()
