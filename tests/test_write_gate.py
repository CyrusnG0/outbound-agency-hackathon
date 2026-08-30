"""
Tests for app.write_gate — the sole entry point for every core-table write.

These tests verify the three gate rules from docs/gates.md §1.2:
1. Unknown action types are refused (no silent new write paths).
2. Invalid actors are refused (only "system" and "operator" may write).
3. The write and its audit row are atomic — a failed SQL statement must not
   leave a dangling write_log row or a partial write in the target table.
"""

import json

import pytest

# Import the public interfaces this module exposes to the rest of the codebase.
# commit() is the single choke-point function that every tool must call to write.
# WriteGateRefused is the exception raised when the gate blocks an attempt.
from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.write_gate import KNOWN_ACTIONS, commit, WriteGateRefused


@pytest.fixture
def conn(scratch_db_target):
    """Create a fresh in-tmpdir database with the full schema applied.

    Each test gets its own empty database file inside pytest's tmp_path,
    so tests never share state or interfere with each other."""
    # Open a connection to a temporary SQLite file — this will be created on first write (WAL).
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    # Apply the full DDL so all tables exist; idempotent (CREATE TABLE IF NOT EXISTS).
    apply_schema(c)
    # Seed the agent registry (plan A3): commit() refuses writes from
    # unregistered agents, so the "system" agent must be registered before
    # any of the writes below are attempted.
    seed_agent_registry(c, run_id="r0", step_id="s0")
    yield c
    # Close the connection after the test so SQLite releases its lock on the temp file.
    c.close()


def test_commit_writes_record_and_write_log_row(conn):
    """Happy path: a valid system write inserts a row AND its audit log atomically."""
    # Arrange/Act: call commit() with a valid action, table, actor, agent, and SQL.
    write_id = commit(
        conn,
        action="insert_offer",
        table_name="offers",
        record_id="off_1",
        payload={"slug": "acme"},
        run_id="run_1",
        step_id="step_1",
        actor="system",
        agent_id="system",  # The seeded deterministic principal (plan A3).
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "acme", 1),
    )
    # Assert: commit() returns a non-empty write_id string (the audit row's primary key).
    assert write_id

    # Verify data row: the offer row must exist with the slug we inserted.
    offer = conn.execute("SELECT * FROM offers WHERE offer_id = 'off_1';").fetchone()
    assert offer["slug"] == "acme"

    # Verify audit row: one write_log row must exist for this write, with
    # the correct action, actor, agent, and payload stored as JSON.
    log_row = conn.execute(
        "SELECT * FROM write_log WHERE write_id = ?;", (write_id,)
    ).fetchone()
    assert log_row["action"] == "insert_offer"
    assert log_row["actor"] == "system"
    assert log_row["agent_id"] == "system"
    # payload_json is stored as a JSON string — parse it back to compare.
    assert json.loads(log_row["payload_json"]) == {"slug": "acme"}


def test_commit_refuses_unknown_action(conn):
    """Gate rule 1: actions not in KNOWN_ACTIONS must raise WriteGateRefused.

    The action "delete_everything" is not in the allowlist. If commit() silently
    accepted it, any tool could introduce a new write path without updating the
    gate — this test ensures that can't happen."""
    # Use pytest.raises to assert the exact exception type and that its message
    # includes the phrase "unknown action type" (matching the raise in commit()).
    with pytest.raises(WriteGateRefused, match="unknown action type"):
        commit(
            conn,
            action="delete_everything",  # Not in KNOWN_ACTIONS — must be refused.
            table_name="offers",
            record_id="off_1",
            payload={},
            run_id="run_1",
            step_id="step_1",
            actor="system",
            agent_id="system",
            sql="DELETE FROM offers;",
            params=(),
        )


def test_commit_refuses_invalid_actor(conn):
    """Gate rule 2: actors not in ("system", "operator") must raise WriteGateRefused.

    An actor of "the_llm_decided" represents the scenario where an LLM or untrusted
    agent tries to write directly — the gate must refuse it. Only deterministic
    pipeline code ("system") or the human operator should be writing."""
    # Use pytest.raises to assert the exact exception type and that its message
    # includes the phrase "invalid actor".
    with pytest.raises(WriteGateRefused, match="invalid actor"):
        commit(
            conn,
            action="insert_offer",
            table_name="offers",
            record_id="off_1",
            payload={},
            run_id="run_1",
            step_id="step_1",
            actor="the_llm_decided",  # Not "system" or "operator" — must be refused.
            agent_id="system",
            sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
            params=("off_1", "acme", 1),
        )


def test_commit_rolls_back_on_sql_error(conn):
    """Gate rule 3: a SQL error must roll back BOTH the data write AND the audit row.

    The INSERT uses a nonexistent column, which SQLite will reject. After the
    error, neither the offers table nor the write_log table should have any rows —
    this is the atomicity guarantee that prevents a dangling audit row when the
    actual write failed."""
    # Act: attempt a commit with a SQL statement that references a column that
    # doesn't exist — the database will raise an error on execute().
    with pytest.raises(Exception):
        commit(
            conn,
            action="insert_offer",
            table_name="offers",
            record_id="off_1",
            payload={},
            run_id="run_1",
            step_id="step_1",
            actor="system",
            agent_id="system",
            sql="INSERT INTO offers (nonexistent_column) VALUES (?);",
            params=("x",),
        )
    # Assert: after the error, the offers table must be empty (the INSERT was rolled back).
    row = conn.execute("SELECT * FROM offers;").fetchone()
    assert row is None
    # Assert: no audit row exists FOR THIS FAILED WRITE — the atomicity
    # guarantee. Scoped to table_name='offers' rather than asserting write_log
    # is globally empty: since A3 the fixture seeds agent_registry so that
    # agent_id="system" is registered, and that seeding is itself a real
    # audited write, so write_log legitimately holds two insert_agent_registry
    # rows before this test even runs. A global emptiness check would fail on
    # those and hide what is actually being tested.
    orphans = conn.execute(
        "SELECT count(*) AS n FROM write_log WHERE table_name='offers';"
    ).fetchone()
    assert orphans["n"] == 0


def test_b4b_review_actions_are_registered():
    """Ticket B4b's two new write paths must be registered in KNOWN_ACTIONS
    — the allowlist is the gate's whole contract, so a missing registration
    means the review gate's writes would be refused as unknown actions.
    Membership is asserted (not the full set) so unrelated future additions
    cannot break this test."""
    assert "insert_review_decision" in KNOWN_ACTIONS
    assert "insert_suppression" in KNOWN_ACTIONS


def test_b5_send_actions_are_registered():
    """Ticket B5's two new write paths — the DRY_RUN messages row
    (app/tools/send_email.py) and the send-gate verdict row
    (app/send_gate.py) — must be registered in KNOWN_ACTIONS: the allowlist
    is the gate's whole contract, and an unregistered send write would be
    refused as an unknown action mid-send.  Membership is asserted (not the
    full set) so unrelated future additions cannot break this test."""
    assert "insert_message" in KNOWN_ACTIONS
    assert "insert_send_gate_decision" in KNOWN_ACTIONS


# ── Ticket H8: suppression removal requires the operator flag ───────────────
#
# docs/gates.md §1.2: "any suppression removal without an operator flag" is
# refused.  H8 moved the enforcement INTO the gate: commit() refuses
# action="delete_suppression" unless operator_confirmed=True AND
# actor="operator", before any SQL runs.  These tests pin that the rule
# holds for ANY caller of commit() — not just scripts/add_suppression.py —
# and that the flag is meaningless for inserts.


def _seed_suppression(conn) -> str:
    """Insert one suppression row through the gate so a removal test has
    something to remove.  The seed is itself a normal gated write
    (action="insert_suppression"), so it exercises the same actor/capability
    path the real writers use."""
    return commit(
        conn,
        action="insert_suppression",
        table_name="suppressions",
        record_id="a@b.test",
        payload={"email": "a@b.test", "email_normalized": "a@b.test"},
        run_id="run_1",
        step_id="step_1",
        actor="operator",   # the human performs this write
        agent_id="operator",  # attributed to the registered operator principal
        sql="""
            INSERT INTO suppressions (email, email_normalized, domain, reason, added_at, added_by, notes)
            VALUES (?,?,?,?,datetime('now'),?,?)
        """,
        params=("a@b.test", "a@b.test", None, "manual", "operator", None),
    )


def test_delete_suppression_without_flag_is_refused(conn):
    """H8 gate rule: a delete_suppression with operator_confirmed left at
    its safe default False must be refused BEFORE any SQL runs — the row
    survives and no write_log row claims the removal."""
    _seed_suppression(conn)
    # Act: the removal attempt, with the flag deliberately NOT passed
    # (operator_confirmed defaults to False — the safe value).
    with pytest.raises(WriteGateRefused, match="operator_confirmed"):
        commit(
            conn,
            action="delete_suppression",
            table_name="suppressions",
            record_id="a@b.test",
            payload={"email_normalized": "a@b.test"},
            run_id="run_1",
            step_id="step_1",
            actor="operator",
            agent_id="operator",
            sql="DELETE FROM suppressions WHERE email_normalized=?;",
            params=("a@b.test",),
        )
    # Assert: the suppression row is still present — no row was deleted.
    row = conn.execute(
        "SELECT 1 FROM suppressions WHERE email_normalized='a@b.test';"
    ).fetchone()
    assert row is not None
    # Assert: no write_log row claims the removal — the refusal was silent
    # in the audit trail exactly because it never opened a write.
    n = conn.execute(
        "SELECT count(*) AS n FROM write_log WHERE action='delete_suppression';"
    ).fetchone()
    assert n["n"] == 0


def test_delete_suppression_with_flag_but_system_actor_is_refused(conn):
    """H8 gate rule: the flag alone is not enough — actor must be
    'operator', so a pipeline ('system') removal that passes the flag is
    still refused.  An LLM or deterministic principal must never be able to
    lift a suppression."""
    _seed_suppression(conn)
    # Act: the removal with operator_confirmed=True but actor='system'.
    with pytest.raises(WriteGateRefused, match="operator"):
        commit(
            conn,
            action="delete_suppression",
            table_name="suppressions",
            record_id="a@b.test",
            payload={"email_normalized": "a@b.test"},
            run_id="run_1",
            step_id="step_1",
            actor="system",  # NOT the operator — refused even with the flag.
            agent_id="system",
            sql="DELETE FROM suppressions WHERE email_normalized=?;",
            params=("a@b.test",),
            operator_confirmed=True,
        )
    # Assert: the suppression row is still present, and no write_log row
    # claims the removal.
    row = conn.execute(
        "SELECT 1 FROM suppressions WHERE email_normalized='a@b.test';"
    ).fetchone()
    assert row is not None
    n = conn.execute(
        "SELECT count(*) AS n FROM write_log WHERE action='delete_suppression';"
    ).fetchone()
    assert n["n"] == 0


def test_delete_suppression_with_flag_and_operator_actor_succeeds(conn):
    """H8 gate rule: the allowed path — flag set AND actor='operator' —
    removes the row and writes the write_log audit row atomically."""
    _seed_suppression(conn)
    # Act: the valid removal.
    write_id = commit(
        conn,
        action="delete_suppression",
        table_name="suppressions",
        record_id="a@b.test",
        payload={"email_normalized": "a@b.test"},
        run_id="run_1",
        step_id="step_1",
        actor="operator",
        agent_id="operator",
        sql="DELETE FROM suppressions WHERE email_normalized=?;",
        params=("a@b.test",),
        operator_confirmed=True,
    )
    # Assert: the write_id is the audit row's key, the row is gone, and the
    # audit row exists and is attributed to the operator.
    assert write_id
    row = conn.execute(
        "SELECT 1 FROM suppressions WHERE email_normalized='a@b.test';"
    ).fetchone()
    assert row is None
    audit = conn.execute(
        "SELECT action, actor, agent_id FROM write_log WHERE write_id=?;",
        (write_id,),
    ).fetchone()
    assert audit["action"] == "delete_suppression"
    assert audit["actor"] == "operator"
    assert audit["agent_id"] == "operator"


def test_insert_action_unaffected_by_operator_flag_default(conn):
    """H8 gate rule: operator_confirmed is meaningful ONLY for
    delete_suppression — an insert with the flag left at its default False
    must pass exactly as before, so the flag never gates non-removal
    writes."""
    # Act: a plain insert_offer, flag deliberately NOT passed.
    write_id = commit(
        conn,
        action="insert_offer",
        table_name="offers",
        record_id="off_flag_1",
        payload={"slug": "acme"},
        run_id="run_1",
        step_id="step_1",
        actor="system",
        agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_flag_1", "acme", 1),
    )
    # Assert: the insert succeeded and the row exists.
    assert write_id
    row = conn.execute(
        "SELECT * FROM offers WHERE offer_id='off_flag_1';"
    ).fetchone()
    assert row["slug"] == "acme"
