"""
Tests for plan task A3 — agent_registry and per-agent write attribution.

Covers the write gate's new per-agent capability layer:
- seeding is idempotent (two runs, two rows, no error)
- a registered agent with the action allowed writes, and write_log.agent_id
  records which agent wrote
- an unregistered agent is refused with no data row written
- a disabled agent (enabled=0) is refused
- an action outside an agent's allowed_actions is refused
- log_step records steps.agent_id
"""

import json

import pytest

from app.agents_registry import seed_agent_registry
from app.db import apply_schema, connect
from app.tools.log_step import log_step
from app.write_gate import WriteGateRefused, commit


@pytest.fixture
def conn(scratch_db_target):
    """Fresh in-tmpdir SQLite database with the full schema applied.

    Deliberately NOT seeded — tests that need a populated registry call the
    `seeded` fixture, so the unregistered-agent test can run against a
    registry that only contains what the seeder put there."""
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    yield c
    c.close()


@pytest.fixture
def seeded(conn):
    """The conn fixture plus the two bootstrap principals registered."""
    seed_agent_registry(conn, run_id="seed_run", step_id="seed_step")
    return conn


def test_seeding_is_idempotent(conn):
    """Two seeder runs must leave exactly the eight existing principals.

    The ON CONFLICT upsert turns the second run into an UPDATE of the same
    rows, so there is no error and no duplicate row.  B2c: the set now
    includes icp_judge, the LLM judge principal added by ticket B2c; B3:
    draft_writer and draft_critic, the writer⇄critic loop principals; C1:
    reply_classifier, the reply classifier principal; C4: taskmaster, the
    natural-language root agent (whose allowed_actions is deliberately
    empty — see test_taskmaster.py); demo 2026-08-30: meeting_scheduler,
    the real-calendar scheduling principal (see test_follow_up_draft.py)."""
    seed_agent_registry(conn, run_id="r1", step_id="s1")
    seed_agent_registry(conn, run_id="r2", step_id="s2")
    rows = conn.execute("SELECT agent_id FROM agent_registry;").fetchall()
    assert sorted(r["agent_id"] for r in rows) == [
        "draft_critic", "draft_writer", "icp_judge", "meeting_scheduler",
        "operator", "reply_classifier", "system", "taskmaster",
    ]


def test_icp_judge_is_registered_as_an_llm_principal(conn):
    """B2c: the judge must be a registered principal with its own agent_id
    (never reusing "system"), enabled by default, and carrying its model
    alias — the registry's model_alias column is NULL only for deterministic
    principals, and the judge is an LLM principal, so its row must name the
    judge_model role alias."""
    seed_agent_registry(conn, run_id="r1", step_id="s1")
    row = conn.execute(
        "SELECT agent_id, model_alias, enabled FROM agent_registry "
        "WHERE agent_id='icp_judge';"
    ).fetchone()
    assert row is not None, "icp_judge must be seeded in agent_registry"
    assert row["model_alias"] == "judge_model", (
        "the judge is an LLM principal — its model_alias must name the "
        "config/models.yaml role it calls through"
    )
    assert row["enabled"] == 1, "the judge starts enabled; the kill switch is opt-out"


def test_reply_classifier_is_registered_as_an_llm_principal(conn):
    """C1: the reply classifier must be a registered principal with its
    own agent_id (never reusing "system"), enabled by default, and
    carrying the reply_classifier_model role alias — the same contract
    the judge's principal satisfies (model_alias is NULL only for
    deterministic principals, and the classifier is an LLM principal)."""
    seed_agent_registry(conn, run_id="r1", step_id="s1")
    row = conn.execute(
        "SELECT agent_id, model_alias, enabled FROM agent_registry "
        "WHERE agent_id='reply_classifier';"
    ).fetchone()
    assert row is not None, "reply_classifier must be seeded in agent_registry"
    assert row["model_alias"] == "reply_classifier_model", (
        "the classifier is an LLM principal — its model_alias must name "
        "the config/models.yaml role it calls through"
    )
    assert row["enabled"] == 1, "the classifier starts enabled; the kill switch is opt-out"


def test_registered_agent_with_allowed_action_writes_and_logs_agent_id(seeded):
    """Happy path: a registered, enabled agent writing an allowed action
    succeeds, and the write_log audit row records its agent_id."""
    write_id = commit(
        seeded,
        action="insert_offer",
        table_name="offers",
        record_id="off_1",
        payload={"slug": "acme"},
        run_id="run_1",
        step_id="step_1",
        actor="system",
        agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "acme", 1),
    )
    # The data row landed.
    offer = seeded.execute("SELECT slug FROM offers WHERE offer_id='off_1';").fetchone()
    assert offer["slug"] == "acme"
    # The audit row names which specific agent wrote.
    log_row = seeded.execute(
        "SELECT agent_id FROM write_log WHERE write_id=?;", (write_id,)
    ).fetchone()
    assert log_row["agent_id"] == "system"


def test_unregistered_agent_is_refused_and_writes_nothing(seeded):
    """An agent_id with no registry row must be refused before any SQL runs —
    no data row may exist afterwards."""
    with pytest.raises(WriteGateRefused, match="not registered"):
        commit(
            seeded,
            action="insert_offer",
            table_name="offers",
            record_id="off_1",
            payload={},
            run_id="run_1",
            step_id="step_1",
            actor="system",
            agent_id="ghost_agent",  # Never seeded — must be refused.
            sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
            params=("off_1", "acme", 1),
        )
    # The refusal happened before the mutation — the offers table stays empty.
    assert seeded.execute("SELECT * FROM offers;").fetchone() is None


def test_disabled_agent_is_refused(seeded):
    """enabled=0 is the per-agent kill switch — even an otherwise-authorized
    write must be refused. The UPDATE here is direct test setup, not a
    pipeline write path."""
    seeded.execute("UPDATE agent_registry SET enabled=0 WHERE agent_id='system';")
    with pytest.raises(WriteGateRefused, match="disabled"):
        commit(
            seeded,
            action="insert_offer",
            table_name="offers",
            record_id="off_1",
            payload={},
            run_id="run_1",
            step_id="step_1",
            actor="system",
            agent_id="system",
            sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
            params=("off_1", "acme", 1),
        )


def test_action_outside_allowed_actions_is_refused(seeded):
    """A registered agent attempting an action outside its allowed_actions
    JSON array must be refused — the KNOWN_ACTIONS allowlist is per-agent now.
    The limited agent row is inserted directly as fixture setup."""
    seeded.execute(
        "INSERT INTO agent_registry (agent_id, display_name, description, model_alias, "
        "allowed_actions, allowed_transitions, enabled, created_at) "
        "VALUES (?,?,?,?,?,?,?,datetime('now'))",
        (
            "limited_agent", "Limited test agent",
            "Fixture agent that may only insert offers", None,
            json.dumps(["insert_offer"]),  # A deliberately narrow capability set.
            "*", 1,
        ),
    )
    with pytest.raises(WriteGateRefused, match="allowed_actions"):
        commit(
            seeded,
            action="update_target_score",  # NOT in limited_agent's allowed_actions.
            table_name="targets",
            record_id="tgt_1",
            payload={},
            run_id="run_1",
            step_id="step_1",
            actor="system",
            agent_id="limited_agent",
            sql="UPDATE targets SET score=? WHERE target_id=?;",
            params=(10, "tgt_1"),
        )


def test_log_step_records_agent_id(conn):
    """log_step writes the calling agent's id into steps.agent_id, so the
    trace log stays attributable once multiple agents write concurrently."""
    log_step(
        conn,
        run_id="run_1",
        step_id="step_1",
        target_id="tgt_1",
        tool_name="score_lead",
        agent_id="system",
        input_data={"has_contact_data": False},
        output_data={"fit_score": 65},
        status="success",
    )
    row = conn.execute("SELECT agent_id FROM steps WHERE step_id='step_1';").fetchone()
    assert row["agent_id"] == "system"
