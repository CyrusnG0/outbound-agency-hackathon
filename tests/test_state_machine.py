import pytest

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.write_gate import commit
from app.state_machine import transition, StateTransitionRefused, PHASE_1_REACHABLE_STATES


@pytest.fixture
def conn(scratch_db_target):
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # Register the system agent (plan A3) — commit() refuses unregistered agents.
    seed_agent_registry(c, run_id="r1", step_id="s1")
    # seed a minimal account + offer + target row for FK integrity
    commit(
        c, action="insert_offer", table_name="offers", record_id="off_1",
        payload={}, run_id="r1", step_id="s1", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "acme", 1),
    )
    commit(
        c, action="insert_account", table_name="accounts", record_id="acc_1",
        payload={}, run_id="r1", step_id="s1", actor="system", agent_id="system",
        sql="""INSERT INTO accounts
               (account_id, company_name, domain, normalized_domain, created_at, updated_at)
               VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_1", "Acme", "acme.test", "acme.test"),
    )
    commit(
        c, action="insert_target", table_name="targets", record_id="tgt_1",
        payload={}, run_id="r1", step_id="s1", actor="system", agent_id="system",
        sql="""INSERT INTO targets
               (target_id, account_id, offer_id, source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("tgt_1", "acc_1", "off_1", "csv", "new"),
    )
    yield c
    c.close()


def test_valid_transition_updates_state_and_logs(conn):
    transition_id = transition(
        conn, target_id="tgt_1", from_state="new", to_state="researched",
        reason="research success, no enrichment (Phase 1)", actor="system",
        run_id="r1", step_id="s2",
    )
    assert transition_id

    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "researched"

    log = conn.execute(
        "SELECT * FROM state_transitions WHERE transition_id=?;", (transition_id,)
    ).fetchone()
    assert log["previous_state"] == "new"
    assert log["new_state"] == "researched"


def test_invalid_transition_is_refused(conn):
    with pytest.raises(StateTransitionRefused):
        transition(
            conn, target_id="tgt_1", from_state="new", to_state="sent",
            reason="not a real path", actor="system", run_id="r1", step_id="s2",
        )
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "new"


def test_any_to_failed_is_always_valid(conn):
    transition(
        conn, target_id="tgt_1", from_state="new", to_state="failed",
        reason="no_sources_available", actor="system", run_id="r1", step_id="s2",
    )
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"


def test_dry_run_sent_to_replied_is_valid(conn):
    """The C1 edge (docs/state-machine.md §3/§7j): a simulated inbound
    message linked to a DRY_RUN send moves the target dry_run_sent →
    replied — the DRY_RUN mirror of sent → replied, so B5's dead-end
    state has its one outbound edge."""
    # Move the fixture target into dry_run_sent first (a valid hop from
    # its seeded "new" state is not possible, so seed it directly — this
    # test is about the C1 edge, not about reaching dry_run_sent).
    conn.execute("UPDATE targets SET state='dry_run_sent' WHERE target_id='tgt_1';")
    transition_id = transition(
        conn, target_id="tgt_1", from_state="dry_run_sent", to_state="replied",
        reason="inbound_message_linked", actor="system", run_id="r1", step_id="s2",
    )
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "replied"
    log = conn.execute(
        "SELECT previous_state, new_state FROM state_transitions WHERE transition_id=?;",
        (transition_id,),
    ).fetchone()
    assert (log["previous_state"], log["new_state"]) == ("dry_run_sent", "replied")


def test_transitions_carry_monotonic_insert_seq(conn):
    """Ticket C1 extended B5's insert_seq fix to state_transitions — the
    state machine's own audit log.  Two hops landing in the same second
    (exactly what the reply router does: replied → routed → suppressed
    inside one classify_and_route_reply call) share a second-precision
    created_at, so created_at alone cannot order them.  The audit rows
    must carry strictly increasing insert_seq values written by the
    transition INSERT's MAX+1 subquery — asserted here at the source
    (transition()), not via a downstream read."""
    first = transition(
        conn, target_id="tgt_1", from_state="new", to_state="researched",
        reason="research success, no enrichment (Phase 1)", actor="system",
        run_id="r1", step_id="s2",
    )
    second = transition(
        conn, target_id="tgt_1", from_state="researched", to_state="scored",
        reason="scoring success", actor="system", run_id="r1", step_id="s3",
    )
    rows = conn.execute(
        "SELECT insert_seq FROM state_transitions "
        "WHERE transition_id IN (?,?) ORDER BY insert_seq;",
        (first, second),
    ).fetchall()
    seqs = [r["insert_seq"] for r in rows]
    # Both rows carry a value (the writer always populates it) and the
    # second transition's sequence is strictly greater — the monotonic
    # order the history reads rely on.
    assert all(s is not None for s in seqs)
    assert len(seqs) == 2
    assert seqs[0] < seqs[1]


def test_phase_1_reachable_states_excludes_phase_1b_states():
    assert PHASE_1_REACHABLE_STATES == {
        "new", "enriched", "researched", "scored", "watchlist", "not_target", "failed",
    }
    assert "drafted" not in PHASE_1_REACHABLE_STATES
    assert "sent" not in PHASE_1_REACHABLE_STATES
