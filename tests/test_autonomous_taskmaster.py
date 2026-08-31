# tests/test_autonomous_taskmaster.py — the bounded outer loop (2026-08-31).
#
# What this file exists to prove:
#   - nothing pending at start -> reports complete WITHOUT ever calling the
#     (expensive, real-Gemini) taskmaster_cli entry point at all.
#   - something pending -> the loop calls taskmaster_cli repeatedly, and
#     stops the SAME iteration the deterministic selector counts both hit
#     zero — not because a model claimed to be finished.
#   - the loop is bounded: if the mocked taskmaster_cli never actually
#     clears the pending work, --max-iterations is respected and the run
#     exits non-zero with a loud, specific message (never silently forever).
#
# taskmaster_cli.main is ALWAYS mocked here — this file must never place a
# real call to Gemini/Vertex; the point under test is the loop's own
# stopping logic, not the agent's behaviour (that is test_taskmaster.py's
# job). Every seeded row goes through the write gate / state machine, never
# a raw INSERT or UPDATE — the same discipline every other test file in
# this repo follows.
from unittest.mock import patch

from app.agents_registry import seed_agent_registry
from app.autonomous_taskmaster import main
from app.db import apply_schema, connect
from app.ids import new_id
from app.state_machine import transition
from app.write_gate import commit


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _conn(scratch_db_target):
    c = connect(scratch_db_target)
    apply_schema(c)
    # The write gate refuses any actor/agent_id not registered in
    # agent_registry (app/write_gate.py) — seed it the same way every
    # other test file's `seeded` fixture does before any commit()/
    # transition() call below.
    seed_agent_registry(c, run_id="r0", step_id="s0")
    return c


def _seed_target(c, *, target_id: str, state: str) -> None:
    """Seed one offer/account/target triple at the given state — the
    minimal shape select_research_pending_targets/select_draft_eligible_targets
    read. Copied from tests/test_taskmaster.py's own helper rather than
    cross-imported, matching that file's own precedent
    (_insert_policy_decision's docstring notes it was copied from
    test_draft_agent.py for the same reason: test files don't import each
    other in this repo)."""
    if c.execute("SELECT 1 FROM offers WHERE offer_id='off_1';").fetchone() is None:
        commit(
            c, action="insert_offer", table_name="offers", record_id="off_1",
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
            params=("off_1", "acme", 1),
        )
    if c.execute("SELECT 1 FROM accounts WHERE account_id='acc_1';").fetchone() is None:
        commit(
            c, action="insert_account", table_name="accounts", record_id="acc_1",
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
                   industry, estimated_size, geo, company_summary, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=("acc_1", "Acme", "acme.test", "acme.test", "Logistics", "11-50", "HK",
                    "Acme coordinates logistics bookings."),
        )
    commit(
        c, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, offer_id, source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, "acc_1", "off_1", "csv", state),
    )


def _insert_policy_decision(c, target_id: str, decision: str) -> None:
    """Insert one policy_decisions row through the write gate — copied from
    tests/test_taskmaster.py's own fixture helper of the same name (test
    files in this repo don't cross-import; see _seed_target's docstring
    above for the precedent)."""
    commit(
        c, action="insert_policy_decision", table_name="policy_decisions",
        record_id=new_id("pol"), payload={"decision": decision},
        run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO policy_decisions
               (policy_decision_id, run_id, step_id, target_id, action, decision,
                risk_level, reasons_json, matched_rules_json, missing_fields_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=(new_id("pol"), "r0", "s0", target_id, "policy_check_phase1", decision,
                "low", "[]", "[]", "[]"),
    )


def _advance(c, target_id: str, from_state: str, to_state: str) -> None:
    """One legal hop through the real state machine — never a raw UPDATE —
    so these tests exercise the exact same VALID_TRANSITIONS table the
    live pipeline is bound by."""
    transition(
        c, target_id=target_id, from_state=from_state, to_state=to_state,
        reason="test: simulating what a real taskmaster_cli call would have done",
        actor="system", run_id="r0", step_id=new_id("step"),
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_nothing_pending_at_start_reports_complete_without_calling_taskmaster(scratch_db_target, capsys):
    """An empty/fully-processed DB must report done on its own — the whole
    point of the pre-loop check is to never spend a Taskmaster (real
    Gemini) call confirming what the selectors already answered for free."""
    conn = _conn(scratch_db_target)
    conn.close()  # nothing seeded — apply_schema alone leaves zero targets

    with patch("app.autonomous_taskmaster.taskmaster_main") as mock_main:
        exit_code = main(["--db", scratch_db_target])

    mock_main.assert_not_called()  # the expensive path must never fire when there's nothing to do
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "AUTONOMOUS RUN COMPLETE" in out
    assert "nothing was pending at start" in out


def test_loop_calls_taskmaster_repeatedly_until_nothing_pending(scratch_db_target, capsys):
    """One target starting at 'new' needs two rounds of real pipeline
    progress (new->researched->scored, then scored->drafted->awaiting_review)
    before either selector reads empty. The mock simulates exactly what a
    successful taskmaster_cli call would have left behind — advancing the
    SAME target through the SAME state machine every real call would use —
    without spending a real Gemini call. The loop must call the mock
    exactly twice: once while research is still pending, once while only
    drafting is, and stop on the THIRD check (which needs no third call)."""
    conn = _conn(scratch_db_target)
    _seed_target(conn, target_id="tgt_1", state="new")
    conn.close()

    calls = {"n": 0}

    def fake_taskmaster_main(argv):
        calls["n"] += 1
        c = connect(scratch_db_target)
        try:
            if calls["n"] == 1:
                # what resume_pending_research would have done: research
                # the stuck target through to 'scored'. A real research run
                # always ends with policy_check_phase1 writing a
                # policy_decisions row (app/agents/phase1.py's "policy_gate"
                # node) before the target can be seen as scored -- the mock
                # must write one too, or the H4 fix (has_allow_policy_decision)
                # would fail-closed and read this target as not draft-eligible.
                _advance(c, "tgt_1", "new", "researched")
                _advance(c, "tgt_1", "researched", "scored")
                _insert_policy_decision(c, "tgt_1", "allow")
            elif calls["n"] == 2:
                # what draft_for_scored would have done: draft it, landing
                # at the human-review gate.
                _advance(c, "tgt_1", "scored", "drafted")
                _advance(c, "tgt_1", "drafted", "awaiting_review")
            else:
                raise AssertionError("the loop must not call taskmaster_main a third time")
        finally:
            c.close()
        return 0

    with patch("app.autonomous_taskmaster.taskmaster_main", side_effect=fake_taskmaster_main) as mock_main:
        exit_code = main(["--db", scratch_db_target, "--max-iterations", "10"])

    assert mock_main.call_count == 2
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "AUTONOMOUS RUN COMPLETE after 2 iteration(s)" in out
    # the state really landed at the human-review gate, not just "not pending"
    conn = _conn(scratch_db_target)
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "awaiting_review"
    conn.close()


def test_permanently_policy_denied_target_does_not_reloop_to_the_bound(scratch_db_target, capsys):
    """The bug reported live 2026-09-01 (ticket H4): a target sitting at
    'scored' with a policy_denied decision stays in state='scored' forever
    -- a refusal is not a state transition, by design, so it must surface
    to the operator rather than silently vanish from selection. Before this
    fix, _pending_count counted that target as draft-eligible purely by
    state, so the stopping check never went to zero and the loop burned
    every iteration up to --max-iterations re-discovering the same refusal.
    After the fix, has_allow_policy_decision excludes it from the pending
    count -- the PRE-loop check (before any taskmaster_main call is even
    made) already reads zero, so the loop reports done without spending a
    single Gemini call re-discovering a refusal that already happened."""
    conn = _conn(scratch_db_target)
    _seed_target(conn, target_id="tgt_1", state="scored")
    _insert_policy_decision(conn, "tgt_1", "deny")
    conn.close()

    with patch("app.autonomous_taskmaster.taskmaster_main") as mock_main:
        exit_code = main(["--db", scratch_db_target, "--max-iterations", "30"])

    mock_main.assert_not_called()  # not 30 calls, not even 1 -- policy_denied is recognised as unwinnable before the loop starts
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "AUTONOMOUS RUN COMPLETE" in out
    assert "nothing was pending at start" in out


def test_bound_exceeded_stops_and_returns_1(scratch_db_target, capsys):
    """A target that a mocked taskmaster_cli never actually advances must
    NOT loop forever — --max-iterations must be honoured exactly, and the
    failure must be loud (non-zero exit, a clear stderr message), never a
    silent hang. This is the CLAUDE.md 'retries must be bounded' guarantee,
    exercised end-to-end rather than just asserted in prose."""
    conn = _conn(scratch_db_target)
    _seed_target(conn, target_id="tgt_1", state="new")
    conn.close()

    with patch("app.autonomous_taskmaster.taskmaster_main", return_value=0) as mock_main:
        exit_code = main(["--db", scratch_db_target, "--max-iterations", "3"])

    assert mock_main.call_count == 3  # exactly the bound — never one more, never one fewer
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "BOUND EXCEEDED" in err
    assert "3 iterations" in err
