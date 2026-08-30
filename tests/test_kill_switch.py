"""Tests for the kill switch (ticket B4a): the fail-closed reader
(app/kill_switch.py), the P6 policy gate wiring (app/policy.py), the
agent-entry guardrail (app/agents/guardrail.py), and the Dockerfile
config/ copy.

The suite's hermeticity guard (tests/conftest.py) refuses any real
google.genai.Client construction, so no test here can make a live billable
call.  The two tests that build the REAL draft agents
(test_guardrail_halts_real_draft_run, test_per_agent_disabled_refused_at_entry)
rely on that guard as a backstop: if the kill-switch halt ever failed to
fire, the writer LlmAgent would attempt a model call and the test would
fail loudly instead of spending money.

Every test that engages the switch points the reader at a tmp file via the
OUTBOUND_KILL_SWITCH_PATH env var — the committed config/kill_switch.json
stays enabled=false, so a normal run (and every other test in the suite)
is unaffected.
"""

import asyncio  # asyncio.run drives the minimal ADK pipelines in the guardrail unit tests
import json  # switch files are JSON on disk; the Dockerfile test parses the committed file
from pathlib import Path  # repo-root anchoring for the deploy-artifact assertions

import pytest  # fixtures, tmp_path, monkeypatch

from app.agents.draft import build_draft_agent, run_target_through_draft  # the real draft pipeline the halt tests run against
from app.agents.guardrail import make_kill_switch_callback  # the guardrail under test
from app.agents_registry import seed_agent_registry  # the five principals — the write gate refuses unregistered writers
from app.db import apply_schema, connect  # fresh per-test SQLite database
from app.ids import new_id  # fresh ids for seeded policy rows
from app.kill_switch import read_kill_switch  # the fail-closed reader under test
from app.policy import policy_check_phase1  # the P6-wired policy gate under test
from app.schemas import CompanyProfile, ICPAssessment  # passing P3a/P4 inputs for the P6-dominance test
from app.write_gate import commit  # every seeded core-table row goes through the gate, never a raw INSERT
from google.adk.agents import BaseAgent, SequentialAgent  # minimal-pipeline harness for the guardrail unit tests
from google.adk.events import Event, EventActions  # how the marker node publishes its state delta
from google.adk.runners import Runner  # executes the minimal pipeline against an in-memory session
from google.adk.sessions import InMemorySessionService  # in-memory session store (the same one run_target_through_draft uses)
from google.genai import types  # the synthetic "run" user message ADK's Runner requires


# ── Switch-file helpers ──────────────────────────────────────────────────────
# One writer so every test produces the documented three-field shape
# (runbook.md §1) without drift; tests that need a MALFORMED file bypass it
# deliberately.

def _write_switch(path: Path, payload: dict) -> None:
    """Write a kill-switch file with the documented shape."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _engaged_switch(path: Path) -> Path:
    """Write an engaged switch file and return its path."""
    _write_switch(path, {"enabled": True, "updated_at": "2026-08-23T00:00:00Z", "updated_by": "test"})
    return path


def _disengaged_switch(path: Path) -> Path:
    """Write a disengaged switch file and return its path."""
    _write_switch(path, {"enabled": False, "updated_at": "2026-08-23T00:00:00Z", "updated_by": "test"})
    return path


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conn(scratch_db_target):
    """Fresh SQLite DB with schema, the five seeded principals, and one
    offer/account/target at "scored" with a policy "allow" decision — the
    same shape tests/test_draft_agent.py uses, so the draft-pipeline halt
    tests run against a realistic precondition set."""
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # Register the five principals (system/operator/icp_judge/draft_writer/
    # draft_critic) — commit() refuses unregistered agents, and the
    # per-agent halt test disables draft_writer's row afterwards.
    seed_agent_registry(c, run_id="r0", step_id="s0")
    commit(
        c, action="insert_offer", table_name="offers", record_id="off_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "acme", 1),
    )
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
        c, action="insert_target", table_name="targets", record_id="tgt_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, offer_id, source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("tgt_1", "acc_1", "off_1", "csv", "scored"),
    )
    # tgt_1 gets a policy "allow" decision — the draft runner's second
    # precondition (latest policy_decisions row must be allow), so the
    # halt tests prove the HALT (not a precondition refusal) stopped the run.
    commit(
        c, action="insert_policy_decision", table_name="policy_decisions",
        record_id=new_id("pol"), payload={"decision": "allow"},
        run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO policy_decisions
               (policy_decision_id, run_id, step_id, target_id, action, decision,
                risk_level, reasons_json, matched_rules_json, missing_fields_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=(new_id("pol"), "r0", "s0", "tgt_1", "score_lead", "allow",
                "low", "[]", "[]", "[]"),
    )
    yield c
    c.close()


def _disable_agent_via_gate(c, agent_id: str) -> None:
    """Set agent_registry.enabled=0 for one principal THROUGH the write
    gate (core-table writes never bypass it, in tests or in code).  Uses
    the seeder's own upsert action with an ON CONFLICT clause that DOES
    update enabled — the deliberate inverse of the seeder's clause, which
    preserves enabled on conflict so an operator's disable survives
    re-seeds."""
    commit(
        c, action="insert_agent_registry", table_name="agent_registry",
        record_id=agent_id, payload={"enabled": 0},
        run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO agent_registry
               (agent_id, display_name, description, model_alias,
                allowed_actions, allowed_transitions, enabled, created_at)
               VALUES (?,?,?,?,?,?,?,datetime('now'))
               ON CONFLICT(agent_id) DO UPDATE SET enabled=0""",
        params=(agent_id, agent_id, "disabled in test", None, "[]", "*", 0),
    )


# ── 1. The reader: healthy states ────────────────────────────────────────────

def test_committed_switch_file_reads_disengaged():
    # The repo's committed config/kill_switch.json is the default read path
    # (repo-root anchored, cwd-independent) and MUST be disengaged — the
    # ticket's definition of done: "committed with enabled: false, so a
    # normal run is unaffected".  If this fails, someone flipped the real
    # switch (or the file vanished, which would fail closed and trip the
    # engaged=True assertion) — both are exactly what this test exists to
    # catch.
    state = read_kill_switch()  # default path — no env var, no argument
    assert state.engaged is False
    assert state.reason == ""  # healthy disengaged state carries no fault


def test_engaged_switch_reads_engaged_true(tmp_path):
    # The operator's deliberate halt: enabled=true must read engaged with
    # a reason that distinguishes "someone flipped the switch" from a
    # fail-closed fault (the reason names who and when).
    path = _engaged_switch(tmp_path / "on.json")
    state = read_kill_switch(str(path))
    assert state.engaged is True
    assert "engaged" in state.reason
    assert state.updated_by == "test"  # metadata carried through for the halt message


# ── 2. The reader: fail-closed faults (one test per fault, reasons
# distinguishable) ────────────────────────────────────────────────────────────

def test_fail_closed_missing_file(tmp_path):
    # A DELETED switch file must halt, never disable the halt — deleting
    # the file is the first thing a bad deploy (or an attacker) would try,
    # and a fail-open reader would invert the control into a liability.
    missing = tmp_path / "nope.json"
    state = read_kill_switch(str(missing))
    assert state.engaged is True
    assert "not found" in state.reason  # names the specific fault


def test_fail_closed_invalid_json(tmp_path):
    # A file that exists but is not JSON is a corrupt switch — its intent
    # must not be guessed, so it fails closed with the parser's message.
    bad = tmp_path / "bad.json"
    bad.write_text("{not json at all", encoding="utf-8")
    state = read_kill_switch(str(bad))
    assert state.engaged is True
    assert "not valid JSON" in state.reason


def test_fail_closed_missing_enabled_key(tmp_path):
    # A JSON object without the operative field is a malformed switch —
    # "no enabled key" must not read as "not engaged".
    no_key = tmp_path / "no_key.json"
    _write_switch(no_key, {"updated_at": "2026-08-23T00:00:00Z", "updated_by": "test"})
    state = read_kill_switch(str(no_key))
    assert state.engaged is True
    assert "no 'enabled' field" in state.reason


def test_fail_closed_enabled_as_string(tmp_path):
    # THE string trap: in Python the string "false" is truthy, so a
    # truthiness test would misread `"enabled": "false"` — and a naive
    # `== False` comparison would misread it the other way.  The only
    # correct reading of a non-boolean enabled is MALFORMED, so it fails
    # closed with a reason naming the type, and `"enabled": "0"` (also
    # truthy) cannot slip through either.
    str_false = tmp_path / "str_false.json"
    _write_switch(str_false, {"enabled": "false", "updated_at": "", "updated_by": ""})
    state = read_kill_switch(str(str_false))
    assert state.engaged is True
    assert "not a JSON boolean" in state.reason
    assert "'false'" in state.reason  # the specific fault is named, not just truthy


# ── 3. The reader: no caching, env override ──────────────────────────────────

def test_read_is_not_cached(tmp_path):
    # THE test that fails if someone adds an lru_cache: read once, REWRITE
    # the same path, read again — the second read must reflect the new
    # value.  A cached read would keep reporting the stale disengaged
    # state, which is exactly the failure mode that makes a kill switch
    # useless mid-run (the only moment it matters).
    path = tmp_path / "flip.json"
    _disengaged_switch(path)
    assert read_kill_switch(str(path)).engaged is False
    _engaged_switch(path)  # rewrite the SAME file — the flip an operator makes mid-run
    assert read_kill_switch(str(path)).engaged is True


def test_env_var_path_override_and_explicit_path(tmp_path, monkeypatch):
    # OUTBOUND_KILL_SWITCH_PATH repoints the FILE (it never sets the
    # state — no file-says-off / env-says-on split-brain, runbook.md §1).
    # Read at call time, so a repoint between reads takes effect without a
    # restart; an explicit path argument outranks the env var.
    engaged = _engaged_switch(tmp_path / "engaged.json")
    disengaged = _disengaged_switch(tmp_path / "disengaged.json")
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(engaged))
    assert read_kill_switch().engaged is True  # env var honoured when no argument is given
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(disengaged))
    assert read_kill_switch().engaged is False  # repointed env var takes effect immediately
    assert read_kill_switch(str(engaged)).engaged is True  # explicit path outranks the env var


# ── 4. P6 in policy_check_phase1 ─────────────────────────────────────────────

def test_p6_dominates_a_passing_p3a_p4(conn, tmp_path, monkeypatch):
    # Per docs/policy-matrix.md P6: "If the kill switch is on, all outbound
    # actions → deny, unconditionally."  Unconditionally means a target
    # that PASSES P3a (complete profile) and P4 (score 80 >= 60) is still
    # denied when the switch is engaged — the switch outranks a clean
    # score.  The decision row must still be PERSISTED (never skip logs —
    # a silent deny is indistinguishable from a broken gate).
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(_engaged_switch(tmp_path / "on.json")))
    profile = CompanyProfile(one_line_summary="A real summary", confidence=0.7)
    assessment = ICPAssessment(fit_label="good_fit", fit_score=80, fit_reasons=["x"], non_fit_reasons=[])
    decision = policy_check_phase1(
        conn, company_profile=profile, icp_assessment=assessment, signals=[],
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    assert decision.decision == "deny"  # P6's mandated outcome, regardless of score
    assert decision.risk_level == "high"  # deny always maps to high
    assert decision.matched_rules == ["P6"]  # P6 alone — P3a/P4 never evaluated under P6
    assert decision.required_fields_missing == []  # no completeness verdict exists under P6
    assert decision.reasons and "engaged" in decision.reasons[0]  # the switch's reason travels in the decision
    # The deny is persisted to policy_decisions (through the write gate,
    # same as every P3a/P4 row) so the audit trail names why the target
    # was denied even though its score cleared the floor.  Scoped to
    # run_id='r1' — the fixture's pre-seeded allow row is 'r0', and
    # created_at is second-precision TEXT so a same-second ORDER BY is
    # ambiguous (the same caveat the draft runner documents).
    row = conn.execute(
        "SELECT * FROM policy_decisions WHERE target_id='tgt_1' AND run_id='r1';"
    ).fetchone()
    assert row is not None
    assert json.loads(row["matched_rules_json"]) == ["P6"]
    assert json.loads(row["reasons_json"])[0] == decision.reasons[0]


def test_p6_disengaged_leaves_p3a_p4_behaviour_unchanged(conn, tmp_path, monkeypatch):
    # The byte-identical contract: with a disengaged switch the gate must
    # behave exactly as before B4a — a complete, high-scoring target is
    # allowed, and a low score still denies on P4 alone (no P6 anywhere).
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(_disengaged_switch(tmp_path / "off.json")))
    profile = CompanyProfile(one_line_summary="A real summary", confidence=0.7)
    allow_assessment = ICPAssessment(fit_label="good_fit", fit_score=80, fit_reasons=["x"], non_fit_reasons=[])
    allow_decision = policy_check_phase1(
        conn, company_profile=profile, icp_assessment=allow_assessment, signals=[],
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    assert allow_decision.decision == "allow"
    assert "P6" not in allow_decision.matched_rules  # the disengaged switch leaves no trace in the verdict
    deny_assessment = ICPAssessment(fit_label="watchlist", fit_score=45, fit_reasons=[], non_fit_reasons=[])
    deny_decision = policy_check_phase1(
        conn, company_profile=profile, icp_assessment=deny_assessment, signals=[],
        target_id="tgt_1", run_id="r1", step_id="s2",
    )
    assert deny_decision.decision == "deny"
    assert deny_decision.matched_rules == ["P4"]  # P4 alone fires — unchanged from before B4a


# ── 5. The guardrail: global halt on the real draft pipeline ─────────────────

def test_guardrail_halts_real_draft_run(conn, tmp_path, monkeypatch):
    # The ticket's core observable: flip the switch file, run the REAL
    # draft pipeline (build_draft_agent + run_target_through_draft), and
    # assert the agent's work did not happen — no state transition, no
    # draft version row — while a failed step row records the halt.
    #
    # GEMINI_FLASH_MODEL is pinned so build_draft_agent can CONSTRUCT the
    # real writer/critic LlmAgents (construction never builds a genai
    # client — measured, tests/conftest.py); if the halt ever failed to
    # fire, the writer's first model turn would trip the autouse guard and
    # fail this test loudly instead of spending money.
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(_engaged_switch(tmp_path / "on.json")))
    agent = build_draft_agent(conn)  # the real pipeline with the guardrail attached at its root
    outcome = run_target_through_draft(agent, conn=conn, target_id="tgt_1", run_id="r1")
    assert outcome == "scored"  # the halt left the target where it was — the runner's honest degrade
    # The agent's work did not happen: no revision persisted, no state
    # changed (the switch aborts the RUN, not the target — the target
    # stays in scored and is re-run after disengaging).
    assert conn.execute("SELECT COUNT(*) AS n FROM message_draft_versions;").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM state_transitions;").fetchone()["n"] == 0
    # The halt IS observable: one failed kill_switch step row naming the
    # halted agent and the reason (never skip logging — a silent halt is
    # indistinguishable from a broken pipeline).
    rows = conn.execute("SELECT * FROM steps WHERE tool_name='kill_switch';").fetchall()
    assert len(rows) == 1  # the root halt fired once; no sub-agent ever entered
    assert rows[0]["status"] == "failed"
    output = json.loads(rows[0]["output_json"])
    assert output["halted_agent"] == "draft_loop"
    assert output["scope"] == "global"
    assert "engaged" in output["reason"]


# ── 6. The guardrail: per-agent refusal at entry ─────────────────────────────

def test_per_agent_disabled_refused_at_entry(conn, tmp_path, monkeypatch):
    # docs/policy-matrix.md §3a's per-agent kill switch, moved from
    # write-time to ENTRY: with draft_writer disabled in agent_registry,
    # the writer must be refused BEFORE it does any work — no model tokens
    # burned (today it runs and only its attributed writes are refused deep
    # into the work).  The refusal is logged and the loop exits via the
    # escalate signal, so the critic and persist node never run either.
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")  # real writer/critic construction needs the pin (see the global-halt test)
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(_disengaged_switch(tmp_path / "off.json")))  # the GLOBAL switch stays off — only the per-agent check fires
    _disable_agent_via_gate(conn, "draft_writer")  # the operator's disable, written through the gate
    agent = build_draft_agent(conn)  # the real pipeline: root + writer + critic guardrails attached
    outcome = run_target_through_draft(agent, conn=conn, target_id="tgt_1", run_id="r1")
    assert outcome == "scored"  # the loop produced nothing persistable — the honest degrade
    # No work happened: no revisions, no transitions.  (The writer never
    # ran at all — had it run, its first model turn would have tripped the
    # autouse live-client guard and failed this test.)
    assert conn.execute("SELECT COUNT(*) AS n FROM message_draft_versions;").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM state_transitions;").fetchone()["n"] == 0
    # The refusal is logged with the disabled agent's name and the reason.
    rows = conn.execute("SELECT * FROM steps WHERE tool_name='kill_switch';").fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    output = json.loads(rows[0]["output_json"])
    assert output["halted_agent"] == "draft_writer"
    assert output["scope"] == "per_agent"
    assert "disabled" in output["reason"]


# ── 7. The guardrail: sentinel + passthrough on a minimal pipeline ───────────

class _MarkerNode(BaseAgent):
    """Minimal deterministic node: records that it ran (via a private attr —
    BaseAgent is pydantic extra='forbid', so public attribute assignment
    would raise) and publishes a marker state delta."""

    def __init__(self):
        super().__init__(name="marker")  # not a registered principal — the per-agent check passes it
        self._ran = False  # private attr: read by the tests to prove the node ran (or did not)

    async def _run_async_impl(self, ctx):
        self._ran = True  # record BEFORE publishing, so a halted run leaves this False
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"marker_ran": True}),
        )


def _run_minimal_pipeline(agent, conn) -> dict:
    """Drive a minimal agent through ADK's Runner the same way
    run_target_through_phase1 does: in-memory session service, seeded
    run_id/target_id (the guardrail's log_step reads them from session
    state), terminal state read back from the session."""
    async def _run():
        session_service = InMemorySessionService()
        runner = Runner(
            app_name="outbound", agent=agent,
            session_service=session_service, auto_create_session=True,
        )
        async for _ in runner.run_async(
            user_id="operator", session_id="tgt_1",
            new_message=types.Content(role="user", parts=[types.Part(text="run")]),
            state_delta={"run_id": "r1", "target_id": "tgt_1"},
        ):
            pass  # events consumed for their side effects; terminal state read below
        session = await session_service.get_session(
            app_name="outbound", user_id="operator", session_id="tgt_1",
        )
        return session.state

    return asyncio.run(_run())


def test_guardrail_engaged_publishes_sentinel_and_skips_node(conn, tmp_path, monkeypatch):
    # The sentinel half of the halt contract (the plan's "must do both"):
    # a halt at the ROOT must publish final_state="failed" into session
    # state — without it, run_target_through_phase1 would KeyError on the
    # missing terminal state instead of returning a clean "failed", and a
    # SUB-agent halt could not short-circuit downstream deterministic
    # nodes (their existing guards all test final_state).  The reason
    # travels in kill_switch_reason so the terminal state names the cause.
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(_engaged_switch(tmp_path / "on.json")))
    node = _MarkerNode()
    agent = SequentialAgent(name="phase1", sub_agents=[node])
    agent.before_agent_callback = make_kill_switch_callback(conn=conn)
    state = _run_minimal_pipeline(agent, conn)
    assert node._ran is False  # the agent's work did not happen — the node never ran
    assert state.get("final_state") == "failed"  # the sentinel every downstream short-circuit honours
    assert "engaged" in state.get("kill_switch_reason", "")  # the why, in the terminal state
    rows = conn.execute("SELECT * FROM steps WHERE tool_name='kill_switch';").fetchall()
    assert len(rows) == 1  # the halt is logged (never skip logging)
    assert rows[0]["status"] == "failed"


def test_guardrail_disengaged_lets_the_node_run(conn, tmp_path, monkeypatch):
    # The passthrough half: a disengaged switch (and an unregistered node
    # name, which the per-agent check passes) must return None and let the
    # agent run normally — no halt row, the node's work happens.
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(_disengaged_switch(tmp_path / "off.json")))
    node = _MarkerNode()
    agent = SequentialAgent(name="phase1", sub_agents=[node])
    agent.before_agent_callback = make_kill_switch_callback(conn=conn)
    state = _run_minimal_pipeline(agent, conn)
    assert node._ran is True  # the node ran — the guardrail allowed it
    assert state.get("marker_ran") is True  # its published state delta landed
    assert state.get("final_state") is None  # no sentinel — nothing was halted
    assert conn.execute("SELECT COUNT(*) AS n FROM steps WHERE tool_name='kill_switch';").fetchone()["n"] == 0  # no halt row — nothing to trace


# ── 8. The deploy artifact: config/ ships in the image ───────────────────────

ROOT = Path(__file__).resolve().parent.parent  # the artifacts live at the repo root, never under tests/
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"


def _dockerfile_instructions() -> list[str]:
    # Only Dockerfile INSTRUCTION lines (not comments) — the same
    # comment-trap avoidance tests/test_deploy_artifacts.py uses.
    return [
        line for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _dockerignore_patterns() -> list[str]:
    # One pattern per non-blank, non-comment line — the same parser
    # tests/test_deploy_artifacts.py uses, so a comment mentioning
    # "config" cannot satisfy (or trip) an assertion.
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_dockerfile_copies_config_and_switch_file_is_committed():
    # Failure prevented: the image shipping app/ only, while
    # app/kill_switch.py resolves config/kill_switch.json relative to the
    # REPO ROOT and FAILS CLOSED — a deployed container without the file
    # would read the switch as engaged and halt every pipeline entry on
    # boot (ticket B4a's §3.6).  config/ must be copied, .dockerignore
    # must not exclude it (config/ holds no secrets — model aliases and
    # offer YAMLs only), and the committed file must be present and
    # disengaged so a normal run is unaffected.
    instructions = _dockerfile_instructions()
    assert "COPY config/ ./config/" in instructions
    assert "COPY app/ ./app/" in instructions
    patterns = _dockerignore_patterns()
    assert "config/" not in patterns
    assert "config" not in patterns
    switch_file = ROOT / "config" / "kill_switch.json"
    assert switch_file.is_file(), "config/kill_switch.json missing — the fail-closed reader would engage on boot"
    committed = json.loads(switch_file.read_text(encoding="utf-8"))
    assert committed["enabled"] is False  # the DoD: committed disengaged, so a normal run is unaffected
