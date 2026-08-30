# tests/test_agents_phase1.py — A4a port of tests/test_graph.py: the same
# Phase 1 pipeline tests now run against the ADK agent instead of the
# LangGraph StateGraph, plus tests for ADK's state_delta plumbing, the A6
# step-id fix, and the A4b policy gate.  B1b (research LlmAgent) adaptations:
# the pipeline's first node is now a REAL LlmAgent that would make live
# billable LLM calls, so every pipeline test patches
# app.agents.phase1.build_research_agent with an offline stand-in (see
# _StubResearchAgent below).  The real research agent's behaviour is covered
# in tests/test_research_agent.py; these tests keep covering the pipeline
# contract downstream of it, with every assertion intact.
#
# B2c adaptation: the score node now runs the ICP judge after score_lead,
# so every test that reaches scoring also patches
# app.agents.phase1.judge_icp_module.judge_icp with return_value=None — the
# documented judge-failure degradation (deterministic label stands), which
# keeps these tests' pre-B2c assertions byte-identical.  The judge's real
# happy path (divergence, persistence, attribution) is covered in
# tests/test_judge_icp.py, which patches the LLM seam
# (app.tools.judge_icp._call_judge_llm) instead and lets the real function
# run against the database.
import asyncio  # drives ADK's async Runner from a synchronous test
from unittest.mock import patch  # mock every network/LLM boundary, same as the old graph tests

import pytest

from app.agents.phase1 import (  # the pipeline under test, plus the retained node the A6 test drives directly
    FetchAndNormalizeNode,
    build_phase1_agent,
    run_target_through_phase1,
)
from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.write_gate import commit
from app.schemas import CompanyProfile, Signal
from app.tools.fetch_sources import NormalizedSource
from app.tools.log_step import log_step  # needed only by the A6 step-id-collision test, whose fake fetch_sources must write a REAL steps row
from google.adk.agents import BaseAgent  # base class of the offline research stand-in
from google.adk.events import Event, EventActions  # how the stand-in publishes extracted_text, exactly like output_key would
from google.adk.runners import Runner  # needed only by the state_delta plumbing test (inline runner)
from google.adk.sessions import InMemorySessionService  # needed only by the state_delta plumbing test
from google.genai import types  # builds the synthetic user message for the inline runner


class _StubResearchAgent(BaseAgent):
    """Offline stand-in for the B1b research LlmAgent.

    The real agent (app/agents/research.py) makes live billable LLM calls;
    these pipeline tests must stay offline, so they patch build_research_agent
    with this stub, which publishes a fixed extracted_text (or nothing, when
    findings is None) through the same state_delta mechanism the real agent's
    output_key uses.  findings=None reproduces "the agent found nothing" —
    the shape research_bookkeeping must route to failed."""

    def __init__(self, findings: str | None):
        super().__init__(name="research")  # same stable name as the real agent
        self._findings = findings  # private attr — pydantic forbids public assignment (same as the nodes' _conn)

    async def _run_async_impl(self, ctx):
        # Mimic output_key: publish the findings into session state under
        # extracted_text.  findings=None yields nothing — the key stays
        # absent, which is the real agent's behaviour on an empty final turn.
        if self._findings is not None:
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"extracted_text": self._findings}),
            )


@pytest.fixture
def conn(scratch_db_target):
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # Register the system agent (plan A3) — commit() refuses unregistered agents.
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
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain, created_at, updated_at)
               VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_1", "Acme", "acme.test", "acme.test"),
    )
    commit(
        c, action="insert_target", table_name="targets", record_id="tgt_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, offer_id, source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("tgt_1", "acc_1", "off_1", "csv", "new"),
    )
    yield c
    c.close()


def test_full_pipeline_reaches_scored_or_watchlist_or_not_target(conn):
    # B1b: the research stage is stubbed offline (see _StubResearchAgent) —
    # its findings stand in for what the real agent's output_key would
    # publish.  Everything downstream of research_bookkeeping is unchanged
    # and still driven for real with the LLM boundaries mocked.
    fake_profile = CompanyProfile(one_line_summary="Acme does logistics", industry="Logistics", confidence=0.8)
    fake_signals = [Signal(
        signal_type="hiring_relevant_role", signal_value="Hiring ops manager",
        signal_strength=0.8,
        # B2a: every Signal now requires an evidence quote — a placeholder
        # here; these tests assert on pipeline routing and step logging, not
        # on quote verification.
        evidence_quote="hiring an operations manager for the team",
    )]

    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=fake_profile), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1")

    assert final_state in ("scored", "watchlist", "not_target")
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == final_state


def test_pipeline_with_zero_sources_reaches_failed(conn):
    # B1b: "zero sources" now manifests as "the research agent found
    # nothing" (findings=None → no extracted_text in state), which
    # research_bookkeeping routes to failed.  NOTE (B1d): this shape is
    # "the agent produced no output", so its reason is
    # research_agent_no_output_phase1, NOT §7c's no_sources_available —
    # that one is reserved for the agent honestly reporting the sentinel.
    # This test asserts the state, not the reason; the reason cases are
    # pinned in tests/test_research_agent.py.  The summarize LLM boundary is patched to assert the
    # ticket's "no downstream node runs" guarantee AND to keep the suite
    # offline even if bookkeeping ever regressed.
    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings=None)), \
         patch("app.tools.summarize_company.call_structured") as fake_summarize, \
         patch("app.tools.detect_signals._call_detect_signals") as fake_detect:
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1")
    assert final_state == "failed"
    # No downstream LLM node may run after a bookkeeping failure — the
    # short-circuit is the whole point of the governance node.
    fake_summarize.assert_not_called()
    fake_detect.assert_not_called()


def test_fetch_and_normalize_write_two_distinct_step_rows(conn):
    """A6 regression: FetchAndNormalizeNode must hand EACH tool call its OWN
    step id.  steps.step_id is the PRIMARY KEY (app/db.py) and both
    fetch_sources and normalize_sources call log_step, so the pre-A6 code —
    one step_id passed to both tools — made the SECOND insert raise
    sqlite3.IntegrityError: UNIQUE constraint failed: steps.step_id on the
    very first target with a fetchable website, killing the whole run before
    any LLM call.

    IMPORTANT — why this test is not like its neighbors: every OTHER test in
    this file patches fetch_sources with a plain return_value=[...], so the
    mock never calls log_step and only ONE row is ever inserted for this
    node — the collision is structurally impossible in those tests, which is
    exactly why the suite stayed green at 132 passed while the real pipeline
    crashed.  Do NOT "simplify" this test back into that mocked pattern: the
    side_effect fake below is the whole point.  It reproduces the REAL
    fetch_sources' only behavior that matters here — writing a steps row via
    log_step with the step_id the node handed it — so a collision can
    actually occur (and did, pre-fix: this test failed with IntegrityError).

    B1b adaptation: FetchAndNormalizeNode is retained in app/agents/phase1.py
    (superseded in the pipeline by the research LlmAgent), and this test keeps
    exercising it by standing it in FOR the research stage — the
    build_research_agent patch returns the real retained node, which drives
    the mocked fetch_sources and the real normalize_sources exactly as
    before.  research_bookkeeping still runs after it (logging its own row
    and re-issuing the new→researched transition, both harmless for these
    assertions), so the pipeline contract downstream is unchanged.
    """
    # Fake replaces only fetch_sources' network layer; its logging behavior
    # is kept REAL so the collision this test exists to catch is possible.
    def fake_fetch_sources(conn, *, domain, target_id, run_id, step_id):
        # Mirror the real success path in app/tools/fetch_sources.py: log one
        # "fetch_company_page" steps row keyed by the step_id the node passed
        # in.  A second insert with the same id (the bug) violates the PK.
        log_step(
            conn, run_id=run_id, step_id=step_id, target_id=target_id,
            tool_name="fetch_company_page", agent_id="system",
            input_data={"domain": domain},
            output_data={"chars_extracted": 27}, status="success",
        )
        # Return one usable source so normalize_sources takes its happy path
        # and writes its own "normalize_sources" step row.
        return [NormalizedSource("company_website", "https://acme.test", "Acme does logistics. Hiring ops manager.", "t", 0.8, 1, "static")]

    # Downstream LLM boundaries are mocked exactly like the other pipeline
    # tests — irrelevant to the collision, but they must not make real
    # billable LLM calls.
    fake_profile = CompanyProfile(one_line_summary="Acme does logistics", industry="Logistics", confidence=0.8)
    fake_signals = [Signal(
        signal_type="hiring_relevant_role", signal_value="Hiring ops manager",
        signal_strength=0.8,
        # B2a: every Signal now requires an evidence quote — a placeholder
        # here; these tests assert on pipeline routing and step logging, not
        # on quote verification.
        evidence_quote="hiring an operations manager for the team",
    )]

    # NOTE the side_effect fake above (NOT return_value) — this is the one
    # test in the file where fetch_sources actually writes to steps.
    with patch("app.agents.phase1.build_research_agent",
               return_value=FetchAndNormalizeNode(name="research", conn=conn)), \
         patch("app.tools.fetch_sources.fetch_sources", side_effect=fake_fetch_sources), \
         patch("app.tools.summarize_company.call_structured", return_value=fake_profile), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
        agent = build_phase1_agent(conn)
        # Must NOT raise IntegrityError — pre-fix this call crashed with
        # sqlite3.IntegrityError: UNIQUE constraint failed: steps.step_id.
        final_state = run_target_through_phase1(agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1")

    # The run completed normally on the happy path (not just "didn't crash").
    assert final_state in ("scored", "watchlist", "not_target")
    # Two tools ran = two rows in steps, one per tool, with DISTINCT step_ids.
    rows = conn.execute(
        "SELECT step_id, tool_name FROM steps "
        "WHERE target_id='tgt_1' AND tool_name IN ('fetch_company_page','normalize_sources') "
        "ORDER BY tool_name;"
    ).fetchall()
    assert [r["tool_name"] for r in rows] == ["fetch_company_page", "normalize_sources"]
    assert rows[0]["step_id"] != rows[1]["step_id"]


def test_pipeline_never_reaches_a_phase_1b_state(conn):
    """Phase 1 must never produce drafted/awaiting_review/approved/sent/etc."""
    from app.state_machine import PHASE_1_REACHABLE_STATES

    fake_profile = CompanyProfile(one_line_summary="x", confidence=0.5)

    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="text")), \
         patch("app.tools.summarize_company.call_structured", return_value=fake_profile), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=[]), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1")

    assert final_state in PHASE_1_REACHABLE_STATES


def test_adk_state_delta_extracted_text_reaches_later_node_and_session_state(conn):
    """A4a's genuinely new mechanism: a node's Event state_delta must be
    visible to LATER nodes (via ctx.session.state) and must land in the FINAL
    session state.  The research stage writes extracted_text; summarize
    reads it.  Capture what summarize's LLM call actually received, and read
    the terminal session state directly — both must contain the findings.

    B1b adaptation: the writer of extracted_text is now the research agent
    (stubbed offline here — its state_delta is the same mechanism the real
    agent's output_key uses), and the agent build moved INSIDE the patch
    context so the stub is in place when build_phase1_agent constructs the
    pipeline."""
    fake_profile = CompanyProfile(one_line_summary="Acme does logistics", industry="Logistics", confidence=0.8)
    captured: dict = {}

    def fake_call_structured(*, model_alias, system_prompt, user_content, response_schema):
        # summarize_company passes the text it read from session state as
        # user_content — recording it here proves the research stage's
        # state_delta was visible to the summarize node (a LATER node in the
        # pipeline).
        captured["user_content"] = user_content
        return fake_profile

    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics.")), \
         patch("app.tools.summarize_company.call_structured", side_effect=fake_call_structured), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=[]), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
        agent = build_phase1_agent(conn)
        # Drive the agent with an inline runner so the FINAL session state is
        # inspectable — run_target_through_phase1 hides it, returning only
        # final_state.  The seeding mirrors run_target_through_phase1 exactly
        # (state_delta= carries the initial state; the fixture's target has no
        # contact linked, so has_contact_data is False).
        async def _run() -> dict:
            session_service = InMemorySessionService()
            runner = Runner(
                app_name="outbound", agent=agent, session_service=session_service,
                auto_create_session=True,
            )
            async for _ in runner.run_async(
                user_id="operator",
                session_id="tgt_1",
                new_message=types.Content(role="user", parts=[types.Part(text="run")]),
                state_delta={
                    "target_id": "tgt_1", "domain": "acme.test", "run_id": "r1",
                    "has_contact_data": False,
                },
            ):
                pass
            session = await session_service.get_session(
                app_name="outbound", user_id="operator", session_id="tgt_1",
            )
            return session.state

        final_session_state = asyncio.run(_run())

    # The stub publishes its findings verbatim, so the extracted_text the
    # summarize node consumed and the terminal session state's copy both
    # equal the findings string.
    assert captured["user_content"] == "Acme does logistics."
    assert final_session_state["extracted_text"] == "Acme does logistics."
    # The run also reached a terminal state — the plumbing test runs the real pipeline.
    assert final_session_state["final_state"] in ("scored", "watchlist", "not_target")


def test_policy_gate_creates_exactly_one_decision_row_on_success(conn):
    """A4b: on a successful run the policy gate records exactly ONE
    policy_decisions row for the target, and the gate does NOT change the
    routing — final_state is still one of scored/watchlist/not_target
    (score_lead owns the terminal state; the gate is an audit record,
    not a router)."""
    # Same mocking style as the existing pipeline tests: mock every network/
    # LLM boundary (the research stage is stubbed offline), let the real
    # deterministic nodes (including the gate) run.
    fake_profile = CompanyProfile(one_line_summary="Acme does logistics", industry="Logistics", confidence=0.8)
    fake_signals = [Signal(
        signal_type="hiring_relevant_role", signal_value="Hiring ops manager",
        signal_strength=0.8,
        # B2a: every Signal now requires an evidence quote — a placeholder
        # here; these tests assert on pipeline routing and step logging, not
        # on quote verification.
        evidence_quote="hiring an operations manager for the team",
    )]

    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=fake_profile), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1")

    # The gate records the decision but must not route: score_lead's terminal
    # state must survive untouched.
    assert final_state in ("scored", "watchlist", "not_target")
    # Exactly one row — the gate runs once per successful target, and its
    # internal write_gate.commit is the only write to this table.
    row = conn.execute("SELECT COUNT(*) AS n FROM policy_decisions WHERE target_id='tgt_1';").fetchone()
    assert row["n"] == 1


def test_policy_gate_runs_on_success_path_regression(conn):
    """A4b regression test for the guard-clause trap: the gate's guard must
    be `final_state == "failed"`, NOT the other nodes' `state.get("final_state")`
    idiom.  score ALWAYS sets final_state on success, so the common idiom
    would skip the gate on every successful run — reverting the guard to
    `if ctx.session.state.get("final_state"):` makes THIS test fail (the row
    below would not exist)."""
    # These mocks produce a strong fit: fit_score 70 (30 company + 0 persona
    # + 25 signal + 5 completeness + 10 evidence per score_lead's formula)
    # → good_fit, which clears the P4 floor and resolves to decision "allow" —
    # asserting that specific value proves the row came from a REAL
    # policy_check_phase1 evaluation, not a stub.
    fake_profile = CompanyProfile(one_line_summary="Acme does logistics", industry="Logistics", estimated_size="50-200", confidence=1.0)
    fake_signals = [
        # B2a: every Signal now requires an evidence quote — placeholders
        # here; these tests assert on the policy gate's evaluation, not on
        # quote verification.
        Signal(
            signal_type="hiring_relevant_role", signal_value="Hiring ops manager",
            signal_strength=1.0,
            evidence_quote="hiring an operations manager for the team",
        ),
        Signal(
            signal_type="product_or_ops_change", signal_value="Expanding ops team",
            signal_strength=1.0,
            evidence_quote="expanding the operations team this quarter",
        ),
        Signal(
            signal_type="recent_launch_or_expansion", signal_value="New warehouse",
            signal_strength=1.0,
            evidence_quote="opened a new warehouse in the region",
        ),
        Signal(
            signal_type="workflow_complexity_evidence", signal_value="Multi-step workflow",
            signal_strength=1.0,
            evidence_quote="a multi-step workflow across several tools",
        ),
    ]

    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=fake_profile), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
        agent = build_phase1_agent(conn)
        run_target_through_phase1(agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1")

    # If the gate were skipped (guard reverted to the common idiom), this row
    # would not exist — the assertion below IS the regression test.
    row = conn.execute("SELECT * FROM policy_decisions WHERE target_id='tgt_1';").fetchone()
    assert row is not None
    # All P3a fields present and fit_score 70 clears the P4 floor, so the
    # gate's real evaluation must resolve to "allow".
    assert row["decision"] == "allow"


def test_policy_gate_skipped_on_upstream_failure(conn):
    """A4b: an upstream failure (zero sources → final_state="failed") must
    NOT produce a policy_decisions row — the gate's guard is
    `final_state == "failed"`, so the genuine failure path skips it."""
    # B1b's equivalent of the old zero-sources mock: the research agent found
    # nothing (findings=None), so research_bookkeeping publishes the failure
    # delta — every downstream node including the gate short-circuits.  The
    # summarize boundary is patched (and asserted uncalled) so the
    # short-circuit guarantee is checked and the suite stays offline.
    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings=None)), \
         patch("app.tools.summarize_company.call_structured") as fake_summarize:
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1")
    assert final_state == "failed"
    fake_summarize.assert_not_called()  # the failure short-circuit must reach all the way past summarize
    # No decision row: the gate never ran, so nothing was written.
    row = conn.execute("SELECT COUNT(*) AS n FROM policy_decisions WHERE target_id='tgt_1';").fetchone()
    assert row["n"] == 0


def test_policy_decision_lands_in_session_state(conn):
    """A4b: the gate publishes decision.decision into session state under the
    key "policy_decision" — the caller reads it from the terminal session
    state, exactly like final_state."""
    # Strong-fit mocks (same as the regression test above) so the decision is
    # deterministically "allow" and the assertion can be exact.
    fake_profile = CompanyProfile(one_line_summary="Acme does logistics", industry="Logistics", estimated_size="50-200", confidence=1.0)
    fake_signals = [
        # B2a: every Signal now requires an evidence quote — placeholders
        # here; these tests assert on the policy gate's evaluation, not on
        # quote verification.
        Signal(
            signal_type="hiring_relevant_role", signal_value="Hiring ops manager",
            signal_strength=1.0,
            evidence_quote="hiring an operations manager for the team",
        ),
        Signal(
            signal_type="product_or_ops_change", signal_value="Expanding ops team",
            signal_strength=1.0,
            evidence_quote="expanding the operations team this quarter",
        ),
        Signal(
            signal_type="recent_launch_or_expansion", signal_value="New warehouse",
            signal_strength=1.0,
            evidence_quote="opened a new warehouse in the region",
        ),
        Signal(
            signal_type="workflow_complexity_evidence", signal_value="Multi-step workflow",
            signal_strength=1.0,
            evidence_quote="a multi-step workflow across several tools",
        ),
    ]

    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=fake_profile), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
        agent = build_phase1_agent(conn)  # built INSIDE the patch so the research stub is wired in (B1b)
        # Inline runner — run_target_through_phase1 hides the session state,
        # returning only final_state, so the full terminal state dict must be
        # read directly to see policy_decision (same pattern as the A4a
        # state_delta plumbing test above).
        async def _run() -> dict:
            session_service = InMemorySessionService()
            runner = Runner(
                app_name="outbound", agent=agent, session_service=session_service,
                auto_create_session=True,
            )
            async for _ in runner.run_async(
                user_id="operator",
                session_id="tgt_1",
                new_message=types.Content(role="user", parts=[types.Part(text="run")]),
                state_delta={
                    "target_id": "tgt_1", "domain": "acme.test", "run_id": "r1",
                    "has_contact_data": False,
                },
            ):
                pass
            session = await session_service.get_session(
                app_name="outbound", user_id="operator", session_id="tgt_1",
            )
            return session.state

        final_session_state = asyncio.run(_run())

    # The gate's delta must be merged into the terminal session state.
    assert final_session_state["policy_decision"] == "allow"
    # And the run still ends at a terminal Phase 1 state — the gate changed
    # nothing about routing.
    assert final_session_state["final_state"] in ("scored", "watchlist", "not_target")
