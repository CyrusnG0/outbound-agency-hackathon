"""Phase 1 ADK agent — wires research → research_bookkeeping → summarize →
detect → score → policy_check as a SequentialAgent of five deterministic
BaseAgent nodes plus one tool-choosing LlmAgent (task A4a runtime swap, plus
the task A4b policy gate as the final node, plus the ticket B1b research
agent replacing the static fetch node, plus the ticket B2c ICP judge running
inside the score node between the deterministic score and the final routing).

Up to B1b this module was a behaviour-identical port of the former LangGraph
StateGraph in app/graph.py (deleted in task A4a).  B1b changes the HEAD of
the pipeline: the deterministic ``FetchAndNormalizeNode`` (one static GET,
then give up) is superseded in the pipeline by the ``research`` LlmAgent
built in ``app/agents/research.py`` — an agent that chooses among
fetch_page / google_search / url_context and falls back when one fails —
followed by a NEW deterministic ``ResearchBookkeepingNode`` that owns every
governed side effect the agent must not have: the new→researched (or
→failed) transition and the char-count step row.  The old
``FetchAndNormalizeNode`` class is KEPT IN THIS FILE (see its comment) so
the existing tests that exercise it directly keep working, but it is no
longer part of ``build_phase1_agent``.

The remaining plumbing is unchanged from A4a: same tool calls, same argument
names, same order for the nodes that survive, same short-circuit semantics,
same `new_id("step")` per log_step write, and every state change through
transition().  Nodes publish state by yielding an Event carrying
EventActions(state_delta=...); the runner merges that delta into
ctx.session.state before the next node runs (verified A4a).  The research
LlmAgent publishes its final text the same way, through ADK's output_key
(verified B1b: output_key writes the agent's final text response into
session state under the named key).

Each node short-circuits if an upstream step failed, so a failure at any
point skips the remaining nodes without running further LLM calls or writes.
The first five nodes test `if state.get("final_state")` because for them any
set final_state means failure; the policy_gate node instead tests
`== "failed"` because score — the node immediately before it — ALWAYS sets
final_state on its success path (see the guard comment in PolicyGateNode).
In the LangGraph version a SqliteSaver checkpointer
stored in-flight state after every node so a process crash mid-run doesn't
lose progress (per docs/data-flow.md §9); that crash-recovery role is
deliberately deferred to Task A5 rather than reimplemented here — see the
comment at the InMemorySessionService in run_target_through_phase1.
"""

import asyncio  # asyncio.run bridges ADK's async runner to our synchronous entry point; wait_for bounds it (B1g)
import httpx  # httpx.TimeoutException — the SDK-level timeout family that must land in the same clean timeout bucket as the ceiling (B1g)
import json  # steps.output_json is a JSON string — the bookkeeping node parses it to read back upstream char counts (B1d)
import os  # PHASE1_TARGET_TIMEOUT_SECONDS — the operator-facing override of the per-target ceiling (B1g)
from typing import AsyncGenerator  # return annotation of every node's _run_async_impl

from google.adk.agents import BaseAgent, SequentialAgent  # node base class + ordered sub-agent container
from google.adk.agents.invocation_context import InvocationContext  # type of ctx: per-run handle to session state
from google.adk.events import Event, EventActions  # how a node publishes its state_delta to the session
from google.adk.runners import Runner  # executes the agent against a session service
from google.adk.sessions import InMemorySessionService  # in-memory session state store (see run_target_through_phase1)
from google.genai import types  # Content/Part builders for the synthetic "run" user message

from app.agents.guardrail import make_kill_switch_callback  # B4a: the agent-entry kill-switch guardrail, attached to this pipeline's root
from app.agents.research import (  # B1b: the tool-choosing research LlmAgent + the nothing-found sentinel its bookkeeping reacts to
    NO_RESEARCH_FINDINGS_SENTINEL,
    build_research_agent,
)
from app.config import load_offer_configs  # B2c: the ScoreNode loads the offer's icp block + pitch for the judge
from app.ids import new_id  # unique step IDs so each node's audit entries are distinct
from app.policy import policy_check_phase1  # the Phase 1 policy gate (P3a + P4) — wired in as the final node (task A4b)
from app.state_machine import transition  # THE state-change gate — every target transition goes through it
from app.tools.detect_signals import detect_signals
# Imported as a module, not `from app.tools.fetch_sources import fetch_sources` —
# the test patches "app.tools.fetch_sources.fetch_sources", and that patch is
# only honored if the call site looks the name up on the module object at
# call time, not at import time (classic unittest.mock "patch where it's
# used" gotcha).
import app.tools.fetch_sources as fetch_sources_module
# B2c: the ICP judge, imported as a MODULE for the same "patch where it's
# used" reason as fetch_sources — tests patch
# "app.agents.phase1.judge_icp_module.judge_icp" (offline fallback) and
# "app.tools.judge_icp._call_judge_llm" (offline verdicts).
import app.tools.judge_icp as judge_icp_module
from app.tools.judge_icp import JUDGE_AGENT_ID  # the judge's registered agent_id — the ScoreNode attributes judge-driven routing to it
from app.tools.log_step import log_step  # the bookkeeping node writes its own char-count step row
from app.tools.normalize_sources import normalize_sources
from app.tools.score_lead import apply_final_fit_label, score_lead  # B2c: score_lead computes the evidence; apply_final_fit_label routes with the FINAL label
from app.tools.summarize_company import summarize_company


class FetchAndNormalizeNode(BaseAgent):
    """Node "fetch_and_normalize": fetch sources for the target's domain,
    normalize them to one text blob, and transition the target from "new" to
    "researched" so the audit trail is truthful.

    RETAINED FOR TESTS, SUPERSEDED IN THE PIPELINE (ticket B1b): this node
    is no longer used by build_phase1_agent — the research LlmAgent in
    app/agents/research.py replaced its single-shot static fetch with a
    tool-choosing agent, and ResearchBookkeepingNode (below) took over its
    governed transitions.  The class stays in this file because
    tests/test_agents_phase1.py's A6 regression test
    (test_fetch_and_normalize_write_two_distinct_step_rows) exercises it
    directly — deleting it would delete that coverage.  Do NOT wire it back
    into the pipeline and do NOT delete it; if the test it serves is ever
    retired, this class goes with it.

    ADK plumbing: a node publishes state by yielding an Event carrying
    EventActions(state_delta=...); the runner merges that delta into
    ctx.session.state before the next node fires.
    """

    def __init__(self, name: str, conn):
        # BaseAgent is a pydantic model with model_config extra='forbid' —
        # PUBLIC attribute assignment (self.conn = x) raises ValueError, so
        # the DB connection is stored on a PRIVATE attribute instead, which
        # pydantic allows.  It must NOT go into session state: a live
        # sqlite3/pg8000 connection is not serializable, and session state is
        # exactly the kind of thing ADK (and LangGraph's checkpointer before
        # it) expects to persist.  This is the same non-serializable-
        # connection constraint that Phase1State's docstring documented in
        # the old app/graph.py.
        super().__init__(name=name)  # registers the node under its stable pipeline name (preserved from LangGraph)
        self._conn = conn  # private attr: visible to this node's logic, never serialized into state

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # ADK calls this when the node fires.  ctx.session.state is the
        # running state dict seeded by run_target_through_phase1 and extended
        # by every upstream node's state_delta — the ADK equivalent of
        # LangGraph's `state` parameter.
        conn = self._conn  # pull the live DB connection from the private attr (see __init__)
        state = ctx.session.state  # local alias: this node's working state for the target
        # ONE step id PER TOOL CALL — the invariant this node enforces, and
        # the fix for ticket A6.  steps.step_id is the PRIMARY KEY (app/db.py)
        # and BOTH tools below call log_step, so handing the SAME id to two
        # DIFFERENT tools makes the second insert raise sqlite3.IntegrityError:
        # UNIQUE constraint failed: steps.step_id — observed on the first real
        # run: 10 targets imported, 2 steps logged, run dead before any LLM
        # call.  Two tools ran = two steps = two rows = two ids.
        # No suffix scheme here (f"{step_id}_a{attempt}"): suffixes are for
        # ONE tool retried twice — summarize_company / detect_signals do that
        # internally with ids they derive themselves.  fetch_sources and
        # normalize_sources are two DIFFERENT tools, so two independent ids
        # are the honest representation.  If a third tool call is ever added
        # to this node, it must get its OWN new_id too — same rule, and the
        # regression test test_fetch_and_normalize_write_two_distinct_step_rows
        # will catch anyone who reintroduces the single-id pattern.
        fetch_step_id = new_id("step")  # the fetch step — fetch_sources' log_step writes this row
        normalize_step_id = new_id("step")  # the normalize step — normalize_sources' log_step writes this row
        # STEP A: fetch raw sources from web/Tavily/static crawl
        sources = fetch_sources_module.fetch_sources(
            conn, domain=state["domain"], target_id=state["target_id"],
            run_id=state["run_id"], step_id=fetch_step_id,
        )
        # STEP B: normalize sources into a single extracted-text blob
        text = normalize_sources(
            conn, sources=sources, target_id=state["target_id"],
            run_id=state["run_id"], step_id=normalize_step_id,
        )
        # STEP C: if normalize returned None (zero usable sources), fail immediately
        if text is None:
            # Publish the failure delta.  The runner merges this delta into
            # ctx.session.state, then this generator returns WITHOUT further
            # yields — downstream nodes see final_state set and short-circuit.
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"extracted_text": None, "final_state": "failed"}),
            )
            return  # end the node here; nothing downstream should run
        # normalize_sources' happy path does NOT itself move the target's state
        # out of "new" (only its zero-sources failure path calls transition()).
        # Nothing downstream does either, until score_lead assumes the target is
        # already "researched". transition() trusts the caller's from_state as
        # given rather than re-reading the DB, so skipping this call wouldn't
        # crash anything — it would just silently write a false previous_state
        # into the state_transitions audit trail. Phase 1 targets always start
        # at "new" (no enrichment node exists yet), so "new" is a safe hardcode
        # here, matching the same precedent already used in normalize_sources.py.
        # step_id=normalize_step_id links this transition to the NORMALIZE
        # step in the audit trail: research is only "complete" once the
        # sources have been normalized into text, so the new→researched hop
        # is the normalize step's outcome.  This also keeps the node's two
        # transition paths consistent — on the zero-sources failure path,
        # normalize_sources transitions the target to "failed" internally
        # using the SAME normalize_step_id it was given (ticket A6: that id
        # must be the normalize one, not the fetch one) — so both transitions
        # this node produces point at the normalize step.  transition() writes
        # state_transitions (its own PK), never steps, so reusing an id here
        # cannot collide.
        transition(
            conn, target_id=state["target_id"], from_state="new", to_state="researched",
            reason="research_complete_no_enrichment", actor="system",
            run_id=state["run_id"], step_id=normalize_step_id,
        )
        # STEP D: pass extracted text downstream to summarize→detect→score
        # (state_delta carries only the changed key; the runner merges it into
        # the session state the downstream nodes will read).
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"extracted_text": text}),
        )


def _sum_fetched_chars(conn, target_id: str, run_id: str) -> int:
    """Best-effort sum of ``chars_extracted`` across the target's
    ``fetch_company_page`` step rows in THIS run (ticket B1d).

    Why the bookkeeping node reads the steps table: its failure row must
    make the Mark Boyden contradiction visible in ONE row — "14,828 chars
    fetched, 0 chars out" — and an empty agent output carries no fetch
    information of its own.  The count is read back from the rows the
    agent's own fetch_page tool already wrote (tool_name="fetch_company_page",
    output_json={"chars_extracted": N}), scoped to (target, run) so a re-run
    of the same target never mixes in a previous run's counts.

    Deliberately best-effort: this is diagnostic enrichment of the trace,
    not a governed write — any malformed or legacy row simply contributes 0
    rather than breaking bookkeeping's real job (the transition and the
    failure row)."""
    total = 0
    # Read only THIS run's fetch rows for THIS target: a re-run of the same
    # target must not inherit character counts from an earlier run.
    rows = conn.execute(
        "SELECT output_json FROM steps WHERE target_id=? AND run_id=? AND tool_name='fetch_company_page';",
        (target_id, run_id),
    ).fetchall()
    for row in rows:
        try:
            # steps.output_json is a JSON string (see app/tools/log_step.py);
            # fetch_sources writes {"chars_extracted": N} on its success
            # path.  `or 0` handles a missing key or JSON null, and the try
            # guards the whole parse so one bad legacy row cannot crash the
            # bookkeeping node — the diagnostic count is only lowered, never
            # made wrong.
            total += int(json.loads(row["output_json"]).get("chars_extracted") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue  # malformed/legacy row — skip it rather than fail the node's real job
    return total


class ResearchBookkeepingNode(BaseAgent):
    """Node "research_bookkeeping": the deterministic governance node that
    runs immediately AFTER the research LlmAgent (ticket B1b).

    WHY BOOKKEEPING IS A SEPARATE DETERMINISTIC NODE RATHER THAN SOMETHING
    THE AGENT DOES — this split is the ticket's central governance rule and
    breaking it breaks the trust model.  The research LlmAgent is allowed to
    CHOOSE TOOLS and PRODUCE TEXT; it owns no governed side effects.  Every
    state change must go through state_machine.transition() and every step
    must be logged — both are deterministic writes the agent has no tool
    for, and both happen HERE: this node reads the agent's published
    extracted_text (written into session state by ADK's output_key) and
    either (a) transitions new→researched with
    reason="research_complete_no_enrichment" — the same call the old
    FetchAndNormalizeNode made — or (b) routes the target to failed.  The
    failure route now DISCRIMINATES (ticket B1d) between two opposite
    diagnoses that previously collapsed into one reason:

    - the agent published the NO_RESEARCH_FINDINGS_SENTINEL (an honest
      "company not findable" verdict) → failed with
      reason="no_sources_available" (state-machine.md §7c, the exact reason
      A7 established for "no usable text").  Here the claim is TRUE.
    - the agent published nothing at all — extracted_text absent, or
      whitespace-only — → failed with reason="research_agent_no_output_phase1"
      (a new reason string, same "failed" state; precedent: A4c's
      llm_transport_error_phase1).  This is the Mark Boyden Associates
      regression: the fetch had succeeded (14,828 chars) and the target was
      recorded as "no_sources_available", pointing a reader at the wrong
      layer — the sources WERE available; the agent's output was what failed
      to materialize.  An operator must be able to tell the two apart from
      state_transitions.reason alone.  No new state is invented.
    """

    def __init__(self, name: str, conn):
        super().__init__(name=name)  # register under the stable pipeline name "research_bookkeeping"
        self._conn = conn  # private attr — same non-serializable-connection rationale as FetchAndNormalizeNode

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Short-circuit if an upstream node already set final_state
        # (consistent with the other nodes; with the current wiring this node
        # is second and nothing upstream sets final_state, but the guard
        # keeps the node safe if the order ever changes).
        if ctx.session.state.get("final_state"):
            return  # already failed upstream, nothing to bookkeep
        conn = self._conn  # pull the live DB connection from the private attr (see __init__)
        state = ctx.session.state  # local alias: the running state for this target
        # The agent's deliverable: whatever the LlmAgent's output_key wrote,
        # or None if the agent produced no text at all (a failed/empty agent
        # turn leaves the key absent from state).
        text = state.get("extracted_text")
        # One fresh step id for this node's audit entries — the log_step row
        # AND the transition below share it (transition writes
        # state_transitions, its own PK, so the reuse cannot collide — same
        # pattern as FetchAndNormalizeNode).
        step_id = new_id("step")
        # Character count of what came back, so an operator scanning the
        # trace can see how much research each target produced — 0 means the
        # agent published nothing at all.
        chars = len(text) if isinstance(text, str) else 0
        # ── Governance decision: is there usable research to pass on? ──────
        # Three shapes of "nothing usable" arrive here, and they are TWO
        # different diagnoses (ticket B1d), NOT one shared route:
        #   1. the key is absent (agent produced no final text), or
        #   2. the text strips to nothing (whitespace-only response)
        # — both mean the agent (or the model) FAILED TO PRODUCE OUTPUT at
        # all, possibly after a fully successful upstream fetch.  The Mark
        # Boyden Associates run is the concrete case: fetch_company_page
        # succeeded with chars_extracted 14828 and normalize_sources with
        # chars 14828, and the old code then recorded "no_sources_available"
        # — false, and pointing a reader at the wrong layer entirely.  These
        # route to failed with reason="research_agent_no_output_phase1".
        #   3. the text is exactly the nothing-found sentinel the research
        #      instruction defines (NO_RESEARCH_FINDINGS_SENTINEL in
        #      app/agents/research.py — a contract string; the equality test
        #      is exact after .strip() so a page that merely contains the
        #      phrase as data cannot trip it) — the agent worked correctly
        #      and honestly reports the company is not findable, so the §7c
        #      reason "no_sources_available" is TRUE here and stays.
        if text is None or not text.strip():
            outcome = "no_output"  # the agent produced nothing — B1d's newly separated failure shape
            reason = "research_agent_no_output_phase1"  # new reason string, same "failed" state (precedent: A4c's llm_transport_error_phase1)
        elif text.strip() == NO_RESEARCH_FINDINGS_SENTINEL:
            outcome = "sentinel"  # the agent's honest nothing-found verdict — the §7c vocabulary is correct here
            reason = "no_sources_available"  # unchanged from before B1d: this case must not move
        else:
            outcome = None  # usable findings — the happy path below owns this case
            reason = None
        if outcome is not None:
            # Route the target to failed.  from_state="new" is the same known
            # simplification the old node and normalize_sources used (Phase 1
            # targets always start at "new"; no enrichment node exists).
            # to_state="failed" is allowed from any state (ANY_TARGET_TRANSITIONS).
            # reason is the discriminated value above — an operator must be
            # able to tell the two causes apart from state_transitions.reason
            # alone, without opening the steps table.
            transition(
                conn, target_id=state["target_id"], from_state="new", to_state="failed",
                reason=reason, actor="system",
                run_id=state["run_id"], step_id=step_id,
            )
            # chars_fetched: how much upstream research materialized BEFORE
            # the agent went silent.  Read back from the fetch_company_page
            # steps rows the agent's own fetch_page tool wrote — an empty
            # agent output carries no fetch information of its own, so the
            # contradiction ("14,828 chars fetched, 0 chars out") is only
            # visible if the node looks the fetch count up here.
            chars_fetched = _sum_fetched_chars(conn, state["target_id"], state["run_id"])
            # The failure-path step row.  output_data carries the outcome
            # discriminator AND both char counts so this one row shows the
            # contradiction: "sentinel" vs "no_output" names what the agent
            # did, chars quantifies what came out, chars_fetched what went in
            # upstream.
            log_step(
                conn, run_id=state["run_id"], step_id=step_id, target_id=state["target_id"],
                tool_name="research_bookkeeping",
                agent_id="system",  # deterministic pipeline code — the registered system agent
                input_data={"stage": "research_output_check", "outcome": outcome},
                output_data={"outcome": outcome, "chars": chars, "chars_fetched": chars_fetched},
                status="failed",
            )
            # Publish the failure delta; the runner merges it into session
            # state and every downstream node (summarize onward) short-circuits
            # on it — this yield IS the "no downstream node runs" guarantee.
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"final_state": "failed"}),
            )
            return  # end the node; nothing downstream should run
        # ── Happy path: the agent produced usable findings ─────────────────
        # The same transition the old FetchAndNormalizeNode made — new→
        # researched, same reason string, so the audit trail's vocabulary is
        # unchanged and every downstream consumer of
        # "research_complete_no_enrichment" keeps working.  step_id links the
        # transition to THIS node's step in the trace (research is only
        # "complete" once bookkeeping has verified the agent's output).
        transition(
            conn, target_id=state["target_id"], from_state="new", to_state="researched",
            reason="research_complete_no_enrichment", actor="system",
            run_id=state["run_id"], step_id=step_id,
        )
        # B2b: persist the findings themselves to the sources table — the
        # SAME table the raw fetched pages live in — so the `findings` tier
        # is checkable after the run and a retroactive fact-checker has the
        # text the signal agent actually read.  source_type is
        # FINDINGS_SOURCE_TYPE ("research_findings") so detect_signals can
        # tell agent prose apart from raw pages when it loads verification
        # texts: raw rows are checked FIRST, findings only as the fallback.
        # The row carries NULL source_url/source_confidence/source_priority
        # (agent prose has no single URL, no measured confidence, no
        # normalization priority) and extraction_method="agent" as its
        # provenance marker.  Written through the shared persist_source_row
        # seam — the same single write path fetch_sources uses — never a raw
        # conn.execute.  Like fetch_sources, a persistence failure here
        # PROPAGATES: silently continuing would leave detect_signals with no
        # findings text to fall back to, mis-tiering every
        # findings-backed signal as unverified (see docs/data-flow.md §9i).
        # Ordered after the transition and before the log step on purpose:
        # the transition is the node's primary job and must land first, and
        # the step row below reports the state of a run that already has its
        # evidence persisted.
        _findings_source_id = fetch_sources_module.persist_source_row(
            conn,
            source_id=new_id("src"),  # same "src" id family as fetch_sources — one evidence namespace per table
            run_id=state["run_id"],
            target_id=state["target_id"],
            step_id=step_id,  # links the evidence row to this node's audit step
            source_type=fetch_sources_module.FINDINGS_SOURCE_TYPE,  # the shared constant — marks the row as agent prose, never a raw page
            source_url=None,  # no single URL: the agent consolidated many sources (some server-side, uncapturable)
            extracted_text=text,  # the verbatim findings downstream nodes will read from session state
            source_confidence=None,  # agent prose has no measured confidence — NULL means "not independently assessed"
            source_priority=None,  # findings are not a normalization input — no priority applies
            extraction_method="agent",  # provenance marker: produced by the research agent, not a fetcher
        )
        # The success-path step row with the char count — the operator-facing
        # "how much research came back" signal the ticket requires, plus the
        # persisted findings row id so the trace links straight to the stored
        # evidence.
        log_step(
            conn, run_id=state["run_id"], step_id=step_id, target_id=state["target_id"],
            tool_name="research_bookkeeping",
            agent_id="system",
            input_data={"stage": "research_output_check", "outcome": "usable_findings"},
            output_data={"chars": chars, "findings_source_id": _findings_source_id},
            status="success",
        )
        # NO state_delta on the happy path: extracted_text is already in
        # session state (the agent's output_key wrote it), so the downstream
        # nodes read it unchanged, exactly as before B1b.  Returning without
        # yielding ends the node — the runner then fires summarize.


class SummarizeNode(BaseAgent):
    """Node "summarize": call summarize_company on the extracted text.
    Short-circuits if an upstream node already set final_state (failure path).

    Fires second in the pipeline, right after fetch_and_normalize.
    """

    def __init__(self, name: str, conn):
        super().__init__(name=name)  # register under the stable pipeline name "summarize"
        self._conn = conn  # private attr — same non-serializable-connection rationale as FetchAndNormalizeNode

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Check if an upstream node already failed — skip the LLM call if so.
        # Returning from an async generator without yielding ends the node
        # with no state change (the happy path below provides the yield that
        # makes this function a generator at all).
        if ctx.session.state.get("final_state"):
            return  # already failed upstream, short-circuit
        conn = self._conn  # pull the live DB connection from the private attr
        state = ctx.session.state  # local alias: this node's working state
        step_id = new_id("step")  # unique step for this node's audit entries
        # Call the LLM-backed summarizer; returns None on failure
        profile = summarize_company(
            conn, extracted_text=state["extracted_text"], target_id=state["target_id"],
            run_id=state["run_id"], step_id=step_id,
        )
        # If summarization failed (LLM error / empty result), mark as failed
        if profile is None:
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"final_state": "failed"}),
            )
            return  # stop the node; downstream nodes will short-circuit
        # Pass the structured CompanyProfile downstream to score (only the
        # changed key goes in the delta; the runner merges it into state).
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"company_profile": profile}),
        )


class DetectSignalsNode(BaseAgent):
    """Node "detect_signals": call detect_signals on the extracted text.
    Short-circuits if upstream already failed.

    Fires third in the pipeline, after summarize.
    """

    def __init__(self, name: str, conn):
        super().__init__(name=name)  # register under the stable pipeline name "detect_signals"
        self._conn = conn  # private attr — same non-serializable-connection rationale as FetchAndNormalizeNode

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Short-circuit if an upstream node already set final_state (failure)
        if ctx.session.state.get("final_state"):
            return  # already failed upstream, skip the LLM call entirely
        conn = self._conn  # pull the live DB connection from the private attr
        state = ctx.session.state  # local alias: this node's working state
        step_id = new_id("step")  # unique step for this node's audit entries
        # Call the LLM-backed signal detector; returns None on failure
        signals = detect_signals(
            conn, extracted_text=state["extracted_text"], target_id=state["target_id"],
            run_id=state["run_id"], step_id=step_id,
        )
        # If signal detection failed (LLM error / empty structured output), fail
        if signals is None:
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"final_state": "failed"}),
            )
            return  # stop the node; the score node will short-circuit
        # Pass the detected signals list downstream to score (changed key only).
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"signals": signals}),
        )


# ── Offer context for the ICP judge (ticket B2c) ─────────────────────────────
# The pipeline's default offers directory — the same default phase1_cli's
# --offers-dir uses, so a bare run_target_through_phase1 call sees the real
# config/offers without extra plumbing.  Callers (the CLI, tests) can
# override it per run.
DEFAULT_OFFERS_DIR = "config/offers"


def _load_offer_context(conn, target_id: str, offers_dir: str) -> tuple[object, str | None]:
    """Load the target's offer icp block and pitch from its YAML definition.

    The judge compares the researched company against the CAMPAIGN — that
    comparison is the whole reason B2c exists (the London practitioner who
    scored good_fit in a Hong Kong campaign), so the offer's icp block and
    pitch must reach the judge.  The path is offer_id → slug → YAML config,
    read fresh per target: offer YAMLs are operator-edited between runs, and
    caching them would silently judge against a stale ICP.

    Failure behaviour is deliberately lenient — every branch returns
    (None, None) rather than raising: a target whose offer row is missing,
    whose slug has no YAML file, or whose offers dir does not exist must
    still be scored (the judge simply works with less to go on — a missing
    icp block is a documented, supported configuration).
    """
    # offer_id lives on the target row; the YAML config is keyed by slug —
    # join through offers to get the slug (the console does the same join).
    row = conn.execute(
        "SELECT o.slug FROM targets t JOIN offers o ON t.offer_id = o.offer_id "
        "WHERE t.target_id=?;",
        (target_id,),
    ).fetchone()
    if row is None:
        # No target row (or no offer linked) — nothing to compare against.
        # The judge still runs; the caller decides what "no offer context"
        # means for its verdict.
        return None, None
    try:
        configs = load_offer_configs(offers_dir)
    except OSError:
        # The offers dir is missing/unreadable (e.g. a test environment or a
        # misconfigured operator path) — degrade to "no offer context"
        # rather than failing the target's scoring: the deterministic score
        # and the judge's other inputs are all still available.
        return None, None
    # The config dict for this slug, or an empty dict when the slug has no
    # YAML file (e.g. an offer synced from a directory that no longer holds
    # it).  .get() with defaults keeps both the icp block and the pitch
    # optional — an offer without them is legitimate.
    offer_config = configs.get(row["slug"], {})
    return offer_config.get("icp"), offer_config.get("pitch")


class ScoreNode(BaseAgent):
    """Node "score": call score_lead (deterministic evidence), then run the
    ICP judge (B2c), then route the target to its FINAL terminal Phase 1
    state: scored, watchlist, or not_target.

    The final label is the judge's when the judge produced a verdict, and
    the deterministic label when the judge failed after its bounded retries
    (a broken judge degrades to today's behaviour — it never fails the
    target).  The routing hop happens exactly ONCE, here, via
    apply_final_fit_label — never a route-then-reroute double-hop.

    Fires fourth in the pipeline, immediately before policy_gate.
    Short-circuits if upstream already failed.
    """

    def __init__(self, name: str, conn):
        super().__init__(name=name)  # register under the stable pipeline name "score"
        self._conn = conn  # private attr — same non-serializable-connection rationale as FetchAndNormalizeNode

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Short-circuit if an upstream node already set final_state (failure)
        if ctx.session.state.get("final_state"):
            return  # already failed upstream, skip scoring and its writes
        conn = self._conn  # pull the live DB connection from the private attr
        state = ctx.session.state  # local alias: this node's working state
        step_id = new_id("step")  # unique step for this node's audit entries
        # score_lead computes the deterministic assessment and performs the
        # researched→scored transition internally (scoring completed — that
        # fact never changes).  It does NOT route scored→watchlist/
        # not_target anymore (B2c): that hop happens once, below, with the
        # FINAL label.
        assessment = score_lead(
            conn, company_profile=state["company_profile"], signals=state["signals"],
            has_contact_data=state.get("has_contact_data", False), target_id=state["target_id"],
            run_id=state["run_id"], step_id=step_id,
        )
        # ── B2c: the ICP judge runs AFTER the deterministic score ─────────
        # It consumes the assessment (its evidence), the signals (with their
        # persisted B2b tiers, looked up inside judge_icp), and the offer's
        # icp + pitch (loaded fresh from the offers dir seeded into session
        # state).  The judge gets its OWN step id — one tool call, one row,
        # attributable to agent icp_judge.
        offer_icp, offer_pitch = _load_offer_context(
            conn, state["target_id"], state.get("offers_dir", DEFAULT_OFFERS_DIR)
        )
        judge_step_id = new_id("step")
        verdict = judge_icp_module.judge_icp(
            conn,
            company_profile=state["company_profile"],
            signals=state["signals"],
            icp_assessment=assessment,
            offer_icp=offer_icp,
            offer_pitch=offer_pitch,
            target_id=state["target_id"],
            run_id=state["run_id"],
            step_id=judge_step_id,
        )
        # ── Decide the FINAL label, then route exactly once ───────────────
        # Judge produced a verdict → its label is final, and the routing
        # transition is attributed to the judge principal (write_log.agent_id
        # = icp_judge).  Judge failed (None) → the deterministic label
        # stands, attributed to system — today's pre-B2c behaviour, the
        # documented degradation.  Either way the target is scored.
        if verdict is not None:
            final_label = verdict.fit_label
            routing_agent_id = JUDGE_AGENT_ID
            routing_step_id = judge_step_id  # the hop belongs to the judge's step
        else:
            final_label = assessment.fit_label
            routing_agent_id = "system"
            routing_step_id = step_id  # the hop belongs to the scoring step
        # The single routing hop: strong_fit/good_fit stay at "scored" (no
        # transition), watchlist/not_target route scored→<label> with a
        # reason that names any judge override (greppable divergence).
        apply_final_fit_label(
            conn, target_id=state["target_id"], run_id=state["run_id"],
            step_id=routing_step_id, final_label=final_label,
            deterministic_label=assessment.fit_label, agent_id=routing_agent_id,
        )
        # Map the FINAL label to the target's terminal Phase 1 state.
        to_state = "watchlist" if final_label == "watchlist" else (
            "not_target" if final_label == "not_target" else "scored"
        )
        # final_state is the Phase 1 terminal state; run_target_through_phase1
        # reads it out of session state after the run completes.
        # icp_assessment travels in the SAME delta because the policy_gate
        # node runs immediately after this one and needs the full assessment
        # object (fit_label AND fit_score) to evaluate P3a/P4 — publishing it
        # here is what turns policy_check_phase1 from dead code into live code
        # (task A4b).  DELIBERATELY the DETERMINISTIC assessment, not the
        # judge's verdict (B2c): P4's floor must read the number the formula
        # produced — the judge's label travels nowhere near the gate, so a
        # judge cannot talk a low-scoring target past the floor.  The judge's
        # verdict is persisted to accounts.judge_* and the steps trace; the
        # gate has no use for it.  final_state stays in the delta unchanged.
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={"final_state": to_state, "icp_assessment": assessment},
            ),
        )


class PolicyGateNode(BaseAgent):
    """Node "policy_gate": run policy_check_phase1 (P3a data completeness +
    P4 score floor) on the scored target and persist one policy_decisions row.

    Fires LAST in the pipeline, immediately after score.  In Phase 1 this
    node is an audit/decision record, not a router: the score node
    (score_lead's deterministic formula + the B2c ICP judge) owns the
    target's terminal state (scored/watchlist/not_target), and the gate does
    NOT change it — PHASE_1_REACHABLE_STATES is {new, enriched, researched,
    scored, watchlist, not_target, failed} with no "policy_denied" state, and
    inventing one would break every downstream consumer of the state machine.
    The decision string is published to session state as "policy_decision" so
    the caller (phase1_cli / later phases) can inspect allow/deny/
    review_required and decide whether to advance the target further.

    B2c note: the gate reads the DETERMINISTIC icp_assessment published by
    the score node — never the judge's verdict — so P4's floor is computed
    from the formula's fit_score alone.  A judge that labels a low-scoring
    target strong_fit changes the target's state, but the gate still denies
    it: the judge may set the label, never the number policy reads.
    """

    def __init__(self, name: str, conn):
        super().__init__(name=name)  # register under the stable pipeline name "policy_gate"
        self._conn = conn  # private attr — same non-serializable-connection rationale as FetchAndNormalizeNode

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # GUARD — DELIBERATELY DIFFERENT from the other four nodes, and it
        # must stay different.  They short-circuit on `state.get("final_state")`
        # because for them a set final_state ALWAYS means an upstream failure.
        # That idiom is WRONG here: score — the node immediately before this
        # one — sets final_state on its SUCCESS path ("scored"/"watchlist"/
        # "not_target"), so the common idiom would skip the gate on every
        # successful run.  policy_check_phase1 would stay dead code — no
        # policy_decisions row, no P4 floor enforced — a silently dead gate,
        # which is exactly the bug this task (A4b) exists to fix.
        # The gate must therefore skip ONLY the genuine failure case.
        # Do NOT "tidy" this back to `if state.get("final_state"):` — that
        # one-line change silently disables the entire policy gate.
        if ctx.session.state.get("final_state") == "failed":
            return  # a real upstream failure (zero sources / LLM error) — there is no assessment to police, skip the gate
        conn = self._conn  # pull the live DB connection from the private attr (see __init__)
        state = ctx.session.state  # local alias: the accumulated pipeline state for this target
        step_id = new_id("step")  # fresh step ID — this node's audit entries are distinct from score's
        # Run the Phase 1 policy check.  actor is NOT passed: policy_check_phase1
        # defaults it to "system", the registered principal for pipeline-internal
        # writes.  The function persists its own policy_decisions row through
        # write_gate.commit internally, so this call is the ONLY write path —
        # adding a second insert here (or a raw conn.execute) would double-log
        # the same decision and violate the single-write-path rule.
        decision = policy_check_phase1(
            conn,
            company_profile=state["company_profile"],  # published by summarize (upstream state)
            icp_assessment=state["icp_assessment"],  # published by ScoreNode's delta (task A4b)
            signals=state["signals"],  # published by detect_signals (upstream state)
            target_id=state["target_id"],  # seeded by run_target_through_phase1
            run_id=state["run_id"],  # seeded by run_target_through_phase1
            step_id=step_id,  # this node's own audit step
        )
        # NO state_machine.transition() call here — the gate does not change
        # the target's state.  The score node already performed the terminal
        # transitions (researched→scored via score_lead, then
        # scored→watchlist/not_target via apply_final_fit_label with the
        # judge's — or deterministic — final label) and owns the target's
        # terminal Phase 1 state; the policy decision is recorded ALONGSIDE
        # that state, not instead of it.
        # Calling transition() here would either duplicate an already-made hop
        # (and raise StateTransitionRefused) or require inventing a
        # "policy_denied" state that PHASE_1_REACHABLE_STATES does not contain.
        # Publish ONLY the decision string ("allow" / "review_required" /
        # "deny") so it is inspectable in the final session state — the runner
        # merges this delta in before the run ends.
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"policy_decision": decision.decision}),
        )


def build_phase1_agent(conn) -> SequentialAgent:
    """Build the Phase 1 ADK agent: one tool-choosing research LlmAgent plus
    five deterministic nodes inside a SequentialAgent, executed strictly in
    pipeline order (B1b changed the head of the pipeline; A4a/A4b built the
    rest).

    Node order: research (LlmAgent, autonomy) → research_bookkeeping
    (governance) → summarize → detect_signals → score → policy_gate.  The
    agent is built ONCE per run, before any target is known — the research
    agent's fetch_page tool therefore reads target_id/run_id from the
    injected ToolContext at call time (see app/agents/research.py), which is
    also what keeps one shared agent from mis-attributing across targets.

    SequentialAgent emits a DeprecationWarning in ADK 2.7.1 ("deprecated in
    favor of Workflow"), but using it is a deliberate, verified decision, not
    an oversight: the replacement, google.adk.workflow.Workflow, is a
    BaseNode, and LlmAgent(sub_agents=[Workflow(...)]) raises a pydantic
    ValidationError, while LlmAgent(sub_agents=[SequentialAgent(...)]) works.
    Milestone C4 needs an LlmAgent root with these as sub-agents, so Workflow
    is unusable here.  LlmAgent itself is NOT deprecated.  google-adk is
    therefore pinned to exactly 2.7.1 (see pyproject.toml) so an upgrade
    cannot remove SequentialAgent mid-hackathon.
    """
    # Construct the nodes with the pipeline's live DB connection (stored on
    # each node's private _conn attr) and register them under their stable
    # node names.  SequentialAgent runs sub_agents in list order and merges
    # each one's state_delta into session state before the next fires — that
    # ordering IS the pipeline.  research is FIRST (index 0) because every
    # downstream node consumes its extracted_text; research_bookkeeping is
    # SECOND so the governed transition happens before any further LLM call;
    # policy_gate is LAST so it runs after score has published both
    # final_state and icp_assessment (the gate's two inputs downstream).
    agent = SequentialAgent(
        name="phase1",
        description="Phase 1 research pipeline: agentic research -> bookkeeping -> summarize -> detect signals -> score -> policy gate",
        sub_agents=[
            build_research_agent(conn),  # B1b: the tool-choosing research LlmAgent (autonomy — owns no governed writes)
            ResearchBookkeepingNode(name="research_bookkeeping", conn=conn),  # B1b: the deterministic governance node
            SummarizeNode(name="summarize", conn=conn),
            DetectSignalsNode(name="detect_signals", conn=conn),
            ScoreNode(name="score", conn=conn),
            PolicyGateNode(name="policy_gate", conn=conn),
        ],
    )
    # ── B4a: attach the kill-switch guardrail at the ROOT only ─────────────
    # Root-level attachment is sufficient for the GLOBAL switch (measured
    # against the pinned google-adk==2.7.1, documented in
    # app/agents/guardrail.py): the root callback fires before any sub-agent
    # runs, and its returned Content sets ctx.end_invocation, which halts
    # the WHOLE invocation — attaching the callback to all six sub-agents
    # would buy nothing for the global halt.  The PER-AGENT
    # (agent_registry.enabled=0) check has different reach, and here it is
    # honestly limited: no Phase 1 sub-agent is a registered principal (the
    # research agent attributes its steps to "system"; the deterministic
    # nodes are not registry rows), so there is no per-agent entry check to
    # fire in this pipeline — a disabled principal's ATTRIBUTED WRITES are
    # still refused by the write gate, and the P6 gate in policy_check_phase1
    # still denies the target.  Do not overstate the guarantee: the
    # enabled=0 entry refusal is delivered for the draft stage's registered
    # principals (app/agents/draft.py), not for Phase 1's.
    agent.before_agent_callback = make_kill_switch_callback(conn=conn)
    return agent


# ── Per-target wall-clock ceiling (ticket B1g) ────────────────────────────────
# WHY A CEILING EXISTS: the 2026-08-22 hang — one real Phase 1 run sat for
# 9h48m on 1.09s of CPU, parked in the asyncio selector with two
# ESTABLISHED-but-idle connections to a Google endpoint, having completed 1 of
# 10 targets.  A hung socket await never raises, so A4c's transport-error
# retry (which only sees exceptions) and B1f's per-target crash guard (which
# only sees exceptions) both stood by while the batch stalled forever.  The
# only bound that can stop a silent hang is a wall-clock deadline on the whole
# per-target invocation.
#
# WHY THE DEFAULT IS 600s: measured healthy per-target wall time on the real
# runs (docs/data-flow.md §9g) is roughly 20–60 seconds.  600s is 10x the top
# of that range — generous headroom for a slow-but-alive target (the research
# agent may run several model turns plus up to 8 fetch_page tool calls) —
# while still being 59x below the observed 9h48m hang.  "Far below forever"
# matters more than tightness: a ceiling that fires on healthy targets costs
# a re-run; a ceiling that never fires costs the whole batch.  The operator
# can override it per environment via PHASE1_TARGET_TIMEOUT_SECONDS.
DEFAULT_PHASE1_TARGET_TIMEOUT_SECONDS = 600.0

# The env var that overrides the default.  Read at CALL time (see
# _resolve_target_timeout_seconds) so a change takes effect on the next run
# without restarting anything — the same per-call resolution discipline
# app.llm._resolve_model applies to model pins.
PHASE1_TARGET_TIMEOUT_ENV_VAR = "PHASE1_TARGET_TIMEOUT_SECONDS"


def _resolve_target_timeout_seconds() -> float:
    """Resolve the per-target ceiling: env var if set, else the default.

    Fails loudly on a non-numeric value — same refuse-to-boot discipline as
    model pins: a typo like ``PHASE1_TARGET_TIMEOUT_SECONDS=ten`` must be an
    immediate error, not a silent fallback to the default that hides the
    operator's misconfiguration until a run hangs again."""
    raw = os.environ.get(PHASE1_TARGET_TIMEOUT_ENV_VAR)  # None when unset — the default applies
    if raw is None:
        return DEFAULT_PHASE1_TARGET_TIMEOUT_SECONDS  # unconfigured environment: the documented default
    try:
        return float(raw)  # float: wait_for accepts fractional seconds, which the tests use
    except ValueError:
        # Garbage in the env var is a wiring mistake — name the var and the
        # value so the operator can fix it without reading this code.
        raise ValueError(
            f"{PHASE1_TARGET_TIMEOUT_ENV_VAR}={raw!r} is not a number of seconds — "
            f"unset it to use the default ({DEFAULT_PHASE1_TARGET_TIMEOUT_SECONDS}s) "
            f"or set it to a positive float"
        ) from None


def _record_phase1_timeout(
    conn, *, target_id: str, run_id: str, timeout_seconds: float, detail: str
) -> str:
    """Record a timed-out target: ``failed`` + ``phase1_timeout`` transition +
    a failed step row, then return ``"failed"``.

    WHY THIS IS A CLEAN FAILURE, NOT A CRASH (the ticket's core requirement):
    a timed-out target is a normal Phase 1 outcome — the target exceeded its
    time budget — so it must look like one in the audit trail and in the CLI
    summary (``failed`` line, exit code 0), NOT like B1f's CRASHED line (exit
    code 1).  The reason string is NEW (``phase1_timeout``) so an operator can
    tell a wall-clock timeout apart from ``unhandled_error_phase1`` (a real
    crash) and from ``llm_transport_error_phase1`` (a raised transport error)
    by reading state_transitions.reason alone — A7 was opened by exactly that
    kind of mislabelling, and this must not reintroduce it."""
    # One fresh step id shared by the transition and the log_step row — the
    # same pattern every other failure path in this file uses, so the timeout's
    # audit entries hang together under one step.
    step_id = new_id("step")
    # A timeout can fire at ANY stage (research, summarize, detect, ...), so
    # READ the target's current state from the DB rather than hardcoding
    # "new" — the state_transitions row must record where the target actually
    # was when the ceiling hit, or the audit trail lies about the timeout
    # point (the exact B1f lesson for crash transitions).
    current = conn.execute(
        "SELECT state FROM targets WHERE target_id=?;", (target_id,)
    ).fetchone()
    if current is None:
        # The row must exist — target_ids always come from import_csv in this
        # same run — and a transition for a phantom target would be a lying
        # audit row (same guard B1f's crash path uses).
        raise ValueError(f"target {target_id} has no targets row")
    from_state = current["state"]
    # The state change goes through THE gate, never a raw UPDATE.  Any state
    # → failed is valid (ANY_TARGET_TRANSITIONS); the NEW reason string names
    # the cause (precedent: A4c's llm_transport_error_phase1, B1d's
    # research_agent_no_output_phase1 — new reasons, no new states).
    transition(
        conn, target_id=target_id, from_state=from_state, to_state="failed",
        reason="phase1_timeout", actor="system",
        run_id=run_id, step_id=step_id,
    )
    # Golden Rule "never skip logging": the timeout gets its own failed step
    # row, carrying the ceiling value and the detail discriminator — so the
    # trace shows whether the CEILING fired (wait_for) or an SDK-level timeout
    # fired first, without opening the code.
    log_step(
        conn, run_id=run_id, step_id=step_id, target_id=target_id,
        tool_name="phase1_target_timeout",  # distinct tool_name so the row is greppable in the trace
        agent_id="system",  # deterministic pipeline code — the registered system agent
        input_data={"stage": "phase1_target_run", "timeout_seconds": timeout_seconds},
        output_data={"timeout_seconds": timeout_seconds, "detail": detail},
        status="failed",
    )
    # "failed" is the honest terminal Phase 1 state for a target that ran out
    # of time — it lands in the CLI's results dict, not its crashed dict.
    return "failed"


async def run_target_through_phase1_async(
    agent, *, conn, target_id: str, domain: str, run_id: str,
    offers_dir: str = DEFAULT_OFFERS_DIR,
) -> str:
    """The async core of ``run_target_through_phase1`` — same behaviour,
    same return contract, but a coroutine instead of a blocking call.

    Split out (taskmaster follow-up to B1g) because ``run_target_through_phase1``
    itself calls ``asyncio.run(...)``, which is only legal when NO event loop
    is already running on the calling thread. ``app/phase1_cli.py`` is that
    case (a bare synchronous script) and keeps calling the sync wrapper
    below unchanged. ``app/agents/taskmaster.py``'s ``import_and_research``
    tool is NOT that case — ADK's ``Runner`` invokes it from INSIDE an
    already-running event loop (confirmed by reading
    ``google/adk/tools/function_tool.py``: a sync tool callable is invoked
    directly on the caller's thread, no ``to_thread`` hop), so a second
    nested ``asyncio.run()`` there raises
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``
    on every single target — a real, previously-undiscovered bug (found by
    running a real 12-target batch live, 2026-08-29; existing tests never
    caught it because they patch ``run_target_through_phase1`` itself at the
    module boundary, one layer above where the nesting actually happens).
    ``import_and_research`` is now itself ``async def`` and ``await``s this
    function directly on the SAME thread/loop ADK already gave it — no
    second loop, no SQLite cross-thread hazard either (the DB connection
    never leaves its original thread).
    """
    # Check if the target has contact data linked (for scoring's persona fit)
    contact = conn.execute(
        "SELECT contact_id FROM targets WHERE target_id=?;", (target_id,)
    ).fetchone()
    has_contact_data = contact["contact_id"] is not None

    async def _run() -> dict:
        # Fresh in-memory session service per run.  Deliberately NOT
        # DatabaseSessionService: ADK's persistent session store would
        # duplicate the durable audit trail that already lives in the
        # steps / write_log / state_transitions tables, and the LangGraph
        # checkpointer's crash-recovery role is deliberately deferred to
        # Task A5 rather than reimplemented here.  In-memory sessions also
        # sidestep the serialization constraint: the live DB connection must
        # never enter session state (see the node __init__ rationale).
        session_service = InMemorySessionService()
        # Runner executes the agent against the session service;
        # auto_create_session=True lets run_async create the session on first
        # use instead of requiring a separate create_session call.
        runner = Runner(
            app_name="outbound",
            agent=agent,
            session_service=session_service,
            auto_create_session=True,
        )
        # Drive the agent once.  The "run" user message is a placeholder —
        # the deterministic nodes never read message content; it exists only
        # because ADK's Runner starts an invocation with a user turn.
        # session_id=target_id keys the session per target (the LangGraph
        # version used thread_id=target_id for the same reason: one target's
        # state must never collide with another's).  state_delta seeds the
        # initial state (target_id/domain/run_id/has_contact_data) exactly as
        # the old graph.invoke payload did.
        async for _ in runner.run_async(
            user_id="operator",
            session_id=target_id,
            new_message=types.Content(role="user", parts=[types.Part(text="run")]),
            state_delta={
                "target_id": target_id,
                "domain": domain,
                "run_id": run_id,
                "has_contact_data": has_contact_data,
                # B2c: the offers dir the judge reads offer context from —
                # seeded like every other per-run input, so the ScoreNode
                # reads it from session state exactly as it reads target_id.
                "offers_dir": offers_dir,
            },
        ):
            pass  # events are consumed only for their side effects; the terminal state is read from the session below
        # Read the terminal state straight from the session-state dict — NOT
        # by scraping the event stream — it is the merged result of every
        # node's state_delta, including the final "final_state" write.
        session = await session_service.get_session(
            app_name="outbound", user_id="operator", session_id=target_id,
        )
        return session.state

    # ── The B1g ceiling: bound the WHOLE per-target invocation in wall-clock ─
    # WHY THE CEILING SITS HERE, AT THE asyncio.run SEAM, AND NOT DEEPER: the
    # hang this ticket fixes is a network await that never resolves, and ADK
    # owns that await somewhere inside Runner.run_async — there is no single
    # deeper await to bound, and no SDK knob (if one even exists) covers
    # every tool the agent may use.  asyncio.wait_for around the whole
    # coroutine is the ONE point that spans the entire per-target run, and
    # cancelling the task cancels whatever await is pending — SDK-independent
    # by construction.  That is the guarantee: even if every SDK-level timeout
    # (part 2 of this ticket) is missing or misconfigured, a target cannot
    # stall the batch past this deadline.
    #
    # WHY wait_for AND the SDK-level timeouts, not just one: the SDK-level
    # timeout (app/llm.py, app/agents/research.py) fires per REQUEST and
    # carries the provider's own exception for attribution; this ceiling fires
    # per TARGET and guarantees the property.  A single stalled request
    # surfaces as the SDK's timeout exception, caught below and routed into
    # the same phase1_timeout bucket; anything else that stalls — a hang
    # outside any timed request, several sequential stalls — is cut off here.
    timeout_seconds = _resolve_target_timeout_seconds()  # env override or the documented default, resolved per call
    try:
        # wait_for adds the wall-clock deadline; when it fires it cancels the
        # inner task, which cancels the pending network await inside ADK,
        # and then raises TimeoutError here.  No asyncio.run() at this seam
        # any more — this function IS a coroutine now, and the caller (either
        # the sync run_target_through_phase1 wrapper below, or taskmaster.py
        # directly) owns the event loop.
        state = await asyncio.wait_for(_run(), timeout=timeout_seconds)
    except TimeoutError as exc:
        # The ceiling fired.  TimeoutError here IS asyncio.TimeoutError (an
        # alias since Python 3.11, and this repo runs 3.14) — deliberately
        # caught by its builtin name because wait_for is documented to raise
        # it, and because the same name also covers any timeout exception the
        # inner run itself may raise (e.g. aiohttp's ServerTimeoutError,
        # which subclasses asyncio.TimeoutError).  exc carries no useful
        # message (wait_for raises a bare TimeoutError), so the detail string
        # states the ceiling fact explicitly.
        return _record_phase1_timeout(
            conn, target_id=target_id, run_id=run_id,
            timeout_seconds=timeout_seconds,
            detail=f"per-target wall-clock ceiling of {timeout_seconds}s exceeded "
                   f"(asyncio.wait_for cancelled the run)",
        )
    except httpx.TimeoutException as exc:
        # An SDK-level timeout fired BEFORE the ceiling (a single stalled
        # model request): the google-genai client raises httpx.ReadTimeout /
        # httpx.ConnectTimeout unwrapped (measured, app/llm.py §9c), and ADK
        # lets it propagate out of the agent loop.  Without this clause it
        # would land in B1f's CRASHED bucket — but a timed-out target must not
        # look like a crash (the ticket's Golden Rule).  Route it into the
        # SAME phase1_timeout bucket, keeping the SDK's exception text in the
        # step row so the trace still shows which layer fired.  httpx is
        # already a declared direct dependency (pyproject.toml); catching its
        # timeout family here — not its whole RequestError family — keeps
        # every other transport failure flowing to B1f exactly as before.
        return _record_phase1_timeout(
            conn, target_id=target_id, run_id=run_id,
            timeout_seconds=timeout_seconds,
            detail=f"{type(exc).__name__}: {exc}",
        )
    # The run ended with the full Phase 1 state; extract the terminal state string
    return state["final_state"]


def select_research_pending_targets(conn, *, limit: int) -> list[str]:
    """The set of targets imported but never researched — state 'new'.

    Exists for exactly one recovery scenario: an import_and_research call
    imports a whole CSV successfully, then runs out of wall-clock time (or
    crashes) before researching every row it just created. Those targets
    sit at 'new' forever unless something processes them directly —
    re-running the SAME CSV through import_csv fails outright the moment
    any of its domains already exist (accounts.normalized_domain is
    UNIQUE), so a fresh import is never the fix for a target already sitting
    in the database. This is the single query the Taskmaster's
    resume_pending_research tool selects through, mirroring the same
    "one selector, one source of truth" precedent
    select_draft_eligible_targets already set for the draft stage.
    """
    # state='new' is the ONLY terminal-of-import state a target can be
    # stuck at before Phase 1 ever ran on it — 'scored'/'watchlist'/
    # 'not_target'/'failed' all mean Phase 1 already concluded.
    rows = conn.execute(
        "SELECT target_id FROM targets WHERE state = 'new' ORDER BY created_at LIMIT ?;",
        (limit,),
    ).fetchall()
    return [r["target_id"] for r in rows]


def run_target_through_phase1(
    agent, *, conn, target_id: str, domain: str, run_id: str,
    offers_dir: str = DEFAULT_OFFERS_DIR,
) -> str:
    """Synchronous entry point — app/phase1_cli.py's unchanged call site.

    A thin asyncio.run() wrapper around run_target_through_phase1_async
    (see that function's docstring for why the split exists). This is the
    ONLY place asyncio.run() is called for this stage now — phase1_cli.py
    is a bare synchronous script with no event loop of its own running, so
    starting one here is legal, unlike inside taskmaster.py's tool.
    """
    return asyncio.run(
        run_target_through_phase1_async(
            agent, conn=conn, target_id=target_id, domain=domain,
            run_id=run_id, offers_dir=offers_dir,
        )
    )
