"""The agent-entry guardrail (ticket B4a): ``make_kill_switch_callback()``.

WHY THIS HOOK — this module turns the kill-switch file into a live control
by attaching to ADK's ``before_agent_callback``.  That hook is the ONLY
ADK mechanism that can stop a running agent BEFORE it does any work, and it
was chosen over ``before_tool_callback`` deliberately — the A4b rescope
found before_tool_callback unusable for P3a/P4 because those rules need an
``icp_assessment`` that only exists after scoring (docs/policy-matrix.md
§2, "Where P3a and P4 actually run").  The kill switch has no such
dependency: it is a global pre-condition readable before any work happens,
so the agent-ENTRY hook is the correct placement.  This is not a repeat of
a rejected approach.

MEASURED ADK 2.7.1 CONTRACT (pinned wheel, verified by probe, not from
docs):
- ``before_agent_callback`` is a field on ``BaseAgent`` (line ~150), so it
  exists on every agent: the LlmAgents, the deterministic BaseAgent nodes,
  SequentialAgent and LoopAgent alike.
- The callback must accept a keyword argument named exactly
  ``callback_context`` (CallbackContext is a type alias of Context in
  2.7.1); it may be sync or async; returning ``Optional[types.Content]``.
- Returning non-None content HALTS THE ENTIRE INVOCATION when the callback
  is attached to the ROOT agent: ``_handle_before_agent_callback`` sets
  ``ctx.end_invocation = True`` and ``run_async`` returns without running
  ``_run_async_impl`` — the root's sub-agents never start.  That whole-run
  halt IS the kill-switch semantic.
- A callback attached to a SUB-agent skips ONLY that sub-agent: the child
  invocation context is a ``model_copy`` of the parent's, so
  ``end_invocation`` does not propagate to the parent, and neither
  SequentialAgent nor LoopAgent re-checks it between sub-agents (measured).
  Per-agent refusal is therefore NOT implemented by attaching to
  sub-agents — see below.

TWO CONTROLS IN ONE CALLBACK — on every agent entry it re-reads the switch
file (uncached — the only moment the switch matters is a flip mid-run) and:
1. GLOBAL: engaged → halt the whole invocation with a logged refusal.
2. PER-AGENT: the entered agent's name, plus every id named in
   ``check_agent_ids``, is looked up in ``agent_registry``; ``enabled=0``
   refuses before any of them does work — instead of today's behaviour,
   where a disabled agent still runs and only its write-gate-attributed
   writes are refused deep into the work.  ``check_agent_ids`` exists
   because the registered principals are SUB-agents (draft_writer,
   draft_critic) while the guardrail is attached to the container ROOT
   (draft_loop) — the root entry is the one place a refusal cleanly stops
   the whole loop, and a loop whose writer is disabled cannot run
   meaningfully anyway (the critic's {draft} placeholder would go
   unfilled).  A name with no registry row (the containers themselves and
   the deterministic nodes) is NOT a registered principal, so it is not
   refused — the write gate still governs its writes.  An UNREADABLE
   registry fails closed (halt) for the same reason the switch file does:
   an agent that runs because we could not check its permission is the
   failure mode the whole gate exists to prevent.

ATTACHMENT RULE (measured, load-bearing): the guardrail is attached ONLY
to container roots (SequentialAgent/LoopAgent — plain BaseAgents), never
directly to an LlmAgent with ``output_schema``.  LlmAgent's
``_handle_before_agent_callback`` feeds callback-returned Content through
``__maybe_save_output_to_state``, which validates it against
``output_schema`` (measured: a halt on the draft writer crashed with
"Invalid JSON ... for EmailDraft").  Containers have no output_schema, so
their halt Content is safe.

EVERY HALT IS LOGGED — Golden Rule "never skip logging" applied to the one
event an operator most needs after the fact: a failed ``steps`` row with
``tool_name="kill_switch"`` carrying the halted agent's name, the scope
(global vs per-agent), and the reason, so the trace shows exactly which
agent was stopped and why even though no work happened.
"""

from collections.abc import Callable  # the callback factory's return type
from typing import TYPE_CHECKING  # ADK types only in annotations — no ADK import needed to use this module

from google.genai import types  # types.Content: the halt message returned to the user, ADK's "skip this agent" signal

from app.ids import new_id  # one fresh step id per halt — each refusal is its own trace row (the A6 one-id-per-write rule)
from app.kill_switch import read_kill_switch  # the fail-closed, uncached reader — this module's source of truth
from app.tools.log_step import log_step  # the steps-trace writer — every halt lands in the trace

if TYPE_CHECKING:
    # The ADK callback protocol — used only in annotations so this module
    # does not need the ADK runtime at import time.
    from google.adk.agents.base_agent import BeforeAgentCallback

# The steps.tool_name every kill-switch refusal row carries — distinct from
# every pipeline tool name so the trace log shows "the kill switch fired
# here" at a glance, greppable across the whole steps table.
KILL_SWITCH_TOOL_NAME = "kill_switch"

# The steps.agent_id on halt rows: the guardrail is deterministic pipeline
# code, the same "system" principal every node attributes its own
# bookkeeping to — the HALTED agent's name travels in output_data, not in
# agent_id (agent_id records who WROTE the row, not who was stopped).
_GUARDRAIL_AGENT_ID = "system"


def _halt(callback_context, *, conn, agent_name: str, scope: str, reason: str) -> types.Content:
    """Log the halt, publish the short-circuit sentinel, and return the
    Content that skips the agent / ends the invocation.

    The parts are the measured 2.7.1 mechanics, and each is load-bearing:
    - the ``log_step`` row is the observable trace (a silent halt is
      indistinguishable from a broken pipeline — Golden Rule);
    - ``callback_context.state["final_state"] = "failed"`` is the sentinel
      every deterministic node's existing short-circuit checks, so the run
      ends with a clean terminal state instead of a KeyError on the
      missing key.  The value is "failed" specifically because the Phase 1
      policy_gate node's guard tests ``== "failed"`` — any other value
      would let that node run against a state the halted run never
      produced.  The target's DB row is deliberately NOT transitioned (the
      switch aborts the RUN, not the target — it stays in its current
      state and is re-run after disengaging; the run reports "failed" via
      this sentinel);
    - returning the Content is what makes ADK skip THIS agent's
      ``_run_async_impl`` and, at a container root, end the whole
      invocation (end_invocation).
    """
    # run_id/target_id come from the seeded session state (both pipelines
    # seed them before the first agent entry); "unknown" is the honest
    # fallback for a halt that fires before any seed — the row still lands
    # in the trace rather than being skipped.
    run_id = callback_context.state.get("run_id") or "unknown"
    target_id = callback_context.state.get("target_id")  # None is legal for log_step — a global halt may precede any target
    # One fresh step id per halt: steps.step_id is the PRIMARY KEY and a
    # run can be halted many times (once per agent entry while engaged) —
    # reusing an id would raise IntegrityError on the second halt.
    step_id = new_id("step")
    # The trace row.  status="failed" is the steps vocabulary's honest
    # refusal ("the run was stopped here"), the same status every other
    # refusal path in the repo uses (e.g. _record_draft_refusal).
    log_step(
        conn,
        run_id=run_id,
        step_id=step_id,
        target_id=target_id,
        tool_name=KILL_SWITCH_TOOL_NAME,
        agent_id=_GUARDRAIL_AGENT_ID,  # deterministic guardrail code wrote this row
        input_data={"stage": "before_agent_entry", "scope": scope, "agent": agent_name},
        output_data={"halted_agent": agent_name, "scope": scope, "reason": reason},
        status="failed",
    )
    # The sentinel pair: final_state short-circuits every deterministic
    # node downstream (and gives run_target_through_phase1 a clean
    # terminal value); kill_switch_reason leaves the WHY in the final
    # session state so the terminal state names the cause, not just the
    # halt.
    callback_context.state["final_state"] = "failed"
    callback_context.state["kill_switch_reason"] = reason
    # The Content ADK returns to the user instead of running the agent —
    # self-explanatory so the operator reading the event stream sees the
    # halt and its reason without opening the steps table.  SAFE here
    # because the guardrail is only ever attached to container roots (see
    # the module docstring's attachment rule).
    return types.Content(
        role="model",
        parts=[types.Part(text=f"HALTED by kill switch ({scope}): {reason}")],
    )


def make_kill_switch_callback(
    *,
    conn,
    kill_switch_path: str | None = None,
    check_agent_ids: tuple[str, ...] = (),
) -> "BeforeAgentCallback":
    """Build a ``before_agent_callback`` enforcing the global kill switch
    and the per-agent registry switch.

    ``conn`` is captured in the closure the same way
    ``make_tool_budget_callback`` in app/agents/adk_support.py captures its
    budget state: the callback fires inside ADK's event loop, far from the
    caller's locals, so everything it needs must travel in the closure.
    ``kill_switch_path``, when given, pins the switch file for this
    callback (tests, future callers); None means the default/env
    resolution inside ``read_kill_switch``.  ``check_agent_ids`` names the
    registered principals the CONTAINER this callback sits on will run
    (e.g. the draft loop's writer and critic) — each is refused at the
    container's entry when disabled, before any of them burns a token.

    Returns a SYNC callback (ADK awaits it only if awaitable — measured) —
    the work here is a small file read plus primary-key SELECTs, both
    blocking-safe on sqlite3 and pg8000 connections.
    """

    def _before_agent(*, callback_context) -> types.Content | None:
        # ── The agent's identity, resolved from the context — NOT guessed
        # and NOT threaded through a constructor argument.  Measured 2.7.1:
        # CallbackContext is a type alias of Context, whose
        # get_invocation_context() returns the InvocationContext whose
        # .agent is the agent being entered (its .name is the agent's
        # stable trace identity).  agent is None only when the Runner
        # drives a BaseNode instead of a BaseAgent — never in this repo's
        # pipelines, but the fallback keeps the halt self-describing.
        invocation_ctx = callback_context.get_invocation_context()
        agent_name = invocation_ctx.agent.name if invocation_ctx.agent is not None else "unknown_agent"

        # ── CONTROL 1: the global switch.  Read UNCACHED on every agent
        # entry (runbook.md §1 — a cached read cannot see a mid-run flip,
        # and mid-run is the only moment the switch matters).  Engaged
        # halts unconditionally — it outranks everything, including a
        # healthy agent registry (P6 dominates: docs/policy-matrix.md).
        kill_state = read_kill_switch(kill_switch_path)
        if kill_state.engaged:
            return _halt(
                callback_context,
                conn=conn,
                agent_name=agent_name,
                scope="global",
                reason=kill_state.reason,
            )

        # ── CONTROL 2: the per-agent registry switch.  Two lookups:
        # the entered agent itself (covers a future pipeline that attaches
        # this callback directly to a registered principal) and every id
        # the caller named in check_agent_ids (covers TODAY's wiring,
        # where the registered principals are sub-agents of the container
        # this callback sits on).  Names with no registry row — the
        # containers themselves and the deterministic nodes — are not
        # registered principals, so they are not refused here (the write
        # gate still governs their writes); refusing them would halt every
        # run on a lookup that has no row to read.
        for candidate_id in (agent_name, *check_agent_ids):
            try:
                row = conn.execute(
                    "SELECT enabled FROM agent_registry WHERE agent_id=?;",
                    (candidate_id,),
                ).fetchone()
            except Exception as exc:
                # FAIL CLOSED (§4 of the ticket): an unreadable registry
                # must halt, never allow — an agent that runs because its
                # permission could not be checked is the exact failure mode
                # this gate exists to prevent.  The exception type and
                # message go into the reason so the trace shows it was a DB
                # fault, not an operator's disable.
                return _halt(
                    callback_context,
                    conn=conn,
                    agent_name=candidate_id,
                    scope="per_agent",
                    reason=(
                        f"agent_registry unreadable while checking {candidate_id!r} "
                        f"({type(exc).__name__}: {exc}) — failing closed"
                    ),
                )
            if row is not None and row["enabled"] == 0:
                # The per-agent kill switch (docs/policy-matrix.md §3a):
                # the operator disabled this principal.  Refuse at the
                # container's entry — the disabled agent never runs, so it
                # burns no model tokens, and the refusal names the agent so
                # the trace shows WHO was stopped and why.
                return _halt(
                    callback_context,
                    conn=conn,
                    agent_name=candidate_id,
                    scope="per_agent",
                    reason=f"agent {candidate_id!r} is disabled (agent_registry.enabled=0)",
                )

        # ── Disengaged, every checked agent enabled (or not a registered
        # principal): None lets ADK run the agent normally.  Returning
        # None is the ONLY "allow" signal in the callback contract —
        # measured.
        return None

    return _before_agent
