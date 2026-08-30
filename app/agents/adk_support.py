"""ADK glue foundations for Milestone B (ticket B1a).

Two small pieces that keep Milestone A's hard-won disciplines intact once
Milestone B starts constructing ``google.adk`` ``LlmAgent``\\s:

1. ``resolve_adk_model`` — ADK's ``LlmAgent(model=...)`` takes a hardcoded
   model string and builds its own ``google.genai`` client from the PROCESS
   environment; it never consults this repo's alias file or env pin.  This
   function routes the repo's ``config/models.yaml`` alias through the SAME
   resolution path every other LLM call uses (``app.llm._resolve_model``), so
   an ADK agent is pinned exactly like ``call_structured`` is — and it refuses
   non-gemini providers, which ADK's agent loop cannot speak.

2. ``make_tool_budget_callback`` — an ``LlmAgent(before_tool_callback=...)``
   that bounds how many tool calls one agent invocation may make.  Measured
   against the pinned google-adk==2.7.1: ``LlmAgent`` has NO ``max_iterations``
   field (only ``LoopAgent`` does), so a tool-using agent can loop calling
   tools forever.  ``before_tool_callback`` fires before each tool call and a
   dict return value blocks the call — that is the hook used here.

This module is pure glue: no DB access, no network access, no agent
construction, no file reads of its own (only what ``_resolve_model`` does).
"""

# U2-fix2 enabling fix (pre-existing bug disclosed in docs/current_status.md's
# D3 writeup): `BeforeToolCallback` is imported only under TYPE_CHECKING yet
# used as a LIVE return annotation below, and without deferred evaluation the
# annotation is evaluated at function-definition time -> NameError, which broke
# pytest collection on 14 test files.  `from __future__ import annotations`
# turns every annotation in this module into a lazily-evaluated string, so the
# TYPE_CHECKING-only ADK names stay out of the runtime import graph (the
# module's deliberate no-ADK-runtime-import design) while the annotations stop
# crashing at import.  Behavior is unchanged: nothing in the repo inspects
# these annotations via typing.get_type_hints().
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# Load-bearing import, and NOT only for the function it provides: importing
# app.llm runs its module-level ``load_dotenv()``, which copies .env's
# GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION /
# GOOGLE_API_KEY into the process environment.  google.adk's LlmAgent builds
# its own google.genai client FROM THE PROCESS ENVIRONMENT at construction
# time (measured during B1a design: an agent built before app.llm was ever
# imported fails with "ValueError: No API key was provided" even though
# call_structured works in the same interpreter — the env is only populated
# once something imports app.llm).  Keeping this import at the top of THIS
# module makes the dependency explicit: importing app.agents.adk_support
# guarantees the env is populated before any agent is constructed.  Do NOT
# "tidy" it into a local import inside resolve_adk_model() — an agent
# constructed before the first resolution call would then see an empty
# environment and fail at the API boundary instead of booting cleanly.
from app.llm import _resolve_model

if TYPE_CHECKING:
    # ADK types used only in annotations, so this glue module does not need
    # to import the ADK runtime itself.  Signature facts below were measured
    # against the pinned google-adk==2.7.1 installation, not from docs.
    from google.adk.agents.llm_agent import BeforeToolCallback
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext


def resolve_adk_model(model_alias: str) -> str:
    """Resolve a role alias (e.g. "research_model") to the pinned model string
    an ADK ``LlmAgent(model=...)`` needs.

    Delegates to ``app.llm._resolve_model`` — deliberately NOT reimplementing
    the alias lookup and NOT reading config/models.yaml directly, so every
    model choice in the repo (call_structured and every ADK agent) shares ONE
    resolution path and one source of truth.

    ADK's ``LlmAgent`` speaks gemini only: it hands its model string to the
    google.genai SDK it constructs internally.  config/models.yaml may legally
    resolve to "deepseek" (still a supported fallback provider in app/llm.py);
    handing that string to LlmAgent would fail confusingly deep inside ADK at
    the API boundary.  A non-gemini resolution therefore fails LOUDLY here, at
    construction time, naming the alias, the resolved provider, and the model.

    Raises:
        RuntimeError: the alias resolved to a non-gemini provider (raised
            here), or the alias's env pin is unset (raised by _resolve_model
            itself and deliberately NOT caught — refusing to boot on an
            unpinned model is exactly the discipline this function exists to
            preserve for the ADK path).
    """
    # The one resolution path: same two-step alias→env lookup every other
    # LLM call in the repo uses.  Returns (provider, model); provider is the
    # prefix before the first "." in the alias value (e.g. "gemini").
    provider, model = _resolve_model(model_alias)
    if provider != "gemini":
        # A fallback-provider pin reached the ADK boundary.  Fail at
        # construction with a self-explanatory message instead of letting
        # LlmAgent build a client for a model string its SDK cannot serve —
        # the alternative is a confusing SDK error on the first live call.
        raise RuntimeError(
            f"resolve_adk_model({model_alias!r}) resolved to provider "
            f"{provider!r} (model {model!r}), but ADK LlmAgent agents require "
            f"a gemini provider — repoint the {model_alias!r} alias in "
            f"config/models.yaml to gemini.* or use call_structured for "
            f"non-gemini providers"
        )
    # The pinned model string, exactly as LlmAgent(model=...) expects it.
    return model


# State key the budget counter lives under.  The "temp:" prefix does two jobs
# (both measured against ADK 2.7.1's State implementation):
# 1. It bypasses state-schema validation — _validate_state_entry skips any
#    key containing ":", so the counter cannot raise StateSchemaError when a
#    future agent declares a state schema with no budget field.
# 2. It is stripped from the persisted event delta by the session service
#    (_trim_temp_delta_state), so budget bookkeeping never reaches storage.
_ADK_BUDGET_STATE_KEY = "temp:adk_tool_call_budget"


def make_tool_budget_callback(
    max_tool_calls: int,
    *,
    on_call: Callable[[str], None] | None = None,
) -> BeforeToolCallback:
    """Build a ``before_tool_callback`` that bounds one agent invocation's
    tool calls to ``max_tool_calls``.

    ADK 2.7.1 contract (measured in flows/llm_flows/functions.py of the pinned
    wheel, not from docs): each before_tool_callback is invoked with keyword
    arguments ``tool=``, ``args=``, ``tool_context=``; it may be sync or
    async; a truthy dict return value REPLACES the real tool call and is fed
    back to the model as that call's function response, while returning None
    lets the tool run.  This callback therefore returns a short dict naming
    the limit once the budget is spent — the agent is told it is out of
    budget and allowed to finish with what it has — rather than raising,
    which would abort the whole run.

    The counter lives in ``tool_context.state``, but that store is the
    SESSION's state (ADK's Context.__init__ hands the session's state dict to
    State as its backing value and State.__setitem__ writes through to it), so
    it survives across invocations within one session.  A bare counter there
    would be the same shared-state bug as a module-level global: the second
    agent invocation in a session would inherit the first's spent budget and
    be refused before doing any work.  The record is therefore keyed by
    (agent_name, invocation_id): invocation_id is fresh for every
    runner.run_async() call, and agent_name keeps two agents chained inside
    ONE invocation from sharing a budget (this repo's Phase 1 SequentialAgent
    already runs every node inside a single invocation, and Milestone C4
    chains agents as sub_agents the same way).  A new (agent, invocation)
    pair starts at zero — no Python global, no cross-invocation leakage.

    ``on_call``, when given, is invoked with the tool name on every attempt —
    allowed or blocked.  B1b will pass a log_step writer through it so every
    tool attempt lands in the steps trace; this module deliberately does NOT
    import log_step or touch any database itself.

    Raises:
        ValueError: ``max_tool_calls`` is not positive — a programming error
            (the caller wired the agent wrong), not a "no tools allowed"
            configuration.
    """
    if max_tool_calls <= 0:
        # Fail at agent-construction time, not on the first tool call: a
        # non-positive budget is a wiring mistake the operator must fix
        # before the agent ever runs.
        raise ValueError(
            f"max_tool_calls must be a positive int, got {max_tool_calls!r} — "
            f"a non-positive budget is a programming error, not a "
            f"'no tools allowed' configuration"
        )

    def _before_tool_call(
        *,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        # Which agent invocation is making this attempt?  agent_name is the
        # agent whose tool ADK is about to run; invocation_id is the current
        # runner.run_async() call.  Together they define the "one agent
        # invocation" the budget is scoped to — both come from the
        # InvocationContext ADK creates fresh for each run.
        agent_name = tool_context.agent_name
        invocation_id = tool_context.invocation_id
        # The record left by an earlier tool call in the SAME agent
        # invocation, if any: a (agent, invocation, count) tuple stored in
        # the session state.  None on the very first tool call of a run.
        record = tool_context.state.get(_ADK_BUDGET_STATE_KEY)
        if (
            record is not None
            and record[0] == agent_name
            and record[1] == invocation_id
        ):
            # Same agent, same invocation — continue counting from where the
            # previous tool call left off.
            calls_so_far = record[2]
        else:
            # First call of this agent invocation (or the previous record
            # belonged to a different agent/invocation) — the budget restarts
            # at zero, which is the entire point of keying by
            # (agent, invocation) instead of a bare counter.
            calls_so_far = 0
        call_number = calls_so_far + 1
        # Persist the updated count for the NEXT tool call of this agent
        # invocation to read.  Written through tool_context.state so it never
        # touches a Python global and dies with the session, not the process.
        tool_context.state[_ADK_BUDGET_STATE_KEY] = (
            agent_name,
            invocation_id,
            call_number,
        )
        if on_call is not None:
            # Notify the observer (B1b's log_step writer) on EVERY attempt,
            # allowed or blocked, so the trace log shows tools that were
            # refused as well as tools that ran.
            on_call(tool.name)
        if call_number > max_tool_calls:
            # Budget exhausted: return a dict, the value ADK 2.7.1 treats as
            # "skip this tool" — the dict becomes the tool's function
            # response and the model is told the limit, so it can finish with
            # what it has instead of the run being aborted by an exception.
            return {
                "result": (
                    f"Tool call '{tool.name}' blocked: the budget of "
                    f"{max_tool_calls} tool call(s) per agent invocation is "
                    f"exhausted — finish the task with what you have"
                )
            }
        # Under budget: None tells ADK to run the tool normally.
        return None

    return _before_tool_call
