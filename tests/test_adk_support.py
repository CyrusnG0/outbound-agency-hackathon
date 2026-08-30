# tests/test_adk_support.py — tests for app/agents/adk_support.py (ticket B1a).
#
# No live API calls anywhere: model resolution runs against a temp
# config/models.yaml plus monkeypatched env vars (same pattern as
# tests/test_llm.py's _resolve_model tests), and the budget callback runs
# against fake tool / tool_context objects matching the signature measured
# in the pinned google-adk==2.7.1 — ADK invokes before_tool_callback with
# keyword args tool=, args=, tool_context=, and a dict return blocks the tool
# call while None lets it run.
from types import SimpleNamespace

import pytest

from app.agents.adk_support import make_tool_budget_callback, resolve_adk_model


def _write_temp_models_config(tmp_path, content: str) -> str:
    """Create config/models.yaml under tmp_path.

    resolve_adk_model calls app.llm._resolve_model with its default
    config_path="config/models.yaml" (a cwd-relative path) — the tests use
    monkeypatch.chdir(tmp_path) so that default resolves to a temp file
    instead of the real repo config, mirroring tests/test_llm.py's pattern of
    never touching the real config/models.yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "models.yaml"
    config_path.write_text(content)
    return str(config_path)


# ── resolve_adk_model ─────────────────────────────────────────────────────────


def test_resolve_adk_model_returns_pinned_model_for_gemini_alias(tmp_path, monkeypatch):
    # A gemini alias in a temp config plus its env pin must come back as the
    # bare pinned model string — exactly what LlmAgent(model=...) expects.
    _write_temp_models_config(tmp_path, "research_model: gemini.flash\n")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash")
    monkeypatch.chdir(tmp_path)  # point _resolve_model's default relative path at the temp config
    assert resolve_adk_model("research_model") == "gemini-3.5-flash"


def test_resolve_adk_model_refuses_deepseek_provider_naming_it(tmp_path, monkeypatch):
    # deepseek.v4_flash is a legal value in config/models.yaml (supported
    # fallback in app/llm.py) but ADK's LlmAgent cannot speak it — the guard
    # must fail loudly at construction, naming the resolved provider.
    _write_temp_models_config(tmp_path, "research_model: deepseek.v4_flash\n")
    monkeypatch.setenv("DEEPSEEK_V4_FLASH_MODEL", "deepseek-v4-flash")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError) as exc_info:
        resolve_adk_model("research_model")
    message = str(exc_info.value)
    assert "deepseek" in message  # names the provider that reached the ADK boundary
    assert "research_model" in message  # names the alias, so the operator knows which role to repoint


def test_resolve_adk_model_still_raises_when_env_var_unset(tmp_path, monkeypatch):
    # The pinning discipline is preserved, not softened: an unset env var
    # must still refuse to boot, exactly as _resolve_model does for every
    # other caller — resolve_adk_model must not catch or paper over it.
    _write_temp_models_config(tmp_path, "research_model: gemini.flash\n")
    monkeypatch.delenv("GEMINI_FLASH_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError):
        resolve_adk_model("research_model")


# ── make_tool_budget_callback ─────────────────────────────────────────────────


class _FakeToolContext:
    """Minimal stand-in for google.adk's ToolContext (an alias of Context).

    Carries only the three things the budget callback reads, mirroring what
    ADK 2.7.1 really provides: a delta-aware session state (a plain dict has
    the same .get / []= interface the callback uses), the fresh-per-run
    invocation_id, and the name of the agent whose tool is about to run."""

    def __init__(self, state: dict, invocation_id: str, agent_name: str):
        self.state = state
        self.invocation_id = invocation_id
        self.agent_name = agent_name


_FAKE_TOOL = SimpleNamespace(name="search_web")


def test_budget_allows_exactly_max_tool_calls_then_blocks():
    # Budget of 2: the first two attempts must return None (ADK runs the
    # tool normally); the third must return a dict (ADK skips the tool and
    # feeds the dict to the model as the call's function response).
    callback = make_tool_budget_callback(2)
    context = _FakeToolContext(state={}, invocation_id="inv-1", agent_name="research_agent")
    assert callback(tool=_FAKE_TOOL, args={}, tool_context=context) is None  # attempt 1: allowed
    assert callback(tool=_FAKE_TOOL, args={}, tool_context=context) is None  # attempt 2: allowed
    blocked = callback(tool=_FAKE_TOOL, args={}, tool_context=context)  # attempt 3: blocked
    assert isinstance(blocked, dict)  # a dict is what ADK 2.7.1 treats as "skip this tool"
    assert "2" in blocked["result"]  # the message names the limit


def test_budget_is_per_invocation_not_shared_across_agents():
    # The regression test for the subtle bug: two invocations that share ONE
    # session state dict (what ADK really does — the state store survives
    # across invocations within a session) must each get a full budget.  A
    # module-level global OR a bare session-state counter would let the
    # second invocation inherit the first's spent budget; keying by
    # (agent, invocation) is what makes the second invocation start at zero.
    callback = make_tool_budget_callback(1)
    shared_state = {}  # deliberately shared, like two run_async calls on one session
    first_invocation = _FakeToolContext(state=shared_state, invocation_id="inv-1", agent_name="research_agent")
    second_invocation = _FakeToolContext(state=shared_state, invocation_id="inv-2", agent_name="research_agent")
    assert callback(tool=_FAKE_TOOL, args={}, tool_context=first_invocation) is None  # target 1, call 1: allowed
    assert callback(tool=_FAKE_TOOL, args={}, tool_context=first_invocation) is not None  # target 1, call 2: blocked
    assert callback(tool=_FAKE_TOOL, args={}, tool_context=second_invocation) is None  # target 2, call 1: full budget again


def test_budget_is_per_agent_within_one_invocation():
    # Two agents chained inside ONE invocation (this repo's SequentialAgent
    # runs every node in a single invocation) must not share a budget either:
    # the record is keyed by (agent, invocation), not invocation alone.
    callback = make_tool_budget_callback(1)
    shared_state = {}  # same session state, same invocation
    research_ctx = _FakeToolContext(state=shared_state, invocation_id="inv-1", agent_name="research_agent")
    judge_ctx = _FakeToolContext(state=shared_state, invocation_id="inv-1", agent_name="icp_judge_agent")
    assert callback(tool=_FAKE_TOOL, args={}, tool_context=research_ctx) is None  # research: allowed
    assert callback(tool=_FAKE_TOOL, args={}, tool_context=research_ctx) is not None  # research: blocked
    assert callback(tool=_FAKE_TOOL, args={}, tool_context=judge_ctx) is None  # judge: full budget of its own


def test_on_call_fires_for_allowed_and_blocked_attempts():
    # on_call is the seam B1b's log_step writer plugs into: it must observe
    # every attempt, including the one that gets refused, so the steps trace
    # shows refused tool calls as well as executed ones.
    seen = []
    callback = make_tool_budget_callback(1, on_call=seen.append)
    context = _FakeToolContext(state={}, invocation_id="inv-1", agent_name="research_agent")
    callback(tool=_FAKE_TOOL, args={}, tool_context=context)  # allowed attempt
    callback(tool=_FAKE_TOOL, args={}, tool_context=context)  # blocked attempt
    assert seen == ["search_web", "search_web"]


@pytest.mark.parametrize("bad_budget", [0, -1])
def test_non_positive_budget_raises_value_error(bad_budget):
    # A non-positive budget is a programming error in the agent wiring, not
    # a "no tools allowed" configuration — it must fail at factory time,
    # before any agent is constructed around the callback.
    with pytest.raises(ValueError):
        make_tool_budget_callback(bad_budget)
