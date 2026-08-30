# tests/test_research_agent.py — tests for app/agents/research.py (ticket B1b).
#
# No live API calls anywhere.  The fetch_page tool is driven directly with a
# fake ToolContext (the same minimal stand-in pattern as test_adk_support.py),
# the underlying network tool is mocked at "app.tools.fetch_sources.fetch_sources",
# and agent construction runs offline: model resolution is patched and the
# Google env vars LlmAgent construction reads are set to inert values (no
# request is ever built, let alone sent).
import asyncio  # the bookkeeping node's _run_async_impl is an async generator
import json  # steps.output_json is a JSON string — parse it to assert the char count
from types import SimpleNamespace  # fake tool object for the budget-callback test
from unittest.mock import patch  # patch every network/model boundary so the suite stays offline

import pytest
from google.adk.models import LlmResponse  # build REAL response objects so tests pin the actual SDK field names the callback reads
from google.genai import types  # real GroundingMetadata / UsageMetadata / FinishReason instances — offline pure Pydantic models, no client

from app.agents.phase1 import ResearchBookkeepingNode  # the B1b governance node under test
from app.agents.research import (  # the B1b module under test
    NO_RESEARCH_FINDINGS_SENTINEL,
    _make_research_after_model_callback,
    build_research_agent,
    make_fetch_page_tool,
)
from app.agents_registry import seed_agent_registry  # register "system" so transition()'s write_gate accepts writes
from app.db import apply_schema, connect  # per-test temp sqlite database
from app.ids import new_id  # fresh step ids for the fetch rows the B1d contradiction test seeds into steps
from app.tools.fetch_sources import NormalizedSource  # the source shape fetch_sources returns
from app.tools.log_step import log_step  # seed the fetch_company_page row the B1d contradiction test reads back
from app.write_gate import commit  # seed offer/account/target rows through the single write path


@pytest.fixture
def conn(scratch_db_target):
    """Temp DB with schema + one new target — the same seeding pattern as
    tests/test_agents_phase1.py (registry seed, offer, account, target)."""
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # Register the system agent (plan A3) — transition()'s write gate refuses
    # unregistered agents, so the bookkeeping node's writes would fail without it.
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


@pytest.fixture
def adk_env(monkeypatch):
    """Inert Google env vars so LlmAgent construction (which builds a genai
    client from the process environment) succeeds offline in tests 3/4 —
    no request is ever sent, so the values are placeholders."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")


class _FakeToolContext:
    """Minimal stand-in for ADK's ToolContext (an alias of Context).

    Carries the three things the tool and the budget callback read: the
    session state (a plain dict — same .get / []= interface), the fresh
    invocation_id, and the agent name (the budget counter is keyed by
    (agent_name, invocation_id), so both must be present).
    """

    def __init__(self, state: dict, invocation_id: str = "inv-1", agent_name: str = "research"):
        self.state = state
        self.invocation_id = invocation_id
        self.agent_name = agent_name


async def _collect_events(node, ctx):
    """Drive one node's _run_async_impl to completion and collect its events."""
    return [event async for event in node._run_async_impl(ctx)]


# ── fetch_page: the static path as a model-facing FunctionTool ───────────────


@pytest.mark.parametrize(
    # The two failure shapes the real run produced: every fetch failed
    # (403/timeout), or a page fetched but extracted to nothing (JS shell).
    "fake_sources, expected",
    [
        ([], "FETCH FAILED: no page content could be fetched from acme.test"),
        (
            [NormalizedSource("company_website", "https://acme.test", "", "t", 0.8, 1, "static")],
            "FETCH FAILED: acme.test was fetched but contained no readable text",
        ),
    ],
)
def test_fetch_page_returns_descriptive_string_and_never_raises(conn, fake_sources, expected):
    # A raise would abort the agent; a descriptive string lets the model read
    # the failure and pick a different tool — the behaviour measured in
    # data-flow.md §9e.  The tool's wrapped function is reached through the
    # FunctionTool's .func attribute (ADK 2.7.1 stores the wrapped callable
    # there); calling it directly keeps this test offline and unit-scoped.
    tool = make_fetch_page_tool(conn)
    fetch_page = tool.func
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    with patch("app.tools.fetch_sources.fetch_sources", return_value=fake_sources):
        result = fetch_page("acme.test", ctx)  # must not raise
    # The failure must come back as data, in the exact "FETCH FAILED:" shape
    # the instruction teaches the model to recognize as the fallback trigger.
    assert result == expected
    # Load-bearing governance assertion: the failed fetch must NOT have
    # transitioned the target.  The old normalize_sources zero-text path
    # would have moved it to failed here — which would end the run before
    # the agent could fall back to google_search/url_context.  The tool
    # reports; the bookkeeping node owns the transition.
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "new"


def test_fetch_page_gets_a_fresh_step_id_per_call(conn):
    """A6 regression, re-armed for the loop: the agent may call fetch_page
    more than once, and each call's inner normalize_sources writes a steps
    row keyed by the step_id the tool handed it.  steps.step_id is the
    PRIMARY KEY — a reused id makes the SECOND call raise sqlite3
    IntegrityError: UNIQUE constraint failed: steps.step_id, the exact bug
    A6 fixed in FetchAndNormalizeNode one loop iteration away from
    returning here."""
    tool = make_fetch_page_tool(conn)
    fetch_page = tool.func
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    fake_sources = [NormalizedSource("company_website", "https://acme.test", "Acme does logistics.", "t", 0.8, 1, "static")]
    with patch("app.tools.fetch_sources.fetch_sources", return_value=fake_sources):
        first = fetch_page("acme.test", ctx)  # call 1 — must not collide with call 2
        second = fetch_page("acme.test", ctx)  # call 2 — the pre-fix code would crash HERE
    # Happy path: the normalized text comes back to the model as the response.
    assert first == "Acme does logistics."
    assert second == "Acme does logistics."
    # Two calls = two "normalize_sources" rows (the mocked fetch writes none),
    # with DISTINCT step_ids — the whole point of generating one per call.
    rows = conn.execute(
        "SELECT step_id FROM steps WHERE target_id='tgt_1' AND tool_name='normalize_sources';"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["step_id"] != rows[1]["step_id"]


# ── build_research_agent: construction contract ──────────────────────────────


def test_build_research_agent_resolves_model_through_resolve_adk_model(conn, adk_env):
    # The model string must come from the one shared resolution path — never
    # a hardcoded literal in this module.  Patching resolve_adk_model proves
    # both that it is consulted (called once, with the role alias) and that
    # its return value — not any embedded string — is what the agent carries.
    fake_model = "gemini-fake-pinned-model"
    with patch("app.agents.research.resolve_adk_model", return_value=fake_model) as resolve:
        agent = build_research_agent(conn)
    resolve.assert_called_once_with("research_model")
    assert agent.model == fake_model
    # The stable audit/trace name and the output key the bookkeeping node
    # reads — both are part of the pipeline contract, so pin them here.
    assert agent.name == "research"
    assert agent.output_key == "extracted_text"
    # All three tools coexist on the one agent (measured fact, data-flow.md
    # §9e — the plan's AgentTool isolation workaround is retired).
    assert {t.name for t in agent.tools} == {"fetch_page", "google_search", "url_context"}
    # B1e wiring: the after-model observer must be attached — without it
    # the server-side grounding path (google_search/url_context) stays
    # invisible to the audit trail, the exact defect this ticket exists to
    # close.
    assert callable(agent.after_model_callback)


def test_research_agent_carries_the_output_token_budget(conn, adk_env):
    # B1d Part 1 regression: the research LlmAgent must carry an explicit
    # output-token budget via generate_content_config.  Without one, ADK runs
    # on its own default limit and Gemini's thinking tokens can consume the
    # entire budget on a large input (docs/data-flow.md §9a, finding 2 —
    # measured: 979 of 1024 tokens spent thinking, finish_reason=MAX_TOKENS),
    # which is the Mark Boyden Associates failure shape: 14,828 chars fetched
    # successfully and the agent's response came back empty.  The value is
    # asserted by reading it off the CONSTRUCTED agent, not by importing the
    # module constant, so this test pins the wiring rather than the name.
    with patch("app.agents.research.resolve_adk_model", return_value="gemini-fake-pinned-model"):
        agent = build_research_agent(conn)
    # The config ADK copies into every request (verified: LlmAgent's field
    # validator accepts max_output_tokens; flows/llm_flows/basic.py::
    # _build_basic_request copies this config into llm_request.config, which
    # google_llm.py hands to genai's generate_content verbatim).
    config = agent.generate_content_config
    assert config is not None  # a missing config means ADK's default limit silently applies
    assert config.max_output_tokens == 16384  # the B1d budget: 2x app/llm.py's 8192 — a multi-source dossier, not a single CompanyProfile


def test_research_agent_tool_budget_is_attached_and_bounded_at_eight(conn, adk_env):
    # LlmAgent has NO max_iterations (only LoopAgent does — measured, B1a),
    # so the budget callback is the only thing standing between the agent and
    # an unbounded tool loop.  Drive the built agent's before_tool_callback
    # directly: the first 8 attempts must be allowed (None) and the 9th must
    # be blocked with a dict — ADK 2.7.1's "skip this tool" signal — naming
    # the limit of 8.
    with patch("app.agents.research.resolve_adk_model", return_value="gemini-fake-pinned-model"):
        agent = build_research_agent(conn)
    callback = agent.before_tool_callback
    assert callback is not None  # the budget callback must be attached to the agent
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    fake_tool = SimpleNamespace(name="fetch_page")
    for _ in range(8):
        assert callback(tool=fake_tool, args={}, tool_context=ctx) is None  # attempts 1-8: allowed
    blocked = callback(tool=fake_tool, args={}, tool_context=ctx)  # attempt 9: budget spent
    assert isinstance(blocked, dict)
    assert "8" in blocked["result"]
    # Golden Rule "never skip logging": every attempt — allowed or blocked —
    # must have written a steps row, and each row must have its own PK.
    rows = conn.execute("SELECT step_id FROM steps WHERE tool_name='research.fetch_page';").fetchall()
    assert len(rows) == 9
    assert len({r["step_id"] for r in rows}) == 9


# ── after_model_callback: the server-side grounding audit trail (B1e) ────────


def _make_fake_llm_response(*, with_grounding: bool = True, with_usage: bool = True) -> LlmResponse:
    """Build a REAL LlmResponse (the Pydantic model ADK 2.7.1 yields) with
    the payloads the research agent produces.

    Using the real types — not SimpleNamespace stand-ins — pins the
    callback's getattr field names to the installed SDK: if the SDK ever
    renames a field and the callback keeps reading the old name, these
    tests fail instead of silently passing against a fake that shares the
    bug.  All three models are pure offline Pydantic objects; nothing here
    touches a client, so the B1c hermeticity guard is not involved.
    """
    grounding = None
    if with_grounding:
        # Real GroundingMetadata (field names read off the installed
        # google.genai types): web_search_queries = the queries the turn
        # issued to Google Search; grounding_chunks[].web.uri = the web
        # pages grounding actually retrieved.
        grounding = types.GroundingMetadata(
            web_search_queries=["acme.test staff", "acme.test booking"],
            grounding_chunks=[
                types.GroundingChunk(web=types.GroundingChunkWeb(uri="https://acme.test/about", title="About")),
                types.GroundingChunk(web=types.GroundingChunkWeb(uri="https://acme.test/jobs", title="Jobs")),
            ],
        )
    usage = None
    if with_usage:
        # Real GenerateContentResponseUsageMetadata: thoughts_token_count
        # is Gemini's THINKING spend (billed against max_output_tokens),
        # candidates_token_count the OUTPUT spend.
        usage = types.GenerateContentResponseUsageMetadata(
            thoughts_token_count=1200,
            candidates_token_count=800,
        )
    # LlmResponse constructor takes the snake_case field names
    # (populate_by_name=True in the installed model) — the same names the
    # callback reads.
    return LlmResponse(
        grounding_metadata=grounding,
        finish_reason=types.FinishReason.STOP,
        usage_metadata=usage,
    )


def test_after_model_callback_logs_one_row_with_queries_urls_and_finish_reason(conn):
    # The production failure this prevents (ticket B1e, measured on the
    # real run 2026-08-21): Inner Compass Psychotherapy's static fetch
    # failed with 403 and the agent still produced 6,945 characters of
    # research — from google_search and url_context, which run server-side
    # and never pass through before_tool_callback, so NONE of that
    # activity reached the audit trail (exactly one tool row logged
    # against a budget of 8).  This row is the record: the queries the
    # turn issued, the URLs grounding actually retrieved, and how the turn
    # ended.
    callback = _make_research_after_model_callback(conn)
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    response = _make_fake_llm_response()
    # Keyword call — ADK 2.7.1 invokes after_model_callbacks by keyword
    # (callback_context=..., llm_response=...), so this pins the names the
    # framework passes.
    callback(callback_context=ctx, llm_response=response)
    rows = conn.execute(
        "SELECT output_json, status, target_id, run_id FROM steps WHERE tool_name='research.model_turn';"
    ).fetchall()
    assert len(rows) == 1  # one row per model turn — never skipped
    out = json.loads(rows[0]["output_json"])
    assert out["search_queries"] == ["acme.test staff", "acme.test booking"]  # the queries the turn issued
    assert out["retrieved_urls"] == ["https://acme.test/about", "https://acme.test/jobs"]  # the pages actually retrieved
    assert out["grounding_present"] is True
    assert out["finish_reason"] == "STOP"  # how the turn ended, as the wire string
    assert rows[0]["status"] == "success"  # the turn was fully observed
    assert rows[0]["target_id"] == "tgt_1" and rows[0]["run_id"] == "r1"  # honest attribution, same source as before_tool


def test_after_model_callback_plain_answer_turn_logs_grounding_absent_explicitly(conn):
    # A plain answer turn (no tool use, no search) carries NO
    # grounding_metadata and no usage_metadata at all.  The row must still
    # be written — the turn happened and "never skip logs" is absolute —
    # and must record the absence EXPLICITLY: grounding_present=False plus
    # status="success" means "the agent grounded nothing", which a trace
    # reader must be able to tell apart from "our logging failed"
    # (status="failed" + extraction_error, test 6).  A silently empty
    # field would make the two indistinguishable — the exact ambiguity
    # this ticket exists to kill.
    callback = _make_research_after_model_callback(conn)
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    response = _make_fake_llm_response(with_grounding=False, with_usage=False)
    callback(callback_context=ctx, llm_response=response)  # must not raise
    rows = conn.execute(
        "SELECT output_json, status FROM steps WHERE tool_name='research.model_turn';"
    ).fetchall()
    assert len(rows) == 1  # absence is recorded, not skipped — exactly one row either way
    out = json.loads(rows[0]["output_json"])
    assert out["grounding_present"] is False  # the explicit discriminator...
    assert out["usage_present"] is False
    assert out["search_queries"] is None and out["retrieved_urls"] is None
    assert out["finish_reason"] == "STOP"  # ...alongside the fields that ARE present
    assert "extraction_error" not in out  # absent metadata is a fact about the turn, not an error
    assert rows[0]["status"] == "success"  # logging itself succeeded


def test_after_model_callback_logs_thinking_and_output_token_counts(conn):
    # The self-diagnostic for the intermittent
    # research_agent_no_output_phase1 failure (Momentum Counselling on the
    # real run: 18,348 chars fetched, 0 out): Gemini bills THINKING tokens
    # against max_output_tokens, so a turn can spend the whole budget
    # thinking and leave no output.  B1d could only guess at a budget
    # because no trace carried these counts; these fields make the next
    # occurrence self-diagnosing — a no-output turn with thoughts near the
    # budget and output 0 is budget exhaustion, not an empty research
    # verdict.
    callback = _make_research_after_model_callback(conn)
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    callback(callback_context=ctx, llm_response=_make_fake_llm_response())
    row = conn.execute(
        "SELECT output_json FROM steps WHERE tool_name='research.model_turn';"
    ).fetchone()
    out = json.loads(row["output_json"])
    assert out["thoughts_token_count"] == 1200  # thinking spend — the budget eater
    assert out["output_token_count"] == 800  # actual output spend
    assert out["usage_present"] is True


def test_after_model_callback_two_turns_write_two_rows_with_distinct_step_ids(conn):
    # A6 regression, re-armed for the MODEL loop: the callback fires once
    # per model turn and one target takes several turns (search, then
    # fetch context, then answer).  steps.step_id is the PRIMARY KEY — a
    # reused id makes the second turn's INSERT raise IntegrityError:
    # UNIQUE constraint failed: steps.step_id, the exact bug A6 fixed in
    # the fetch node, one loop iteration away from returning here.
    callback = _make_research_after_model_callback(conn)
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    callback(callback_context=ctx, llm_response=_make_fake_llm_response())  # turn 1
    callback(callback_context=ctx, llm_response=_make_fake_llm_response())  # turn 2 — pre-fix code would crash HERE
    rows = conn.execute(
        "SELECT step_id FROM steps WHERE tool_name='research.model_turn';"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["step_id"] != rows[1]["step_id"]  # distinct PKs per turn


def test_after_model_callback_returns_none_and_never_mutates_the_response(conn):
    # Observer, not interceptor: ADK 2.7.1 treats a TRUTHY callback return
    # as "replace the model's response with this" (its call site does
    # ``llm_response = altered``), so returning the response itself — or
    # any other truthy value — would silently corrupt the research the
    # model produced.  None is the only acceptable return, and the
    # response object must pass through byte-identical (model_dump before
    # vs after): a callback that mutates what it audits defeats the entire
    # audit trail it writes.
    callback = _make_research_after_model_callback(conn)
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    response = _make_fake_llm_response()
    before = response.model_dump()
    returned = callback(callback_context=ctx, llm_response=response)
    assert returned is None  # the no-change sentinel — the ONLY acceptable return
    assert response.model_dump() == before  # pure observation: the model's response is untouched


class _ExplodingChunk:
    """A chunk-shaped stand-in whose EVERY attribute lookup raises — the
    worst payload the SDK could hand the callback.  getattr(chunk, "web",
    None) therefore raises instead of returning the default."""

    def __getattr__(self, name):
        raise RuntimeError(f"malformed grounding chunk: {name}")


def test_after_model_callback_malformed_metadata_does_not_raise(conn):
    # Requirement 5's malformed half: a response whose metadata cannot be
    # interpreted must not take the model loop down with it — the turn's
    # research is already produced, the run must continue — but the
    # failure must be VISIBLE in the trace: status="failed" plus an
    # extraction_error field, so a scanner can tell "our observer failed"
    # apart from "the agent grounded nothing" (status="success" with
    # grounding_present=False, test 2).  model_construct bypasses
    # pydantic validation so the malformed chunk can be injected into a
    # real LlmResponse — the SDK itself would never build one like this,
    # but an audit observer must not assume it cannot happen.
    callback = _make_research_after_model_callback(conn)
    ctx = _FakeToolContext(state={"target_id": "tgt_1", "run_id": "r1"})
    grounding = types.GroundingMetadata.model_construct(
        web_search_queries=["acme.test"],
        grounding_chunks=[_ExplodingChunk()],
    )
    response = LlmResponse.model_construct(grounding_metadata=grounding)
    callback(callback_context=ctx, llm_response=response)  # must not raise
    row = conn.execute(
        "SELECT output_json, status FROM steps WHERE tool_name='research.model_turn';"
    ).fetchone()
    assert row["status"] == "failed"  # the observer's failure is surfaced
    out = json.loads(row["output_json"])
    assert out["grounding_present"] is True  # salvaged: grounding WAS present
    assert "extraction_error" in out  # and the row says why the extraction failed


# ── ResearchBookkeepingNode: the deterministic governance node ───────────────


def test_bookkeeping_good_text_transitions_to_researched_and_logs_char_count(conn):
    # The happy path: the agent published usable findings, so bookkeeping
    # must make the SAME transition the old FetchAndNormalizeNode made —
    # new→researched, reason "research_complete_no_enrichment" — and log a
    # step row carrying the character count of what came back.
    node = ResearchBookkeepingNode(name="research_bookkeeping", conn=conn)
    text = "Intake is manual paper forms; 3 locations; hiring 2 practitioners."
    ctx = SimpleNamespace(
        invocation_id="inv-1",
        session=SimpleNamespace(state={"target_id": "tgt_1", "run_id": "r1", "extracted_text": text}),
    )
    events = asyncio.run(_collect_events(node, ctx))
    # No state_delta on the happy path: extracted_text is already in session
    # state (the agent's output_key wrote it) — the downstream nodes read it
    # unchanged, exactly as before B1b.
    assert events == []
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "researched"
    trn = conn.execute(
        "SELECT reason, new_state FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert trn["reason"] == "research_complete_no_enrichment"  # the exact reason the old node used — unchanged vocabulary
    assert trn["new_state"] == "researched"
    step = conn.execute(
        "SELECT output_json, status FROM steps WHERE tool_name='research_bookkeeping';"
    ).fetchone()
    assert step["status"] == "success"
    # B2b: the output now also carries the persisted findings row's id (the
    # trace link to the stored evidence), alongside the char count.
    out = json.loads(step["output_json"])
    assert out["chars"] == len(text)  # the operator-facing char count
    assert out["findings_source_id"].startswith("src")  # the B2b trace link to the sources row


def test_bookkeeping_persists_findings_as_source_row(conn):
    # B2b: the research agent's findings must be persisted to the sources
    # table (source_type='research_findings') so the `findings` tier is
    # checkable AFTER the run and a retroactive fact-checker has the text
    # the signal agent actually read.  Asserted against the DATABASE, not
    # just the return events — the persistence is the ticket's deliverable.
    node = ResearchBookkeepingNode(name="research_bookkeeping", conn=conn)
    text = "Intake is manual paper forms; 3 locations; hiring 2 practitioners."
    ctx = SimpleNamespace(
        invocation_id="inv-1",
        session=SimpleNamespace(state={"target_id": "tgt_1", "run_id": "r1", "extracted_text": text}),
    )
    asyncio.run(_collect_events(node, ctx))
    rows = conn.execute(
        "SELECT * FROM sources WHERE target_id='tgt_1' AND run_id='r1';"
    ).fetchall()
    assert len(rows) == 1  # exactly one evidence row for the findings
    src = rows[0]
    assert src["source_type"] == "research_findings"  # marked as agent prose, never a raw page
    assert src["extracted_text"] == text  # the verbatim findings downstream nodes will read
    assert src["source_url"] is None  # no single URL — the agent consolidated many (possibly server-side) sources
    assert src["source_confidence"] is None  # agent prose has no measured confidence
    assert src["source_priority"] is None  # findings are not a normalization input
    assert src["extraction_method"] == "agent"  # provenance marker
    # The write went through the gate — the write_log row proves the single
    # write path, never a raw INSERT, and the audit row must be attributed
    # to the SAME step as the bookkeeping step row (the trace link the
    # output_data's findings_source_id points at).
    write = conn.execute(
        "SELECT * FROM write_log WHERE action='insert_source' AND record_id=?;",
        (src["source_id"],),
    ).fetchone()
    assert write is not None
    step = conn.execute(
        "SELECT step_id FROM steps WHERE tool_name='research_bookkeeping';"
    ).fetchone()
    assert write["step_id"] == step["step_id"]


def test_bookkeeping_sentinel_transitions_to_failed_no_sources_available(conn):
    # Case 1 of the B1d discrimination (the regression guard): the sentinel
    # is the agent's HONEST "company not findable" verdict, and its route
    # must stay exactly as it was — failed with reason "no_sources_available"
    # (state-machine.md §7c).  If the B1d discrimination ever collapses this
    # case back into the agent-silence branch or renames its reason, this
    # test fails.
    node = ResearchBookkeepingNode(name="research_bookkeeping", conn=conn)
    ctx = SimpleNamespace(
        invocation_id="inv-1",
        session=SimpleNamespace(
            state={"target_id": "tgt_1", "run_id": "r1", "extracted_text": NO_RESEARCH_FINDINGS_SENTINEL},
        ),
    )
    events = asyncio.run(_collect_events(node, ctx))
    # Exactly ONE event, and it is the failure delta — this is the node-level
    # half of the "no downstream node runs" guarantee (the pipeline-level
    # half is asserted in test_agents_phase1.py, where the summarize LLM
    # call must never fire after a bookkeeping failure).
    assert len(events) == 1
    assert events[0].actions.state_delta == {"final_state": "failed"}
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    trn = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert trn["reason"] == "no_sources_available"  # the §7c reason — the sentinel case must not move (ticket B1d Part 2.1)
    # The step row names the discriminator so the trace shows this was an
    # honest verdict, not a silent agent.
    step = conn.execute(
        "SELECT output_json, status FROM steps WHERE tool_name='research_bookkeeping';"
    ).fetchone()
    assert step["status"] == "failed"  # the failure-path step row must exist either way (never skip logging)
    assert json.loads(step["output_json"]) == {
        "outcome": "sentinel",
        "chars": len(NO_RESEARCH_FINDINGS_SENTINEL),
        "chars_fetched": 0,
    }


@pytest.mark.parametrize(
    # Both whitespace shapes are the same diagnosis: the agent produced no
    # output at all — neither a verdict nor usable text.
    "state_text",
    ["", "   "],
)
def test_bookkeeping_empty_output_transitions_to_failed_research_agent_no_output(conn, state_text):
    # Case 3 of the B1d discrimination — the production failure this prevents
    # is Mark Boyden Associates' contradictory trace: fetch_company_page
    # succeeded with chars_extracted 14828 and normalize_sources with chars
    # 14828, then the old code recorded "no_sources_available" — false,
    # because the sources WERE available; the agent's output is what failed
    # to materialize.  A whitespace-only response is exactly that shape: not
    # the sentinel, not usable text.  Seed the same successful fetch row,
    # then assert the transition names the agent's silence
    # (research_agent_no_output_phase1) and the step row shows the
    # contradiction in ONE row: 14,828 fetched / 0 out.
    fetch_step_id = new_id("step")  # the fetch row's PK, mirroring what the real fetch_page tool writes
    log_step(
        conn, run_id="r1", step_id=fetch_step_id, target_id="tgt_1",
        tool_name="fetch_company_page", agent_id="system",
        input_data={"domain": "acme.test"},
        output_data={"chars_extracted": 14828},  # Mark Boyden's measured upstream success
        status="success",
    )
    node = ResearchBookkeepingNode(name="research_bookkeeping", conn=conn)
    ctx = SimpleNamespace(
        invocation_id="inv-1",
        session=SimpleNamespace(
            state={"target_id": "tgt_1", "run_id": "r1", "extracted_text": state_text},
        ),
    )
    events = asyncio.run(_collect_events(node, ctx))
    assert len(events) == 1  # the failure delta — the "no downstream node runs" guarantee
    assert events[0].actions.state_delta == {"final_state": "failed"}
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    trn = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert trn["reason"] == "research_agent_no_output_phase1"  # B1d's new reason — causes must be distinguishable from reason alone
    # The step row must make the contradiction visible in one row: the
    # discriminator ("no_output"), 0 chars out, and the upstream fetch count
    # read back from steps — 14,828 fetched, 0 out.
    step = conn.execute(
        "SELECT output_json, status FROM steps WHERE tool_name='research_bookkeeping';"
    ).fetchone()
    assert step["status"] == "failed"  # never skip logging
    # chars is the RAW count of what the agent published (a whitespace-only
    # response still counts its spaces) — the "outcome" discriminator is what
    # classifies it as no output.  The empty-string case pins the ticket's
    # exact example: 14,828 fetched / 0 out.
    assert json.loads(step["output_json"]) == {
        "outcome": "no_output",
        "chars": len(state_text),
        "chars_fetched": 14828,
    }


def test_bookkeeping_missing_key_transitions_to_failed_research_agent_no_output(conn):
    # Case 4 of the B1d discrimination: the key is entirely absent (a failed
    # or empty agent turn leaves extracted_text out of session state).  This
    # is the SAME diagnosis as the whitespace case — the agent produced no
    # output — so it must share reason research_agent_no_output_phase1, NOT
    # the sentinel's no_sources_available.
    node = ResearchBookkeepingNode(name="research_bookkeeping", conn=conn)
    ctx = SimpleNamespace(
        invocation_id="inv-1",
        session=SimpleNamespace(state={"target_id": "tgt_1", "run_id": "r1"}),  # no extracted_text key at all
    )
    events = asyncio.run(_collect_events(node, ctx))
    assert len(events) == 1  # the failure delta, same as every failure shape
    assert events[0].actions.state_delta == {"final_state": "failed"}
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    trn = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert trn["reason"] == "research_agent_no_output_phase1"  # key absence is agent silence — same reason as the whitespace case
    step = conn.execute(
        "SELECT output_json FROM steps WHERE tool_name='research_bookkeeping';"
    ).fetchone()
    assert json.loads(step["output_json"])["outcome"] == "no_output"  # the discriminator, not the sentinel
