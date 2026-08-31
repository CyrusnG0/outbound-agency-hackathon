# tests/test_taskmaster.py — C4: the TaskmasterAgent's zero-trust boundary.
#
# What this file exists to prove (each in its own test, per the ticket):
#   C4-Z1 — no tool of the Taskmaster can approve anything: the toolset has
#           no approval capability (AST + tool-introspection), and a run
#           that reaches awaiting_review leaves targets there with zero
#           review_decisions rows.
#   C4-Z3 — no tool can write the kill switch: AST/introspection, plus a
#           behavioural run against an ENGAGED switch halts at entry, logs,
#           and leaves the switch file byte-identical.
#   C4-Z4 — a batch over MAX_BATCH_SIZE is refused BY THE TOOL, with no
#           core-table rows written — deterministically, no matter how
#           nicely the model asks (the check is code, not negotiation).
#   C4-Z2 — the two new modules import no mail transport (a focused AST
#           check; tests/test_send_gate.py's whole-package walk also covers
#           them, unmodified).
# Plus: the bounded tool budget is wired, report_pipeline_status is
# read-only (zero write_log rows), each stage tool dispatches the EXISTING
# runner (patched, not reimplemented), every tool call lands in `steps`
# under the taskmaster principal, and the registry seeds the taskmaster
# row — enabled, aliased, and with the deliberately EMPTY allowed_actions.
#
# Every test keeps the suite offline: agents are constructed (construction
# never builds a genai client — tests/conftest.py documents the measured
# fact), model boundaries are patched, and the autouse live-client guard
# stays intact.  GEMINI_FLASH_MODEL is pinned in the agent fixture so
# resolve_adk_model can construct (the same pattern test_kill_switch.py's
# draft-halt tests use).

import ast  # the structural Z1/Z2/Z3 checks parse the new modules' source
import asyncio  # asyncio.run drives the minimal ADK invocations in the Z3 halt test
import json  # reading steps output_json and the switch file
from pathlib import Path  # locating the new modules' sources for the AST checks
from types import SimpleNamespace  # the fake ToolContext the budget-callback test drives
from unittest.mock import patch  # patching the stage runners to prove dispatch-not-reimplement

import pytest  # fixtures, tmp_path, monkeypatch

from app.agents.taskmaster import (  # the module under test
    TASKMASTER_AGENT_ID,
    _TASKMASTER_MAX_TOOL_CALLS,
    build_taskmaster_agent,
)
from app.agents_registry import seed_agent_registry  # the registry the write gate checks
from app.db import apply_schema, connect  # fresh per-test SQLite database
from app.ids import new_id  # fresh ids for seeded rows
from app.phase1_cli import MAX_BATCH_SIZE  # the cap Z4 enforces — the test must fail if the cap itself drifts
from app.schemas import CompanyProfile, DraftCritique, EmailDraft, Signal  # valid offline stand-in payloads for the draft/research stages
from app.tools.send_email import SendEmailResult  # the send runner's outcome shape the dispatch test fakes
from app.write_gate import commit  # every seeded core-table row goes through the gate, never a raw INSERT
from google.adk.agents import BaseAgent  # base class of the offline draft stand-ins (B1b pattern)
from google.adk.events import Event, EventActions  # how the stand-ins publish their output dicts
from google.adk.runners import Runner  # executes the agent in the Z3 halt test
from google.adk.sessions import InMemorySessionService  # in-memory session store (the same one the runners use)
from google.genai import types  # the synthetic "run" user message ADK's Runner requires


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conn(scratch_db_target):
    """Fresh in-tmpdir SQLite database with the full schema applied.

    Deliberately NOT seeded — tests that need a populated registry call the
    `seeded` fixture (the same split test_agent_registry.py uses)."""
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    yield c
    c.close()


@pytest.fixture
def seeded(conn):
    """The conn fixture plus the seeded principals (incl. taskmaster)."""
    seed_agent_registry(conn, run_id="seed_run", step_id="seed_step")
    return conn


@pytest.fixture
def disengaged_switch(tmp_path):
    """A tmp kill-switch file in the DISENGAGED state — the normal state of
    a normal run.  The agent fixture pins this path so the tests never read
    the repo's real config/kill_switch.json."""
    path = tmp_path / "switch_off.json"
    path.write_text(json.dumps({"enabled": False, "updated_by": "test", "updated_at": "t"}))
    return path


@pytest.fixture
def offers_dir(tmp_path):
    """A tmp offers directory with one offer yaml — the same shape
    test_draft_agent's fixture writes, because the draft brief and the
    deterministic footer read their offer context from here."""
    d = tmp_path / "offers"
    d.mkdir()
    (d / "acme.yaml").write_text(
        "pitch: We cut intake admin time in half.\n"
        "persona_hint: Operations lead at a mid-size practice.\n"
        "from_address: outreach@acme.test\n"
        "icp:\n  geography: HK\n  disqualifiers:\n    - outside HK\n"
    )
    return d


@pytest.fixture
def taskmaster_agent(seeded, offers_dir, disengaged_switch, monkeypatch, tmp_path):
    """The REAL Taskmaster agent, built offline.

    Construction never builds a genai client (the measured conftest fact),
    so this fixture needs no model patch — only the env pin
    resolve_adk_model reads, exactly like test_kill_switch.py's draft-halt
    tests.  The switch path and the outbox/inbox dirs are pinned to tmp
    values so nothing reads or writes the repo's real files."""
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")  # the model pin resolve_adk_model refuses to boot without
    agent = build_taskmaster_agent(
        seeded,
        run_id="run_1",  # the run every dispatched write and step row of these tests shares
        offers_dir=str(offers_dir),
        outbox_dir=str(tmp_path / "outbox"),
        inbox_dir=str(tmp_path / "inbox"),
        kill_switch_path=str(disengaged_switch),  # the pinned DISENGAGED switch
    )
    return agent


def _tool_by_name(agent, name: str):
    """Fetch one of the agent's FunctionTools by its model-visible name —
    tests call the underlying function directly (tool.func), which is the
    deterministic half of every boundary this file checks."""
    for tool in agent.tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"tool {name!r} not in the taskmaster toolset")


def _insert_policy_decision(c, target_id: str, decision: str) -> None:
    """Insert one policy_decisions row through the write gate — the same
    path policy_check_phase1 uses (core table, gated write), copied from
    test_draft_agent.py's fixture helper."""
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


def _seed_target(c, *, target_id: str, state: str, policy: str | None = None) -> None:
    """Seed one offer/account/target triple at the given state (plus an
    optional policy decision) — the minimal shape the stage tools' eligible
    SELECTs and the draft runner's preconditions read.  All rows go through
    the write gate, never a raw INSERT."""
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
    if policy is not None:
        _insert_policy_decision(c, target_id, policy)


# ── The C4-Z1 structural boundary: no approval capability exists ─────────────

# Identifiers that may NEVER appear as executable code in the taskmaster
# module — the write ACTION names of the approval path, the review writer,
# and the kill-switch writer.  Prose (docstrings) may discuss them; code
# may not (the docstring-skip helper below enforces exactly that split,
# modelled on test_send_gate.py's transport scanner).  Since C4b the same
# names may not appear as IMPORTED ALIASES either (check 2b): a bare
# from-import trips nothing in the Name/Attribute scan — an ImportFrom
# alias node is neither — so the alias names are asserted directly.
# Importing a forbidden symbol IS referencing it, used or not.
_FORBIDDEN_IDENTIFIERS = ("record_review_decision", "insert_review_decision", "write_kill_switch")

# The one module import the taskmaster must never make: app.review is the
# operator-approval write path (C4-Z1).  app.kill_switch IS imported — but
# only its READER (read_kill_switch), which the forbidden-identifier check
# above already splits from its WRITER.
_FORBIDDEN_IMPORT_ROOTS = ("app.review",)


def _taskmaster_module_files() -> list[Path]:
    """The two C4 modules' source files — the enforcement surface for the
    structural checks (a forbidden capability hidden in the CLI is still a
    forbidden capability)."""
    app_dir = Path(__file__).resolve().parent.parent / "app"
    return [app_dir / "agents" / "taskmaster.py", app_dir / "taskmaster_cli.py"]


def _docstring_constant_ids(tree: ast.Module) -> set[int]:
    """Collect docstring string-Constant ids so the string-literal scan
    skips prose — only strings that could actually execute are checked
    (the same helper test_send_gate.py uses for its transport scan)."""
    containers: list = [tree]
    containers.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    ids: set[int] = set()
    for container in containers:
        body = getattr(container, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _collect_transition_to_states(tree: ast.Module) -> list[str]:
    """Every ``transition(...)`` call's literal to_state= value in the
    module — the C4-Z1 contract is that the ONLY state the Taskmaster's own
    code may ever transition a target to is "failed" (the B1f crash
    discipline); "approved" (or anything else) here is the boundary
    breaking.  Both call spellings are collected: the module-attribute form
    (state_machine_module.transition) and a direct name (were a future edit
    to switch to a from-import)."""
    states: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # func is either Name("transition") or Attribute(..., "transition").
        func = node.func
        is_transition_call = (
            (isinstance(func, ast.Name) and func.id == "transition")
            or (isinstance(func, ast.Attribute) and func.attr == "transition")
        )
        if not is_transition_call:
            continue
        for kw in node.keywords:
            if kw.arg == "to_state" and isinstance(kw.value, ast.Constant):
                states.append(kw.value.value)
    return states


def test_no_approval_or_killswitch_capability_in_toolset(taskmaster_agent):
    """C4-Z1 + C4-Z3, structural half: the registered toolset is EXACTLY
    the six stage/status tools — no approval tool, no review tool, no
    kill-switch tool — and no tool callable references the forbidden write
    identifiers.  The tool NAME list is the first thing a judge reads; the
    callable scan is what fails if someone adds the capability under a
    friendly name.

    resume_pending_research (2026-08-31) is the sixth: it researches
    targets already in the database at state "new" (never re-importing),
    the recovery path for a batch that timed out partway through. It
    carries no more capability than import_and_research/draft_for_scored
    already had — same write_gate, same state_machine, no approval and no
    switch write — so this list was updated deliberately, in the same
    change that added the tool, per the assertion message below."""
    names = sorted(t.name for t in taskmaster_agent.tools)
    assert names == [
        "draft_for_scored",
        "dry_run_send_approved",
        "fetch_and_classify_replies",
        "import_and_research",
        "report_pipeline_status",
        "resume_pending_research",
    ], "the toolset is the capability set — an added tool is an added capability, reviewed deliberately"
    for tool in taskmaster_agent.tools:
        # co_names: every name the tool function's own code resolves at
        # call time (closure variables and globals alike).  A reference to
        # a review/kill-switch writer would appear here.
        for forbidden in _FORBIDDEN_IDENTIFIERS:
            assert forbidden not in tool.func.__code__.co_names, (
                f"tool {tool.name!r} references {forbidden!r} — the Taskmaster "
                f"must be structurally incapable of it (C4-Z1/Z3)"
            )


def test_taskmaster_module_ast_has_no_approval_or_switch_write_path():
    """C4-Z1 + C4-Z3, module-level half: an AST walk over BOTH new modules
    (agent + CLI).  Three assertions, each a line a future 'helpful' edit
    must cross deliberately:
    1. no executable reference to the approval/switch-write identifiers
       (Name/Attribute nodes, and non-docstring string constants — the
       dynamic-import bypass);
    2. no import of app.review (the approval write path) — and, since
       C4b, no import of any forbidden SYMBOL either: a bare
       ``from app.kill_switch import write_kill_switch`` (imported, never
       called) is an ImportFrom alias — neither a Name nor an Attribute —
       so the alias names themselves are asserted directly (check 2b);
    3. every transition() call the Taskmaster's own code makes passes
       to_state="failed" literally — the ONLY state change it may make."""
    all_to_states: list[str] = []  # the union across files: the crash discipline must exist SOMEWHERE in the module pair
    for path in _taskmaster_module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        skip = _docstring_constant_ids(tree)
        for node in ast.walk(tree):
            # ── Check 1a: identifier references ─────────────────────────
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            if name is not None:
                assert name not in _FORBIDDEN_IDENTIFIERS, (
                    f"{path.name} references {name!r} — a capability the "
                    f"Taskmaster must not hold (C4-Z1/Z3)"
                )
            # ── Check 1b: executable string constants (the dynamic-import
            # bypass — importlib.import_module("app.review") never appears
            # as an Import node) ────────────────────────────────────────
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in skip
            ):
                for forbidden in _FORBIDDEN_IDENTIFIERS:
                    assert forbidden not in node.value, (
                        f"{path.name} carries an executable string naming "
                        f"{forbidden!r} — a capability the Taskmaster must "
                        f"not hold (C4-Z1/Z3)"
                    )
                for root in _FORBIDDEN_IMPORT_ROOTS:
                    assert root not in node.value, (
                        f"{path.name} carries an executable string naming "
                        f"{root!r} — the approval path is off-limits (C4-Z1)"
                    )
            # ── Check 2: static imports ────────────────────────────────
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module
            elif isinstance(node, ast.Import):
                imported = ",".join(alias.name for alias in node.names)
            else:
                imported = None
            if imported is not None:
                for root in _FORBIDDEN_IMPORT_ROOTS:
                    assert not (imported == root or imported.startswith(root + ".")), (
                        f"{path.name} imports {imported!r} — the approval "
                        f"path is off-limits (C4-Z1)"
                    )
            # ── Check 2b: imported SYMBOL names (the C4b gap) ──────────
            # A bare ``from app.kill_switch import write_kill_switch`` —
            # imported but never called — parses to an ImportFrom whose
            # alias node is neither a Name nor an Attribute, so checks
            # 1a/1b/2a above never see the symbol; C4's sabotage test
            # proved exactly that hole.  The alias NAME is therefore
            # asserted directly: importing a forbidden symbol IS
            # referencing it, and the Z1/Z3 boundary forbids the
            # reference itself, used or not.  The other spelling —
            # ``import app.kill_switch`` followed by attribute use — is
            # already caught by check 1a's Attribute half (the use
            # ``app.kill_switch.write_kill_switch`` is an Attribute node
            # whose attr is forbidden), so only the alias needs this new
            # check.  read_kill_switch stays legal: it is not in
            # _FORBIDDEN_IDENTIFIERS, so no alias check fires on it (Z3
            # is a WRITE-only prohibition — reading must stay possible).
            alias_names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                alias_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                alias_names = [alias.name for alias in node.names]
            for alias_name in alias_names:
                for forbidden in _FORBIDDEN_IDENTIFIERS:
                    assert alias_name != forbidden, (
                        f"{path.name} imports {alias_name!r} — a capability the "
                        f"Taskmaster must not hold, even un-called (C4-Z1/Z3)"
                    )
        # ── Check 3: the ONLY permitted transition is -> failed ─────────
        to_states = _collect_transition_to_states(tree)
        all_to_states.extend(to_states)
        for state in to_states:
            assert state == "failed", (
                f"{path.name} transitions a target to {state!r} — the Taskmaster "
                f"may only transition to 'failed' (C4-Z1: 'approved' is unreachable "
                f"by construction)"
            )
    # The crash discipline must exist somewhere in the pair (the agent
    # module owns it — the CLI delegates to the stage tools) — its absence
    # would mean a crashed target no longer goes to failed through the
    # state machine, i.e. the discipline itself was removed.
    assert all_to_states, (
        "neither taskmaster module calls transition() — the B1f crash "
        "discipline (a crashed target goes to failed through the state "
        "machine) appears to have been removed"
    )


def test_run_to_awaiting_review_stops_there_with_zero_review_rows(seeded, offers_dir, taskmaster_agent):
    """C4-Z1, behavioural half: drive the REAL draft stage (offline stand-in
    writer/critic, the test_draft_agent pattern) through the Taskmaster's
    own draft tool.  The target must land in awaiting_review and STAY there
    — zero review_decisions rows — and the tool's summary must say review
    is required, because that stop-and-report is the product."""
    _seed_target(seeded, target_id="tgt_1", state="scored", policy="allow")
    # ── Offline stand-ins for the draft loop's two LLM agents ─────────────
    # The real writer/critic are LlmAgents that would make live billable
    # calls; these publish fixed dicts under the same state keys
    # ("draft"/"critique") the real output_schema + output_key write.
    class _StubWriter(BaseAgent):
        def __init__(self):
            super().__init__(name="draft_writer")  # the real agent's stable name

        async def _run_async_impl(self, ctx):
            # A valid EmailDraft dict — the real persist node validates it.
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"draft": EmailDraft(
                    subject="A question about your intake admin",
                    body="Hello, I help practices cut intake admin time, and your booking volume stands out. Worth a short call?",
                    rationale="Matches the offer's ICP geography and the intake pain the pitch speaks to.",
                    confidence=0.8,
                ).model_dump()}),
            )

    class _StubCritic(BaseAgent):
        def __init__(self):
            super().__init__(name="draft_critic")  # the real agent's stable name

        async def _run_async_impl(self, ctx):
            # A clean pass — the loop exits after one revision.
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"critique": DraftCritique(
                    passed=True, issues=[], required_changes="", severity="none",
                ).model_dump()}),
            )

    tool = _tool_by_name(taskmaster_agent, "draft_for_scored")
    with patch("app.agents.draft._build_writer_agent", return_value=_StubWriter()), \
         patch("app.agents.draft._build_critic_agent", return_value=_StubCritic()):
        # tool.func is async now (2026-08-29 fix — see phase1.py's
        # run_target_through_phase1_async docstring): asyncio.run() here is
        # legal because THIS call site has no event loop already running
        # (a plain sync test function) — unlike production, where ADK's own
        # Runner is the already-running loop this tool must be awaited into.
        summary = asyncio.run(tool.func(limit=5))  # the tool's own call — the deterministic half under test
    # ── The gate held ─────────────────────────────────────────────────────
    row = seeded.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "awaiting_review", "the draft stage must move the target to awaiting_review"
    assert seeded.execute("SELECT COUNT(*) AS n FROM review_decisions;").fetchone()["n"] == 0, (
        "ZERO review_decisions rows: nothing the Taskmaster ran may record an "
        "approval decision — the human gate is the only writer"
    )
    # ── The report says review is required (the honest-report product) ────
    assert "awaiting_review" in summary and "REVIEW" in summary.upper(), (
        f"the summary must tell the operator review is required; got: {summary!r}"
    )
    # ── The tool call is audited under the taskmaster principal ───────────
    step = seeded.execute(
        "SELECT agent_id, status FROM steps WHERE tool_name='taskmaster.draft_for_scored';"
    ).fetchone()
    assert step is not None and step["agent_id"] == TASKMASTER_AGENT_ID and step["status"] == "success"


# ── resume_pending_research: the stuck-batch recovery tool (2026-08-31) ──────

def test_resume_pending_research_moves_a_stuck_new_target_off_new_without_importing(seeded, offers_dir):
    """The behavioural core of the recovery tool: a target already sitting
    at state 'new' (simulating a prior import_and_research call that timed
    out before researching it) is driven to a real terminal Phase 1 state
    — WITHOUT any CSV path, WITHOUT calling import_csv, and WITHOUT
    creating a second accounts/targets row. This is the exact scenario a
    real run hit live: re-importing would have collided on the domain
    that already exists; this tool must never even attempt that INSERT."""
    _seed_target(seeded, target_id="tgt_1", state="new")
    accounts_before = seeded.execute("SELECT COUNT(*) AS n FROM accounts;").fetchone()["n"]
    targets_before = seeded.execute("SELECT COUNT(*) AS n FROM targets;").fetchone()["n"]

    # Offline stand-ins for the research stage's LLM boundaries — the SAME
    # recipe tests/test_agents_phase1.py's full-pipeline test uses, since
    # this tool calls the identical run_target_through_phase1_async.
    class _StubResearchAgent(BaseAgent):
        def __init__(self):
            super().__init__(name="research")

        async def _run_async_impl(self, ctx):
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"extracted_text": "Acme does logistics. Hiring ops manager."}),
            )

    fake_profile = CompanyProfile(one_line_summary="Acme does logistics", industry="Logistics", confidence=0.8)
    fake_signals = [Signal(
        signal_type="hiring_relevant_role", signal_value="Hiring ops manager",
        signal_strength=0.8, evidence_quote="hiring an operations manager for the team",
    )]

    agent = _tool_by_name(build_taskmaster_agent(
        seeded, run_id="run_1", offers_dir=str(offers_dir),
        outbox_dir="unused", inbox_dir="unused",
    ), "resume_pending_research")
    with patch("app.agents.phase1.build_research_agent", return_value=_StubResearchAgent()), \
         patch("app.tools.summarize_company.call_structured", return_value=fake_profile), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
        # tool.func is async — see draft_for_scored's test above for why
        # asyncio.run() here (no event loop already running) is legal.
        summary = asyncio.run(agent.func(limit=5))

    row = seeded.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] in ("scored", "watchlist", "not_target"), (
        f"the target must reach a real terminal Phase 1 state, not stay 'new'; got {row['state']!r}"
    )
    assert row["state"] in summary, f"the summary must name the state reached; got: {summary!r}"
    # ── The no-reimport proof: exactly zero new accounts/targets rows ──────
    assert seeded.execute("SELECT COUNT(*) AS n FROM accounts;").fetchone()["n"] == accounts_before, (
        "resume_pending_research must never create a new accounts row — it processes an EXISTING target"
    )
    assert seeded.execute("SELECT COUNT(*) AS n FROM targets;").fetchone()["n"] == targets_before, (
        "resume_pending_research must never create a new targets row — no import happens here"
    )
    # ── Audited under the taskmaster principal, same convention as every
    # other tool's outcome row ─────────────────────────────────────────────
    step = seeded.execute(
        "SELECT agent_id, status FROM steps WHERE tool_name='taskmaster.resume_pending_research';"
    ).fetchone()
    assert step is not None and step["agent_id"] == TASKMASTER_AGENT_ID and step["status"] == "success"


def test_resume_pending_research_with_nothing_stuck_reports_so_and_writes_nothing(seeded, offers_dir):
    """The empty-set path: no target at 'new' means nothing to resume — the
    tool must say so plainly (never invent work) and touch no core table."""
    write_log_before = seeded.execute("SELECT COUNT(*) AS n FROM write_log;").fetchone()["n"]
    agent = _tool_by_name(build_taskmaster_agent(
        seeded, run_id="run_1", offers_dir=str(offers_dir),
        outbox_dir="unused", inbox_dir="unused",
    ), "resume_pending_research")
    summary = asyncio.run(agent.func(limit=5))
    assert "no targets stuck" in summary
    assert seeded.execute("SELECT COUNT(*) AS n FROM write_log;").fetchone()["n"] == write_log_before


def test_resume_pending_research_over_cap_refused_with_no_rows_written(seeded, offers_dir):
    """C4-Z4 applies here too: the same deterministic cap every stage tool
    enforces, checked before any DB read of the stuck set."""
    agent = _tool_by_name(build_taskmaster_agent(
        seeded, run_id="run_1", offers_dir=str(offers_dir),
        outbox_dir="unused", inbox_dir="unused",
    ), "resume_pending_research")
    summary = asyncio.run(agent.func(limit=MAX_BATCH_SIZE + 1))
    assert summary.startswith("refused:") and str(MAX_BATCH_SIZE) in summary


# ── C4-Z3 behavioural half: an engaged switch halts at entry ─────────────────

def test_engaged_switch_halts_invocation_and_leaves_file_untouched(seeded, offers_dir, monkeypatch, tmp_path):
    """C4-Z3: with the switch ENGAGED, a real Taskmaster invocation ends at
    agent entry — no tool runs, no model token is spent, the halt is logged
    (a kill_switch step row naming the taskmaster), and the switch file's
    bytes are untouched (reading must never become writing)."""
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
    switch_path = tmp_path / "switch_on.json"
    switch_path.write_text(json.dumps({"enabled": True, "updated_by": "test", "updated_at": "t"}))
    before_bytes = switch_path.read_bytes()  # the byte-identity assertion's baseline
    agent = build_taskmaster_agent(
        seeded, run_id="run_1", offers_dir=str(offers_dir),
        outbox_dir=str(tmp_path / "outbox"), inbox_dir=str(tmp_path / "inbox"),
        kill_switch_path=str(switch_path),  # the ENGAGED switch
    )

    async def _run() -> dict:
        session_service = InMemorySessionService()
        runner = Runner(app_name="outbound", agent=agent, session_service=session_service, auto_create_session=True)
        async for _ in runner.run_async(
            user_id="operator", session_id="run_1",
            new_message=types.Content(role="user", parts=[types.Part(text="run outreach")]),
            state_delta={"run_id": "run_1"},
        ):
            pass  # events consumed only for side effects; state read from the session below
        session = await session_service.get_session(app_name="outbound", user_id="operator", session_id="run_1")
        return session.state

    state = asyncio.run(_run())
    # ── The halt's sentinels (the guardrail's published short-circuit) ────
    assert state.get("final_state") == "failed", "the guardrail must short-circuit the run"
    assert state.get("kill_switch_reason") is not None, "the halt must leave its reason in state"
    # ── The halt IS observable in the trace (never skip logs) ─────────────
    halt_rows = seeded.execute("SELECT * FROM steps WHERE tool_name='kill_switch';").fetchall()
    assert len(halt_rows) == 1, "the root halt fires once, before any work"
    assert json.loads(halt_rows[0]["output_json"])["halted_agent"] == TASKMASTER_AGENT_ID
    # ── No Taskmaster tool ever ran (the whole point of entry refusal) ────
    tool_rows = seeded.execute(
        "SELECT COUNT(*) AS n FROM steps WHERE tool_name LIKE 'taskmaster.%';"
    ).fetchone()["n"]
    assert tool_rows == 0, "an engaged switch must halt BEFORE any tool call"
    # ── Reading never became writing (Z3's file half) ─────────────────────
    assert switch_path.read_bytes() == before_bytes, "the switch file must be byte-identical — the Taskmaster may read it, never write it"


# ── C4-Z4: the deterministic batch cap ───────────────────────────────────────

def test_batch_over_cap_refused_by_the_tool_with_no_rows_written(taskmaster_agent, seeded, tmp_path):
    """C4-Z4: a request for more than MAX_BATCH_SIZE is refused BY THE TOOL
    — the refusal string names the cap, and NO core-table rows are written
    (no targets, no accounts, no write_log rows: the refusal fires before
    any I/O).  The determinism half: the check is code, so the refusal is
    identical no matter how 'nicely' the model frames the request — there
    is no phrasing to negotiate with, the function simply refuses."""
    tool = _tool_by_name(taskmaster_agent, "import_and_research")
    # Snapshot BEFORE the calls: write_log already holds the seeder's own
    # registry rows (7 principals), so the write_log assertion below is a
    # BEFORE/AFTER delta — the refusals must add zero rows to it.
    write_log_before = seeded.execute("SELECT COUNT(*) AS n FROM write_log;").fetchone()["n"]
    # A nonexistent CSV proves the refusal fires BEFORE any file open: the
    # limit check precedes the CSV read, so a polite 500-target request for
    # a file that does not even exist still gets the cap refusal, not a
    # file error.
    # tool.func is async now (2026-08-29 fix — see phase1.py's
    # run_target_through_phase1_async docstring); asyncio.run() per call is
    # legal here (no event loop already running in a plain sync test).
    refusal_1 = asyncio.run(tool.func(csv_path=str(tmp_path / "does_not_exist.csv"), offer_slug="hk", limit=500))
    # Determinism half: the same request produces the same refusal — there
    # is no phrasing for the model to vary, because the tool's signature
    # has no message parameter and the check is code.  The model "asking
    # nicely" cannot reach this function with anything but the same
    # arguments.
    refusal_again = asyncio.run(tool.func(csv_path=str(tmp_path / "does_not_exist.csv"), offer_slug="hk", limit=500))
    assert refusal_1 == refusal_again, "the refusal is deterministic code, not negotiation — identical requests refuse identically"
    assert refusal_1.startswith("refused:") and str(MAX_BATCH_SIZE) in refusal_1, (
        f"the refusal must name the cap; got: {refusal_1!r}"
    )
    # And the boundary itself: exactly MAX_BATCH_SIZE+1 is over the cap,
    # even though it is only one target over — the cap is a hard line.
    refusal_2 = asyncio.run(tool.func(csv_path=str(tmp_path / "does_not_exist.csv"), offer_slug="hk", limit=MAX_BATCH_SIZE + 1))
    assert refusal_2.startswith("refused:"), "MAX_BATCH_SIZE+1 must refuse — the cap is a hard line"
    # ── No core-table rows, no NEW write_log rows ─────────────────────────
    # targets/accounts/contacts start empty in this fixture; the write_log
    # delta compares against the snapshot taken before the calls.
    for table in ("targets", "accounts", "contacts"):
        assert seeded.execute(f"SELECT COUNT(*) AS n FROM {table};").fetchone()["n"] == 0, (
            f"a refused batch must not write {table}"
        )
    assert seeded.execute("SELECT COUNT(*) AS n FROM write_log;").fetchone()["n"] == write_log_before, (
        "a refused batch must add no write_log rows (the refusal fires before any gated write)"
    )
    # ── The refusal IS logged (never skip logs — a silent refusal is
    # indistinguishable from a broken tool) under the taskmaster principal ─
    row = seeded.execute(
        "SELECT agent_id, status FROM steps WHERE tool_name='taskmaster.import_and_research';"
    ).fetchone()
    assert row is not None and row["agent_id"] == TASKMASTER_AGENT_ID and row["status"] == "failed"


def test_csv_rows_over_cap_refused_even_when_limit_is_small(taskmaster_agent, seeded, tmp_path):
    """C4-Z4's second half: the model could ask for limit=5 while pointing
    at a 16-row CSV — the batch is what the FILE holds, so the file is
    counted too (the phase1_cli check), before any DB or network I/O."""
    csv_path = tmp_path / "big.csv"
    csv_path.write_text("company,website\n" + "".join(f"Company {i},https://c{i}.test\n" for i in range(MAX_BATCH_SIZE + 1)))
    tool = _tool_by_name(taskmaster_agent, "import_and_research")
    summary = asyncio.run(tool.func(csv_path=str(csv_path), offer_slug="hk", limit=5))
    assert summary.startswith("refused:") and str(MAX_BATCH_SIZE) in summary, (
        f"an over-cap CSV must be refused by row count; got: {summary!r}"
    )
    assert seeded.execute("SELECT COUNT(*) AS n FROM targets;").fetchone()["n"] == 0


# ── C4-Z2: no transport in the new modules (focused) ────────────────────────

def test_taskmaster_modules_import_no_mail_transport():
    """C4-Z2, focused half: the two new modules import no mail transport —
    an AST import scan with the same forbidden roots
    tests/test_send_gate.py uses for the whole package (that test ALSO walks
    these files, unmodified; this one fails fast with a message naming the
    C4 file if a transport ever sneaks in here specifically)."""
    forbidden = (
        "smtplib", "aiosmtplib", "poplib", "imaplib", "smtpd",
        "googleapiclient", "google_auth_oauthlib", "google.oauth2",
        "yagmail", "redmail", "sendgrid", "mailgun", "exchangelib",
        "imapclient", "imbox",
    )
    for path in _taskmaster_module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                imported = []
            for name in imported:
                for bad in forbidden:
                    assert not (name == bad or name.startswith(bad + ".")), (
                        f"{path.name} imports {name!r} — a mail transport. "
                        f"The only send is the DRY_RUN .eml write (C4-Z2)."
                    )


# ── The bounded tool budget ──────────────────────────────────────────────────

def test_tool_budget_blocks_after_limit(taskmaster_agent):
    """The B1a bound: LlmAgent has no max_iterations, so the budget
    callback must block the (N+1)th tool call.  Driven directly against the
    agent's wired before_tool_callback with a fake ToolContext — the same
    (agent, invocation) keying the callback counts, so the block fires
    exactly at the configured ceiling."""
    callback = taskmaster_agent.before_tool_callback
    assert callback is not None, "the Taskmaster MUST carry a tool budget (B1a)"
    ctx = SimpleNamespace(  # the callback reads .agent_name / .invocation_id / .state — a dict-backed stand-in matches the session-state contract
        agent_name=TASKMASTER_AGENT_ID,
        invocation_id="inv_1",
        state={},
    )
    fake_tool = SimpleNamespace(name="report_pipeline_status")
    for _ in range(_TASKMASTER_MAX_TOOL_CALLS):
        # Every in-budget call returns None — the ONLY allow signal in the
        # callback contract (measured, app/agents/adk_support.py).
        assert callback(tool=fake_tool, args={}, tool_context=ctx) is None
    # The (N+1)th call returns the block dict the model is told about.
    blocked = callback(tool=fake_tool, args={}, tool_context=ctx)
    assert isinstance(blocked, dict) and "blocked" in blocked["result"], (
        f"call {_TASKMASTER_MAX_TOOL_CALLS + 1} must be blocked; got {blocked!r}"
    )
    # A FRESH invocation starts at zero — the budget is per (agent,
    # invocation), never a process-global that would leak across runs.
    ctx2 = SimpleNamespace(agent_name=TASKMASTER_AGENT_ID, invocation_id="inv_2", state={})
    assert callback(tool=fake_tool, args={}, tool_context=ctx2) is None


# ── report_pipeline_status is read-only ──────────────────────────────────────

def test_report_pipeline_status_writes_no_gated_rows(taskmaster_agent, seeded):
    """The status tool must not write ANY core-table row: zero new
    write_log rows from calling it (its only side effect is its own steps
    trace row, which is what makes the report itself auditable)."""
    _seed_target(seeded, target_id="tgt_1", state="awaiting_review", policy="allow")
    before = seeded.execute("SELECT COUNT(*) AS n FROM write_log;").fetchone()["n"]
    tool = _tool_by_name(taskmaster_agent, "report_pipeline_status")
    summary = tool.func()
    after = seeded.execute("SELECT COUNT(*) AS n FROM write_log;").fetchone()["n"]
    assert after == before, "report_pipeline_status must be read-only — no gated writes"
    # The report names the state and the review requirement (the honest
    # status the model plans from).
    assert "awaiting_review" in summary and "awaiting review: 1" in summary
    # Its one side effect: the trace row under the taskmaster principal.
    row = seeded.execute(
        "SELECT agent_id, status FROM steps WHERE tool_name='taskmaster.report_pipeline_status';"
    ).fetchone()
    assert row is not None and row["agent_id"] == TASKMASTER_AGENT_ID and row["status"] == "success"


# ── Dispatch, not reimplement: each tool calls the EXISTING runner ───────────

def test_import_tool_dispatches_existing_phase1_runners(taskmaster_agent, seeded, offers_dir, tmp_path):
    """The import tool must call sync_offers_table / import_csv /
    build_phase1_agent / run_target_through_phase1 — the phase1_cli
    internals, patched here — with the right arguments, proving C4 wraps
    rather than reimplements.  ``offers_dir`` is the fixture value the
    agent was built with, so the closure-bound arguments are asserted
    exactly."""
    csv_path = tmp_path / "targets.csv"
    csv_path.write_text("company,website\nAcme,acme.test\n")
    _seed_target(seeded, target_id="tgt_1", state="new")  # the domain lookup the tool runs after import
    tool = _tool_by_name(taskmaster_agent, "import_and_research")
    with patch("app.config.sync_offers_table") as sync, \
         patch("app.tools.get_targets.import_csv", return_value=["tgt_1"]) as imp, \
         patch("app.agents.phase1.build_phase1_agent", return_value=object()) as build, \
         patch("app.agents.phase1.run_target_through_phase1_async", return_value="scored") as run:
        # tool.func is async now; patch's target is an async function too,
        # so unittest.mock auto-detects and uses AsyncMock — return_value
        # comes back correctly from `await run(...)` inside the tool.
        summary = asyncio.run(tool.func(csv_path=str(csv_path), offer_slug="hk", limit=1))
    sync.assert_called_once()  # the offer table is synced before import (the phase1_cli order)
    assert sync.call_args.kwargs["run_id"] == "run_1"  # the closure-bound run id — never a model-supplied value
    assert sync.call_args.args[1] == str(offers_dir)  # the closure-bound offers dir, same for import and research below
    imp.assert_called_once_with(
        seeded, csv_path=str(csv_path), cli_offer_slug="hk",
        run_id="run_1", step_id=imp.call_args.kwargs["step_id"],
    )
    build.assert_called_once_with(seeded)  # one agent for the batch (the phase1_cli pattern)
    run.assert_called_once_with(
        build.return_value, conn=seeded, target_id="tgt_1", domain="acme.test",
        run_id="run_1", offers_dir=str(offers_dir),
    )
    assert "scored" in summary and "processed 1" in summary


def test_draft_tool_dispatches_existing_draft_runner(taskmaster_agent, seeded, offers_dir):
    """The draft tool must call build_draft_agent + run_target_through_draft
    — the draft_cli internals — with the closure-bound offers dir."""
    _seed_target(seeded, target_id="tgt_1", state="scored", policy="allow")
    tool = _tool_by_name(taskmaster_agent, "draft_for_scored")
    with patch("app.agents.draft.build_draft_agent", return_value=object()) as build, \
         patch("app.agents.draft.run_target_through_draft_async", return_value="awaiting_review") as run:
        summary = asyncio.run(tool.func(limit=5))
    build.assert_called_once_with(seeded)
    run.assert_called_once_with(
        build.return_value, conn=seeded, target_id="tgt_1",
        run_id="run_1", offers_dir=str(offers_dir),
    )
    assert "awaiting_review" in summary and "REVIEW" in summary.upper()


def test_send_tool_dispatches_existing_send_runner(taskmaster_agent, seeded):
    """The send tool must call send_email — the send_cli runner — with the
    closure-bound outbox/offers dirs.  The patched runner returns a
    successful DRY_RUN result; the tool's summary reports it as such."""
    _seed_target(seeded, target_id="tgt_1", state="approved", policy="allow")
    tool = _tool_by_name(taskmaster_agent, "dry_run_send_approved")
    fake = SendEmailResult(
        target_id="tgt_1", refused=False, refusal_reason="",
        message_id="msg_1", outbox_path="/tmp/outbox/msg_1.eml", new_state="dry_run_sent",
    )
    with patch("app.tools.send_email.send_email", return_value=fake) as send:
        summary = tool.func(limit=5)
    send.assert_called_once_with(
        seeded, target_id="tgt_1", run_id="run_1",
        outbox_dir=send.call_args.kwargs["outbox_dir"],  # the closure-bound outbox — the model cannot redirect it
        offers_dir=send.call_args.kwargs["offers_dir"],
    )
    assert "dry_run_sent" in summary and "DRY_RUN" in summary


def test_reply_tool_dispatches_existing_reply_runners(taskmaster_agent, seeded):
    """The reply tool must call fetch_inbox + build_reply_agent +
    classify_and_route_reply — the reply_cli internals — with the
    closure-bound inbox dir and the tool's limit."""
    tool = _tool_by_name(taskmaster_agent, "fetch_and_classify_replies")
    from app.tools.fetch_inbox import InboxFetchResult  # the fetch runner's outcome shape
    fetched = InboxFetchResult(files_seen=1, replies_created=["rep_1"], skipped=[], errors=[])
    with patch("app.tools.fetch_inbox.fetch_inbox", return_value=fetched) as fetch, \
         patch("app.agents.reply.build_reply_agent", return_value=object()) as build, \
         patch("app.agents.reply.classify_and_route_reply_async", return_value="routed") as classify:
        summary = asyncio.run(tool.func(limit=5))
    fetch.assert_called_once_with(
        seeded, inbox_dir=fetch.call_args.kwargs["inbox_dir"],  # the closure-bound inbox — the model cannot point the fetch elsewhere
        run_id="run_1", limit=5,
    )
    build.assert_called_once_with(seeded)
    classify.assert_called_once_with(build.return_value, conn=seeded, reply_id="rep_1", run_id="run_1")
    assert "routed" in summary and "rep_1" in summary


# ── The registry row ─────────────────────────────────────────────────────────

def test_taskmaster_is_registered_enabled_and_capability_narrowed(seeded):
    """The registry must seed the taskmaster principal: registered (the
    guardrail's per-agent check and the write gate both read it), enabled
    (the kill switch is opt-out), carrying the taskmaster_model alias — and
    with the deliberately EMPTY allowed_actions array, the C4 §3.3
    narrowing that makes any gated write attributed to the Taskmaster
    refused by the gate before any SQL runs."""
    row = seeded.execute(
        "SELECT agent_id, model_alias, enabled, allowed_actions "
        "FROM agent_registry WHERE agent_id='taskmaster';"
    ).fetchone()
    assert row is not None, "taskmaster must be seeded in agent_registry"
    assert row["enabled"] == 1, "the Taskmaster starts enabled; the kill switch is opt-out"
    assert row["model_alias"] == "taskmaster_model", (
        "the Taskmaster is an LLM principal — its model_alias must name the "
        "config/models.yaml role it resolves"
    )
    assert json.loads(row["allowed_actions"]) == [], (
        "C4 §3.3: the Taskmaster's allowed_actions must be EMPTY — it performs "
        "no gated writes of its own, and the empty set refuses any write "
        "attributed to it (the Z1/Z3 structural backstop)"
    )
