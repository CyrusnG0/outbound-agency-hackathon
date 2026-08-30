# app/taskmaster_cli.py -- the natural-language entry point (operator-facing)
# This module is the single CLI command the operator runs to give the
# TaskmasterAgent one task in plain language and have it drive the whole
# pipeline:
#
#   python -m app.taskmaster_cli --task "run outreach for the HK therapy clinics offer, 10 targets"
#
# The agent plans, dispatches the EXISTING stage runners through its tools
# (import+research -> draft -> [human review] -> DRY_RUN send -> reply
# classification), and prints its own report.  It is structurally incapable
# of approving drafts, disabling the kill switch, or sending real email —
# those boundaries live in app/agents/taskmaster.py and are tested.
#
# THE FOUR CLIs REMAIN THE DETERMINISTIC ESCAPE HATCH (plan §1.5): this
# entry point does NOT replace phase1_cli / draft_cli / send_cli /
# reply_cli — they stay exactly as they are, one auditable code path per
# stage, for the operator to drive by hand when the agent's judgement is
# not wanted.  This CLI is an ADDITIONAL front door over the same runners.
import argparse  # stdlib argument parser — no new dependency for the operator
import asyncio  # asyncio.run + wait_for: the B1g wall-clock ceiling over the whole agent invocation
import sys  # stderr for error messages, argv for the default None sentinel

import httpx  # declared direct dependency (pyproject.toml): the SDK-level timeout type the B1g ceiling routes (same as phase1.py)

from app.agents.phase1 import _resolve_target_timeout_seconds  # the ONE env-var ceiling resolver (PHASE1_TARGET_TIMEOUT_SECONDS) — reused, not duplicated (ticket §3.2)
from app.agents.taskmaster import TASKMASTER_AGENT_ID, build_taskmaster_agent  # the root agent this CLI runs
from app.agents_registry import seed_agent_registry  # registers the principals (incl. taskmaster) the write gate checks
from app.db import apply_schema, connect  # opens the DB and applies the DDL (idempotent)
from app.ids import new_id  # generates unique prefixed IDs for the run and each failure step
from app.tools.fetch_inbox import DEFAULT_INBOX_DIR  # the simulated inbox default the reply tool reads
from app.tools.log_step import log_step  # steps-table trace writer — a failed invocation still gets its step row (never skip logs)
from app.tools.send_email import DEFAULT_OUTBOX_DIR  # the DRY_RUN outbox default the send tool writes
from google.adk.runners import Runner  # executes the agent against an in-memory session
from google.adk.sessions import InMemorySessionService  # in-memory sessions: the durable audit trail already lives in steps/write_log/state_transitions (A4a fact 6)
from google.genai import types  # the user-turn message ADK's Runner starts an invocation with


def _collect_report(agent, conn, *, run_id: str, task_text: str) -> tuple[str, dict]:
    """Run the Taskmaster once and return (report text, final session state).

    One invocation, one session (session_id=run_id — the run's sessions
    must never collide with another run's).  The report text is the
    concatenation of every non-thought text part the taskmaster authored —
    model turns interleave with tool calls, so this is the agent's full
    account of the run, in order.  The session state is read afterwards for
    the kill-switch sentinel (the guardrail publishes
    ``kill_switch_reason`` there when it halts the invocation).
    """
    async def _run() -> tuple[str, dict]:
        # Fresh in-memory session service — the same deliberate
        # InMemorySessionService rationale as every per-target runner: the
        # durable audit trail lives in the DB tables, and the live
        # connection must never enter session state.
        session_service = InMemorySessionService()
        # Runner executes the agent against the session service;
        # auto_create_session=True lets run_async create the session on
        # first use instead of a separate create_session call.
        runner = Runner(
            app_name="outbound",
            agent=agent,
            session_service=session_service,
            auto_create_session=True,
        )
        report_parts: list[str] = []  # the agent's own text turns, in order
        # Drive the agent once.  The user message IS the operator's
        # natural-language task — unlike the deterministic pipelines, whose
        # runners ignore message content, this agent's whole input is this
        # text.  state_delta seeds run_id so the guardrail's halt rows (and
        # any future state-templated instruction) carry the real run id
        # instead of the "unknown" fallback.
        async for event in runner.run_async(
            user_id="operator",
            session_id=run_id,
            new_message=types.Content(role="user", parts=[types.Part(text=task_text)]),
            state_delta={"run_id": run_id},
        ):
            # Events stream for every step — user turn, model turns, tool
            # calls/responses.  Only the taskmaster's own text parts are
            # the report; tool-call events carry function_call/response
            # parts with no text, and the user turn is the operator's own
            # words (not the report).  "thought" parts (Gemini thinking)
            # are skipped — they are not the agent's answer.
            if event.author == TASKMASTER_AGENT_ID and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text and not getattr(part, "thought", False):
                        report_parts.append(part.text)
        # Terminal state read straight from the session — NOT by scraping
        # the event stream (the established A4a pattern) — carrying the
        # guardrail's kill_switch_reason sentinel when a halt fired.
        session = await session_service.get_session(
            app_name="outbound", user_id="operator", session_id=run_id,
        )
        return "\n".join(report_parts), session.state

    # ── The B1g ceiling: bound the WHOLE invocation in wall clock ─────────
    # The same guarantee and rationale as run_target_through_phase1: a hung
    # Vertex connection must not stall the run, and asyncio.wait_for around
    # the whole coroutine is the ONE point that spans the entire agent
    # invocation — SDK-independent by construction.  The ceiling value is
    # the SAME env-var knob every per-target runner uses
    # (PHASE1_TARGET_TIMEOUT_SECONDS, default 600s) — one knob for the
    # whole repo, not a new one to document (ticket §3.2).
    timeout_seconds = _resolve_target_timeout_seconds()  # env override or the documented default, resolved per call
    try:
        # asyncio.run bridges ADK's async runner to this synchronous entry
        # point; wait_for adds the wall-clock deadline and cancels the
        # pending network await inside ADK when it fires.
        return asyncio.run(asyncio.wait_for(_run(), timeout=timeout_seconds))
    except TimeoutError as exc:
        # The ceiling fired (asyncio.TimeoutError — the same alias
        # reasoning as phase1.py).  A timed-out Taskmaster produced NO
        # report, which is a lost run: log it, tell the operator, and
        # exit non-zero (unlike the per-target timeouts, which are normal
        # outcomes — this is the whole task, not one target).
        _record_invocation_failure(
            conn, run_id=run_id, tool_name="taskmaster_target_timeout",
            output_data={
                "timeout_seconds": timeout_seconds,
                "detail": "per-invocation wall-clock ceiling exceeded (asyncio.wait_for cancelled the run)",
            },
        )
        print(
            f"ERROR: the Taskmaster exceeded its {timeout_seconds}s wall-clock ceiling "
            f"and produced no final report. Raise PHASE1_TARGET_TIMEOUT_SECONDS and retry.",
            file=sys.stderr,
        )
        raise _InvocationFailed() from exc
    except httpx.TimeoutException as exc:
        # An SDK-level timeout fired BEFORE the ceiling (a single stalled
        # model request) — the same unwrapped-httpx fact as phase1.py, the
        # same bucket, keeping the SDK's exception text in the step row.
        _record_invocation_failure(
            conn, run_id=run_id, tool_name="taskmaster_target_timeout",
            output_data={"timeout_seconds": timeout_seconds, "detail": f"{type(exc).__name__}: {exc}"},
        )
        print(
            f"ERROR: a model request timed out before the wall-clock ceiling — "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise _InvocationFailed() from exc


class _InvocationFailed(Exception):
    """Internal signal: the agent invocation failed (timeout) — main()
    catches it to return a non-zero exit after the failure was logged and
    printed.  A private control-flow exception keeps the failure handling
    in one place instead of threading sentinel tuples through
    _collect_report."""


def _record_invocation_failure(conn, *, run_id: str, tool_name: str, output_data: dict) -> None:
    """Log one failed invocation step row (never skip logs — a lost run
    must leave SOME trace of the loss, the B1f/Golden-Rule discipline
    applied to the whole run).  Best-effort: a broken connection must not
    mask the original failure."""
    try:
        log_step(
            conn, run_id=run_id, step_id=new_id("step"), target_id=None,
            tool_name=tool_name,
            agent_id=TASKMASTER_AGENT_ID,  # the taskmaster principal — the run belongs to it
            input_data={"stage": "taskmaster_invocation"},
            output_data=output_data,
            status="failed",
        )
    except Exception as bookkeeping_exc:
        print(
            f"ERROR: could not log the failed invocation — log_step also failed: {bookkeeping_exc}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    # ----- Parse CLI args -----
    # No mode flag, no safety dial: the task text plus the run's inputs.
    # Safety is structural (no approval tool, no transport, no switch
    # writer in the agent's reach) — never configurational.
    parser = argparse.ArgumentParser(prog="python -m app.taskmaster_cli")
    parser.add_argument(
        "--task", required=True,
        help="the task in plain language, e.g. 'run outreach for the HK therapy clinics offer, 10 targets'",
    )
    parser.add_argument("--db", default="data/outbound.db")  # main operational DB
    parser.add_argument("--offers-dir", default="config/offers")  # YAML offer definitions directory (same default as phase1_cli)
    parser.add_argument(
        "--inbox", default=DEFAULT_INBOX_DIR,
        help="directory the simulated inbox .eml files are read from (data/inbox/ by default)",
    )
    parser.add_argument(
        "--outbox", default=DEFAULT_OUTBOX_DIR,
        help="directory the DRY_RUN .eml artifacts are written to (data/outbox/ by default)",
    )
    args = parser.parse_args(argv)
    if not args.task.strip():
        # An empty task is a wiring mistake — refuse before any DB I/O
        # rather than paying for a model turn that has nothing to do.
        print("ERROR: --task must be a non-empty string.", file=sys.stderr)
        return 1

    # ----- Open DB and apply schema -----
    conn = connect(args.db)  # app.db.Conn — sqlite file path or postgresql:// / cloudsql:// URL
    apply_schema(conn)  # idempotent DDL; safe to call on every run

    # ----- Seed the agent registry so the write gate accepts this run -----
    # The stage runners' inner writes carry their own principals
    # (system / draft_writer / reply_classifier), and the taskmaster row
    # itself must exist for the guardrail's per-agent check and for this
    # run's own step attribution.  Idempotent upsert — the same startup
    # sequence every other CLI uses.
    run_id = new_id("run")  # unique run identifier ties together all steps in this invocation
    seed_agent_registry(conn, run_id=run_id, step_id=new_id("step"))

    # ----- Build the agent ONCE, closure-bound to this run's context -----
    # conn/run_id/offers_dir/inbox_dir/outbox_dir travel in the closures —
    # never in session state (the A4a constraint) and never as model-
    # settable tool parameters (the make_fetch_page_tool trust pattern).
    agent = build_taskmaster_agent(
        conn,
        run_id=run_id,
        offers_dir=args.offers_dir,
        outbox_dir=args.outbox,
        inbox_dir=args.inbox,
    )

    # ----- Run it, bounded, and print whatever it reports -----
    print(f"Taskmaster run {run_id} — task: {args.task}")
    try:
        report, state = _collect_report(agent, conn, run_id=run_id, task_text=args.task)
    except _InvocationFailed:
        conn.close()  # explicit close — though CPython would close on exit, be explicit
        return 1  # a lost run is not a clean run
    except Exception as exc:
        # An unhandled crash of the invocation itself (not a per-target
        # stage crash — those are contained inside the stage tools).  Log
        # it (never skip logs) and exit non-zero.
        _record_invocation_failure(
            conn, run_id=run_id, tool_name="taskmaster_target_run",
            output_data={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        print(f"ERROR: the Taskmaster run crashed — {type(exc).__name__}: {exc}", file=sys.stderr)
        conn.close()
        return 1

    # ----- The guardrail's halt sentinel (B4a) -----
    # The guardrail publishes kill_switch_reason into session state when it
    # ends the invocation at entry.  A halt is the harness working
    # correctly — loud, logged, and exit 0 (the same convention as the
    # per-stage CLIs, where a switch-driven refusal is a normal outcome,
    # not a crash).
    kill_reason = state.get("kill_switch_reason")
    if kill_reason is not None:
        print(f"Run {run_id} HALTED by the kill switch: {kill_reason}")
        print("Disengage the switch (runbook.md §1) and re-run the task.")
        conn.close()
        return 0
    if not report.strip():
        # The invocation ended with no model text and no halt — an empty
        # report is a lost run (the B1d honesty lesson: never mislabel an
        # agent's silence as success).
        print(f"ERROR: run {run_id} produced no report — the agent emitted no final text.", file=sys.stderr)
        conn.close()
        return 1
    # ----- The report is the product -----
    # Printed verbatim — refusals, crashes, review requirements and all.
    # The agent was ordered to report what actually happened, and the
    # operator reads exactly that.
    print(report)

    conn.close()  # explicit close — though CPython would close on exit, be explicit
    return 0  # the run completed and reported; refusals inside the report are correct behaviour, not failures


# Guard so `python app/taskmaster_cli.py` also works, not just `python -m app.taskmaster_cli`.
# Uses SystemExit instead of sys.exit() to stay testable (pytest can catch SystemExit).
if __name__ == "__main__":
    raise SystemExit(main())
