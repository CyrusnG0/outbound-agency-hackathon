# app/phase1_cli.py -- Phase 1 pipeline entry point (operator-facing)
# This module is the single CLI command the operator runs to import targets,
# research them, score them, and classify them through the full Phase 1
# pipeline (a Google ADK SequentialAgent since task A4a; previously a
# LangGraph StateGraph).  It exists so every Phase 1 run goes through one
# auditable code path with a hard batch-size cap, not ad-hoc scripts.
import argparse  # stdlib argument parser — no new dependency for the operator
import csv  # used by _count_csv_rows to count data rows without loading them into memory
import sys  # stderr for error messages, argv for the default None sentinel

from app.agents_registry import seed_agent_registry  # registers the system/operator agents the write gate checks
from app.config import sync_offers_table  # ensures every offer in the CSV exists in the DB before import
from app.db import apply_schema, connect  # opens the DB and applies the DDL (idempotent)
from app.agents.phase1 import (  # ADK Phase 1 agent + single-target runner (task A4a)
    build_phase1_agent,
    run_target_through_phase1,
)
from app.ids import new_id  # generates unique prefixed IDs for the run and each step
from app.state_machine import transition  # the single state-change gate — a crashed target is marked failed through it, never a raw UPDATE
from app.tools.get_targets import import_csv  # CSV → accounts/contacts/targets rows
from app.tools.log_step import log_step  # steps-table trace writer — a crashed target still gets its step row (never skip logs)

# Self-use conservative scale cap — the operator is one person sending to a
# handful of targets at a time, not a marketing platform.  This cap fires
# before any DB or network I/O so oversized batches are rejected cheaply.
MAX_BATCH_SIZE = 15  # see docs/PROJECT-REFERENCE.md for reasoning


def _count_csv_rows(csv_path: str) -> int:
    # Count data rows (excluding header) so the batch-size check happens
    # BEFORE opening the DB or touching the network — fail-fast on oversized
    # batches.  Uses DictReader so the count matches import_csv's row count.
    with open(csv_path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    # ----- Parse CLI args -----
    parser = argparse.ArgumentParser(prog="python -m app.phase1_cli")
    parser.add_argument("--csv", required=True, help="path to targets CSV")  # the batch of targets to import & research
    parser.add_argument(
        "--offer", default=None,
        help="offer slug, used when the CSV has no offer_id column",  # fallback offer for CSVs without explicit offers
    )
    parser.add_argument("--db", default="data/outbound.db")  # main operational DB
    parser.add_argument("--offers-dir", default="config/offers")  # YAML offer definitions directory
    args = parser.parse_args(argv)

    # ----- Pre-flight: batch-size cap -----
    # Reject oversized batches before any DB or network I/O.  This enforces
    # the self-use scale posture from gates.md / PROJECT-REFERENCE.md at the
    # outermost edge of the pipeline.
    row_count = _count_csv_rows(args.csv)
    if row_count > MAX_BATCH_SIZE:
        print(
            f"ERROR: batch of {row_count} exceeds the {MAX_BATCH_SIZE}-target cap per run "
            f"(gates.md's conservative posture, extended to research batches — see "
            f"docs/PROJECT-REFERENCE.md's self-use scale reasoning). "
            f"Split into smaller CSV files.",
            file=sys.stderr,
        )
        return 1  # non-zero exit = rejected before any work started

    # ----- Open DB and apply schema -----
    conn = connect(args.db)  # app.db.Conn — sqlite file path or postgresql:// / cloudsql:// URL
    apply_schema(conn)  # idempotent DDL; safe to call on every run

    # ----- Seed the agent registry so the write gate accepts this run -----
    # Every write in this run carries agent_id="system", and the write gate
    # refuses writes from unregistered agents. The seed is idempotent
    # (upsert), so running it on every invocation is safe and cheap.
    run_id = new_id("run")  # unique run identifier ties together all steps in this invocation
    seed_agent_registry(conn, run_id=run_id, step_id=new_id("step"))

    # ----- Sync offers so every CSV offer_id resolves -----
    sync_offers_table(
        conn, args.offers_dir, run_id=run_id, step_id=new_id("step"),
    )

    # ----- Import CSV into accounts/contacts/targets -----
    try:
        target_ids = import_csv(
            conn,
            csv_path=args.csv,
            cli_offer_slug=args.offer,
            run_id=run_id,
            step_id=new_id("step"),
        )
    except Exception as exc:
        # import_csv raises typed errors (MissingOfferIdError, etc.) — catch
        # them all so we can print to stderr and exit non-zero without a traceback.
        print(f"ERROR: import failed — {exc}", file=sys.stderr)
        return 1  # import failure is fatal — nothing to run

    # ----- Build the Phase 1 ADK agent and run every imported target -----
    # The agent is built once (with this run's DB connection) and shared
    # across all targets; each target gets its own in-memory ADK session.
    agent = build_phase1_agent(conn)

    results: dict[str, str] = {}  # target_id -> terminal Phase 1 state for targets the pipeline concluded normally
    # target_id -> "ExceptionType: message" for targets killed by an unhandled
    # exception.  Kept separate from results so the summary can mark a crash
    # distinctly from an ordinary "failed" outcome — a lost target must never
    # look like a normal result (ticket B1f: the summary must not lie).
    crashed: dict[str, str] = {}
    for target_id in target_ids:
        # Look up the normalized domain for this target (set during import).
        row = conn.execute(
            "SELECT a.normalized_domain FROM targets t "
            "JOIN accounts a ON t.account_id = a.account_id "
            "WHERE t.target_id = ?;",
            (target_id,),
        ).fetchone()
        domain = row["normalized_domain"]
        try:
            # Run the full Phase 1 pipeline: fetch sources → summarize → detect
            # signals → score → classify.  Returns the terminal Phase 1 state.
            final_state = run_target_through_phase1(
                agent, conn=conn, target_id=target_id, domain=domain, run_id=run_id,
                # B2c: the ICP judge reads the offer's icp block + pitch from
                # the same offers dir this run synced the offers table from —
                # the judge must compare against the same definitions the run
                # imported, never a different directory.
                offers_dir=args.offers_dir,
            )
            results[target_id] = final_state
        except Exception as exc:
            # ---- Crash containment: one target's crash must never abort the batch ----
            # (ticket B1f: a 10-target run died at target 4 with
            # ServerDisconnectedError and six targets were left at state "new".)
            #
            # Catch Exception, NOT BaseException: KeyboardInterrupt (operator
            # pressed Ctrl-C) and SystemExit must still propagate out of main()
            # and abort the run — swallowing a Ctrl-C would leave a batch the
            # operator believes is stopped silently spending money, which is
            # worse than the bug this guard fixes.  Exception covers every
            # failure the pipeline itself can produce, including ADK's aiohttp
            # transport errors (e.g. ServerDisconnectedError) that completely
            # bypass app/llm.py's LLMTransportError wrapper.
            error_type = type(exc).__name__  # type NAME, not str(exc): makes ServerDisconnectedError distinguishable from a KeyError in our own code when reading the trace later
            error_message = str(exc)
            # Print the ORIGINAL error immediately, before any bookkeeping, so
            # the operator always learns the real cause — the bookkeeping below
            # can itself fail (see the second guard) and must never be the only
            # thing that reaches the operator.
            print(
                f"ERROR: target {target_id} crashed during Phase 1 — "
                f"{error_type}: {error_message}",
                file=sys.stderr,
            )
            crashed[target_id] = f"{error_type}: {error_message}"
            # One fresh step id shared by the transition and the log_step row
            # below — the same pattern phase1.py's failure paths use, so this
            # crash's audit entries hang together under one step.
            step_id = new_id("step")
            try:
                # transition() requires an explicit from_state, and a crash can
                # happen at ANY stage (new, researched, scored, ...), so READ
                # the target's current state from the DB instead of hardcoding
                # "new" — the state_transitions row must record where the
                # target actually was when it died, or the audit trail lies
                # about the crash point.
                current = conn.execute(
                    "SELECT state FROM targets WHERE target_id=?;", (target_id,)
                ).fetchone()
                if current is None:
                    # The row must exist (target_ids came from import_csv in
                    # this same run) — recording a transition for a phantom
                    # target would write a lying audit row.
                    raise ValueError(f"target {target_id} has no targets row")
                from_state = current["state"]
                # Any state -> failed is valid (ANY_TARGET_TRANSITIONS); the
                # NEW reason string names the cause without inventing a new
                # state (precedent: A4c's llm_transport_error_phase1, B1d's
                # research_agent_no_output_phase1 — an operator can tell the
                # causes apart from state_transitions.reason alone).
                transition(
                    conn, target_id=target_id, from_state=from_state, to_state="failed",
                    reason="unhandled_error_phase1", actor="system",
                    run_id=run_id, step_id=step_id,
                )
            except Exception as bookkeeping_exc:
                # ---- Second guard (transition half): bookkeeping must not kill the batch ----
                # If the DB connection is what broke (the most likely reason a
                # transition fails right after a crash), this guard keeps the
                # loop moving to the next target instead of dying inside its
                # own error handling and taking the remaining targets with it.
                # The message repeats the ORIGINAL exception type and message
                # so a cleanup failure can never mask the real cause.
                print(
                    f"ERROR: could not mark target {target_id} failed after "
                    f"({error_type}: {error_message}) — transition also failed: {bookkeeping_exc}",
                    file=sys.stderr,
                )
            try:
                # Second guard (log half), separate from the transition: even
                # if the state change could not be written, the step row still
                # gets its own best-effort attempt — a crashed target must
                # leave SOME trace of the crash in the audit trail (never skip
                # logs, CLAUDE.md §3).
                log_step(
                    conn, run_id=run_id, step_id=step_id, target_id=target_id,
                    tool_name="phase1_target_run",
                    agent_id="system",  # deterministic CLI code — the registered system agent
                    input_data={"stage": "phase1_target_run"},
                    # output_data carries the exception type name and message
                    # so the trace shows WHAT killed the target, not just that
                    # it died.
                    output_data={"error_type": error_type, "error_message": error_message},
                    status="failed",
                )
            except Exception as bookkeeping_exc:
                print(
                    f"ERROR: could not log the crash for target {target_id} "
                    f"({error_type}: {error_message}) — log_step also failed: {bookkeeping_exc}",
                    file=sys.stderr,
                )

    # ----- Print summary -----
    # Count how many targets reached a terminal scoring/classification state
    # so the operator can see at a glance how many need review vs. were
    # disqualified.
    scored_count = sum(
        1 for s in results.values()
        if s in ("scored", "watchlist", "not_target")
    )
    # Denominator is len(target_ids), NOT len(results): crashed targets are
    # deliberately absent from results, so len(results) would silently print
    # e.g. "9/9" after a crash ate one target of ten — the summary must not
    # lie about how much of the batch actually completed.
    print(
        f"Phase 1 run {run_id} complete. "
        f"{scored_count}/{len(target_ids)} targets reached a terminal Phase 1 state.",
    )
    for target_id, state in results.items():
        print(f"  {target_id}: {state}")
    # Crashed targets get their own lines, marked CRASHED with the exception
    # type and message — visually distinct from an ordinary "failed" target so
    # a lost target can never hide as a normal outcome in the summary.
    for target_id, error in crashed.items():
        print(f"  {target_id}: CRASHED ({error})")

    conn.close()  # explicit close — though CPython would close on exit, be explicit
    # Non-zero exit if ANY target crashed: a batch that lost a target to an
    # unhandled error is not a clean run and must not look like one to a
    # script or a human checking $?.  Ordinary failed/watchlist/not_target
    # outcomes do NOT change the exit code — those are normal Phase 1
    # results, not batch damage.
    return 1 if crashed else 0


# Guard so `python app/phase1_cli.py` also works, not just `python -m app.phase1_cli`.
# Uses SystemExit instead of sys.exit() to stay testable (pytest can catch SystemExit).
if __name__ == "__main__":
    raise SystemExit(main())
