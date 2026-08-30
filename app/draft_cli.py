# app/draft_cli.py -- Phase 1b drafting entry point (operator-facing)
# This module is the single CLI command the operator runs to draft outreach
# emails for every target the Phase 1 pipeline left in state "scored" (with
# a policy "allow").  It exists so every drafting run goes through one
# auditable code path — the writer⇄critic LoopAgent in app/agents/draft.py —
# rather than ad-hoc scripts.  It NEVER sends and NEVER approves anything:
# the target ends in "awaiting_review", and human review (ticket B4) is
# mandatory before any send (B5).
import argparse  # stdlib argument parser — no new dependency for the operator
import sys  # stderr for error messages, argv for the default None sentinel

from app.agents.draft import (  # the ADK draft LoopAgent + single-target runner (ticket B3) + the shared eligible-set selector (ticket E1)
    build_draft_agent,
    run_target_through_draft,
    select_draft_eligible_targets,
)
from app.agents_registry import seed_agent_registry  # registers the system/operator/judge/draft principals the write gate checks
from app.db import apply_schema, connect  # opens the DB and applies the DDL (idempotent, incl. B3's migration columns)
from app.ids import new_id  # generates unique prefixed IDs for the run and each step
from app.state_machine import transition  # the single state-change gate — a crashed target is marked failed through it, never a raw UPDATE
from app.tools.log_step import log_step  # steps-table trace writer — a crashed target still gets its step row (never skip logs)


def main(argv: list[str] | None = None) -> int:
    # ----- Parse CLI args -----
    parser = argparse.ArgumentParser(prog="python -m app.draft_cli")
    parser.add_argument("--db", default="data/outbound.db")  # main operational DB
    parser.add_argument("--offers-dir", default="config/offers")  # YAML offer definitions directory
    parser.add_argument(
        "--limit", type=int, default=10,
        help="maximum number of scored targets to draft in this run",  # self-use conservative default: the operator drafts a handful at a time
    )
    args = parser.parse_args(argv)
    if args.limit < 1:
        # A non-positive limit is a wiring mistake — refuse before any DB
        # I/O rather than silently selecting nothing (or everything).
        print("ERROR: --limit must be a positive integer.", file=sys.stderr)
        return 1

    # ----- Open DB and apply schema -----
    conn = connect(args.db)  # app.db.Conn — sqlite file path or postgresql:// / cloudsql:// URL
    apply_schema(conn)  # idempotent DDL; adds B3's critique columns to an already-provisioned database

    # ----- Seed the agent registry so the write gate accepts this run -----
    # Every draft write in this run carries agent_id="draft_writer" (plus
    # "system" for deterministic bookkeeping), and the write gate refuses
    # writes from unregistered agents.  The seed is idempotent (upsert), so
    # running it on every invocation is safe and cheap.
    run_id = new_id("run")  # unique run identifier ties together all steps in this invocation
    seed_agent_registry(conn, run_id=run_id, step_id=new_id("step"))

    # NOTE: no sync_offers_table call here (unlike phase1_cli).  The draft
    # stage never imports targets — it reads offer YAML directly per target
    # through load_offer_configs (app/agents/draft.py), and the offers table
    # rows already exist from the Phase 1 run that imported these targets.

    # ----- Select the targets this run will draft -----
    # The eligible set (ticket E1) is the union the shared selector
    # computes: state='scored' (first touch) PLUS state='routed' whose
    # LATEST reply has routed_action='queue_follow_up_draft' (a positive
    # reply queued a follow-up draft).  The policy precondition (latest
    # policy_decisions.decision == "allow") and the 2-follow-up-per-thread
    # cap are enforced PER TARGET inside run_target_through_draft, not
    # pre-filtered here — a refused target still gets its logged outcome
    # line (policy_denied / follow_up_cap_reached / not_draftable), and a
    # silent pre-filter would hide refusals from the operator.
    target_ids = select_draft_eligible_targets(conn, limit=args.limit)

    # ----- Log the batch manifest BEFORE the per-target loop -----
    # The full target list this run intends to draft is known up front, so
    # log it as one batch-level step (target_id NULL — the same convention
    # get_targets uses for phase1_cli's manifest).  The live run view
    # (app/console/app.py) reads this manifest to know the run's FULL
    # intended target set: a target the loop has not reached yet has zero
    # per-target steps/transitions and would otherwise be invisible to a
    # rows-derived completeness check, silently declaring the run complete
    # mid-batch (ticket U2-fix2 — the 12s quiet-period heuristic alone is
    # not safe against this pipeline's ~605s worst-case per-node latency).
    # Logged EVEN when the batch is empty: an empty manifest is a legitimate
    # result (the run had nothing to draft), never a skipped log.
    log_step(
        conn, run_id=run_id, step_id=new_id("step"), target_id=None,
        tool_name="draft_batch_manifest", agent_id="system",
        input_data={"stage": "draft_batch_manifest"},
        output_data={"target_ids": target_ids},
        status="success",
    )

    # ----- Build the draft LoopAgent once and run every selected target -----
    # The agent is built once (with this run's DB connection) and shared
    # across all targets; each target gets its own in-memory ADK session, so
    # the revision counter and draft state are per-target by construction.
    agent = build_draft_agent(conn)

    results: dict[str, str] = {}  # target_id -> outcome for targets the draft run concluded normally
    # target_id -> "ExceptionType: message" for targets killed by an
    # unhandled exception.  Kept separate from results so the summary can
    # mark a crash distinctly from an ordinary outcome — a lost target must
    # never look like a normal result (the B1f lesson, re-applied).
    crashed: dict[str, str] = {}
    for target_id in target_ids:
        try:
            # Run the writer⇄critic loop for this target.  Returns the
            # resulting state ("awaiting_review", "scored", "failed") or a
            # refusal string ("not_draftable", "policy_denied").
            outcome = run_target_through_draft(
                agent, conn=conn, target_id=target_id, run_id=run_id,
                # The draft brief and footer are assembled from the SAME
                # offers dir the operator points this run at — never a
                # different directory than the one on disk right now.
                offers_dir=args.offers_dir,
            )
            results[target_id] = outcome
        except Exception as exc:
            # ---- Crash containment: one target's crash must never abort the batch ----
            # (ticket B1f's per-target isolation, copied from phase1_cli:
            # a 10-target run died at target 4 and six targets were left
            # stranded.)  Catch Exception, NOT BaseException:
            # KeyboardInterrupt and SystemExit must still propagate — a
            # swallowed Ctrl-C would leave a batch the operator believes is
            # stopped silently spending money.
            error_type = type(exc).__name__  # type NAME, not str(exc): makes a transport error distinguishable from a KeyError in our own code
            error_message = str(exc)
            # Print the ORIGINAL error immediately, before any bookkeeping,
            # so the operator always learns the real cause.
            print(
                f"ERROR: target {target_id} crashed during drafting — "
                f"{error_type}: {error_message}",
                file=sys.stderr,
            )
            crashed[target_id] = f"{error_type}: {error_message}"
            # One fresh step id shared by the transition and the log_step row
            # below — the same pattern phase1.py's failure paths use, so this
            # crash's audit entries hang together under one step.
            step_id = new_id("step")
            try:
                # A crash can happen at ANY stage (first iteration, a later
                # revision, ...), so READ the target's current state from
                # the DB instead of hardcoding "scored" — the
                # state_transitions row must record where the target
                # actually was when it died, or the audit trail lies about
                # the crash point (the B1f lesson).
                current = conn.execute(
                    "SELECT state FROM targets WHERE target_id=?;", (target_id,)
                ).fetchone()
                if current is None:
                    # The row must exist (target_ids came from this run's
                    # SELECT) — a transition for a phantom target would be a
                    # lying audit row.
                    raise ValueError(f"target {target_id} has no targets row")
                from_state = current["state"]
                # Any state -> failed is valid (ANY_TARGET_TRANSITIONS); the
                # NEW reason string names the cause without inventing a new
                # state (precedent: phase1's unhandled_error_phase1 — an
                # operator can tell the stages' crashes apart from
                # state_transitions.reason alone).
                transition(
                    conn, target_id=target_id, from_state=from_state, to_state="failed",
                    reason="unhandled_error_draft", actor="system",
                    run_id=run_id, step_id=step_id,
                )
            except Exception as bookkeeping_exc:
                # ---- Second guard (transition half): bookkeeping must not kill the batch ----
                # If the DB connection is what broke, this guard keeps the
                # loop moving to the next target instead of dying inside its
                # own error handling.  The message repeats the ORIGINAL
                # exception so a cleanup failure can never mask the real
                # cause.
                print(
                    f"ERROR: could not mark target {target_id} failed after "
                    f"({error_type}: {error_message}) — transition also failed: {bookkeeping_exc}",
                    file=sys.stderr,
                )
            try:
                # Second guard (log half), separate from the transition: even
                # if the state change could not be written, the step row
                # still gets its own best-effort attempt — a crashed target
                # must leave SOME trace in the audit trail (never skip logs).
                log_step(
                    conn, run_id=run_id, step_id=step_id, target_id=target_id,
                    tool_name="draft_target_run",
                    agent_id="system",  # deterministic CLI code — the registered system agent
                    input_data={"stage": "draft_target_run"},
                    # output_data carries the exception type name and message
                    # so the trace shows WHAT killed the target, not just
                    # that it died.
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
    # Count how many targets reached awaiting_review (the only successful
    # draft outcome) vs were refused/failed, so the operator can see at a
    # glance how many need review.
    reviewed_count = sum(1 for s in results.values() if s == "awaiting_review")
    # Denominator is len(target_ids), NOT len(results): crashed targets are
    # deliberately absent from results, so len(results) would silently print
    # e.g. "9/9" after a crash ate one target — the summary must not lie
    # about how much of the batch actually completed (the B1f lesson).
    print(
        f"Draft run {run_id} complete. "
        f"{reviewed_count}/{len(target_ids)} targets reached awaiting_review.",
    )
    for target_id, outcome in results.items():
        print(f"  {target_id}: {outcome}")
    # Crashed targets get their own lines, marked CRASHED with the exception
    # type and message — visually distinct from an ordinary outcome so a
    # lost target can never hide as a normal result in the summary.
    for target_id, error in crashed.items():
        print(f"  {target_id}: CRASHED ({error})")

    conn.close()  # explicit close — though CPython would close on exit, be explicit
    # Non-zero exit if ANY target crashed: a batch that lost a target to an
    # unhandled error is not a clean run and must not look like one to a
    # script or a human checking $?.  Ordinary outcomes (awaiting_review,
    # not_draftable, policy_denied, scored, failed) do NOT change the exit
    # code — those are normal draft-run results, not batch damage.
    return 1 if crashed else 0


# Guard so `python app/draft_cli.py` also works, not just `python -m app.draft_cli`.
# Uses SystemExit instead of sys.exit() to stay testable (pytest can catch SystemExit).
if __name__ == "__main__":
    raise SystemExit(main())
