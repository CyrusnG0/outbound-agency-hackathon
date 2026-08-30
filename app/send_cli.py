# app/send_cli.py -- Phase 1b DRY_RUN send entry point (operator-facing)
# This module is the single CLI command the operator runs to send every
# target the review gate left in state "approved".  It exists so every
# send attempt goes through one auditable code path — the send gate
# (app/send_gate.py) and the DRY_RUN send (app/tools/send_email.py) —
# rather than ad-hoc scripts.
#
# THE ABSOLUTE RULE: this CLI CANNOT send a real email, by construction.
# There is no mode flag, no --live switch, no transport import anywhere in
# the package.  Every "send" writes data/outbox/{message_id}.eml and
# transitions approved -> dry_run_sent.  That is the whole of it.
import argparse  # stdlib argument parser — no new dependency for the operator
import sys  # stderr for error messages, argv for the default None sentinel

from app.agents_registry import seed_agent_registry  # registers the system/operator/judge/draft principals the write gate checks
from app.db import apply_schema, connect  # opens the DB and applies the DDL (idempotent)
from app.ids import new_id  # generates unique prefixed IDs for the run and each step
from app.state_machine import transition  # the single state-change gate — a crashed target is marked failed through it, never a raw UPDATE
from app.tools.log_step import log_step  # steps-table trace writer — a crashed target still gets its step row (never skip logs)
from app.tools.send_email import DEFAULT_OUTBOX_DIR, send_email  # the DRY_RUN send — the ONLY send in the repo


def main(argv: list[str] | None = None) -> int:
    # ----- Parse CLI args -----
    # Deliberately NO mode flag: there is nothing to switch to.  The three
    # args are the run's inputs, not a safety dial — safety is structural
    # (no transport exists to enable), never configurational.
    parser = argparse.ArgumentParser(prog="python -m app.send_cli")
    parser.add_argument("--db", default="data/outbound.db")  # main operational DB
    parser.add_argument(
        "--outbox", default=DEFAULT_OUTBOX_DIR,
        help="directory the DRY_RUN .eml artifacts are written to (data/outbox/ by default)",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="maximum number of approved targets to dry-run send in this run",  # self-use conservative default: the operator sends a handful at a time
    )
    args = parser.parse_args(argv)
    if args.limit < 1:
        # A non-positive limit is a wiring mistake — refuse before any DB
        # I/O rather than silently selecting nothing (or everything).
        print("ERROR: --limit must be a positive integer.", file=sys.stderr)
        return 1

    # ----- Open DB and apply schema -----
    conn = connect(args.db)  # app.db.Conn — sqlite file path or postgresql:// / cloudsql:// URL
    apply_schema(conn)  # idempotent DDL; the send-gate tables already exist in the DDL

    # ----- Seed the agent registry so the write gate accepts this run -----
    # Every send write in this run carries agent_id="system", and the write
    # gate refuses writes from unregistered agents.  The seed is idempotent
    # (upsert), so running it on every invocation is safe and cheap — the
    # same startup sequence draft_cli uses.
    run_id = new_id("run")  # unique run identifier ties together all steps in this invocation
    seed_agent_registry(conn, run_id=run_id, step_id=new_id("step"))

    # ----- Select the targets this run will send -----
    # The state machine's only inbound edge to dry_run_sent is
    # approved -> dry_run_sent, so "approved" is exactly the eligible set.
    # The full preflight is enforced PER TARGET inside send_email, not
    # pre-filtered here — a refused target still gets its logged outcome
    # line and its send_gate_decisions row, and a silent pre-filter would
    # hide refusals from the operator.
    target_ids = [
        row["target_id"]
        for row in conn.execute(
            "SELECT target_id FROM targets WHERE state='approved' "
            "ORDER BY created_at LIMIT ?;",
            (args.limit,),
        ).fetchall()
    ]

    # ----- Log the batch manifest BEFORE the per-target loop -----
    # The full target list this run intends to send is known up front, so
    # log it as one batch-level step (target_id NULL — the same convention
    # get_targets uses for phase1_cli's manifest).  The live run view
    # (app/console/app.py) reads this manifest to know the run's FULL
    # intended target set: a target the loop has not reached yet has zero
    # per-target steps/transitions and would otherwise be invisible to a
    # rows-derived completeness check, silently declaring the run complete
    # mid-batch (ticket U2-fix2 — the 12s quiet-period heuristic alone is
    # not safe against this pipeline's ~605s worst-case per-node latency).
    # Logged EVEN when the batch is empty: an empty manifest is a legitimate
    # result (the run had nothing to send), never a skipped log.
    log_step(
        conn, run_id=run_id, step_id=new_id("step"), target_id=None,
        tool_name="send_batch_manifest", agent_id="system",
        input_data={"stage": "send_batch_manifest"},
        output_data={"target_ids": target_ids},
        status="success",
    )

    results: dict[str, str] = {}  # target_id -> outcome for targets the send run concluded normally
    # target_id -> "ExceptionType: message" for targets killed by an
    # unhandled exception.  Kept separate from results so the summary can
    # mark a crash distinctly from an ordinary outcome — a lost target must
    # never look like a normal result (the B1f lesson, re-applied).
    crashed: dict[str, str] = {}
    for target_id in target_ids:
        try:
            # Run the gate + DRY_RUN send for this target.  Returns the
            # outcome: refused (with reasons) or dry_run_sent.
            result = send_email(
                conn, target_id=target_id, run_id=run_id,
                outbox_dir=args.outbox,  # the operator's chosen outbox — the artifacts land where they looked
            )
            if result.refused:
                results[target_id] = f"refused ({result.refusal_reason})"
            else:
                results[target_id] = f"dry_run_sent -> {result.outbox_path}"
        except Exception as exc:
            # ---- Crash containment: one target's crash must never abort the batch ----
            # (ticket B1f's per-target isolation, copied from phase1_cli /
            # draft_cli: a 10-target run died at target 4 and six targets
            # were left stranded.)  Catch Exception, NOT BaseException:
            # KeyboardInterrupt and SystemExit must still propagate — a
            # swallowed Ctrl-C would leave a batch the operator believes is
            # stopped silently continuing.
            error_type = type(exc).__name__  # type NAME, not str(exc): makes a transport error distinguishable from a KeyError in our own code
            error_message = str(exc)
            # Print the ORIGINAL error immediately, before any bookkeeping,
            # so the operator always learns the real cause.
            print(
                f"ERROR: target {target_id} crashed during send — "
                f"{error_type}: {error_message}",
                file=sys.stderr,
            )
            crashed[target_id] = f"{error_type}: {error_message}"
            # One fresh step id shared by the transition and the log_step row
            # below — the same pattern draft_cli's failure path uses, so this
            # crash's audit entries hang together under one step.
            step_id = new_id("step")
            try:
                # A crash can happen at ANY point (gate evaluation, file
                # write, row write, transition), so READ the target's
                # current state from the DB instead of hardcoding "approved"
                # — the state_transitions row must record where the target
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
                # state (precedent: unhandled_error_phase1 /
                # unhandled_error_draft — an operator can tell the stages'
                # crashes apart from state_transitions.reason alone).
                transition(
                    conn, target_id=target_id, from_state=from_state, to_state="failed",
                    reason="unhandled_error_send", actor="system",
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
                    tool_name="send_target_run",
                    agent_id="system",  # deterministic CLI code — the registered system agent
                    input_data={"stage": "send_target_run", "simulated": True},
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
    # Count how many targets reached dry_run_sent (the only successful send
    # outcome) vs were refused, so the operator can see at a glance how
    # many artifacts landed in the outbox.
    sent_count = sum(1 for s in results.values() if s.startswith("dry_run_sent"))
    # Denominator is len(target_ids), NOT len(results): crashed targets are
    # deliberately absent from results, so len(results) would silently print
    # e.g. "9/9" after a crash ate one target — the summary must not lie
    # about how much of the batch actually completed (the B1f lesson).
    print(
        f"Send run {run_id} complete (DRY_RUN — no email was sent). "
        f"{sent_count}/{len(target_ids)} targets dry-run-sent.",
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
    # script or a human checking $?.  Ordinary outcomes (dry_run_sent,
    # refused) do NOT change the exit code — those are normal send-run
    # results, not batch damage.
    return 1 if crashed else 0


# Guard so `python app/send_cli.py` also works, not just `python -m app.send_cli`.
# Uses SystemExit instead of sys.exit() to stay testable (pytest can catch SystemExit).
if __name__ == "__main__":
    raise SystemExit(main())
