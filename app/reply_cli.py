# app/reply_cli.py -- the reply-half entry point (operator-facing)
# This module is the single CLI command the operator runs to process the
# simulated inbox: fetch every .eml in data/inbox/, thread it to the
# messages B5 wrote, record a replies row, then classify each new reply
# and apply the routing decision (app/agents/reply.py).  It exists so
# every inbound message goes through one auditable code path — the
# simulated fetch (app/tools/fetch_inbox.py) and the classifier+router —
# rather than ad-hoc scripts.
#
# THE ABSOLUTE RULE: this CLI CANNOT read a real mailbox, by construction.
# There is no mode flag, no --live switch, no IMAP/POP import anywhere in
# the package.  Reading .eml files off disk IS the fetch.  And no reply,
# of any class, can trigger a send: the reply stage has no outbound code
# path at all — the only auto side effect that exists is the unsubscribe
# suppression, an inbound de-escalation.
import argparse  # stdlib argument parser — no new dependency for the operator
import sys  # stderr for error messages, argv for the default None sentinel

from app.agents.reply import build_reply_agent, classify_and_route_reply  # the classifier+router and its per-reply runner
from app.agents_registry import seed_agent_registry  # registers the principals (incl. reply_classifier) the write gate checks
from app.db import apply_schema, connect  # opens the DB and applies the DDL (idempotent)
from app.ids import new_id  # generates unique prefixed IDs for the run and each crash step
from app.state_machine import transition  # the single state-change gate — a crashed reply is marked failed through it, never a raw UPDATE
from app.tools.fetch_inbox import DEFAULT_INBOX_DIR, fetch_inbox  # the simulated inbox fetch — the ONLY inbox that exists
from app.tools.log_step import log_step  # steps-table trace writer — a crashed reply still gets its step row (never skip logs)


def main(argv: list[str] | None = None) -> int:
    # ----- Parse CLI args -----
    # Deliberately NO mode flag: there is nothing to switch to (the same
    # structural stance as send_cli).  The three args are the run's
    # inputs, not a safety dial — safety is structural (no transport
    # exists to import), never configurational.
    parser = argparse.ArgumentParser(prog="python -m app.reply_cli")
    parser.add_argument("--db", default="data/outbound.db")  # main operational DB
    parser.add_argument(
        "--inbox", default=DEFAULT_INBOX_DIR,
        help="directory the simulated inbox .eml files are read from (data/inbox/ by default)",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="maximum number of inbox messages to process in this run",  # self-use conservative default: the operator triages a handful at a time
    )
    args = parser.parse_args(argv)
    if args.limit < 1:
        # A non-positive limit is a wiring mistake — refuse before any DB
        # I/O rather than silently processing nothing (or everything).
        print("ERROR: --limit must be a positive integer.", file=sys.stderr)
        return 1

    # ----- Open DB and apply schema -----
    conn = connect(args.db)  # app.db.Conn — sqlite file path or postgresql:// / cloudsql:// URL
    apply_schema(conn)  # idempotent DDL; the replies/suppressions tables already exist in the DDL

    # ----- Seed the agent registry so the write gate accepts this run -----
    # Every write in this run carries agent_id="system" (the fetch) or
    # agent_id="reply_classifier" (the router), and the write gate refuses
    # writes from unregistered agents.  The seed is idempotent (upsert),
    # so running it on every invocation is safe and cheap — the same
    # startup sequence send_cli uses.
    run_id = new_id("run")  # unique run identifier ties together all steps in this invocation
    seed_agent_registry(conn, run_id=run_id, step_id=new_id("step"))

    # ----- Fetch the simulated inbox -----
    # fetch_inbox has its OWN per-file isolation (one malformed or
    # unmatchable .eml is logged and skipped, never raised), so this call
    # completes the sweep even with a hostile file in the inbox.  --limit
    # caps how many files the sweep processes (sorted by filename, so the
    # cut is deterministic).
    fetched = fetch_inbox(conn, inbox_dir=args.inbox, run_id=run_id, limit=args.limit)

    # ----- Log the batch manifest BEFORE the per-reply loop -----
    # The full reply batch is known up front (fetch_inbox's replies_created),
    # but the manifest must name TARGETS, not replies — the live run view
    # (app/console/app.py) checks target states, so resolve each reply_id to
    # its target via the same replies→messages join the crash path below
    # uses.  Deduplicated: docs/reply-routing.md §5 allows multiple replies
    # on one thread, so two replies can resolve to the SAME target (e.g. a
    # second reply to the same outbound message); the manifest is the set of
    # targets this run will touch, so each target appears once.  A reply
    # whose target row is gone is skipped — there is no target to manifest
    # (the per-reply loop still logs its own outcome for it).  Logged EVEN
    # when the batch is empty: an empty manifest is a legitimate result (the
    # run had nothing to classify), never a skipped log.
    manifest_target_ids: list[str] = []
    _seen_manifest_targets: set[str] = set()
    for _reply_id in fetched.replies_created:
        _manifest_row = conn.execute(
            "SELECT m.target_id FROM replies r JOIN messages m "
            "ON r.message_id = m.message_id WHERE r.reply_id=?;",
            (_reply_id,),
        ).fetchone()
        if _manifest_row is None:
            continue  # Phantom reply/target — nothing to manifest.
        _manifest_target_id = _manifest_row["target_id"]
        if _manifest_target_id not in _seen_manifest_targets:
            _seen_manifest_targets.add(_manifest_target_id)
            manifest_target_ids.append(_manifest_target_id)
    log_step(
        conn, run_id=run_id, step_id=new_id("step"), target_id=None,
        tool_name="reply_batch_manifest", agent_id="system",
        input_data={"stage": "reply_batch_manifest"},
        output_data={"target_ids": manifest_target_ids},
        status="success",
    )

    # ----- Build the classifier+router once, run it per new reply -----
    # One compiled agent for the whole batch (the router reads reply_id /
    # run_id from session state per run — the draft stage's pattern).
    agent = build_reply_agent(conn)

    results: dict[str, str] = {}  # reply_id -> outcome for replies concluded normally
    # reply_id -> "ExceptionType: message" for replies killed by an
    # unhandled exception.  Kept separate from results so the summary can
    # mark a crash distinctly from an ordinary outcome — a lost reply must
    # never look like a normal result (the B1f lesson, re-applied).
    crashed: dict[str, str] = {}
    for reply_id in fetched.replies_created:
        try:
            # Classify this reply and apply the routing decision.  The
            # outcome names what happened: routed / suppressed /
            # review_required / terminal_no_transition /
            # classification_failed / failed.
            outcome = classify_and_route_reply(agent, conn=conn, reply_id=reply_id, run_id=run_id)
            results[reply_id] = outcome
        except Exception as exc:
            # ---- Crash containment: one reply's crash must never abort the batch ----
            # (ticket B1f's per-target isolation, copied from send_cli: a
            # 10-message run died at message 4 and six messages were left
            # stranded.)  Catch Exception, NOT BaseException:
            # KeyboardInterrupt and SystemExit must still propagate — a
            # swallowed Ctrl-C would leave a batch the operator believes
            # is stopped silently continuing.
            error_type = type(exc).__name__  # type NAME, not str(exc): makes a transport error distinguishable from a KeyError in our own code
            error_message = str(exc)
            # Print the ORIGINAL error immediately, before any
            # bookkeeping, so the operator always learns the real cause.
            print(
                f"ERROR: reply {reply_id} crashed during classification — "
                f"{error_type}: {error_message}",
                file=sys.stderr,
            )
            crashed[reply_id] = f"{error_type}: {error_message}"
            # One fresh step id shared by the transition and the log_step
            # row below — the same pattern send_cli's failure path uses.
            step_id = new_id("step")
            # The reply's target for the failure attribution, read fresh
            # (the reply may link to a message whose target no longer
            # exists — then only the step row is written).
            target_row = conn.execute(
                "SELECT m.target_id FROM replies r JOIN messages m "
                "ON r.message_id = m.message_id WHERE r.reply_id=?;",
                (reply_id,),
            ).fetchone()
            try:
                if target_row is None:
                    # No target to mark failed — skip the transition
                    # (a transition for a phantom target would be a
                    # lying audit row), the step row below still lands.
                    raise ValueError(f"reply {reply_id} has no linked target")
                current = conn.execute(
                    "SELECT state FROM targets WHERE target_id=?;", (target_row["target_id"],)
                ).fetchone()
                if current is None:
                    # Same integrity guard — the reply links to a message
                    # whose target row is gone.
                    raise ValueError(f"target {target_row['target_id']} has no targets row")
                from_state = current["state"]
                # Any state -> failed is valid (ANY_TARGET_TRANSITIONS);
                # the NEW reason string names the cause without inventing
                # a new state (precedent: unhandled_error_send).
                transition(
                    conn, target_id=target_row["target_id"],
                    from_state=from_state, to_state="failed",
                    reason="unhandled_error_reply", actor="system",
                    run_id=run_id, step_id=step_id,
                )
            except Exception as bookkeeping_exc:
                # ---- Second guard (transition half): bookkeeping must not kill the batch ----
                # If the DB connection is what broke, this guard keeps
                # the loop moving to the next reply instead of dying
                # inside its own error handling.  The message repeats the
                # ORIGINAL exception so a cleanup failure can never mask
                # the real cause.
                print(
                    f"ERROR: could not mark target for reply {reply_id} failed after "
                    f"({error_type}: {error_message}) — transition also failed: {bookkeeping_exc}",
                    file=sys.stderr,
                )
            try:
                # Second guard (log half), separate from the transition:
                # even if the state change could not be written, the step
                # row still gets its own best-effort attempt — a crashed
                # reply must leave SOME trace in the audit trail (never
                # skip logs).  The payload carries no reply text — only
                # ids and error names (item 18).
                log_step(
                    conn, run_id=run_id, step_id=step_id,
                    target_id=target_row["target_id"] if target_row is not None else None,
                    tool_name="reply_target_run",
                    agent_id="system",  # deterministic CLI code — the registered system agent
                    input_data={"stage": "reply_target_run", "simulated": True, "reply_id": reply_id},
                    output_data={"error_type": error_type, "error_message": error_message},
                    status="failed",
                )
            except Exception as bookkeeping_exc:
                print(
                    f"ERROR: could not log the crash for reply {reply_id} "
                    f"({error_type}: {error_message}) — log_step also failed: {bookkeeping_exc}",
                    file=sys.stderr,
                )

    # ----- Print summary -----
    # Count the notable outcomes so the operator sees at a glance how
    # many replies were routed, suppressed, and sent to review.
    suppressed_count = sum(1 for o in results.values() if o == "suppressed")
    review_count = sum(
        1 for o in results.values() if o in ("review_required", "classification_failed", "unclassified")
    )
    # Denominator is len(fetched.replies_created), NOT len(results):
    # crashed replies are deliberately absent from results, so
    # len(results) would silently print e.g. "9/9" after a crash ate one
    # reply — the summary must not lie (the B1f lesson).
    print(
        f"Reply run {run_id} complete (simulated inbox — no mailbox was "
        f"connected, no email was sent). "
        f"{len(results)}/{len(fetched.replies_created)} replies classified "
        f"({suppressed_count} suppressed, {review_count} review-bound)."
    )
    if fetched.skipped:
        print(f"  Skipped {len(fetched.skipped)} file(s) — see the steps trace for details.")
    for reply_id, outcome in results.items():
        print(f"  {reply_id}: {outcome}")
    # Crashed replies get their own lines, marked CRASHED with the
    # exception type and message — visually distinct from an ordinary
    # outcome so a lost reply can never hide as a normal result.
    for reply_id, error in crashed.items():
        print(f"  {reply_id}: CRASHED ({error})")

    conn.close()  # explicit close — though CPython would close on exit, be explicit
    # Non-zero exit if ANY reply crashed: a batch that lost a reply to an
    # unhandled error is not a clean run and must not look like one to a
    # script or a human checking $?.  Ordinary outcomes (routed,
    # suppressed, review_required, classification_failed) do NOT change
    # the exit code — those are normal reply-run results, not batch
    # damage.
    return 1 if crashed else 0


# Guard so `python app/reply_cli.py` also works, not just `python -m app.reply_cli`.
# Uses SystemExit instead of sys.exit() to stay testable (pytest can catch SystemExit).
if __name__ == "__main__":
    raise SystemExit(main())
