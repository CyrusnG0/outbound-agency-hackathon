# app/autonomous_taskmaster.py -- the bounded autonomous outer loop (2026-08-31)
#
# WHY THIS EXISTS: PHASE1_TARGET_TIMEOUT_SECONDS bounds one whole
# `taskmaster_cli` invocation (app/taskmaster_cli.py's asyncio.wait_for), not
# one target inside it -- a real architectural mismatch this build hit live,
# twice, on a 10-company batch. Raising the ceiling only delays the same
# failure onto a bigger batch; it does not remove it, and CLAUDE.md's
# "retries must be bounded" rule means an unbounded ceiling is off the table
# regardless. The actual fix is to stop treating one invocation as the whole
# job: call the Taskmaster repeatedly, each call bounded by the SAME 600s
# ceiling as always, and let resume_pending_research/report_pipeline_status
# (already built for exactly this) pick up wherever the last call stopped.
#
# THE GOVERNANCE SPLIT, EXTENDED TO "ARE WE DONE": docs/agents.md's central
# claim is "the LLM's judgement decides the outcome; deterministic code
# performs the action." This module applies the same split to loop
# termination. Each iteration, the model is TOLD what to try (research
# anything stuck at 'new', draft anything scored) -- but whether the RUN
# has finished is decided here, in code, by directly reading the same
# selector/precondition functions the stage tools themselves select
# through (select_research_pending_targets, select_draft_eligible_targets,
# has_allow_policy_decision -- the last one added 2026-09-01, ticket H4, so
# a permanently policy_denied target reads as "done", not "still pending"
# forever). A model's own "I think I'm done" is not proof of anything; an
# empty selector result is. This mirrors the "structured outputs over free-form
# reasoning" priority in CLAUDE.md #1.
#
# BOUNDED, NOT "LOOP FOREVER": CLAUDE.md's golden rules say retries must be
# bounded. --max-iterations caps the loop (default 30 iterations * up to
# 600s each = up to 5 hours of wall clock) so a batch that can genuinely
# never finish (a crashing offer config, a permanently broken target) fails
# LOUDLY and says so, instead of spinning silently forever. In the ordinary
# case -- a batch that just ran out of wall-clock time mid-research -- this
# finishes in 1-3 iterations, most of which return long before the ceiling.
#
# NO NEW WRITE PATH: this module writes nothing to the database itself. It
# only (a) calls the EXISTING, tested `app.taskmaster_cli.main` entry point,
# unmodified, once per iteration, and (b) opens short-lived read-only
# connections to run the two existing selector queries. Every actual write
# still goes through write_gate.commit / state_machine.transition inside the
# stage tools, exactly as it always did -- this file adds a stopping
# condition around an unmodified pipeline, not a new capability.
import argparse  # stdlib argument parser -- no new dependency for the operator
import sys  # stderr for the final bound-exceeded message
import time  # time.monotonic() for the elapsed-time report -- never wall-clock time, which can jump

from app.agents.draft import has_allow_policy_decision, select_draft_eligible_targets  # the draft stage's own "who is eligible" / "will policy let this through" sources of truth -- never re-derived here
from app.agents.phase1 import select_research_pending_targets  # the research stage's own "who is stuck at new" source of truth -- same reuse discipline
from app.db import apply_schema, connect  # opens the DB and applies the (idempotent) DDL, exactly like every CLI entry point
from app.taskmaster_cli import main as taskmaster_main  # the UNMODIFIED per-invocation entry point this module loops -- reused, not reimplemented

# Default per-call task: tells the model exactly which two tools to check
# and in which order, so it is never guessing what "resume" means. Mirrors
# the wording of _TASKMASTER_INSTRUCTION's own "RECOVERING A STUCK BATCH"
# paragraph (app/agents/taskmaster.py) rather than inventing new phrasing.
DEFAULT_TASK = (
    "Call report_pipeline_status first. If any targets are stuck at state "
    "'new', call resume_pending_research to research them -- do NOT call "
    "import_and_research again on any CSV. If any targets are at state "
    "'scored' (or 'routed' with a queued follow-up), call draft_for_scored. "
    "Report exactly what changed and what each tool returned."
)

# The bounded loop cap. 30 * up to 600s (the PHASE1_TARGET_TIMEOUT_SECONDS
# default) is ~5 hours of worst-case wall clock -- generous enough that a
# genuinely recoverable batch always finishes well inside it, but a real
# number, not "None" (CLAUDE.md: retries must be bounded, no exceptions
# carved out for "hopefully no more errors").
DEFAULT_MAX_ITERATIONS = 30


def _pending_count(db_path: str, *, offers_dir: str) -> tuple[int, int]:
    """Read (never write) how many targets the Taskmaster's own recovery
    tools would still find work in, right now.

    Opens and closes its OWN short-lived connection rather than sharing one
    across the whole loop -- the same "open, act, close" discipline every
    stage CLI already follows -- so a long-idle connection can never be the
    thing that goes stale mid-loop. `offers_dir` is accepted for call-site
    symmetry with the rest of this module even though neither selector
    reads it; it is not otherwise used here.

    The draft count is further filtered to targets has_allow_policy_decision
    actually lets through (ticket H4, 2026-09-01): select_draft_eligible_targets
    alone answers "who is in state='scored' (or routed-with-follow-up)",
    which stays true FOREVER for a policy_denied target -- a refusal is not
    a state transition, by design (CLAUDE.md: refusals must surface to the
    operator, never be silently pre-filtered away in the selection query
    itself). Left unfiltered, this stopping check would report the same
    permanently-refused targets as "still pending" every iteration, and the
    loop would burn real Gemini calls re-discovering the same refusal up to
    the full --max-iterations bound instead of recognising "nothing left it
    can do" after the first pass. A target excluded here for a bad policy
    decision needs fresh research (a new policy_decisions row), which is
    outside what this task's draft-only recovery can fix -- exactly why it
    must count as NOT pending rather than trigger another iteration.
    """
    del offers_dir  # unused -- kept in the signature for symmetry, not silently dropped without a name
    conn = connect(db_path)
    apply_schema(conn)
    try:
        # limit=1000: this is a COUNT-style read for the stopping decision,
        # not a batch about to be processed -- the real MAX_BATCH_SIZE cap
        # still applies INSIDE resume_pending_research/draft_for_scored
        # themselves, unaffected by this number.
        pending_research = len(select_research_pending_targets(conn, limit=1000))
        draft_eligible = select_draft_eligible_targets(conn, limit=1000)
        # Narrow "eligible by state" down to "can actually still succeed" --
        # the same has_allow_policy_decision the draft stage's own
        # precondition 2 enforces, so this can never drift from what
        # draft_for_scored would really do with these targets.
        pending_draft = sum(
            1 for target_id in draft_eligible if has_allow_policy_decision(conn, target_id)
        )
        return pending_research, pending_draft
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    # ----- Parse CLI args -----
    parser = argparse.ArgumentParser(prog="python -m app.autonomous_taskmaster")
    parser.add_argument("--db", default="data/outbound.db")  # same default every other CLI in this repo uses
    parser.add_argument("--offers-dir", default="config/offers")  # forwarded to each taskmaster_cli invocation unchanged
    parser.add_argument(
        "--task", default=DEFAULT_TASK,
        help="the natural-language task sent to the Taskmaster EACH iteration (default: check status, resume research, draft the scored)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"bounded loop cap -- never truly infinite (default {DEFAULT_MAX_ITERATIONS})",
    )
    args = parser.parse_args(argv)

    start = time.monotonic()  # elapsed-time reporting only -- never used for the stopping decision itself, which is state-based, not clock-based

    # ----- Check BEFORE spending a single model call -----
    # If nothing is pending at all, there is nothing to loop for -- report
    # that honestly and exit, rather than paying for a Taskmaster
    # invocation that would just report the same "nothing to do" itself.
    pending_research, pending_draft = _pending_count(args.db, offers_dir=args.offers_dir)
    if pending_research == 0 and pending_draft == 0:
        print("AUTONOMOUS RUN COMPLETE -- nothing was pending at start (nothing stuck at 'new', nothing draft-eligible).")
        return 0

    # ----- The bounded loop -----
    for iteration in range(1, args.max_iterations + 1):
        print(
            f"\n=== autonomous_taskmaster iteration {iteration}/{args.max_iterations} "
            f"-- {pending_research} stuck at 'new', {pending_draft} draft-eligible ==="
        )
        # One full taskmaster_cli invocation, UNCHANGED -- same 600s
        # per-invocation ceiling, same report-to-stdout behaviour, same
        # exit-code convention (0 = ran and reported, including refusals;
        # 1 = crashed or timed out with no report).
        exit_code = taskmaster_main([
            "--task", args.task,
            "--db", args.db,
            "--offers-dir", args.offers_dir,
        ])
        if exit_code != 0:
            # A non-zero exit here is expected and NOT fatal to the outer
            # loop -- almost always the same per-invocation wall-clock
            # ceiling this whole module exists to work around. The next
            # iteration's report_pipeline_status call picks up exactly
            # where this one stopped; nothing is lost because every
            # completed target's state was already committed through the
            # write gate before the ceiling fired.
            print(
                f"iteration {iteration}: the Taskmaster invocation returned a non-zero exit "
                f"({exit_code}), most likely its own 600s wall-clock ceiling. Re-checking "
                f"pipeline state and continuing -- the next iteration resumes from wherever "
                f"this one stopped."
            )

        # ----- The deterministic stopping check -----
        pending_research, pending_draft = _pending_count(args.db, offers_dir=args.offers_dir)
        if pending_research == 0 and pending_draft == 0:
            elapsed_min = (time.monotonic() - start) / 60
            print(
                f"\nAUTONOMOUS RUN COMPLETE after {iteration} iteration(s), ~{elapsed_min:.1f} min. "
                f"Nothing left stuck at 'new' or draft-eligible. Remaining work (review, send, "
                f"replies) sits behind the human-review gate by design -- check /review/queue."
            )
            return 0

    # ----- Bound exceeded: fail loudly, never fail silently -----
    print(
        f"\nBOUND EXCEEDED: still {pending_research} stuck at 'new' and {pending_draft} "
        f"draft-eligible after {args.max_iterations} iterations. This batch is not "
        f"auto-completable within this bound -- investigate manually (a crashing target or "
        f"a broken offer config are the usual causes) rather than raising --max-iterations "
        f"further without knowing why.",
        file=sys.stderr,
    )
    return 1


# Guard so `python app/autonomous_taskmaster.py` also works, not just
# `python -m app.autonomous_taskmaster`. SystemExit (not sys.exit()) keeps
# this testable -- pytest can catch SystemExit, matching taskmaster_cli.py's
# own convention.
if __name__ == "__main__":
    raise SystemExit(main())
