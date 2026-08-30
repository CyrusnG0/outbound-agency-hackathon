#!/usr/bin/env python3
"""The operator's ONLY permitted manual suppression add/remove path (ticket H4b).

``suppression-policy.md`` §2 and ``runbook.md`` §4 mandate that manual
suppression additions and removals happen ONLY through this script, and that
direct DB edits are forbidden.  Before H4b the script did not exist — the docs
described a writer that was never built, so the operator had no permitted
manual suppression path at all (the C2-era F1b audit finding).  This module is
that writer.

TWO MODES, mutually exclusive (giving BOTH --email and --domain is refused —
the script's idempotency contract is one key per invocation):

    python scripts/add_suppression.py --email a@b.test [--reason manual] [--notes "..."]
    python scripts/add_suppression.py --domain b.test    [--reason manual] [--notes "..."]
    python scripts/add_suppression.py --email a@b.test --remove
    python scripts/add_suppression.py --domain b.test --remove

DESIGN RULES (ticket H4b §2, each enforced in the code below):

- The address is stored AS WRITTEN — the audit record of what arrived
  (``suppression-policy.md`` §1a) — and matched via the ONE shared normaliser
  (``app.db.normalize_email`` / ``normalize_domain``).  There is deliberately
  NO second folding implementation in this file: duplicated folding logic is
  exactly how the C2 suppression-evasion breach happened, and one definition
  is what keeps it from coming back.
- Every write goes through ``write_gate.commit()`` — action
  ``insert_suppression`` for an add, ``delete_suppression`` for a removal,
  ``actor="operator"``, ``agent_id="operator"``.  No raw INSERT/DELETE.
- Idempotent like the existing suppression writers (``app/review.py`` and
  ``app/agents/reply.py`` both check-then-insert): re-adding an existing key
  is a logged no-op, and removing a missing key is a logged no-op.
- ``--remove`` is the explicit operator flag ``gates.md`` §1.2 demands for a
  suppression removal.  Enforcement lives IN the write gate (ticket H8):
  ``commit()`` refuses ``delete_suppression`` unless ``operator_confirmed=True``
  AND ``actor="operator"``.  The script passes both below; its own ``--remove``
  requirement stays as defence in depth (and because the CLI's check gives a
  better error message than the gate's).
- Every operation is logged with ``log_step`` (Golden Rule: never skip logs).
"""

import argparse  # stdlib argument parsing — no new dependency for the operator
import sys  # stderr for error messages, argv for the default None sentinel
from pathlib import Path  # resolving the repo root for the sys.path bootstrap below

# ── sys.path bootstrap ───────────────────────────────────────────────────────
# This file lives in scripts/, but the repo's code lives in app/.  When run
# directly (`python scripts/add_suppression.py`) Python puts scripts/ — not the
# repo root — at sys.path[0], so `import app` would fail.  Inserting the repo
# root (scripts/'s parent) makes the script runnable exactly as the docs tell
# the operator to run it.  The `__file__`-relative resolve keeps this correct
# regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents_registry import seed_agent_registry  # registers the operator principal the write gate checks
from app.db import (  # the DB layer: connect() for the target, apply_schema() for the DDL/migrations
    apply_schema,
    connect,
    normalize_domain,  # THE shared domain fold (lowercase — RFC 1035); never a second copy here
    normalize_email,  # THE shared address fold (case + plus-tags); never a second copy here
)
from app.ids import new_id  # fresh prefixed ids: one run id, one step id per invocation
from app.tools.log_step import log_step  # steps-trace writer — every operation lands in the trace (Golden Rule)
from app.write_gate import commit as write_gate_commit  # THE core-table write path — never a raw INSERT/DELETE

# ── Constants ─────────────────────────────────────────────────────────────────

# The reason vocabulary pinned by the suppressions table's CHECK constraint
# (app/db.py _SUPPRESSIONS_DDL).  The argparse `choices` below validates
# against this exact list, so a bad --reason is refused BEFORE any DB I/O.
VALID_REASONS = ("unsubscribe", "bounce", "complaint", "manual", "legal", "risky_reply")

# The default database target — the operator's real database, the same
# default every operator-facing stage CLI (phase1_cli / draft_cli / send_cli /
# reply_cli) uses.  The script manages the REAL suppression list, so the real
# DB is the intended default, not an accident.
DEFAULT_DB = "data/outbound.db"

# The steps.tool_name carried by add operations — distinct from removals so
# the trace log tells an addition apart from a removal at a glance.
ADD_TOOL_NAME = "add_suppression"

# The steps.tool_name carried by removal operations — see ADD_TOOL_NAME.
REMOVE_TOOL_NAME = "remove_suppression"


# ── Lookup helper (shared by add and remove) ─────────────────────────────────


def _find_suppression(conn, *, email_normalized: str | None, domain: str | None):
    """Return the existing suppression row matching the canonical key(s), or
    None.  The lookup uses the SAME matching keys every reader uses
    (app/send_gate.py's email_normalized / domain probes), so the script's
    idempotency check and the send gate's refusal check can never disagree
    about what counts as "already suppressed"."""
    if email_normalized is not None:
        # Address mode: the matching key is the normalised address — the
        # F1b canonical form, so a re-add of any casing/plus-tag spelling
        # of the same mailbox is a no-op.
        return conn.execute(
            "SELECT email, domain, reason FROM suppressions "
            "WHERE email_normalized=?;",
            (email_normalized,),
        ).fetchone()
    if domain is not None:
        # Domain mode: the matching key is the lowercased domain (the send
        # gate probes with normalize_domain() too).
        return conn.execute(
            "SELECT email, domain, reason FROM suppressions "
            "WHERE domain=?;",
            (domain,),
        ).fetchone()
    return None  # Neither key given — the caller (main) refuses this earlier.


# ── The add path ─────────────────────────────────────────────────────────────


def _add(conn, args, *, run_id: str, step_id: str) -> int:
    """Add one suppression row through the write gate.  Idempotent: if the
    canonical key already exists, print a no-op notice and write nothing."""
    # Compute the row's three identity columns.  The address is stored AS
    # WRITTEN (email) and the matching key (email_normalized) is the shared
    # helper's product; a domain row has no address, so email and
    # email_normalized are NULL and only domain is set.
    email = args.email if args.email else None
    email_normalized = normalize_email(email) if email else None
    domain = normalize_domain(args.domain) if args.domain else None

    # The idempotency check (the app/review.py / app/agents/reply.py
    # precedent): read before you write, so a re-add never relies on the
    # UNIQUE constraint raising.  The row the operator wants already exists
    # — the goal (this address/domain can never be mailed) is already true.
    existing = _find_suppression(
        conn, email_normalized=email_normalized, domain=domain
    )
    if existing is not None:
        print(
            f"already suppressed (email_normalized={email_normalized!r}, "
            f"domain={domain!r}) — no-op"
        )
        return 0

    # The gated INSERT.  record_id is the row's natural identity — the
    # address AS WRITTEN for an address row, "domain:<domain>" for a
    # domain-only row (there is no email to name it by).  actor/agent_id are
    # both "operator": the write gate's actor allowlist and the registry's
    # operator principal, so the write_log row is attributable to the human.
    write_gate_commit(
        conn,
        action="insert_suppression",  # B4b's existing action — REUSED, not a new one (the ticket's explicit instruction)
        table_name="suppressions",
        record_id=email if email is not None else f"domain:{domain}",
        payload={
            "email": email,  # the address as written (audit record)
            "email_normalized": email_normalized,  # the canonical key
            "domain": domain,  # the lowercased domain, when a domain row
            "reason": args.reason,  # the CHECK-constrained reason
            "notes": args.notes,  # the operator's optional context
            "added_by": "operator",  # the CHECK-constrained added_by vocabulary
        },
        run_id=run_id,
        step_id=step_id,
        actor="operator",  # the human performs this write
        agent_id="operator",  # attributed to the registered operator principal
        sql="""
            INSERT INTO suppressions (email, email_normalized, domain, reason, added_at, added_by, notes)
            VALUES (?,?,?,?,datetime('now'),?,?)
        """,
        params=(
            email,  # the address as written — preserved, never overwritten
            email_normalized,  # the shared normaliser's canonical key
            domain,  # NULL for an address row, the lowercased domain otherwise
            args.reason,  # the CHECK-constrained reason vocabulary
            "operator",  # the CHECK-constrained added_by vocabulary
            args.notes,  # the operator's optional context — NULL is "nothing recorded"
        ),
    )

    # The trace row — every operation is logged (Golden Rule), with the
    # input (what was suppressed) and the output (the reason/notes) so the
    # steps table answers "what did the operator add, and why".
    log_step(
        conn,
        run_id=run_id,
        step_id=step_id,
        target_id=None,  # a suppression is not target-scoped — no target to name
        tool_name=ADD_TOOL_NAME,
        agent_id="operator",  # the operator performed this step
        input_data={"email": email, "domain": domain},
        output_data={"reason": args.reason, "notes": args.notes},
        status="success",
    )
    print(
        f"suppressed: email={email!r} domain={domain!r} "
        f"reason={args.reason!r}"
    )
    return 0


# ── The removal path ─────────────────────────────────────────────────────────


def _remove(conn, args, *, run_id: str, step_id: str) -> int:
    """Remove one suppression row through the write gate.  Idempotent: if the
    canonical key does not exist, print a no-op notice and write nothing.

    ``--remove`` is the explicit operator flag gates.md §1.2 demands for a
    suppression removal.  Enforcement now lives IN the write gate (ticket
    H8): commit() refuses delete_suppression unless operator_confirmed=True
    AND actor="operator" — the gate's check fires before any SQL, so a
    removal can never slip through attributed to the wrong principal or a
    caller that forgot the flag.  The script passes operator_confirmed=True
    on the gated DELETE below; main() still requires --remove before this
    function is ever called, which stays as defence in depth and because the
    CLI's check gives the operator a better error message than the gate's.
    """
    # Compute the canonical key(s) exactly like the add path, so a removal
    # finds the same row the add would have matched (any casing/plus-tag
    # spelling of a suppressed mailbox removes it).
    email_normalized = normalize_email(args.email) if args.email else None
    domain = normalize_domain(args.domain) if args.domain else None

    # The idempotency check for a removal: if nothing matches, the operator's
    # goal (this suppression is gone) is already true — a logged no-op, not
    # an error.
    existing = _find_suppression(
        conn, email_normalized=email_normalized, domain=domain
    )
    if existing is None:
        print(
            f"no suppression to remove (email_normalized={email_normalized!r}, "
            f"domain={domain!r}) — no-op"
        )
        return 0

    # The row to delete, keyed by the canonical column so the DELETE removes
    # exactly the row the lookup found.  email_normalized and domain are both
    # UNIQUE, so at most one row matches.
    if email_normalized is not None:
        delete_sql = "DELETE FROM suppressions WHERE email_normalized=?;"
        delete_params = (email_normalized,)
    else:
        delete_sql = "DELETE FROM suppressions WHERE domain=?;"
        delete_params = (domain,)

    # The gated DELETE.  The payload carries what was removed (including the
    # row's recorded reason, so the audit trail shows the removed row's
    # provenance even after the row itself is gone).
    write_gate_commit(
        conn,
        action="delete_suppression",  # H4b's removal action — registered in KNOWN_ACTIONS, documented in gates.md §1.2
        table_name="suppressions",
        record_id=args.email if args.email else f"domain:{domain}",
        payload={
            "email_normalized": email_normalized,  # the canonical key deleted
            "domain": domain,  # the lowercased domain deleted
            "removed_reason": existing["reason"],  # the deleted row's reason, preserved in the audit
            "notes": args.notes,  # the operator's optional removal note
        },
        run_id=run_id,
        step_id=step_id,
        actor="operator",  # the human performs this write
        agent_id="operator",  # attributed to the registered operator principal
        operator_confirmed=True,  # the explicit operator flag gates.md §1.2 demands — the gate refuses delete_suppression without it (H8)
        sql=delete_sql,
        params=delete_params,
    )

    # The trace row — every operation is logged (Golden Rule).
    log_step(
        conn,
        run_id=run_id,
        step_id=step_id,
        target_id=None,  # a suppression is not target-scoped — no target to name
        tool_name=REMOVE_TOOL_NAME,
        agent_id="operator",  # the operator performed this step
        input_data={"email_normalized": email_normalized, "domain": domain},
        output_data={"removed_reason": existing["reason"], "notes": args.notes},
        status="success",
    )
    print(
        f"removed suppression: email_normalized={email_normalized!r} "
        f"domain={domain!r} (was reason={existing['reason']!r})"
    )
    return 0


# ── The CLI entry point ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Parse the operator's flags and dispatch to the add or remove path.

    Exit codes: 0 on success or a logged no-op; 1 on a refused invocation
    (neither --email nor --domain, or --reason with --remove) or an
    unhandled error (which propagates as a traceback — a failed suppression
    must never look like a successful one).
    """
    parser = argparse.ArgumentParser(
        prog="scripts/add_suppression.py",
        description=(
            "The operator's ONLY permitted manual suppression add/remove "
            "path (suppression-policy.md §2 / runbook.md §4)."
        ),
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=(
            "database target — a sqlite file path or a postgresql:// URL. "
            f"(default: {DEFAULT_DB})"
        ),
    )
    parser.add_argument(
        "--email", default=None,
        help="the address AS WRITTEN to suppress (or, with --remove, the "
             "address whose suppression to remove)",
    )
    parser.add_argument(
        "--domain", default=None,
        help="a domain to suppress (lowercased via the shared normaliser) — "
             "or, with --remove, the domain whose suppression to remove",
    )
    parser.add_argument(
        "--reason", default="manual", choices=VALID_REASONS,
        help="one of the CHECK vocabulary (default: manual)",
    )
    parser.add_argument(
        "--notes", default=None,
        help="optional operator note stored in suppressions.notes / the audit payload",
    )
    parser.add_argument(
        "--remove", action="store_true",
        help="REMOVE the matching suppression instead of adding one — the "
             "explicit operator flag gates.md §1.2 demands for a removal; "
             "the write gate enforces it (ticket H8)",
    )
    args = parser.parse_args(argv)

    # ── Refusal 1: a suppression must suppress SOMETHING ──────────────────
    # Mirrors the table-level CHECK (email IS NOT NULL OR domain IS NOT
    # NULL): a row that suppresses nothing would match nothing on every read
    # path, so refuse it before any DB I/O.
    if not args.email and not args.domain:
        print(
            "ERROR: give --email or --domain — a suppression must suppress "
            "something.",
            file=sys.stderr,
        )
        return 1

    # ── Refusal 2: exactly one key per invocation ─────────────────────────
    # The script's idempotency contract is ONE canonical key per call (the
    # check-then-insert lookup is keyed by email_normalized OR domain).  A
    # row that suppresses both is legal in the schema, but doing it in one
    # call would make the no-op semantics ambiguous (which key already
    # exists?); run the script twice for the two independent suppressions.
    if args.email and args.domain:
        print(
            "ERROR: give --email OR --domain, not both — run the script "
            "twice for an address suppression and a domain suppression.",
            file=sys.stderr,
        )
        return 1

    # ── Refusal 3: --reason belongs to additions only ─────────────────────
    # A removal records no reason (the row is deleted, not re-attributed);
    # accepting a --reason with --remove would let the operator believe it
    # was recorded when it is not.
    if args.remove and args.reason != "manual":
        print(
            "ERROR: --reason is only for additions — a removal records no "
            "reason.",
            file=sys.stderr,
        )
        return 1

    # ── Open the DB and prepare the write path ────────────────────────────
    # connect() accepts a sqlite file path or a postgresql:// / cloudsql://
    # URL.  apply_schema() is idempotent and also runs the H4b suppression
    # migration on a provisioned database, so the script works against the
    # operator's existing database.  seed_agent_registry() registers the
    # operator principal the write gate checks (idempotent upsert).
    conn = connect(args.db)
    try:
        apply_schema(conn)
        # One run id and one step id per invocation, generated fresh so this
        # operation's write_log + steps rows hang together under one audit
        # unit — the same pattern every stage CLI uses.
        run_id = new_id("run")
        step_id = new_id("step")
        # The operator principal must be registered before any gated write;
        # the seed is an idempotent upsert, so running it every time is safe.
        seed_agent_registry(conn, run_id=run_id, step_id=new_id("step"))
        print(f"target database: {args.db}")
        # Dispatch to the add or remove path — the --remove flag IS the
        # operator flag, and main() has already refused every ambiguous
        # invocation above.
        if args.remove:
            return _remove(conn, args, run_id=run_id, step_id=step_id)
        return _add(conn, args, run_id=run_id, step_id=step_id)
    finally:
        # Close explicitly — CPython would close on exit, but be explicit
        # (the same discipline the stage CLIs follow).
        conn.close()


# Guard so `python scripts/add_suppression.py` works, not just `python -m`.
# Uses SystemExit instead of sys.exit() to stay testable (pytest can catch
# SystemExit).
if __name__ == "__main__":
    raise SystemExit(main())
