"""The counterparty simulator (ticket E2): a deterministic, scripted
counterparty that lets the pipeline hold a multi-turn conversation.

WHY THIS MODULE EXISTS — E1 made the follow-up draft possible
(``routed → drafted``), so the system can hold a conversation IN
PRINCIPLE.  Nothing has ever exercised more than one exchange: the demo
seed (D3a) generates exactly one reply per target, and the committed
inbox fixtures thread against nothing.  Multi-turn threads — where a
reply is answered by a follow-up send, which draws a second reply, and
so on — have never been walked through the real pipeline.  This module
provides the counterparty for that walk: a set of PERSONAS, each a
turn-by-turn script of inbound replies, and a ``converse`` command that
advances one thread by one turn so the console visibly moves.

THE HONESTY CONTRACT — the counterparty is SCRIPTED, not modelled: no
LLM, no model call, no randomness, no network.  The same thread state
always produces the same next message (byte-identical, including the
Date header, which is derived from the outbound artifact's Date plus the
turn number of hours — never the wall clock).  Every simulated sender is
on an RFC 2606 reserved, non-routable domain (``.test`` / ``.invalid`` /
``.example`` — the same rule demo_seed enforces), so a fabricated sender
can never collide with a real inbox.  Every reply threads for real:
its ``In-Reply-To`` header carries the actual ``Message-ID`` token of
the outbound ``.eml`` artifact, so the production matcher
(``fetch_inbox._match_by_headers``) links it the same way it links a
real reply.

REUSE, NOT DUPLICATION (ticket §2.1) — the outbox→inbox threading is
demo_seed's, already lead-verified: this module imports
``_compose_reply_bytes`` (the RFC-5322 composer), ``_reserved_domain_of``
and ``RESERVED_TLDS`` (the reserved-domain rule), and ``_guard_violation``
(the real-database guard) from ``app/demo_seed.py`` rather than
re-implementing them.  The only matching logic written here is the
"which convo file has already been processed" check, which applies the
SAME substring rule the production matcher documents (a messages-row id
embedded verbatim inside a Message-ID token) — it is a file-hygiene
read, not a second threading implementation.

STRUCTURAL GUARANTEES (asserted by tests/test_conversation_sim.py):
- no raw core-table write anywhere in this file — every ``conn.execute``
  carries SELECT-only SQL; the simulator's only output is ``.eml`` files
  in the inbox, exactly like demo_seed's replies subcommand;
- no mail transport import (the suite-wide AST test walks this file like
  every other app/ module);
- every generated sender address parses to a reserved TLD (tested).

SAFETY GUARDS — the real-database guard runs BEFORE the database is
opened (imported from demo_seed, so the refusal text and the resolved
path comparison are the same ones D3a verified), and a missing sqlite
file is refused rather than silently created (the simulator reads a
seeded database; it never builds one).

Subcommand:
  converse — advance ONE thread (target) by ONE persona turn: write the
             next scripted inbound ``.eml`` into the inbox, threaded
             against the target's latest outbound artifact.  Run
             ``app.reply_cli`` afterwards to classify it (the one
             billable step — this module never calls a model).
"""

import argparse  # stdlib argument parsing — no new dependency for the operator
import sys  # stderr for refusal messages, argv for the default None sentinel
from datetime import datetime, timedelta, timezone  # the reply Date shift (+N hours), the UTC normalization for naive outbound dates, and the unparseable-Date fallback
from email import policy as email_policy  # the stdlib parsing policy for reading outbox .eml files
from email.parser import BytesParser  # RFC-5322 parsing of outbox .eml bytes — PARSING ONLY, never transport
from email.utils import format_datetime, parsedate_to_datetime  # Date-header arithmetic for the generated replies
from pathlib import Path  # outbox/inbox path handling and the real-database guard's resolution
from typing import Literal  # the refusal-reason vocabulary's type — greppable, stable strings

from pydantic import BaseModel  # ScriptedTurn / TurnResult: structured I/O (CLAUDE.md §7)

from app.db import connect  # opens the (already-seeded) database — read-only usage here
# THE REUSED demo_seed helpers — see the module docstring's REUSE section:
# these are the lead-verified outbox→inbox threading pieces, imported
# rather than duplicated (ticket §2.1 says so explicitly).
from app.demo_seed import (  # the counterparty's own composition vocabulary
    RESERVED_TLDS,
    _compose_reply_bytes,
    _guard_violation,
    _reserved_domain_of,
)

# ── Constants ────────────────────────────────────────────────────────────────

# The inbox filename prefix this module writes.  Its own namespace:
# advancing a turn deletes only files with this prefix (processed-file
# hygiene) and never touches the committed 01..05 fixtures or
# demo_seed's demo_reply_* files.
CONVO_FILE_PREFIX = "convo_"

# The Message-ID token suffix for generated replies — deterministic
# (turn number, reserved domain), and it embeds the matched messages-row
# id verbatim so the production substring rule threads the reply.
CONVO_MESSAGE_ID_MARKER = "convo"

# The fallback To address when the outbound artifact carries no To
# header — the same .invalid placeholder demo_seed uses (never a real
# address, never invented).
_CONVO_TO_FALLBACK = "dry-run@outbound-agency.invalid"


# ── The persona scripts ──────────────────────────────────────────────────────


class ScriptedTurn(BaseModel):
    """One scripted inbound reply: the class the REAL classifier is
    expected to assign (the tests stub the classifier and assert this
    mapping), and the body text (worded so a live classifier — the demo
    path — can find that class unambiguously).

    ``reply_class`` is deliberately NOT constrained to a Literal here:
    the persona script is data, and the classifier's schema is the
    authority on the class vocabulary.  The tests pin each persona's
    classes explicitly, so a typo in this data block fails a test.
    """

    reply_class: str  # the §1 class this turn's text is written to elicit
    body: str  # the reply text — written by this script, no model, no real person


# The seven personas (ticket §2.1 plus the D3 trio), four of them
# multi-turn scripts and three single-turn.  GENERATED DATA BLOCK: one
# comment per persona, not per field — the turn bodies are
# self-explanatory.  Every text is fictional; none contains an email
# address, a phone number, or any invented real person.  The classes
# match docs/reply-routing.md §1 verbatim.
PERSONAS: dict[str, tuple[ScriptedTurn, ...]] = {
    # warms_up — the positive thread: interested, then (after the
    # follow-up) a meeting request, then two more positive turns so the
    # E1 follow-up cap (MAX_FOLLOW_UP_DRAFTS_PER_THREAD = 2) is hit on
    # the fourth turn and the refusal is visible in the trace.
    "warms_up": (
        ScriptedTurn(
            reply_class="positive",
            body=(
                "Hi,\n\nThanks for reaching out — this is actually quite "
                "timely for us right now. Could you send over a bit more "
                "detail on how the automation would work, and what the "
                "setup looks like for a practice of our size?"
            ),
        ),
        ScriptedTurn(
            reply_class="meeting_request",
            body=(
                "Thanks for the details — this sounds right for our "
                "workflow. Yes, let's set up a call. Does Thursday at "
                "3pm work for you? I can also share my availability if "
                "that slot is taken."
            ),
        ),
        ScriptedTurn(
            reply_class="positive",
            body=(
                "Great call, thank you for walking us through it. We are "
                "keen to move forward — could you send over the pricing "
                "and the onboarding steps for a practice of our size?"
            ),
        ),
        ScriptedTurn(
            reply_class="positive",
            body=(
                "Apologies for the slow reply — we are still very keen. "
                "Please send the updated materials whenever they are "
                "ready, and we will take it from there."
            ),
        ),
    ),
    # pushes_back_then_leaves — an objection first (draft_hold, review
    # required), then an unsubscribe on the same thread.  The second
    # reply is the multi-turn case D3a never exercised: the target is
    # already in "routed" when the unsubscribe lands.
    "pushes_back_then_leaves": (
        ScriptedTurn(
            reply_class="objection",
            body=(
                "Interesting, but we already use a vendor for scheduling. "
                "What would switching actually cost us in time and "
                "money? Our budget is tight this quarter, so I need a "
                "clear picture before I take this to the partners."
            ),
        ),
        ScriptedTurn(
            reply_class="unsubscribe",
            body=(
                "On reflection, please stop contacting me. Remove this "
                "address from your mailing list and do not send any "
                "further messages to anyone at this practice. We are not "
                "interested and will not be in the future."
            ),
        ),
    ),
    # goes_legal — a risky reply (data-protection complaint, legal tone),
    # then a second, escalating one on the same thread: P5 must hold at
    # every turn, not just the first.
    "goes_legal": (
        ScriptedTurn(
            reply_class="risky",
            body=(
                "To whom it may concern,\n\nYour unsolicited email raises "
                "serious concerns under data-protection law. We require "
                "that you cease all contact immediately, confirm in "
                "writing that you have deleted any data you hold about "
                "our practice, and provide the contact details of your "
                "data-protection officer.\n\nWe reserve all rights."
            ),
        ),
        ScriptedTurn(
            reply_class="risky",
            body=(
                "Further to our previous message, we have not received "
                "your written confirmation. Our legal counsel will be in "
                "touch; please retain all records of this correspondence "
                "and any data you hold about our practice."
            ),
        ),
    ),
    # stays_vague — ambiguous replies that must land under the P4
    # confidence floor (the tests stub a low confidence and assert the
    # review_required routing; the live classifier would likely assign
    # "unclear" to these same texts).
    "stays_vague": (
        ScriptedTurn(
            reply_class="unclear",
            body=(
                "I'm not sure what this is about — I don't understand "
                "what's being proposed."
            ),
        ),
        ScriptedTurn(
            reply_class="unclear",
            body=(
                "ok thanks"
            ),
        ),
    ),
    # negative — a plain, polite decline ("not interested / not a fit,
    # do not contact again") that is deliberately NOT an unsubscribe
    # demand and NOT a legal threat: it must land on the negative
    # class's close_not_target route at high confidence.
    "negative": (
        ScriptedTurn(
            reply_class="negative",
            body=(
                "Thanks for the message, but this is not something we "
                "need. It is not a fit for how our practice works, and "
                "we are not interested in pursuing it further. Please "
                "do not contact us about this again."
            ),
        ),
    ),
    # not_now — interest deferred, not declined: a clear "no, not at
    # this time" (mid-renewal with another vendor, check back later) —
    # distinct from unclear (it is unambiguous) and from objection (no
    # pushback on cost or value, only timing).
    "not_now": (
        ScriptedTurn(
            reply_class="not_now",
            body=(
                "Thanks for reaching out. We are in the middle of our "
                "renewal with another vendor right now, so this is not "
                "the right time. Check back with us in a few months — "
                "we may be open to looking then."
            ),
        ),
    ),
    # wrong_person — the recipient is not the right contact and points
    # elsewhere (no invented names): a clear wrong-recipient reply, not
    # confusion about the offer itself — distinct from unclear.
    "wrong_person": (
        ScriptedTurn(
            reply_class="wrong_person",
            body=(
                "I don't handle procurement or vendor decisions here. "
                "You will want to reach out to our office manager "
                "instead — she is the person who looks after this kind "
                "of thing for the practice."
            ),
        ),
    ),
}


# ── Structured outcome ───────────────────────────────────────────────────────


class TurnResult(BaseModel):
    """What one ``generate_next_turn`` call produced — the success AND
    the refusal shape (refusals are first-class outcomes, never
    exceptions: the CLI prints them, and a scripted counterparty that
    has nothing to say must say so loudly, not silently write nothing).

    ``refusal_reason`` uses a Literal vocabulary so the CLI and the
    tests share one greppable set of outcome strings.
    """

    target_id: str  # which thread the turn was (or was not) written for
    turn_number: int  # 1-based position in the persona script that this call targeted
    written_path: str | None = None  # the inbox .eml path written; None on refusal
    refusal_reason: Literal[
        "persona_exhausted",  # the thread has consumed every scripted turn — the conversation is over
        "unprocessed_reply_exists",  # a previous convo file has no replies row yet — reply_cli must run first
        "no_contact_email",  # the target's contact carries no address to fabricate a sender from
        "non_reserved_sender",  # the contact's address is not on an RFC 2606 reserved domain — refusing to invent a real-looking sender
        "no_outbound_message",  # the target has no recorded outbound send to thread against
        "no_outbox_artifact",  # the recorded send has no .eml artifact — there is no real Message-ID to answer
        "artifact_unparseable",  # the artifact exists but carries no Message-ID header
    ] = ""


# ── The turn generator ───────────────────────────────────────────────────────


def _recorded_reply_count(conn, target_id: str) -> int:
    """How many inbound replies have been RECORDED for this target —
    the deterministic thread state.  A recorded replies row means the
    reply_cli fetch has already processed that turn, so "the next turn"
    is exactly this count (0-based) into the persona script: the same
    thread state always produces the same next message, and the state
    lives in the audit trail, not in a sidecar file."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM replies r "
        "JOIN messages m ON r.message_id = m.message_id "
        "WHERE m.target_id=?;",
        (target_id,),
    ).fetchone()
    return row["n"]


def _cleanup_processed_files(conn, inbox: Path, target_id: str) -> str | None:
    """Delete this module's convo files for the target whose reply has
    ALREADY been recorded (the file answered a recorded send and its
    replies row exists — deleting it keeps the next reply_cli sweep from
    re-fetching the same turn, the demo's duplicate-row hazard).  Return
    a refusal string when an UNPROCESSED convo file exists (its reply
    has no replies row yet — the operator must run reply_cli before the
    thread may advance, or the turn counter would drift from the audit
    trail).

    The processed check applies the SAME substring rule the production
    matcher documents (fetch_inbox._match_by_headers): the messages-row
    id appears verbatim inside the Message-ID token this module embeds,
    so "did a replies row land for the outbound this file answers" is a
    plain substring test over the In-Reply-To header.
    """
    # The target's outbound rows — the id vocabulary the substring test
    # matches against (a convo file only ever answers one of these).
    outbound_ids = [
        row["message_id"]
        for row in conn.execute(
            "SELECT message_id FROM messages WHERE target_id=? AND direction='outbound';",
            (target_id,),
        ).fetchall()
    ]
    # Only THIS module's files for THIS target — the committed fixtures
    # and demo_seed's demo_reply_* namespace are never touched.
    for path in sorted(inbox.glob(f"{CONVO_FILE_PREFIX}{target_id}_*.eml")):
        msg = BytesParser(policy=email_policy.default).parsebytes(path.read_bytes())
        in_reply_to = msg["In-Reply-To"] or ""  # the header whose token embeds the answered row's id
        # The substring rule, applied to the raw header value: the first
        # outbound id found inside it is the message this file answers.
        matched_id = next((mid for mid in outbound_ids if mid in in_reply_to), None)
        if matched_id is None:
            # The file answers none of this target's recorded sends —
            # not one of ours in practice (the filename prefix scopes
            # it); leave it alone rather than guess.
            continue
        recorded = conn.execute(
            "SELECT 1 FROM replies WHERE message_id=?;", (matched_id,)
        ).fetchone()
        if recorded is None:
            # The reply has NOT been fetched yet — advancing now would
            # desynchronize the turn counter from the audit trail.
            return (
                f"unprocessed convo file {path.name!r} exists for target "
                f"{target_id!r} — run reply_cli first, then advance the thread."
            )
        # Processed: the replies row exists, so the artifact is safely
        # redundant — remove only this file, never anything else.
        path.unlink()
    return None  # no unprocessed file — the thread may advance


def _latest_outbound(conn, target_id: str) -> dict | None:
    """The target's latest recorded outbound send — the message the next
    scripted turn answers.  Ordering is created_at DESC with message_id
    DESC as the deterministic tiebreak (messages has no insert_seq; this
    is the exact convention fetch_inbox._match_by_fallback documents for
    the same tie)."""
    return conn.execute(
        "SELECT message_id FROM messages "
        "WHERE target_id=? AND direction='outbound' "
        "ORDER BY created_at DESC, message_id DESC LIMIT 1;",
        (target_id,),
    ).fetchone()


def generate_next_turn(
    conn,
    *,
    outbox_dir: str,
    inbox_dir: str,
    persona_script: tuple[ScriptedTurn, ...],
    target_id: str,
) -> TurnResult:
    """Advance ONE thread by ONE persona turn: write the next scripted
    inbound ``.eml`` into the inbox, threaded via ``In-Reply-To``
    against the target's latest outbound artifact's REAL Message-ID.

    Determinism: the turn index is the count of replies already recorded
    for the target (the audit trail IS the thread state), the reply text
    is the persona script's entry at that index, and the reply Date is
    the outbound artifact's Date plus the turn number of hours — never
    the wall clock — so the same state always produces the same bytes.

    Writes NOTHING to the database: every statement here is a SELECT.
    The only output is the inbox ``.eml`` file (and the deletion of
    already-processed convo files — see _cleanup_processed_files).
    """
    # ── The contact: the reply's sender (a reply comes FROM the person
    # who received the send).  Missing row or empty email → refuse
    # loudly; a fabricated sender needs an address to fabricate from.
    contact = conn.execute(
        "SELECT c.full_name, c.email FROM targets t "
        "JOIN contacts c ON t.contact_id = c.contact_id "
        "WHERE t.target_id=?;",
        (target_id,),
    ).fetchone()
    if contact is None or not contact["email"]:
        return TurnResult(target_id=target_id, turn_number=0, refusal_reason="no_contact_email")
    sender_email = contact["email"]

    # ── THE RESERVED-DOMAIN RULE (demo_seed's, reused): refuse to write
    # a reply From any address that could look real — a fabricated
    # sender on a real domain would be a fake real person.
    sender_domain = _reserved_domain_of(sender_email)
    if sender_domain is None:
        return TurnResult(target_id=target_id, turn_number=0, refusal_reason="non_reserved_sender")

    # ── Processed-file hygiene BEFORE the turn index is derived: the
    # count of recorded replies is only a faithful "turns already shown"
    # if no unprocessed convo file is waiting in the inbox.
    inbox = Path(inbox_dir)
    if inbox.is_dir():
        blocked = _cleanup_processed_files(conn, inbox, target_id)
        if blocked is not None:
            print(f"ERROR: {blocked}", file=sys.stderr)
            return TurnResult(
                target_id=target_id, turn_number=0,
                refusal_reason="unprocessed_reply_exists",
            )

    # ── The turn index: recorded replies = turns already consumed by
    # reply_cli, so the NEXT scripted turn is that count (0-based).
    turn_index = _recorded_reply_count(conn, target_id)
    if turn_index >= len(persona_script):
        return TurnResult(
            target_id=target_id, turn_number=turn_index,
            refusal_reason="persona_exhausted",
        )
    turn = persona_script[turn_index]
    turn_number = turn_index + 1  # 1-based, for humans and filenames

    # ── The outbound the reply answers: the latest recorded send, and
    # its artifact — the .eml is the ONLY source of the real Message-ID
    # token (never recompute or invent it; the file is what a real
    # reply quotes back).
    outbound = _latest_outbound(conn, target_id)
    if outbound is None:
        return TurnResult(target_id=target_id, turn_number=turn_number, refusal_reason="no_outbound_message")
    artifact = Path(outbox_dir) / f"{outbound['message_id']}.eml"
    if not artifact.is_file():
        return TurnResult(target_id=target_id, turn_number=turn_number, refusal_reason="no_outbox_artifact")
    msg = BytesParser(policy=email_policy.default).parsebytes(artifact.read_bytes())
    outbound_token = msg["Message-ID"]
    if not outbound_token:
        return TurnResult(target_id=target_id, turn_number=turn_number, refusal_reason="artifact_unparseable")

    # ── The reply Date: outbound Date + turn_number hours (a reply
    # arrives after the send; later turns arrive later).  A naive
    # outbound date (send_email writes localtime with no zone) is
    # assumed UTC — demo_seed's documented convention — so the
    # arithmetic is environment-independent.  An unparseable Date
    # degrades to now, the same documented fallback demo_seed uses.
    try:
        outbound_dt = parsedate_to_datetime(msg["Date"])
        if outbound_dt.tzinfo is None:
            outbound_dt = outbound_dt.replace(tzinfo=timezone.utc)
        reply_dt = outbound_dt + timedelta(hours=turn_number)
        date_str = format_datetime(reply_dt)
    except (TypeError, ValueError):
        date_str = format_datetime(datetime.now(timezone.utc))  # the documented fallback: an unparseable outbound Date degrades to now

    # ── Compose and write the artifact — demo_seed's composer, reused.
    # The subject strips one leading "Re:" so a reply to a follow-up
    # (whose subject already carries "Re:") does not stack prefixes;
    # fetch_inbox's normalizer would accept either, this is cosmetic.
    subject = msg["Subject"] or "(no subject)"
    if subject.lower().startswith("re:"):
        subject = subject[3:].strip()
    raw = _compose_reply_bytes(
        from_name=contact["full_name"] or sender_email,
        from_email=sender_email,
        to_address=msg["To"] or _CONVO_TO_FALLBACK,
        subject=subject,
        date_str=date_str,
        in_reply_to=outbound_token,  # verbatim, brackets included — never re-wrap
        message_id_token=f"{outbound['message_id']}.{CONVO_MESSAGE_ID_MARKER}-{turn_number:02d}@{sender_domain}",
        body=turn.body,
    )
    inbox.mkdir(parents=True, exist_ok=True)  # the inbox may not exist yet on a first turn
    out_path = inbox / f"{CONVO_FILE_PREFIX}{target_id}_{turn_number:02d}.eml"
    out_path.write_bytes(raw)  # the file write IS the simulated inbound mail
    return TurnResult(target_id=target_id, turn_number=turn_number, written_path=str(out_path))


# ── The CLI ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Dispatch the ``converse`` subcommand: advance one thread one
    turn.  Returns the process exit code (0 on a written turn, 1 on a
    refusal) — the same shape demo_seed's subcommands use.

    THE OPERATOR LOOP (also documented in runbook.md §10)::
        python -m app.conversation_sim converse --db data/demo.db \\
            --persona warms_up --target <target_id>
        python -m app.reply_cli --db data/demo.db     # the one billable step
    then repeat.  ``converse`` itself never calls a model.
    """
    parser = argparse.ArgumentParser(prog="python -m app.conversation_sim")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    converse_p = sub.add_parser(
        "converse", help="advance one target's thread by one persona turn (writes one inbound .eml)"
    )
    converse_p.add_argument("--db", default="data/demo.db")  # the seeded demo database — never data/outbound.db (guarded)
    # --persona/--target are required for a real advance but optional at
    # the parser level: --list-personas must work without them (the
    # manual checks below enforce them with a clearer message).
    converse_p.add_argument("--persona", help="the scripted counterparty to play (see --list-personas)")
    converse_p.add_argument("--target", help="the target_id whose thread advances (see the console)")
    converse_p.add_argument("--outbox", default="data/outbox")
    converse_p.add_argument("--inbox", default="data/inbox")
    converse_p.add_argument("--list-personas", action="store_true", help="print the available personas and exit")
    args = parser.parse_args(argv)

    if args.list_personas:
        # The roster, one line per persona with its scripted class
        # sequence — the operator picks the thread shape they want.
        for name, script in PERSONAS.items():
            classes = " -> ".join(turn.reply_class for turn in script)
            print(f"{name}: {classes}")
        return 0

    # Both operands are required to advance a thread — refuse with the
    # usage hint rather than an argparse traceback.
    if not args.target:
        print("ERROR: --target is required (the target_id whose thread advances).", file=sys.stderr)
        return 1

    # ── The real-database guard runs FIRST, before any connection — the
    # imported demo_seed guard, so the refusal text and the resolved
    # path comparison are the ones D3a verified (zero ways to reach the
    # operator's real run data from a demo tool).
    violation = _guard_violation(args.db)
    if violation is not None:
        print(f"ERROR: {violation}", file=sys.stderr)
        return 1

    # The persona name must be a real script — refuse loudly (listing
    # the valid names) rather than inventing a default counterparty.
    persona_script = PERSONAS.get(args.persona)
    if persona_script is None:
        valid = ", ".join(sorted(PERSONAS))
        print(
            f"ERROR: unknown persona {args.persona!r} — valid personas: {valid}.",
            file=sys.stderr,
        )
        return 1

    # ── Refuse a missing sqlite file rather than letting connect()
    # silently create an empty one — converse READS a seeded database
    # (the same refusal demo_seed's replies subcommand makes).
    if not args.db.startswith(("postgresql://", "postgres://", "cloudsql://")) and not Path(args.db).exists():
        print(
            f"ERROR: database {args.db!r} does not exist — run "
            f"`python -m app.demo_seed seed --db {args.db}` and "
            f"`python -m app.send_cli --db {args.db}` first.",
            file=sys.stderr,
        )
        return 1

    conn = connect(args.db)  # read-only usage: no apply_schema, no registry seed, no writes
    try:
        result = generate_next_turn(
            conn,
            outbox_dir=args.outbox,
            inbox_dir=args.inbox,
            persona_script=persona_script,
            target_id=args.target,
        )
    finally:
        conn.close()  # explicit close — the same hygiene every CLI keeps
    if result.written_path is None:
        # A refusal — the reason was already printed (or is one of the
        # silent structured refusals): exit non-zero so a script can
        # tell "the thread advanced" from "nothing happened".
        print(
            f"converse: target {result.target_id!r} — "
            f"{result.refusal_reason.replace('_', ' ')}.",
        )
        return 1
    print(
        f"converse: wrote {Path(result.written_path).name} — "
        f"persona {args.persona!r} turn {result.turn_number} "
        f"({persona_script[result.turn_number - 1].reply_class}). "
        f"Next: python -m app.reply_cli --db {args.db} (the one billable step)."
    )
    return 0


# Guard so `python app/conversation_sim.py` also works, not just
# `python -m app.conversation_sim`.  Uses SystemExit instead of
# sys.exit() to stay testable (pytest can catch SystemExit) — the same
# pattern as every stage CLI.
if __name__ == "__main__":
    raise SystemExit(main())
