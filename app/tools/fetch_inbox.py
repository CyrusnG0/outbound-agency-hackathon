"""The simulated inbox (ticket C1): read inbound replies from ``.eml``
files, thread them to the ``messages`` rows B5 wrote, redact their text,
and record one ``replies`` row per inbound message through the write gate.

THE ABSOLUTE RULE, EXTENDED TO INBOUND — this module is simulated BY
CONSTRUCTION, not by configuration.  There is no IMAP, no POP, no OAuth,
no mailbox credential, and no network read of any kind: reading ``.eml``
files off disk IS the fetch, and it is the only fetch.  There is no
"live inbox" mode to enable, no provider flag, and no code path to any
mail transport (tests/test_send_gate.py::test_app_imports_no_mail_transport
walks this module like every other app/ module and fails on the import of
one).  Adding a transport would be a deliberate edit to that test, not a
config flip.

THREADING (docs/reply-routing.md §4, resolved open-question 15):
- PRIMARY: match the ``In-Reply-To`` / ``References`` RFC headers against
  the ``messages`` rows B5 wrote.  B5's .eml artifacts set Message-ID via
  ``make_msgid(idstring=message_id, ...)``, which embeds the messages-row
  id verbatim inside the angle-bracket token — so a header token CONTAINING
  a known ``messages.message_id`` is the deterministic link back.
- FALLBACK (headers absent or unmatchable): sender email + normalized
  subject + the most recent outbound message to that contact within a
  14-day window of the reply's Date header.
- ``replies.thread_id`` is the matched outbound message_id on the header
  path (that IS the RFC thread identity, normalized to our id vocabulary),
  and a deterministic ``fallback:{message_id}`` key on the fallback path
  (same inputs → same key, stable across runs, names the method).

An ``.eml`` that matches no known message is NOT an error: it is logged
and skipped, and no row is written — the system never guesses a target.

PII REDACTION IS MANDATORY (docs/open-questions.md item 18,
docs/threat-model.md): ``replies.raw_text`` is a master table and may hold
real data, but ``redacted_text`` — the only copy anything downstream may
see — has email addresses, phone numbers, postal addresses, secrets, and
meeting-link query strings redacted per the threat-model standard.  EVERY
``log_step`` payload and the write-gate payload use redacted forms only;
a trace log that leaks a stranger's phone number is a policy violation,
not a cosmetic issue.

FAILURE PATHS (deliberate, per the ticket):
- unmatched .eml → logged, skipped, no row, not an error;
- malformed .eml → logged failed step, skipped, the batch continues;
- the target of a matched message not in ``dry_run_sent`` → the row is
  still written (every inbound message gets its own record, §5) but NO
  transition fires — a terminal state (``suppressed``/``not_target``/
  ``failed``) can never be overridden by an inbound message.
"""

import re  # redaction patterns and message-id token extraction — stdlib only, no new dependency
from datetime import datetime, timedelta, timezone  # the 14-day fallback window, computed against the reply's Date header
from email import policy as email_policy  # the stdlib parsing policy (default: modern, defects-surviving)
from email.parser import BytesParser  # RFC-5322 parsing of the .eml bytes — PARSING ONLY, never handed to a transport
from email.utils import getaddresses, parsedate_to_datetime  # From-header parsing and Date-header → datetime for the window arithmetic
from pathlib import Path  # inbox directory handling — the overridable, simulated message source

from pydantic import BaseModel  # InboxFetchResult: structured outcome for the CLI and tests (CLAUDE.md §7)

from app.ids import new_id  # one reply id ("rpl"), one step id ("step") per inbound message
from app.state_machine import transition  # THE state-change gate — dry_run_sent → replied goes through it, never a raw UPDATE
from app.tools.log_step import log_step  # steps-trace writer — every processed file lands in the trace (Golden Rule)
from app.write_gate import commit as write_gate_commit  # THE core-table write path — the replies row is written through it

# ── Constants ──────────────────────────────────────────────────────────────

# The default simulated inbox directory.  Overridable per call (inbox_dir)
# and per run (reply_cli's --inbox flag); the committed demo .eml files
# live here (data/inbox/ is NOT gitignored — only data/outbox/ is).  A
# deliberately plain relative default like every other CLI default in this
# repo; the CLI resolves it against the operator's cwd.
DEFAULT_INBOX_DIR = "data/inbox"

# The steps.tool_name every fetch step row carries (successes, skips, AND
# parse failures) — distinct from every pipeline tool so the trace log
# shows "the simulated inbox ran here" at a glance.
FETCH_INBOX_TOOL_NAME = "fetch_inbox"

# The 14-day fallback window (docs/reply-routing.md §4), in days.  A reply
# may only fallback-thread to an outbound message sent within this window
# of the reply's Date header — outside it, the send is too stale to assume
# the reply answers it, and the message is skipped rather than mis-linked.
_FALLBACK_THREAD_WINDOW_DAYS = 14

# The id prefix for replies rows — "rpl", the same short self-describing
# new_id style as tgt/acc/msg/wr/trn/step, matching the table's PK name
# reply_id.
_REPLY_ID_PREFIX = "rpl"


class InboxFetchResult(BaseModel):
    """What one inbox sweep produced — successes, skips, and errors as
    first-class outcomes, never exceptions (one bad message must not kill
    the batch, ticket B1f's isolation rule re-applied to the inbox).

    ``replies_created`` lists the reply_ids the CLI then feeds to the
    classifier; ``skipped`` and ``errors`` name every file that produced
    no row and why, so nothing is silently dropped.
    """

    files_seen: int  # total .eml files found in the inbox this sweep
    replies_created: list[str]  # reply_id per successfully linked inbound message
    skipped: list[str]  # "filename: reason" for unmatched/malformed files
    errors: list[str]  # "filename: ExceptionType: message" for files that raised


# ── Redaction (docs/threat-model.md "PII redaction rule", item 18) ──────────
# The standard this function implements is the repo's own, NOT a from-
# scratch invention: emails keep the domain and the first 2 local-part
# chars (ja***@example.com); phones keep the last 2 digits; postal
# addresses are fully removed; secrets are fully removed; meeting-link
# query strings are fully removed.  Partial redaction is deliberate — the
# redacted copy must stay readable enough for the classifier and the human
# reviewer to do their jobs while the raw values are unrecoverable from
# any log.

# Emails: local part reduced to its first two characters plus "***", the
# domain kept (the domain half is what the classifier and the operator
# need for context; the local part is the PII).  The regex requires a
# dotted TLD so "3.5" or "v1.2" cannot be mistaken for an address.
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# Phones: a run of 6-19 digits with optional +, spaces, dots, parens and
# hyphens (the 6-19 span keeps short codes and 20+-digit account numbers
# out of the phone bucket — those fall to the secret redactor below if
# they look like secrets).  The lookarounds stop a phone from matching
# inside a longer digit string.
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\s().\-]{6,19}\d)(?!\d)")

# The separator-date guard: "2026-08-23" is 8 digits and would otherwise
# match the phone regex — a date is not a phone number and redacting it
# would mangle every dated sentence in the message.
_DATE_GUARD_RE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")

# Postal addresses: a street-number chunk followed by a capitalized name
# and a street/building keyword (Road, Street, Tower, Central, ...).  The
# keyword list is case-sensitive on purpose — "I road-tested the product"
# must not redact, "34 Nathan Road" must.  Deliberately conservative:
# addresses without a street keyword (e.g. a bare "Flat 12B, 5th floor")
# are NOT caught — over-redaction would destroy the classifier's input,
# and the threat model's requirement is the address fully removed from
# LOGS, which this covers for the common written form.
_ADDRESS_RE = re.compile(
    r"\b\d{1,4}[A-Za-z]?\s+[A-Z][A-Za-z0-9\s,]{1,40}?\b"
    r"(?:Road|Rd|Street|St|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|"
    r"Building|Bldg|Tower|Centre|Center|Plaza|Central|House|Estate|Court|Ct)\b"
)

# Secrets: any key=value / key: value assignment whose key names a
# credential.  The \S+ value capture swallows the whole token so nothing
# of the secret survives.  Applies to api_key, access_token, password,
# etc. — the vocabulary of things that must never reach a log.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"secret|password|passwd|pwd)\b\s*[:=]\s*\S+"
)

# Oversized base64-ish blobs (policy P8's marker for smuggled payloads):
# an unbroken run of 60+ base64 alphabet characters with optional
# trailing padding.  60 is high enough that prose and URLs (which contain
# "://", "/", "?" — none in the base64 alphabet) never false-positive.
_BASE64_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/]{60,}={0,2}\b")

# Meeting links: any http(s) URL.  The query string is FULLY removed (the
# threat model's standard) because meeting links carry ?pwd= / ?token=
# secrets there; scheme + host + path stay, which is all the reviewer
# needs to see which meeting tool was proposed.
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def redact_text(raw: str) -> str:
    """Redact one inbound text per docs/threat-model.md's trace standard.

    Order matters: secrets first (their values can contain anything),
    then emails/phones/addresses, then URL query strings last (so any
    redaction inside a URL's path has already happened before the query
    is cut).  Each replacement uses a fixed token ([SECRET], [ADDRESS])
    or the threat model's partial forms — never a variable-length mask
    that could leak the value's length pattern.
    """
    # Secrets and smuggled blobs are removed FIRST: an api_key= value or a
    # base64 blob may itself contain an email-like or phone-like run, and
    # the full-removal standard must win over the partial forms.
    redacted = _SECRET_ASSIGNMENT_RE.sub("[SECRET]", raw)
    redacted = _BASE64_BLOB_RE.sub("[SECRET]", redacted)

    # Emails: first two local chars + *** + @ + domain (the repo's own
    # standard — the classifier can still see the sender's organisation,
    # never the person).
    redacted = _EMAIL_RE.sub(lambda m: f"{m.group(1)[:2]}***@{m.group(2)}", redacted)

    # Phones: keep only the last 2 digits (the repo's standard).  The
    # date guard skips separator dates so "2026-08-23" survives; pure
    # 8-digit YYYYMMDD runs are additionally skipped by the leading-19/20
    # check because an unseparated date is not a phone call.
    def _mask_phone(match: re.Match) -> str:
        # A separator date ("2026-08-23") matches the phone regex's shape
        # but is not a phone — the date guard checks the WHOLE match
        # first, so dated sentences survive unmangled.
        if _DATE_GUARD_RE.fullmatch(match.group(0)):
            return match.group(0)
        # Extract the digits only — formatting (spaces, parens) is not
        # part of the sensitive value and is dropped from the mask.
        digits = re.sub(r"\D", "", match.group(0))
        if not (6 <= len(digits) <= 20):
            # Outside the phone-length sanity span: leave the text alone
            # (an ID or account number is not a phone number; secrets are
            # already handled above).
            return match.group(0)
        if len(digits) == 8 and digits[:2] in ("19", "20"):
            # YYYYMMDD without separators — a date, not a phone.  (The
            # separator form was guarded by _DATE_GUARD_RE above.)
            return match.group(0)
        # The partial mask: asterisks for every digit but the last two.
        return "*" * (len(digits) - 2) + digits[-2:]

    redacted = _PHONE_RE.sub(_mask_phone, redacted)

    # Postal addresses: fully removed (the repo's standard) — the street
    # number and name are exactly the PII that must not reach a log.
    redacted = _ADDRESS_RE.sub("[ADDRESS]", redacted)

    # Meeting-link query strings: fully removed.  The URL itself stays
    # (which meeting tool was proposed is context, not PII); everything
    # after "?" — the pwd/token parameters — is cut.
    redacted = _URL_RE.sub(lambda m: m.group(0).split("?")[0], redacted)
    return redacted


# ── .eml parsing and threading ──────────────────────────────────────────────


def _normalize_subject(subject: str) -> str:
    """Normalize a subject line for fallback threading.

    Strips any number of leading Re:/Fwd:/Fw: prefixes (the reply's
    "Re: Cold subject" and the outbound's "Cold subject" must compare
    equal), collapses surrounding whitespace, and lowercases — so the
    match is about the topic, not the formatting.
    """
    normalized = subject.strip().lower()
    # Loop (not a single pass): "Re: Re: Fwd: topic" is legal mail-client
    # output and must strip down to "topic".
    while True:
        # Match one leading prefix followed by optional whitespace/colon.
        stripped = re.sub(r"^(re|fw|fwd)\s*:\s*", "", normalized)
        if stripped == normalized:
            # No prefix removed this pass — strip the leftover
            # surrounding whitespace ("RE:  cold subject  " leaves a
            # trailing gap after the prefix is cut) and return.
            return normalized.strip()
        normalized = stripped
        # Continue the loop to strip stacked prefixes.


def _extract_message_id_tokens(msg) -> list[str]:
    """Collect every angle-bracket message-id token from the In-Reply-To
    and References headers (the primary threading input, §4).

    ``get_all`` handles repeated headers; tokens are stripped of their
    angle brackets and deduplicated (a References chain often repeats the
    immediate parent).  An empty list means the header path is absent and
    the caller falls back to sender+subject threading.
    """
    tokens: list[str] = []
    for header_name in ("In-Reply-To", "References"):
        # get_all returns None when the header is absent — a legal state
        # (fallback threading exists for exactly that case).
        for value in msg.get_all(header_name, []):
            # Every <...> token in the header value; comments and display
            # names are ignored because message-ids only live in brackets.
            tokens.extend(re.findall(r"<([^<>]+)>", value))
    # Preserve order, drop duplicates — order matters only for readability,
    # dedup keeps the candidate list short.
    return list(dict.fromkeys(tokens))


def _extract_from_email(msg) -> str:
    """Extract the reply's sender address from the From header, lowercased.

    ``getaddresses`` handles "Name <addr>" and bare-addr forms and returns
    (name, addr) tuples; the addr half is what threading and the replies
    row need.  Missing/unparseable From yields "" — the caller refuses the
    message rather than guessing a sender.
    """
    addresses = getaddresses(msg.get_all("From", []))
    if not addresses:
        # No From header at all — cannot thread, cannot record a sender.
        return ""
    # The first address is the sender; lowercased for the contacts.email
    # comparison (emails are case-insensitive in the local part per RFC).
    return (addresses[0][1] or "").lower()


def _parse_reply_date(msg) -> datetime:
    """Parse the reply's Date header into an aware UTC datetime.

    The fallback window is computed against THIS date (a reply answers a
    message sent before it, not after).  An unparseable or absent Date
    degrades to now(UTC) — the window then runs from the current moment,
    which is the honest best guess and keeps the pipeline moving.
    """
    try:
        parsed = parsedate_to_datetime(msg["Date"])
    except (TypeError, ValueError):
        # Absent or malformed Date header — degrade to now, documented
        # above, rather than skipping an otherwise threadable reply.
        return datetime.now(timezone.utc)
    if parsed is None:
        # parsedate_to_datetime returns None for unparseable strings.
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        # A naive date (no zone in the header) is assumed UTC — the same
        # convention the DB's datetime('now') timestamps use, so the
        # string comparisons below stay on one clock.
        return parsed.replace(tzinfo=timezone.utc)
    # Convert any explicit zone to UTC so the TEXT timestamp comparison
    # (which the DB stores in UTC) is apples-to-apples.
    return parsed.astimezone(timezone.utc)


def _match_by_headers(conn, tokens: list[str]) -> str | None:
    """Primary threading: find the outbound messages row whose message_id
    is embedded in any In-Reply-To/References token.

    B5's artifacts set Message-ID via make_msgid(idstring=message_id),
    which produces "<...timestamp...message_id@domain>" — so the
    messages-row id appears VERBATIM inside the token.  The match is
    therefore a substring test (row.message_id in token), and it cannot
    false-positive: a 12-hex new_id like "msg_3f9a2b1c" appearing inside
    a stranger's token is only possible if the token quotes our own
    Message-ID back at us, which is exactly what a real reply does.
    """
    if not tokens:
        # No header tokens — the primary path has nothing to match on.
        return None
    # The candidate set: outbound rows only (an inbound message is never
    # the parent of a thread).  Read the ids into Python and test the
    # substring here — at single-operator scale the table is tiny, and a
    # SQL LIKE over every token would need one OR-term per token.
    rows = conn.execute(
        "SELECT message_id FROM messages WHERE direction='outbound';"
    ).fetchall()
    for token in tokens:
        for row in rows:
            if row["message_id"] in token:
                # The first matching token wins — In-Reply-To (the
                # immediate parent) precedes References in the token list.
                return row["message_id"]
    return None


def _match_by_fallback(conn, *, from_email: str, subject: str, reply_date: datetime) -> str | None:
    """Fallback threading (§4): sender email + normalized subject + the
    most recent outbound message to that contact within 14 days of the
    reply's Date.

    The contact lookup joins messages → contacts (the outbound row stores
    contact_id, and contacts.email is the address B5 sent TO — which must
    equal the reply's From).  The window is [reply_date - 14d, reply_date]
    compared as TEXT (the DB stores second-precision UTC strings, and
    lexicographic order == chronological order in that format).  "Most
    recent" orders by created_at DESC with message_id DESC as the
    deterministic tiebreak for same-second rows (messages has no
    insert_seq; created_at alone is second-precision — the B5 lesson).
    """
    # The window bounds, formatted exactly like the DB's datetime('now')
    # strings so the SQL comparison needs no date functions (and stays
    # identical on both sqlite and postgres).
    window_start = (reply_date - timedelta(days=_FALLBACK_THREAD_WINDOW_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    window_end = reply_date.strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute(
        # Join through contacts so "sender email" means the address the
        # outbound was actually sent TO, not some header text.  The
        # subject comparison happens in Python (normalization needs
        # regex, which SQL cannot express portably).
        "SELECT m.message_id, m.subject, m.created_at FROM messages m "
        "JOIN contacts c ON m.contact_id = c.contact_id "
        "WHERE m.direction='outbound' AND LOWER(c.email)=? "
        "AND m.created_at >= ? AND m.created_at <= ? "
        "ORDER BY m.created_at DESC, m.message_id DESC;",
        (from_email, window_start, window_end),
    ).fetchall()
    normalized_subject = _normalize_subject(subject)
    for row in rows:
        # Normalize the stored outbound subject the same way — "Re: "
        # prefixes never appear on the outbound row (B5 stored the raw
        # revision subject), but normalizing both sides keeps the
        # comparison robust to any future prefix on either side.
        if _normalize_subject(row["subject"]) == normalized_subject:
            # Rows are ordered most-recent-first, so the FIRST subject
            # match IS the most recent outbound in the window.
            return row["message_id"]
    return None


# ── The fetch ───────────────────────────────────────────────────────────────


class ParsedInboxMessage(BaseModel):
    """The deterministic facts extracted from one .eml file — the
    structured I/O between the parser and the row writer (CLAUDE.md §7)."""

    from_email: str  # lowercased sender address; "" when unparseable
    subject: str  # raw subject line (may be "" — a reply with no subject still threads by headers)
    body: str  # the message text (plain part preferred); "" when absent
    in_reply_to_tokens: list[str]  # angle-stripped tokens from In-Reply-To/References
    reply_date: datetime  # aware UTC — the Date header or now() as the documented fallback


def parse_inbox_message(path: Path) -> ParsedInboxMessage:
    """Parse one .eml file with the STDLIB email package — parsing only,
    never transport.  Raises on malformed bytes so the caller's per-file
    handler can log and continue the batch (the ticket's failure path).
    """
    # BytesParser with the default policy: bytes in, a structured message
    # out, and header defects survive as defects rather than killing the
    # parse — a slightly-malformed header should not lose a whole reply.
    msg = BytesParser(policy=email_policy.default).parsebytes(path.read_bytes())
    if not msg.items():
        # Zero recognized headers — the bytes were not an RFC-5322 message
        # at all.  C1-finish choice (the test's side was right, the code
        # was wrong): the stdlib parser is lenient BY DESIGN and does NOT
        # raise on such input — garbage bytes become an empty message, and
        # the fetch would mislabel the file "no parseable From address",
        # hiding that it is malformed.  Raising HERE makes this function's
        # own contract ("raises on malformed bytes") true and routes the
        # file through the caller's malformed-failure path, whose skip
        # vocabulary and failed-step log are what the trace must show
        # (every real RFC-5322 message has at least one header — From is
        # mandatory — so no legitimate message is lost by this check).
        raise ValueError("no RFC-5322 headers — not an email message")
    # ── Body extraction: prefer the plain-text part ─────────────────────
    # get_body walks the MIME tree for a text/plain part (the part a
    # human wrote); html-only replies fall back to get_content(), and a
    # multipart with no usable part falls back to "" rather than raising
    # (an empty body is classified downstream, not crashed here).
    try:
        body_part = msg.get_body(preferencelist=("plain",))
        body = body_part.get_content() if body_part is not None else ""
    except Exception:
        # A defect in the MIME structure — degrade to empty text rather
        # than losing the whole message (the classifier treats "" as
        # unclear; a crash here would skip the reply entirely).
        body = ""
    return ParsedInboxMessage(
        from_email=_extract_from_email(msg),
        subject=msg["Subject"] or "",
        body=body,
        in_reply_to_tokens=_extract_message_id_tokens(msg),
        reply_date=_parse_reply_date(msg),
    )


def fetch_inbox(
    conn,
    *,
    inbox_dir: str = DEFAULT_INBOX_DIR,
    run_id: str,
    limit: int = 100,
) -> InboxFetchResult:
    """Read every ``.eml`` in the simulated inbox, thread it to a known
    outbound message, redact it, record one ``replies`` row, and fire
    ``dry_run_sent → replied`` for the target when it is still in
    ``dry_run_sent``.

    Per-file isolation (the B1f rule, re-applied): one malformed or
    unmatchable file is logged and skipped, never raised — the batch
    always completes.  ``limit`` caps how many files are processed (the
    CLI's --limit), sorted by filename so the sweep is deterministic.
    ``inbox_dir`` overrides the message source (tests use a tmp dir).

    WRITE ORDER — the replies row is written BEFORE the transition, the
    same invariant send_email protects: a ``replied`` target must never
    exist without its reply row, or the audit trail would claim a reply
    that was never recorded.  A crash between row and transition leaves a
    recorded reply plus a target still honestly in ``dry_run_sent`` —
    retryable, and the unmatched-this-time file is simply re-read.
    """
    # The inbox may not exist yet (first run) — that is "no replies", not
    # an error, exactly like an empty mailbox.
    inbox = Path(inbox_dir)
    if not inbox.is_dir():
        return InboxFetchResult(files_seen=0, replies_created=[], skipped=[], errors=[])
    # Sorted filenames make the sweep order (and therefore the --limit
    # cut) deterministic run-to-run.
    files = sorted(inbox.glob("*.eml"))[:limit]
    result = InboxFetchResult(files_seen=len(files), replies_created=[], skipped=[], errors=[])
    for path in files:
        # ── Per-file isolation: nothing a single file does may kill the
        # batch (the B1f rule).  The step id is generated inside the loop
        # so every file — success, skip, or failure — gets its own fresh
        # trace row (the A6 one-id-per-row invariant).
        step_id = new_id("step")
        try:
            parsed = parse_inbox_message(path)
        except Exception as exc:
            # Malformed .eml — log the failure (never skip logs) and
            # continue with the next file.  The error message goes in the
            # trace so the operator can find and fix the bad file.
            log_step(
                conn, run_id=run_id, step_id=step_id, target_id=None,
                tool_name=FETCH_INBOX_TOOL_NAME, agent_id="system",
                input_data={"stage": "fetch_inbox", "file": path.name, "simulated": True},
                output_data={"outcome": "malformed_file", "error_type": type(exc).__name__, "error": str(exc)},
                status="failed",
            )
            result.skipped.append(f"{path.name}: malformed .eml ({type(exc).__name__})")
            continue
        if not parsed.from_email:
            # No parseable sender — nothing to thread on and the replies
            # table's from_email is NOT NULL.  Log and skip; never guess.
            # (A file with zero recognized headers never reaches this
            # branch: parse_inbox_message raises on it — see there — and
            # the malformed-failure path above names it "malformed".)
            log_step(
                conn, run_id=run_id, step_id=step_id, target_id=None,
                tool_name=FETCH_INBOX_TOOL_NAME, agent_id="system",
                input_data={"stage": "fetch_inbox", "file": path.name, "simulated": True},
                output_data={"outcome": "no_sender"},
                status="success",
            )
            result.skipped.append(f"{path.name}: no parseable From address")
            continue

        # ── Threading: header match first, fallback second (§4) ─────────
        # message_id is the outbound messages-row id the reply links to;
        # replies.message_id is NOT NULL and references messages, so an
        # unmatchable file cannot be recorded at all.
        message_id = _match_by_headers(conn, parsed.in_reply_to_tokens)
        match_method = "in_reply_to_header"
        if message_id is None:
            # Headers absent or unmatchable — the sender+subject+14-day
            # fallback.  An empty subject normalizes to "" and only
            # matches an equally-empty outbound subject, so a subjectless
            # reply effectively requires the header path.
            message_id = _match_by_fallback(
                conn, from_email=parsed.from_email,
                subject=parsed.subject, reply_date=parsed.reply_date,
            )
            match_method = "fallback_sender_subject_window"
        if message_id is None:
            # Matches no known message — NOT an error, per the ticket:
            # log it, skip it, write nothing, never guess a target.
            log_step(
                conn, run_id=run_id, step_id=step_id, target_id=None,
                tool_name=FETCH_INBOX_TOOL_NAME, agent_id="system",
                input_data={"stage": "fetch_inbox", "file": path.name, "simulated": True},
                output_data={"outcome": "no_known_message", "match_method": None},
                status="success",
            )
            result.skipped.append(f"{path.name}: matches no known outbound message")
            continue

        # ── The redacted copy — computed BEFORE any log payload exists ───
        # raw_text goes to the master table only; redacted_text is the
        # only form any trace row, prompt, or payload may carry (item 18).
        redacted = redact_text(parsed.body)
        # The row's FK facts: target_id comes from the matched message
        # (replies has no target column — the join is replies→messages→
        # targets), and the thread_id is the message_id on the header
        # path or the deterministic fallback key (§4).
        message_row = conn.execute(
            "SELECT target_id FROM messages WHERE message_id=?;", (message_id,)
        ).fetchone()
        target_id = message_row["target_id"]
        thread_id = message_id if match_method == "in_reply_to_header" else f"fallback:{message_id}"

        # ── Write 1 of 2: the replies row, through the write gate ────────
        # THE write path — never a raw INSERT.  agent_id="system":
        # deterministic code performs the fetch.  The audit payload
        # carries ONLY redacted forms (a redacted subject and a redacted
        # sender) plus lengths — write_log is a trace log and must never
        # hold raw PII, even though the replies row itself may.
        reply_id = new_id(_REPLY_ID_PREFIX)
        write_gate_commit(
            conn,
            action="insert_reply",  # C1's new KNOWN_ACTION — reply arrivals are audited distinctly in write_log
            table_name="replies",
            record_id=reply_id,
            payload={
                "match_method": match_method,
                "thread_id": thread_id,
                # Redacted forms only in the audit row (item 18 — the
                # write_log payload is a trace payload, never the master
                # table).
                "from_email_redacted": redact_text(parsed.from_email),
                "subject_redacted": redact_text(parsed.subject),
                "raw_length": len(parsed.body),
                "redacted_length": len(redacted),
            },
            run_id=run_id,
            step_id=step_id,
            actor="system",  # deterministic code performs the write
            agent_id="system",  # attributed to the registered deterministic principal
            # insert_seq (ticket E1, extending C1's fix to replies) is
            # computed IN the INSERT via the scalar MAX+1 subquery — the
            # same atomic, dialect-neutral monotonic sequence every other
            # insert_seq-carrying table uses.  It makes "which reply on
            # this thread is the LATEST?" deterministic for the follow-up
            # path (created_at is second-precision TEXT; two replies
            # fetched in the same second previously ordered arbitrarily —
            # the B5 lesson, one table further down).
            sql="""
                INSERT INTO replies
                    (reply_id, message_id, thread_id, from_email, raw_text,
                     redacted_text, classification, confidence, routed_action,
                     insert_seq, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,
                        (SELECT COALESCE(MAX(insert_seq),0)+1 FROM replies),
                        datetime('now'))
            """,
            params=(
                reply_id,
                message_id,
                thread_id,
                parsed.from_email,
                parsed.body,  # raw_text: the master table MAY hold real data (threat-model.md) — and only it
                redacted,  # redacted_text: the copy every downstream consumer sees
                None,  # classification: the classifier (C1's router) fills this later — NULL is the honest "not judged yet"
                None,  # confidence: same — no verdict exists at fetch time
                None,  # routed_action: same — routing has not happened yet
            ),
        )

        # ── Write 2 of 2: dry_run_sent → replied, if still applicable ────
        # The state machine's new C1 edge (§7j): the inbound message is
        # linked, so a target B5 left in dry_run_sent moves to replied.
        # READ the state fresh — the caller must never trust its own
        # belief about where the target is (the second reply on a thread
        # finds the target already in replied, a suppressed target is
        # terminal, and neither may be transitioned — §5).
        current = conn.execute(
            "SELECT state FROM targets WHERE target_id=?;", (target_id,)
        ).fetchone()
        new_state = current["state"]
        if current["state"] == "dry_run_sent":
            # The ONE transition this module may fire.  from_state is the
            # state just read, so the transition records the truth even
            # if the target moved between the read and the write (the
            # state machine refuses the stale pair rather than lying).
            transition(
                conn, target_id=target_id,
                from_state="dry_run_sent", to_state="replied",
                reason="inbound_message_linked",  # the §2 trigger vocabulary for the C1 edge
                actor="system",
                run_id=run_id, step_id=step_id,
                # agent_id defaults to actor ("system") — this hop is
                # deterministic code, attributed to the system principal.
            )
            new_state = "replied"
        # Any other state — replied already (a second reply), or a
        # terminal state — records the row and does NOT transition (§5:
        # no reply ever overrides a terminal state).

        # ── The trace row (never skip logs) — REDACTED FORMS ONLY ────────
        # Everything in input/output here has passed through redact_text;
        # raw_text itself never appears in a steps payload (item 18 — a
        # trace log that leaks a stranger's phone number is a policy
        # violation, not a cosmetic issue).
        log_step(
            conn, run_id=run_id, step_id=step_id, target_id=target_id,
            tool_name=FETCH_INBOX_TOOL_NAME, agent_id="system",
            input_data={
                "stage": "fetch_inbox", "file": path.name, "simulated": True,
                "from_email": redact_text(parsed.from_email),
                "subject": redact_text(parsed.subject),
            },
            output_data={
                "reply_id": reply_id,
                "message_id": message_id,
                "thread_id": thread_id,
                "match_method": match_method,
                "target_state": new_state,
                "transitioned": current["state"] == "dry_run_sent",
            },
            status="success",
        )
        result.replies_created.append(reply_id)
    return result
