"""The DRY_RUN "send" — the only send that exists in this repository
(ticket B5).

THE ABSOLUTE RULE — no real email may ever leave this repository.  Not one,
not to anyone, not even to the operator's own address.  This module
therefore contains NO transport of any kind: no smtplib/aiosmtplib, no
Gmail/Google API client, no OAuth flow, no socket, no HTTP POST, no
subprocess.  Writing the .eml file to ``data/outbox/{message_id}.eml`` IS
the send, and it is the only send.  There is no mode flag, no ``LIVE``
branch, and no code path to the state machine's ``approved → sent``
transition (which exists in docs/state-machine.md §3 for a future Phase 2
only — see §7i).  This is enforced structurally, not by configuration:
tests/test_send_gate.py walks every app/ module with ``ast`` and fails if
any mail-transport import appears, and asserts pyproject.toml declares no
mail-transport dependency.  Adding a transport would be a deliberate edit
to that test, not a config flip.

WHAT A DRY_RUN SEND IS (docs/gates.md §2.3a) — the FULL send-gate
preflight (§2.2) always runs, including for DRY_RUN, because the test must
prove the gate logic works, not just that content gets generated.  On
allow, the approved revision is composed into an RFC-5322 message with the
STDLIB ``email`` package — used for FORMATTING ONLY, never handed to a
transport — and written to the outbox.  Then one ``messages`` row records
the simulated send (direction outbound, status ``dry_run_sent``,
``sent_at`` NULL — nothing was sent), and the target transitions
``approved → dry_run_sent``.  No OutcomeRecord is initialized: there is no
real outcome to track (§2.3a).  A refused gate writes NO file, NO messages
row, and NO transition — refusal is the default path, not an exception
case — and the target stays in ``approved`` so a fixed condition can be
retried.

WRITE ORDER (deliberate, ticket §4) — the .eml file is written FIRST, then
the messages row, then the transition.  The invariant this protects: a
``dry_run_sent`` target must never exist without its artifact.  A crash
between file and DB leaves an orphan .eml (harmless — it was never sent
anywhere and has no real-world effect) plus a target still honestly in
``approved``, so the operator simply retries; a fresh message_id means a
fresh filename, so a retry can never clobber the orphan.  The reverse
order (DB first) could leave ``dry_run_sent`` with no artifact — a lying
audit trail about what was "sent".  A filesystem failure raises BEFORE any
DB write, so no partial state is possible (tested).
"""

from email.message import EmailMessage  # RFC-5322 composition — FORMATTING ONLY, never handed to a transport
from email.policy import SMTP as SMTP_POLICY  # wire-format line endings (CRLF) so the .eml reads like a real message artifact
from email.utils import formatdate, make_msgid  # Date and Message-ID headers — deterministic-ish, transport-free
from pathlib import Path  # outbox directory handling — the overridable artifact location

from pydantic import BaseModel  # SendEmailResult: structured outcome for the CLI and tests (CLAUDE.md §7)

from app.config import load_offer_configs  # the offer's from_address for the From header (config-as-code, same source the footer used)
from app.ids import new_id  # one message id ("msg"), one step id ("step") per send
from app.send_gate import DRY_RUN_STATUS, evaluate_send_gate  # the preflight and the status vocabulary the messages row shares
from app.state_machine import transition  # THE state-change gate — approved → dry_run_sent goes through it, never a raw UPDATE
from app.tools.log_step import log_step  # steps-trace writer — every send and every refusal lands in the trace (Golden Rule)
from app.write_gate import commit as write_gate_commit  # THE core-table write path — the messages row is written through it

# ── Constants ──────────────────────────────────────────────────────────────

# The default outbox directory (docs/gates.md §2.3a: data/outbox/{message_id}.eml).
# Overridable per call (the outbox_dir argument) and per run (the CLI's
# --outbox flag) — tests always point it at a tmp dir, so no artifact ever
# lands in the repo's data/ during a test run.  Deliberately a plain
# relative default like every other CLI default in this repo (draft_cli's
# --db/--offers-dir); the CLI resolves it against the operator's cwd.
DEFAULT_OUTBOX_DIR = "data/outbox"

# The offers directory the From header is resolved from — the same default
# app/agents/draft.py uses, so the composed message and the draft's footer
# read the SAME offer config the pipeline drafted from.
DEFAULT_OFFERS_DIR = "config/offers"

# The steps.tool_name every send step row carries (successes AND refusals) —
# distinct from every pipeline tool so the trace log shows "a send-gate
# evaluation happened here" at a glance, mirroring review_decision's role.
SEND_TOOL_NAME = "send_email"

# The fallback From address when the offer config declares no from_address.
# ".invalid" is the RFC 2606 reserved TLD that can never resolve — the
# honest spelling for "this is not a real address and never will be",
# used only because RFC 5322 requires a From header and an artifact must
# still be a well-formed message.  A real sending domain is the operator's
# prerequisite for any future LIVE path — and no LIVE path exists.
_DRY_RUN_FROM_FALLBACK = "dry-run@outbound-agency.invalid"

# The domain half of the fallback Message-ID.  Same .invalid reasoning —
# make_msgid needs a domain; this one cannot exist in DNS.
_DRY_RUN_MESSAGE_ID_DOMAIN = "outbound-agency.invalid"


class SendEmailResult(BaseModel):
    """What one send attempt produced — the success AND the refusal shape.

    Refusals are first-class outcomes (refused=True + refusal_reason), not
    exceptions: the CLI prints them, and the operator always sees WHY
    nothing was sent.  On success, message_id and outbox_path name the
    artifact, and new_state is "dry_run_sent".
    """

    target_id: str  # echoed so a caller can match the outcome to the target without bookkeeping
    refused: bool  # True = the gate said no: no file, no messages row, no transition
    refusal_reason: str  # the gate's reasons joined; "" on success
    message_id: str | None  # the messages row id and .eml filename stem; None on refusal
    outbox_path: str | None  # absolute path of the written .eml; None on refusal
    new_state: str | None  # "dry_run_sent" on success; None on refusal (the target did not move)


def _load_from_address(conn, target_id: str, offers_dir: str) -> str:
    """Resolve the offer's configured from_address for the From header.

    A local sibling of app/agents/draft.py's _load_offer_draft_config,
    deliberately NOT imported from there: that module pulls the whole ADK
    stack (google.adk) at import time, and this deterministic tool must not
    pay that cost or create that dependency edge.  Missing target, missing
    YAML, or a missing from_address key all degrade to the .invalid
    fallback — a missing From address is a configuration gap, not a reason
    to fail the send (the artifact is never delivered anywhere).
    """
    # offer_id lives on the target row; the YAML is keyed by slug — join
    # through offers to find the slug (the same join draft.py uses).
    row = conn.execute(
        "SELECT o.slug FROM targets t JOIN offers o ON t.offer_id = o.offer_id "
        "WHERE t.target_id=?;",
        (target_id,),
    ).fetchone()
    if row is None:
        # No target or no linked offer — nothing to read the address from.
        return _DRY_RUN_FROM_FALLBACK
    try:
        # The offer config for this slug; .get() keeps every key optional
        # (a legitimate offer may carry no from_address).
        config = load_offer_configs(offers_dir).get(row["slug"], {})
    except OSError:
        # The offers dir is missing/unreadable (test environment or a
        # misconfigured path) — degrade to the fallback, never fail the send.
        return _DRY_RUN_FROM_FALLBACK
    # The configured address, or the honest .invalid placeholder.  NOTE the
    # real therapy-app.yaml carries "outreach@REPLACE-ME-BEFORE-SENDING.test"
    # — a .test-domain placeholder the operator must replace.  It is used
    # verbatim: the .eml is never delivered, so the header is cosmetic, and
    # silently rewriting the operator's config would be worse than showing
    # it to them in the artifact.
    return config.get("from_address") or _DRY_RUN_FROM_FALLBACK


def _compose_rfc5322(
    *, message_id: str, from_address: str, to_address: str, subject: str, body: str, footer: str
) -> bytes:
    """Compose the RFC-5322 message bytes from the approved revision.

    The stdlib email package is used FORMATTING ONLY — the returned bytes
    go to a file, never to a transport.  The body is the approved draft
    text plus the deterministic compliance footer (B3-Z1: the footer is
    code-authored and carried on every revision — the model had no field
    for it).  The footer still carries the "[unsubscribe: {UNSUBSCRIBE_URL}]"
    placeholder token: there is no real unsubscribe URL, and inventing a
    URL scheme or domain is forbidden (B3's decision, preserved).
    """
    msg = EmailMessage()  # the stdlib composer — no transport is (or can be) attached
    # The From address is the offer's configured sender; the fallback is
    # the .invalid placeholder (see the constant's comment).
    msg["From"] = from_address
    # The recipient is the contact's verified address — the one the gate
    # checked against suppressions and rate limits.
    msg["To"] = to_address
    msg["Subject"] = subject
    # A real Date header (RFC 5322 requires one) — wall-clock, second
    # precision; the artifact is a snapshot of what WOULD have been sent.
    msg["Date"] = formatdate(localtime=True)
    # Message-ID is the messages-row id, so the artifact and the audit row
    # reference each other.  The domain half comes from the from_address
    # when it parses as one, else the .invalid placeholder domain.
    from_domain = from_address.split("@", 1)[-1] if "@" in from_address else _DRY_RUN_MESSAGE_ID_DOMAIN
    msg["Message-ID"] = make_msgid(idstring=message_id, domain=from_domain)
    # The plain-text content: the approved body, a blank line, then the
    # deterministic footer.  This is exactly the text the operator
    # approved in review (subject and body) plus the code-authored footer
    # they were shown with it.
    msg.set_content(body + "\n\n" + footer)
    # Wire-format bytes (CRLF line endings via the SMTP policy) so the
    # .eml is a faithful artifact of the message that would have been
    # handed to a transport — in a world where one exists.  It does not.
    return msg.as_bytes(policy=SMTP_POLICY)


def send_email(
    conn,
    *,
    target_id: str,
    run_id: str,
    outbox_dir: str = DEFAULT_OUTBOX_DIR,
    offers_dir: str = DEFAULT_OFFERS_DIR,
) -> SendEmailResult:
    """Run one target through the send gate and, on allow, write the DRY_RUN
    .eml artifact, the messages row, and the approved → dry_run_sent
    transition.

    Flow: (1) evaluate_send_gate — refusal returns immediately with no
    file, no row, no transition (the default path); (2) compose the
    RFC-5322 artifact from the approved revision; (3) write the .eml to
    the outbox (FIRST — see the module docstring's WRITE ORDER); (4) insert
    the messages row through the write gate (sent_at NULL: nothing was
    sent); (5) transition approved → dry_run_sent through the state
    machine; (6) log the success step.  A filesystem failure raises before
    any DB write; a DB failure after the file write leaves an orphan
    artifact and an honestly-approved target (documented above).

    ``outbox_dir`` overrides the artifact location (tests use a tmp dir).
    ``offers_dir`` overrides where the From address's YAML is read from.
    """
    # ONE fresh step id shared by the gate's decision row, the messages
    # write, the transition, and the trace row — the established pattern
    # (review.py, draft.py's persist node): the send's audit entries hang
    # together under one step.
    step_id = new_id("step")

    # ── Step 1: the preflight ────────────────────────────────────────────
    # The gate wrote its send_gate_decisions row itself (allow or refuse).
    decision = evaluate_send_gate(conn, target_id=target_id, run_id=run_id, step_id=step_id)
    if not decision.allowed:
        # ── THE REFUSAL PATH — the default, not an exception case ────────
        # No file, no messages row, no transition: the target stays in
        # approved so a fixed condition can be retried.  The refusal is
        # still a logged, observable outcome (Golden Rule: never skip
        # logs) — the failed step row carries every reason the gate gave.
        log_step(
            conn,
            run_id=run_id,
            step_id=step_id,
            target_id=target_id,
            tool_name=SEND_TOOL_NAME,
            agent_id="system",  # deterministic code attempted the send
            input_data={"stage": "send_email", "simulated": True},
            output_data={
                "refusal_reasons": decision.reasons,
                "missing_requirements": decision.missing_requirements,
                "kill_switch_active": decision.kill_switch_active,
            },
            status="failed",  # the steps vocabulary's honest refusal status (no "refused" value exists)
        )
        return SendEmailResult(
            target_id=target_id,
            refused=True,
            refusal_reason="; ".join(decision.reasons),
            message_id=None,
            outbox_path=None,
            new_state=None,
        )

    # ── Step 2: load what the operator approved ──────────────────────────
    # The same lookups the gate used: the approved revision via the latest
    # review row's draft_message_id (the db-schema contract), the contact's
    # address, and the offer's From address.  Re-read rather than passed
    # back through the decision model, whose fields are pinned by the
    # ticket.  The review-row read orders by insert_seq DESC exactly like
    # the gate's (the B5 determinism fix — created_at alone is
    # second-precision and can tie), so the text composed here is
    # guaranteed to be the text the gate verified.
    review_row = conn.execute(
        "SELECT draft_message_id FROM review_decisions WHERE target_id=? "
        "ORDER BY insert_seq DESC, created_at DESC LIMIT 1;",
        (target_id,),
    ).fetchone()
    revision = conn.execute(
        "SELECT subject, body, footer FROM message_draft_versions WHERE draft_version_id=?;",
        (review_row["draft_message_id"],),
    ).fetchone()
    contact = conn.execute(
        "SELECT c.contact_id, c.email FROM targets t JOIN contacts c "
        "ON t.contact_id = c.contact_id WHERE t.target_id=?;",
        (target_id,),
    ).fetchone()
    # The policy row id, re-read so the transition's matched-policy link is
    # the same allow row the gate verified (state-machine.md §4 requires a
    # policy_decision_id on every transition).  Same insert_seq DESC
    # ordering as the gate's policy read (the B5 determinism fix).
    policy_row = conn.execute(
        "SELECT policy_decision_id FROM policy_decisions WHERE target_id=? "
        "ORDER BY insert_seq DESC, created_at DESC LIMIT 1;",
        (target_id,),
    ).fetchone()

    # ── Step 3: compose the artifact ─────────────────────────────────────
    message_id = new_id("msg")  # "msg" prefix: the messages-table id, and the .eml filename stem
    raw = _compose_rfc5322(
        message_id=message_id,
        from_address=_load_from_address(conn, target_id, offers_dir),
        to_address=contact["email"],
        subject=revision["subject"],
        body=revision["body"],
        footer=revision["footer"],
    )

    # ── Step 4: write the .eml FIRST (the WRITE ORDER invariant) ─────────
    # The file is the artifact of the send; the DB rows record that the
    # artifact exists.  A crash between here and the transition leaves an
    # orphan .eml (never sent anywhere — harmless) and a target still
    # honestly in approved (retryable, and the fresh message_id gives the
    # retry a fresh filename).  The reverse order could leave a
    # dry_run_sent target with no artifact, which would lie to the
    # operator about what was "sent".  A failure here raises BEFORE any DB
    # write — no partial state (tested with a blocking outbox path).
    outbox = Path(outbox_dir)
    outbox.mkdir(parents=True, exist_ok=True)  # the outbox may not exist yet on a first run
    outbox_path = outbox / f"{message_id}.eml"
    outbox_path.write_bytes(raw)

    # ── Step 5: record the simulated send through the write gate ─────────
    # The messages row is the audit ledger entry for the DRY_RUN send:
    # direction outbound (it is an outbound-shaped message), status
    # dry_run_sent (the state machine's vocabulary — and the ONLY status
    # the rate-limit counters exclude, §2.3a), sent_at NULL (NOTHING was
    # sent — the column must never claim a send that did not happen),
    # provider_message_id NULL (there is no provider), thread_id NULL (a
    # DRY_RUN opens a fresh thread — threading arrives with C1's reply
    # flow), body = the full composed text (what the .eml contains),
    # body_redacted NULL (nothing to redact: the text is the approved
    # draft, already reviewed by a human).
    write_gate_commit(
        conn,
        action="insert_message",  # B5's new KNOWN_ACTION — message writes are audited distinctly in write_log
        table_name="messages",
        record_id=message_id,
        payload={
            "direction": "outbound",
            "status": DRY_RUN_STATUS,
            "simulated": True,  # the audit payload says it plainly: this row is a simulation
            "sent_at": None,
            "outbox_path": str(outbox_path),
        },
        run_id=run_id,
        step_id=step_id,
        actor="system",  # deterministic code performs the write
        agent_id="system",  # attributed to the registered deterministic principal
        policy_decision_id=policy_row["policy_decision_id"] if policy_row else None,
        sql="""
            INSERT INTO messages
                (message_id, target_id, contact_id, direction, provider_message_id,
                 thread_id, subject, body, body_redacted, status, sent_at, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
        """,
        params=(
            message_id,
            target_id,
            contact["contact_id"],
            "outbound",
            None,  # provider_message_id: no provider exists — nothing was transmitted
            None,  # thread_id: a fresh (threadless) send — no threading exists yet
            revision["subject"],
            revision["body"] + "\n\n" + revision["footer"],  # the full composed text, matching the .eml
            None,  # body_redacted: nothing to redact — the text is human-approved
            DRY_RUN_STATUS,  # the dry-run status — excluded from all §2.2a rate-limit counters
            None,  # sent_at: NOTHING was sent — the honest NULL, never a timestamp
        ),
    )

    # ── Step 6: the transition, through THE state-change gate ────────────
    # approved → dry_run_sent is the state machine's DRY_RUN hop (§3/§7e):
    # it does not count toward rate limits or cooldowns and is terminal for
    # the dry run.  from_state="approved" is safe to hardcode: the gate's
    # approval block read the state fresh and allowed only on "approved".
    # The reason string is this module's vocabulary, greppable in
    # state_transitions alongside the other *_success/dry_run reasons.
    transition(
        conn,
        target_id=target_id,
        from_state="approved",
        to_state="dry_run_sent",
        reason="send_gate_success_dry_run",
        actor="system",
        run_id=run_id,
        step_id=step_id,
        policy_decision_id=policy_row["policy_decision_id"] if policy_row else None,
        # agent_id defaults to actor ("system") inside transition — this
        # hop is deterministic code, attributed to the system principal.
    )

    # ── Step 7: the success trace row (never skip logs) ──────────────────
    # The trace row names the artifact and the row id so the operator can
    # go from steps → outbox file without any joins.
    log_step(
        conn,
        run_id=run_id,
        step_id=step_id,
        target_id=target_id,
        tool_name=SEND_TOOL_NAME,
        agent_id="system",
        input_data={"stage": "send_email", "simulated": True},
        output_data={
            "message_id": message_id,
            "outbox_path": str(outbox_path),
            "new_state": "dry_run_sent",
            "simulated": True,
        },
        status="success",
    )
    return SendEmailResult(
        target_id=target_id,
        refused=False,
        refusal_reason="",
        message_id=message_id,
        outbox_path=str(outbox_path),
        new_state="dry_run_sent",
    )
