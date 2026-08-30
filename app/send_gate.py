"""The send gate — docs/gates.md §2.2's full preflight, the last thing
standing between an LLM-written draft and a real person's inbox (ticket B5).

WHY THIS MODULE EXISTS — an outbound send is the one irreversible action in
the harness: it lands in a real stranger's inbox and cannot be recalled.
Every other stage produces drafts, scores, and rows; this stage is the
checkpoint that decides whether a draft may become a message.  Every item of
the docs/gates.md §2.2 checklist is implemented here as its own check, each
in its own commented block naming the checklist item, so an auditor can diff
this file against the doc line by line.

DRY_RUN-ONLY BY CONSTRUCTION (the standing rule — not a configuration) — this
repo may NEVER send a real email.  Nothing in this file (or anywhere under
app/) imports a mail transport: no smtplib/aiosmtplib/poplib/imaplib, no
Gmail/Google API mail client, no socket, no HTTP POST, no subprocess.  The
only "send" that exists is app/tools/send_email.py writing the composed
message to ``data/outbox/{message_id}.eml`` — that file write IS the send.
This is enforced structurally, not by a flag: tests/test_send_gate.py walks
every app/ module with ``ast`` and fails if any mail-transport import (or a
string that could feed a dynamic import of one) appears, and asserts
pyproject.toml declares no mail-transport dependency.  There is no ``LIVE``
branch and no ``send_mode`` anywhere — adding a transport would be a
deliberate edit to that test, not a config flip.  The ``approved → sent``
transition exists in docs/state-machine.md §3 for a future Phase 2; nothing
in this repo reaches it (state-machine.md §7i records that fact).

REFUSALS ARE THE DEFAULT PATH, AND THEY ARE RECORDED — every evaluation
writes exactly one ``send_gate_decisions`` row (allow OR refuse) through the
write gate, so a refused send leaves an audit trail and the target stays in
``approved`` for retry once the blocking condition is fixed.  TWO
schema-forced exceptions exist, documented at their check sites: an unknown
target_id and a target whose contact_id is NULL cannot produce a
send_gate_decisions row at all, because the table's contact_id column is
``INTEGER``-style NOT NULL — writing a row there would require inventing a
contact that does not exist, which is worse than omitting the row.  Both
cases still return a refusal the caller logs.

STRUCTURAL GAPS (reported, not papered over — see the reason strings): the
gate requires ``contact.email_verified`` (get_targets hardcodes it to 0 and
no verification path exists), ``policy_check_passed`` (no draft-content
policy runner exists in the repo), and ``injection_scan_passed`` (the
Guardrails AI scanner from docs/open-questions.md item 8 is not
implemented).  A NULL gate column is "no check has run", which is NOT
"passed" — so the gate correctly refuses every real target today, DRY_RUN
included (docs/gates.md §2.3a: the test must prove the GATE works, not just
that content gets generated).  The allow path is exercised by tests that
seed the gate columns directly, which is normal fixture practice.

RULE-ID ATTRIBUTION (ticket H6) — every refusal writes its policy-matrix
rule ID(s) into ``send_gate_decisions.matched_rules_json`` (same shape and
semantics as ``policy_decisions.matched_rules_json``) ALONGSIDE the prose
``reasons_json``, never as a replacement, AND returns the same list on the
``SendGateDecision`` model (``matched_rules``), so a caller handling a
refusal sees the rule(s) without re-querying the table.  The mapping is:
  - human-approval item → P1 (no operator approved state)
  - email/domain in suppressions → P2
  - contact.email / email_verified / fit_label / signals → P3 (data completeness)
  - fit_score >= 60 → P4 (confidence floor)
  - kill switch off → P6 (dominates)
  - the §2.2a rate limits / cooldown / per-thread rule → P7
    (P7's literal wording bounds auto-DRAFTS per rolling window; these are
    the send-side analogue of the same no-autonomous-resend principle)
  - draft passed the prompt-injection scan → P8 (injection markers)
Deliberately UNMAPPED (no clean P1-P9 rule — recorded in prose reasons only):
the draft-object content item, the length-and-content-policy item, the
approve_with_edits re-check, and the PolicyGateDecision==allow read (the
latter is the aggregate of the policy gate's own rules, whose rule IDs live
in the policy_decisions row it reads).

RATE LIMITS (docs/gates.md §2.2a) — implemented and testable, but inert on
real data: they count prior REAL sends (messages rows whose status is not
the dry-run status), and only DRY_RUN rows can exist today, so the counters
stay at zero.  The counters deliberately EXCLUDE ``dry_run_sent`` rows —
§2.3a: a DRY_RUN send must not consume rate limits or cooldown windows.
"""

import json  # serializing the decision row's reasons/missing-requirements JSON columns
from datetime import datetime, timedelta, timezone  # the rate-limit window arithmetic (UTC, matching the DB's second-precision UTC timestamps)

from pydantic import BaseModel  # SendGateDecision: the structured verdict every caller consumes (CLAUDE.md §7)

from app.db import normalize_domain, normalize_email  # F1b: THE shared suppression matching-key helpers — one definition, every read/write calls them
from app.ids import new_id  # "sg" ids for send_gate_decisions rows
from app.kill_switch import read_kill_switch  # the fail-closed, uncached switch reader (B4a) — reused, never re-implemented
from app.write_gate import commit as write_gate_commit  # THE core-table write path — the decision row is written through it, never a raw INSERT

# ── Thresholds and vocabulary (docs/gates.md §2.2 / §2.2a) ──────────────────
# Each constant is a checklist value lifted verbatim from the doc, so the doc
# and the enforcement cannot drift without a visible edit here.

# §2.2a per-mailbox limits: 20 sends/day, 5 sends/hour.  "Mailbox" in v1 is
# keyed by the target's offer_id — the messages table has no from-address
# column (adding one is a schema change out of B5's scope), and v1 is
# self-use with one active offer, so offer_id is the only sender proxy that
# exists in the schema.  The approximation FAILS SAFE: if two offers ever
# share one real mailbox, this gate over-refuses (20/offer instead of
# 20/mailbox), which is the conservative direction for an outbound guard.
MAILBOX_DAILY_LIMIT = 20
MAILBOX_HOURLY_LIMIT = 5

# §2.2a per-recipient-domain limit: max 2 sends/day.
DOMAIN_DAILY_LIMIT = 2

# §2.2a per-contact cooldown: 21 days between outbound sends to one contact.
CONTACT_COOLDOWN_DAYS = 21

# §2.2 checklist item: the signals list must hold at least one entry with
# strength >= 0.6 (scoring-rules.md's signal-strength scale).
SIGNAL_STRENGTH_FLOOR = 0.6

# §2.2 checklist item: icp_assessment.fit_score >= 60 (policy-matrix.md P4's
# floor, applied to the persisted deterministic score).
FIT_SCORE_FLOOR = 60

# The status app/tools/send_email.py writes on the messages row of a DRY_RUN
# send — the target state is also named dry_run_sent (state-machine.md §3),
# so the vocabulary stays one word across tables.  This is the ONLY status
# the rate-limit counters exclude (§2.3a: dry runs must not consume limits).
DRY_RUN_STATUS = "dry_run_sent"

# The review decisions that count as an approval for checklist item "a human
# review record exists with decision approved".  approve_with_edits is an
# approval exactly as much as approve is (human-review.md §3; B4b maps both
# to the approved state) — the checklist's word "approved" predates the
# five-decision vocabulary.
APPROVAL_DECISIONS = ("approve", "approve_with_edits")

# The non-fit label the checklist item "icp_assessment.fit_label !=
# \"not_fit\"" corresponds to.  The repo's actual vocabulary (schemas.md's
# FitLabel Literal) names the non-fit tier "not_target" — the checklist's
# "not_fit" spelling is the older spec wording for the same tier, and both
# spellings are refused here so neither can slip through.
NON_FIT_LABELS = ("not_target", "not_fit")

# The unsubscribe token B3's deterministic footer carries
# (_compose_footer in app/agents/draft.py): "[unsubscribe: {UNSUBSCRIBE_URL}]".
# There is no real unsubscribe URL yet — nothing is ever sent — and this
# module must NOT invent a URL scheme or domain.  The token's presence is
# what the checklist's "unsubscribe link" item can honestly require today:
# the footer exists and its unsubscribe slot is present.  Substituting a
# real link is future LIVE-mode work, and a LIVE mode does not exist.
UNSUBSCRIBE_TOKEN = "[unsubscribe:"


class SendGateDecision(BaseModel):
    """The gate's verdict on one send attempt — the structured output every
    caller (the send tool, the CLI, tests) consumes.

    ``allowed`` is the operative bit.  ``reasons`` carries the human-readable
    explanations (the specific reason strings the per-rule tests assert);
    ``missing_requirements`` carries the unmet checklist items by name.
    ``suppression_hit`` / ``approval_verified`` / ``kill_switch_active`` are
    the three boolean columns the send_gate_decisions table records, so the
    row answers "why was this refused?" without re-running the gate.
    ``matched_rules`` (ticket H6) carries the policy-matrix rule IDs behind
    the refusal — the same list that is persisted to the row's
    matched_rules_json — so a caller handling the refusal can see the rule(s)
    on the object it already holds, without re-querying the table.  Empty on
    allow, matching PolicyGateDecision's allow precedent.
    """

    allowed: bool  # True only when every §2.2 checklist item passed — anything else refuses
    reasons: list[str]  # why the verdict is what it is; empty only on allow (where a positive note is appended)
    missing_requirements: list[str]  # the unmet checklist items, named so an auditor can diff against §2.2
    matched_rules: list[str]  # the policy-matrix rule IDs behind the refusal; empty on allow — mirrors PolicyGateDecision.matched_rules (H6)
    suppression_hit: bool  # True when the email OR the domain check hit the suppressions list
    approval_verified: bool  # True when a review approval row exists AND the target is actually in approved
    kill_switch_active: bool  # read_kill_switch().engaged at evaluation time — recorded on the decision row


def _now_utc_str() -> str:
    """The current UTC time formatted EXACTLY like the DB's datetime('now')
    timestamps ("YYYY-MM-DD HH:MM:SS", second precision).  The A2 finding
    pinned both dialects to this byte-identical format, so lexicographic
    string comparison against stored created_at values is a correct time
    comparison — no parsing, no dialect drift."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _cutoff(days: float = 0, hours: float = 0) -> str:
    """A window-start timestamp (UTC, DB format) for the rate-limit checks.
    days/hours are subtracted from now — used to answer "how many real sends
    happened in the last 24h / 1h / 21 days?" by a plain string comparison."""
    return (datetime.now(timezone.utc) - timedelta(days=days, hours=hours)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _record_rule(matched_rules: list[str], rule: str) -> None:
    """Append ``rule`` to the matched-rules accumulator once (ticket H6).

    matched_rules_json is a RULE list, not a checklist-item list: several §2.2
    checklist items can trip the SAME policy-matrix rule (mailbox-daily and
    mailbox-hourly are both P7; email-suppressed and domain-suppressed are
    both P2), and the column should answer "which rules denied this send" —
    a set, not a count of fired items.  The first-occurrence order is
    preserved (the order the checks run in), matching policy_decisions'
    insertion-order precedent.
    """
    if rule not in matched_rules:  # a rule already recorded is not recorded again
        matched_rules.append(rule)


def _record_decision(
    conn,
    *,
    target_id: str,
    contact_id: str,
    run_id: str,
    step_id: str,
    allowed: bool,
    reasons: list[str],
    missing: list[str],
    matched_rules: list[str],
    suppression_hit: bool,
    approval_verified: bool,
    kill_switch_active: bool,
    policy_decision_id: str | None,
) -> SendGateDecision:
    """Write exactly one send_gate_decisions row through the write gate and
    return the verdict.  Called for EVERY evaluation whose target has a
    contact_id — allow and refuse alike (gates.md §2.2: "the reason is
    recorded in send_gate_decisions"; a refused send with no row is the
    failure mode this gate exists to prevent).

    reasons_json carries an OBJECT, not a bare array: {"simulated": bool,
    "reasons": [...]}.  gates.md §2.3a requires the allow row to carry
    ``simulated: true`` INSIDE reasons_json (never in the allowed column,
    which is INTEGER NOT NULL) so a reader of the audit trail can tell an
    allowed row was a dry run, not a real send.  Refusals carry
    simulated: false — nothing was simulated either, the gate said no.

    matched_rules (ticket H6) carries the policy-matrix rule IDs behind the
    refusal, recorded ALONGSIDE the prose reasons in their own JSON column —
    never as a replacement for the reasons.  It is empty on allow, matching
    the policy_decisions allow precedent (app/policy.py appends nothing to
    matched_rules when no rule fires).
    """
    # "sg" is a new id prefix in the established new_id style, matching the
    # table's PK name send_gate_id (same shape as wr/trn/step/tgt/msg/dv).
    send_gate_id = new_id("sg")
    write_gate_commit(
        conn,
        action="insert_send_gate_decision",  # B5's new KNOWN_ACTION — the verdict write is audited distinctly in write_log
        table_name="send_gate_decisions",
        record_id=send_gate_id,
        # The audit payload mirrors the row so the write_log entry is
        # self-describing without joining the decision table.
        payload={
            "allowed": allowed,
            "reasons": reasons,
            "missing_requirements": missing,
            "matched_rules": matched_rules,
            "suppression_hit": suppression_hit,
            "approval_verified": approval_verified,
            "kill_switch_active": kill_switch_active,
            "simulated": allowed,  # allowed rows are ALWAYS simulated: no live path exists in the repo (AST-enforced)
        },
        run_id=run_id,
        step_id=step_id,
        actor="system",  # deterministic code evaluates the gate
        agent_id="system",  # attributed to the registered deterministic principal
        policy_decision_id=policy_decision_id,  # the matched allow row, when one exists — links the verdict to the policy that permitted it
        sql="""
            INSERT INTO send_gate_decisions
                (send_gate_id, run_id, step_id, target_id, contact_id, allowed,
                 reasons_json, missing_requirements_json, matched_rules_json,
                 suppression_hit, approval_verified, kill_switch_active, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
        """,
        params=(
            send_gate_id,
            run_id,
            step_id,
            target_id,
            contact_id,
            1 if allowed else 0,
            json.dumps({"simulated": allowed, "reasons": reasons}),
            json.dumps(missing),
            json.dumps(matched_rules),
            1 if suppression_hit else 0,
            1 if approval_verified else 0,
            1 if kill_switch_active else 0,
        ),
    )
    return SendGateDecision(
        allowed=allowed,
        reasons=reasons,
        missing_requirements=missing,
        matched_rules=matched_rules,
        suppression_hit=suppression_hit,
        approval_verified=approval_verified,
        kill_switch_active=kill_switch_active,
    )


def evaluate_send_gate(conn, *, target_id: str, run_id: str, step_id: str) -> SendGateDecision:
    """Run the FULL docs/gates.md §2.2 preflight for one target and record
    the verdict.  Returns the decision; writes exactly one
    send_gate_decisions row (except the two schema-forced exceptions
    documented at their check sites).

    The checks accumulate — every failing item lands in reasons and
    missing_requirements together, so one evaluation tells the operator
    EVERYTHING that blocks the send, not just the first problem.  The one
    deliberate short-circuit is the kill switch, mirroring P6
    (policy-matrix.md): an engaged switch denies unconditionally and the
    other verdicts are irrelevant while it is on.

    Never raises for a "bad" target — a refusal is the normal outcome, not
    an exception.  Reads only; the single write is the decision row above.
    """
    # Accumulators: built across every check block below and fed into the
    # decision row (and the returned model) at the end.  Lists, not
    # globals — every branch appends without coordination.
    reasons: list[str] = []
    missing: list[str] = []
    # matched_rules (ticket H6): the policy-matrix rule IDs behind each
    # refusal, appended at the same moment the reason/missing lists are, so
    # the audit trail answers "every send P2 refused" without parsing prose.
    # Only checklist items with a clean P1-P9 mapping append — items with no
    # clean rule (draft-content, the PolicyGateDecision read) stay in the
    # prose reasons only, reported in the send_gate module docstring.
    matched_rules: list[str] = []
    suppression_hit = False
    approval_verified = False

    # ── CHECKLIST ITEM: target_id exists and is active ────────────────────
    # "Active" has no dedicated column on targets — liveness is encoded in
    # state, and the operative form of "active enough to send" is
    # state='approved', enforced in the approval block below.  This check
    # establishes the row exists at all; the contact_id it returns is also
    # REQUIRED for the decision row (contact_id is NOT NULL there).
    target_row = conn.execute(
        "SELECT t.contact_id, t.offer_id, t.account_id, t.state "
        "FROM targets t WHERE t.target_id=?;",
        (target_id,),
    ).fetchone()
    if target_row is None:
        # SCHEMA-FORCED EXCEPTION (no decision row): send_gate_decisions.contact_id
        # is NOT NULL and a phantom target has no contact to record.  Inventing
        # one would put a lying row in the audit trail.  The refusal itself is
        # still returned — the caller (send tool / CLI) logs it.
        return SendGateDecision(
            allowed=False,
            reasons=[f"unknown target {target_id!r} — no targets row exists"],
            missing_requirements=["target_id exists and is active"],
            matched_rules=[],  # structural refusal — no policy rule was evaluated (no target row exists to evaluate)
            suppression_hit=False,
            approval_verified=False,
            kill_switch_active=False,  # nothing is recorded, so the switch is not consulted (see the module docstring)
        )
    contact_id = target_row["contact_id"]

    # ── CHECKLIST ITEM: contact.email present (via the contact row) ───────
    # The contact must exist AND carry an email — the address is what the
    # suppression checks, the domain limit, and the message itself need.
    if contact_id is None:
        # SCHEMA-FORCED EXCEPTION (no decision row), second and last: a
        # company-only lead (contact_id NULL, allowed by CSV import) has no
        # contact to write into the NOT NULL column.  Same reasoning as the
        # unknown-target case: no invented contact ids in the audit trail.
        return SendGateDecision(
            allowed=False,
            reasons=[
                f"target {target_id!r} has no contact (contact_id is NULL) — "
                "there is no address to send to"
            ],
            missing_requirements=["contact.email present", "contact.email_verified == true"],
            matched_rules=[],  # structural refusal — no policy rule was evaluated (there is no contact to evaluate)
            suppression_hit=False,
            approval_verified=False,
            kill_switch_active=False,
        )
    contact_row = conn.execute(
        "SELECT email, email_verified FROM contacts WHERE contact_id=?;",
        (contact_id,),
    ).fetchone()
    if contact_row is None:
        # A dangling FK — the target points at a contact that does not
        # exist.  Refuse loudly (and record: contact_id IS available here,
        # so the decision row can be written normally).
        missing.append("contact.email present")
        reasons.append(
            f"contact {contact_id!r} referenced by target {target_id!r} does not exist"
        )
        _record_rule(matched_rules, "P3")  # data completeness: no contact row means contact.email cannot be satisfied
        email = None
    else:
        email = contact_row["email"]
        if not email:
            # No address recorded — nothing to send to, nothing to check
            # against suppressions either.  The send is refused; the
            # email-dependent checks below are skipped (their data does not
            # exist) with this one requirement naming why.
            missing.append("contact.email present")
            reasons.append("the contact has no email address recorded")
            _record_rule(matched_rules, "P3")  # data completeness: contact.email is one of P3's required fields (verified_email's prerequisite)

    # ── CHECKLIST ITEM: kill switch is off (checked FIRST — P6 dominates) ─
    # read_kill_switch() is B4a's reader: uncached and fail-closed (a
    # missing/unreadable/malformed switch file reads ENGAGED), so this item
    # can never be satisfied by deleting a file.  Mirrors policy_check_phase1's
    # P6 placement: an engaged switch denies unconditionally, and the other
    # verdicts are deliberately NOT evaluated while it is on — their
    # reasons would dilute the one thing the operator needs to see.  The
    # switch state is recorded on the decision row (kill_switch_active).
    kill_state = read_kill_switch()
    if kill_state.engaged:
        return _record_decision(
            conn,
            target_id=target_id,
            contact_id=contact_id,
            run_id=run_id,
            step_id=step_id,
            allowed=False,
            reasons=[f"kill switch engaged — all outbound sends denied (P6). Switch reason: {kill_state.reason}"],
            missing=["kill switch is off"],
            matched_rules=["P6"],  # the checklist item maps to policy-matrix P6 — kill switch dominates, the other verdicts are not evaluated
            suppression_hit=False,  # not evaluated — the switch dominates (P6 precedent)
            approval_verified=False,  # not evaluated, same reason
            kill_switch_active=True,
            policy_decision_id=None,
        )

    # ── CHECKLIST ITEM: contact.email_verified == true ────────────────────
    # THE §2 FINDING, enforced honestly: app/tools/get_targets.py hardcodes
    # email_verified=0 on CSV import ("contact email is not yet verified")
    # and the NeverBounce/waterfall verification step
    # (docs/waterfall-enrichment.md) is NOT implemented in this repo — so
    # this check correctly refuses every real target today, DRY_RUN
    # included.  That refusal is the CORRECT behaviour (gates.md §2.3a: the
    # test must prove the gate works) and must not be "fixed" by relaxing
    # the check.  Tests exercise the allow path by seeding email_verified=1
    # in fixtures — normal DB seeding, not a weakening.
    if email and not contact_row["email_verified"]:
        missing.append("contact.email_verified == true")
        reasons.append(
            "contact.email_verified is 0 — no email-verification path has run "
            "(get_targets imports every address unverified and the waterfall "
            "enrichment's verification step is not implemented)"
        )
        _record_rule(matched_rules, "P3")  # verified_email is one of P3's required fields — data completeness

    # ── CHECKLIST ITEMS: email and domain not in suppressions ─────────────
    # suppressions.email is the address AS WRITTEN (the audit record — NOT
    # the primary key since ticket H4b); suppressions.email_normalized is
    # the UNIQUE F1b matching key.  Both are hard refusals — a suppressed
    # address must never be contacted, ever (CLAUDE.md §9).  The probe
    # address is normalised through THE shared helper, so local-part case,
    # domain case, and plus-tag aliases all fold to the same key the writer
    # stored.  The domain is derived from the address via normalize_domain()
    # (case-insensitive per RFC 1035).
    if email:
        email_key = normalize_email(email)  # the probe's canonical key — same folds the suppression writers applied on write
        domain = normalize_domain(email.split("@", 1)[-1]) if "@" in email else None
        email_suppressed = (
            conn.execute(
                "SELECT 1 FROM suppressions WHERE email_normalized=?;", (email_key,)
            ).fetchone()
            is not None
        )
        if email_suppressed:
            suppression_hit = True
            missing.append("contact.email not in suppressions")
            reasons.append(f"contact email {email!r} is on the suppression list")
            _record_rule(matched_rules, "P2")  # P2: "any target present in suppressions → deny" — email half
        domain_suppressed = (
            domain is not None
            and conn.execute(
                "SELECT 1 FROM suppressions WHERE domain=?;", (domain,)
            ).fetchone()
            is not None
        )
        if domain_suppressed:
            suppression_hit = True
            missing.append("contact.domain not in suppressions")
            reasons.append(f"contact domain {domain!r} is on the suppression list")
            _record_rule(matched_rules, "P2")  # P2: "any target present in suppressions → deny" — domain half

    # ── CHECKLIST ITEMS: icp_assessment.fit_label and fit_score ───────────
    # Both read the DETERMINISTIC assessment persisted on the account
    # (accounts.icp_fit_* — what the scoring formula produced).  The judge's
    # verdict (accounts.judge_*) is deliberately not consulted here: the
    # checklist says "icp_assessment", and policy-matrix.md P4 pins the
    # numeric floor to the deterministic score only.
    account_row = conn.execute(
        "SELECT icp_fit_label, icp_fit_score FROM accounts WHERE account_id=?;",
        (target_row["account_id"],),
    ).fetchone()
    label = account_row["icp_fit_label"] if account_row is not None else None
    # H6 mapping choice (ticket H6, verified against policy-matrix.md):
    # the whole checklist item "icp_assessment.fit_label != 'not_fit'" maps
    # to P3 (data completeness), not P4 (confidence floor).  Reasoning:
    # P3's required-field list names icp_assessment.fit_label EXPLICITLY, and
    # P4's list names icp_assessment.fit_score (a different field) — the
    # doc puts the label's presence under P3 and the numeric floor under P4.
    # The gate already records fit_score < 60 as its OWN checklist item
    # tagged P4 below, so tagging the label item P3 keeps each checklist item
    # to exactly one rule and keeps P4 on the field the doc assigns it to.
    if label is None:
        # No label persisted — scoring never ran or never landed on this
        # account.  A send without an ICP verdict is refused: the harness
        # has no basis to contact this company.
        missing.append("icp_assessment.fit_label != 'not_fit'")
        reasons.append("no icp_assessment.fit_label is recorded for this account")
        _record_rule(matched_rules, "P3")  # missing required field — P3's data-completeness list names icp_assessment.fit_label
    elif label in NON_FIT_LABELS:
        # The non-fit tier — contacting a company the scoring pipeline
        # judged non-fit is exactly what the gate exists to stop.
        missing.append("icp_assessment.fit_label != 'not_fit'")
        reasons.append(
            f"icp_assessment.fit_label is {label!r} — a non-fit target must not be contacted"
        )
        _record_rule(matched_rules, "P3")  # the label check is the fit_label field's P3 requirement; P4 is the score item below
    score = account_row["icp_fit_score"] if account_row is not None else None
    if score is None:
        # Same shape as the missing label: no number, no basis, no send.
        missing.append("icp_assessment.fit_score >= 60")
        reasons.append("no icp_assessment.fit_score is recorded for this account")
        _record_rule(matched_rules, "P4")  # P4's floor cannot be cleared without a score — fail closed, same rule
    elif score < FIT_SCORE_FLOOR:
        # P4's floor, applied at the gate: below 60 is a hard refusal
        # regardless of any other condition (a score of exactly 60 passes).
        missing.append("icp_assessment.fit_score >= 60")
        reasons.append(
            f"icp_assessment.fit_score is {score}, below the {FIT_SCORE_FLOOR}-point floor"
        )
        _record_rule(matched_rules, "P4")  # policy-matrix P4: "fit_score < 60 → deny"

    # ── CHECKLIST ITEM: signals has ≥1 entry with strength >= 0.6 ─────────
    # Scoped to the LATEST research run (the run_id whose rows were created
    # last) — the same scoping app/agents/draft.py's brief builder uses, so
    # the gate judges the same evidence the draft was written from.  A
    # target with no signals (or only weak ones) has no trigger to justify
    # a cold outreach.
    strength_rows = conn.execute(
        "SELECT signal_strength FROM signals WHERE target_id=? AND run_id=("
        "SELECT run_id FROM signals WHERE target_id=? ORDER BY created_at DESC LIMIT 1);",
        (target_id, target_id),
    ).fetchall()
    strengths = [r["signal_strength"] for r in strength_rows]
    if not strengths:
        missing.append("signals list has at least 1 entry with strength >= 0.6")
        reasons.append("no signals are recorded for the latest research run")
        _record_rule(matched_rules, "P3")  # P3's signals[].signal_type requirement, extended to the send-side sufficiency floor
    elif not any(s >= SIGNAL_STRENGTH_FLOOR for s in strengths):
        missing.append("signals list has at least 1 entry with strength >= 0.6")
        reasons.append(
            f"no signal reaches strength {SIGNAL_STRENGTH_FLOOR} "
            f"(strongest recorded: {max(strengths)})"
        )
        _record_rule(matched_rules, "P3")  # data completeness on signals — the ticket's mapping of this checklist item

    # ── CHECKLIST ITEM: a human review record exists with decision approved
    # CLAUDE.md §9's non-negotiable rule, enforced literally: the approval
    # is a recorded review_decisions row (B4b wrote it), NOT the target
    # merely sitting in state "approved" — being in the state without a row
    # would be a corrupt audit trail and must not send.  The state is ALSO
    # checked: the recorded approval must have taken effect (B4b moves them
    # together, so a mismatch means something superseded the decision).
    #
    # THE ORDERING FIX (ticket B5) — this read IS the highest-stakes
    # ordering in the repo, and it was wrong: ORDER BY created_at DESC
    # picked an ARBITRARY row when two decisions landed in the same second
    # (created_at is second-precision TEXT — datetime('now')), so an
    # operator who approved and then approved-with-edits inside one second
    # could have the gate resolve to the OLDER decision — refusing valid
    # work, or (in a future LIVE mode) sending text the operator did not
    # approve.  The fix: insert_seq DESC is the primary key — a monotonic
    # integer written by every insert site (review.py) via a scalar
    # MAX+1 subquery, so the order is total and clock-independent.
    # created_at DESC remains as the tiebreaker for legacy rows written
    # before the column existed (their insert_seq is NULL, which sorts
    # LAST in DESC on both SQLite and Postgres — chronologically correct,
    # since legacy rows predate seq-carrying rows).
    review_row = conn.execute(
        "SELECT review_decision_id, decision, edited, draft_message_id "
        "FROM review_decisions WHERE target_id=? "
        "ORDER BY insert_seq DESC, created_at DESC LIMIT 1;",
        (target_id,),
    ).fetchone()
    if review_row is None:
        missing.append("a human review record exists with decision approved")
        reasons.append(
            "no review_decisions row exists for this target — CLAUDE.md §9: "
            "approval is a recorded operator decision, never implied by state"
        )
        _record_rule(matched_rules, "P1")  # P1: "any send_email without an operator approved state → deny" — no approval record at all
    elif review_row["decision"] not in APPROVAL_DECISIONS:
        missing.append("a human review record exists with decision approved")
        reasons.append(
            f"the latest review decision is {review_row['decision']!r}, "
            "not approve/approve_with_edits"
        )
        _record_rule(matched_rules, "P1")  # P1: the recorded verdict is not an approval — still "without an operator approved state"
    elif target_row["state"] != "approved":
        missing.append("a human review record exists with decision approved")
        reasons.append(
            f"target is in state {target_row['state']!r}, not approved — "
            "the recorded approval has not taken effect"
        )
        _record_rule(matched_rules, "P1")  # P1: the approval exists but has not taken effect — the send is still without an approved state
    else:
        approval_verified = True

    # ── CHECKLIST ITEMS: the draft object, its gates, the edited re-check ─
    # All draft checks read the revision the operator actually approved —
    # review_decisions.draft_message_id (the db-schema contract: it holds a
    # draft_version_id, not a messages id).  Sending the approved text, and
    # only the approved text, is the point of the whole review stage.
    if approval_verified:
        revision = conn.execute(
            "SELECT draft_version_id, revision_number, subject, body, footer, "
            "policy_check_passed, injection_scan_passed "
            "FROM message_draft_versions WHERE draft_version_id=?;",
            (review_row["draft_message_id"],),
        ).fetchone()
        if revision is None:
            # The decision references a revision that does not exist — a
            # corrupt reference.  Refuse; nothing was approved to send.
            missing.append("draft_email object (subject, body, footer, unsubscribe link)")
            reasons.append(
                "the approved draft reference points at a missing revision"
            )
        else:
            # ── CHECKLIST ITEM: draft contains subject, body, footer, unsubscribe link
            # The three columns are NOT NULL in the DDL, so their presence
            # is belt-and-braces; the load-bearing half is the unsubscribe
            # token in the deterministic footer (B3-Z1: the model cannot
            # author it).  A real URL cannot be required — none exists, and
            # inventing a URL scheme/domain is forbidden — so the token is
            # what "unsubscribe link" honestly means until a LIVE mode
            # exists (it does not, and must not be added here).
            if not (revision["subject"] and revision["body"] and revision["footer"]):
                missing.append("draft_email object (subject, body, footer, unsubscribe link)")
                reasons.append("the approved revision is missing subject/body/footer text")
            elif UNSUBSCRIBE_TOKEN not in revision["footer"]:
                missing.append("draft_email object (subject, body, footer, unsubscribe link)")
                reasons.append(
                    f"the draft footer carries no unsubscribe token ({UNSUBSCRIBE_TOKEN}...) — "
                    "the compliance footer is missing or was not composed by deterministic code"
                )

            # ── CHECKLIST ITEM: draft passed length and content policy ─────
            # policy_check_passed is the revision's gate column (B3-Z3: the
            # draft agent may not set its own gates).  NULL means "no check
            # has run" — and "no check has run" is NOT "passed", so NULL
            # fails closed exactly like a 0.  This is also a structural gap:
            # no draft-content policy runner exists in the repo, so every
            # real revision is NULL here and is refused (reported, not
            # papered over — the allow path is fixture-seeded).
            if revision["policy_check_passed"] != 1:
                missing.append("draft passed length and content policy")
                reasons.append(
                    "policy_check_passed is not 1 on the approved revision "
                    f"(value: {revision['policy_check_passed']!r}) — NULL means no check "
                    "has run, and no check is not a pass (fail closed)"
                )

            # ── CHECKLIST ITEM: draft passed the prompt-injection scan ─────
            # Same NULL doctrine.  The scanner itself is pinned by
            # docs/open-questions.md item 8 to Guardrails AI for v1 and is
            # NOT implemented — this module deliberately adds no dependency
            # and no fake scanner (a fake pass would prove nothing about
            # the gate).  "Scan not run" is recorded as a missing
            # requirement; a real scanner is future work.
            if revision["injection_scan_passed"] != 1:
                missing.append("draft passed the prompt-injection scan")
                reasons.append(
                    "injection_scan_passed is not 1 on the approved revision "
                    f"(value: {revision['injection_scan_passed']!r}) — the Guardrails AI "
                    "scanner (open-questions.md item 8) is not implemented, and "
                    "'scan not run' is not 'passed' (fail closed)"
                )
                _record_rule(matched_rules, "P8")  # P8: injection markers cannot be used to draft/send outbound — the draft must pass the scan

            # ── CHECKLIST ITEM: approve_with_edits must independently re-pass
            # docs/human-review.md §5: the EDITED revision must re-pass
            # policy, the injection scan, and this checklist in full.  The
            # two column checks above already read the EDITED revision's own
            # columns — B4b wrote them NULL, so an edit that never re-passed
            # is refused here, and the ORIGINAL revision's columns are never
            # consulted (they could be 1 while the edit's are NULL, and the
            # edit is what would be sent).  The one additional integrity
            # check: the approved revision must be the LATEST revision for
            # the target — the operator approved a specific text, and if a
            # newer revision exists the approved text is not what the send
            # would deliver.
            latest_number = conn.execute(
                "SELECT MAX(revision_number) AS n FROM message_draft_versions "
                "WHERE target_id=?;",
                (target_id,),
            ).fetchone()["n"]
            if revision["revision_number"] != latest_number:
                missing.append(
                    "if approve_with_edits was used, the edited revision has "
                    "independently passed policy_check, the injection scan, and "
                    "this checklist again"
                )
                reasons.append(
                    f"the approved revision is #{revision['revision_number']} but the "
                    f"latest revision is #{latest_number} — refusing to send text the "
                    "operator did not approve"
                )

    # ── CHECKLIST ITEM: a PolicyGateDecision.decision == allow, current ───
    # Same shape as B3's drafting precondition (app/agents/draft.py): the
    # LATEST policy_decisions row for the target must be "allow".  No row
    # at all → fail closed (policy-matrix.md: an unmapped action resolves
    # to deny).  The ordering key is insert_seq DESC (the B5 fix — the old
    # ORDER BY created_at could pick an arbitrary same-second row; the
    # hazard is documented at the review-row read above), with created_at
    # as the legacy-row tiebreaker.
    policy_row = conn.execute(
        "SELECT policy_decision_id, decision FROM policy_decisions "
        "WHERE target_id=? ORDER BY insert_seq DESC, created_at DESC LIMIT 1;",
        (target_id,),
    ).fetchone()
    if policy_row is None:
        missing.append("a PolicyGateDecision.decision == allow is present and current")
        reasons.append(
            "no policy_decisions row exists for this target — fail closed "
            "(policy-matrix.md: an unmapped action resolves to deny)"
        )
    elif policy_row["decision"] != "allow":
        missing.append("a PolicyGateDecision.decision == allow is present and current")
        reasons.append(
            f"the latest policy decision is {policy_row['decision']!r}, not allow"
        )

    # ── CHECKLIST ITEMS: §2.2a rate limits ────────────────────────────────
    # The counters read the messages table: a "real send" is an outbound
    # messages row whose status is NOT the dry-run status — i.e. the
    # exclusion IS the §2.3a exemption, so DRY_RUN rows can never consume a
    # limit (tested explicitly).  Only dry-run rows can exist today, so the
    # counters stay at zero on real data; the logic exists and is tested
    # with seeded rows because it is a gate rule, not because it fires.
    # state_transitions is deliberately not used for counting: messages is
    # the send ledger, and counting from two tables would risk double-counts
    # once C1's reply flow starts writing both.
    #
    # H6 mapping: all five §2.2a limits (mailbox daily/hourly, domain daily,
    # cooldown, per-thread) tag as P7.  P7's literal wording bounds
    # auto-DRAFTS per rolling window; these bounds are the send-side analogue
    # of the same no-autonomous-resend principle, and no other policy-matrix
    # rule covers rate limiting — so P7 is the rule these refusals surface.
    cutoff_21d = _cutoff(days=CONTACT_COOLDOWN_DAYS)  # the widest window — one fetch serves all four checks
    recent_real_sends = conn.execute(
        """
        SELECT m.contact_id, m.thread_id, m.created_at, t.offer_id, c.email
        FROM messages m
        JOIN targets t ON m.target_id = t.target_id
        JOIN contacts c ON m.contact_id = c.contact_id
        WHERE m.direction='outbound' AND m.status != ? AND m.created_at >= ?;
        """,
        (DRY_RUN_STATUS, cutoff_21d),
    ).fetchall()

    # Per-mailbox daily + hourly (the offer_id keying is documented at
    # MAILBOX_DAILY_LIMIT).  The window comparison is a plain string
    # comparison — created_at and the cutoffs share the exact DB timestamp
    # format (see _now_utc_str).
    cutoff_24h = _cutoff(hours=24)
    cutoff_1h = _cutoff(hours=1)
    mailbox_rows = [r for r in recent_real_sends if r["offer_id"] == target_row["offer_id"]]
    daily_count = sum(1 for r in mailbox_rows if r["created_at"] >= cutoff_24h)
    if daily_count >= MAILBOX_DAILY_LIMIT:
        missing.append("rate limit per mailbox not exceeded (§2.2a)")
        reasons.append(
            f"per-mailbox daily limit reached: {daily_count} real sends in the "
            f"last 24h (limit {MAILBOX_DAILY_LIMIT})"
        )
        _record_rule(matched_rules, "P7")  # §2.2a mailbox-daily — the send-side analogue of P7's rolling-window bound
    hourly_count = sum(1 for r in mailbox_rows if r["created_at"] >= cutoff_1h)
    if hourly_count >= MAILBOX_HOURLY_LIMIT:
        missing.append("rate limit per mailbox not exceeded (§2.2a)")
        reasons.append(
            f"per-mailbox hourly limit reached: {hourly_count} real sends in the "
            f"last hour (limit {MAILBOX_HOURLY_LIMIT})"
        )
        _record_rule(matched_rules, "P7")  # §2.2a mailbox-hourly — same P7 mapping

    # Per-recipient-domain daily (limit 2).  The domain is derived from each
    # prior send's contact email in Python — no SQL string surgery, so the
    # query stays dialect-agnostic.
    if email and domain is not None:
        domain_rows = [
            r
            for r in recent_real_sends
            if r["email"]
            and r["email"].split("@", 1)[-1].lower() == domain
            and r["created_at"] >= cutoff_24h
        ]
        if len(domain_rows) >= DOMAIN_DAILY_LIMIT:
            missing.append("rate limit per recipient domain not exceeded (§2.2a)")
            reasons.append(
                f"per-domain daily limit reached: {len(domain_rows)} real sends to "
                f"domain {domain!r} in the last 24h (limit {DOMAIN_DAILY_LIMIT})"
            )
            _record_rule(matched_rules, "P7")  # §2.2a per-recipient-domain — same P7 mapping

    # Per-contact cooldown (21 days).  The fetch is already windowed to 21
    # days, so any real send to THIS contact is inside the cooldown window.
    cooldown_rows = [r for r in recent_real_sends if r["contact_id"] == contact_id]
    if cooldown_rows:
        missing.append("no duplicate send to the same recipient inside the cool-down window (§2.2a)")
        reasons.append(
            f"per-contact cooldown active: {len(cooldown_rows)} real send(s) to this "
            f"contact in the last {CONTACT_COOLDOWN_DAYS} days"
        )
        _record_rule(matched_rules, "P7")  # §2.2a per-contact cooldown — same P7 mapping

    # Per-thread rule (§2.2a): no second unprompted outbound on a thread
    # until a reply arrives on that thread.  B5's sends always open a fresh
    # thread (thread_id NULL — no threading exists until C1's reply flow),
    # so this check can only fire against prior rows carrying a thread id.
    # The rule deliberately has NO time window — an unanswered thread stays
    # blocked until a reply lands, however old it is — so this reads its own
    # unwindowed query (scoped to THIS contact: a thread belongs to a
    # conversation, and another contact's unanswered thread must not block
    # this one) instead of piggybacking the 21-day fetch above.
    thread_outbound_rows = conn.execute(
        "SELECT thread_id FROM messages WHERE contact_id=? "
        "AND direction='outbound' AND status != ? AND thread_id IS NOT NULL;",
        (contact_id, DRY_RUN_STATUS),
    ).fetchall()
    inbound_threads = {
        r["thread_id"]
        for r in conn.execute(
            "SELECT DISTINCT thread_id FROM messages WHERE contact_id=? "
            "AND direction='inbound' AND thread_id IS NOT NULL;",
            (contact_id,),
        ).fetchall()
    }
    unanswered_threads = [
        r["thread_id"]
        for r in thread_outbound_rows
        if r["thread_id"] not in inbound_threads
    ]
    if unanswered_threads:
        missing.append(
            "no second unprompted outbound send until a reply is received on that thread (§2.2a)"
        )
        reasons.append(
            f"an unanswered outbound already exists on thread(s) "
            f"{sorted(set(unanswered_threads))} — no second unprompted send until a reply arrives"
        )
        _record_rule(matched_rules, "P7")  # §2.2a per-thread no-second-unprompted — same P7 mapping

    # ── Verdict ───────────────────────────────────────────────────────────
    # allowed iff NOTHING failed.  missing is the authoritative unmet-item
    # list; reasons mirrors it with explanations (every refusal branch
    # appends to both together).  On allow, append the positive note so the
    # decision row reads as a verdict, not an empty list.
    allowed = not missing and not reasons
    if allowed:
        reasons.append(
            "all §2.2 preflight checks passed — DRY_RUN simulated send "
            "(no mail transport exists in this repository)"
        )
    return _record_decision(
        conn,
        target_id=target_id,
        contact_id=contact_id,
        run_id=run_id,
        step_id=step_id,
        allowed=allowed,
        reasons=reasons,
        missing=missing,
        matched_rules=matched_rules,
        suppression_hit=suppression_hit,
        approval_verified=approval_verified,
        kill_switch_active=kill_state.engaged,
        policy_decision_id=policy_row["policy_decision_id"] if policy_row else None,
    )
