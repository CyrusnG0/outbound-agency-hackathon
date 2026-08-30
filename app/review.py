"""Human review gate — ticket B4b: the ONE write path behind the operator's
approval UI, and the gate that makes "all outbound emails require human
approval before send" (CLAUDE.md §9) real.

WHY THIS MODULE EXISTS — the console (app/console/) is structurally
read-only by construction (A5a's two AST tests): its own code issues only
SELECT and imports no write machinery.  B4b must give the operator the five
review decisions through that console WITHOUT reopening the wall.  The
design (docs/console.md §3): every mutating statement — write_gate commits,
state_machine.transition, the kill-switch check, log_step — lives HERE, in
one plain service module with no FastAPI dependency, and the console merely
calls ``record_review_decision``.  The console stays write-free in its own
code; the write path has exactly one door.

THE FIVE DECISIONS (docs/human-review.md §3, docs/state-machine.md §3/§7 —
do not invent a sixth, do not change these mappings):

    approve              awaiting_review -> approved     (no extra effect)
    approve_with_edits   awaiting_review -> approved     + NEW draft revision (§2.1)
    reject               awaiting_review -> not_target   (no extra effect)
    reject_and_suppress  awaiting_review -> suppressed   + suppressions row (§2.2)
    escalate             awaiting_review -> researched   reason=research_escalation (pinned by
                                                         resolved open-question 14 and
                                                         docs/state-machine.md §7 — verbatim)

KILL-SWITCH ASYMMETRY (deliberate — see the comment at the check site):
an engaged switch refuses ONLY approve / approve_with_edits (the two
decisions that authorize an outbound action; policy-matrix.md P6 denies all
outbound actions unconditionally).  reject / reject_and_suppress / escalate
DE-ESCALATE — they move the target away from sending — and an operator who
just hit the emergency stop must still be able to reject and suppress the
very targets that caused it.  A kill switch that blocks the brakes as well
as the accelerator is a bug.

REFUSALS ARE OBSERVABLE, NEVER SILENT — every refusal (unknown decision
string, target not in awaiting_review, kill switch + approve, missing
contact email for suppression, no draft to edit) writes a failed
``steps`` row with tool_name="review_decision" naming the refusal reason,
and returns ReviewOutcome(refused=True).  No exception, no 500, no silent
no-op, and never a fall-through to a default action.

WRITE ORDER — the review_decisions row is written BEFORE the transition.
The invariant that must never break is CLAUDE.md §9's: no path may reach
``approved`` without an operator decision recorded in review_decisions.  If
the transition fails after the decision row landed, the target stays in
awaiting_review and the operator sees an error plus an honest decision row
(a retry creates a second decision row — an audit oddity, not a safety
violation).  The reverse order (transition first, decision row second)
would leave an approved target with no recorded approval if the decision
write failed — the one failure mode this gate exists to prevent.
"""

from pydantic import BaseModel  # ReviewDecisionRequest / ReviewOutcome: structured I/O for the gate (CLAUDE.md §7)

from app.db import normalize_email  # F1b: the ONE suppression matching-key helper — the idempotency read and the INSERT both fold the same way
from app.draft_gate import run_draft_gate  # G2: the deterministic runner that re-passes an EDITED revision's two gate columns (human-review.md §5)
from app.ids import new_id  # fresh prefixed ids: one per decision row, one per edited revision, one per step
from app.kill_switch import read_kill_switch  # the fail-closed, uncached switch reader — both the gate and the recorded state
from app.state_machine import transition  # THE state-change gate — every decision's hop goes through it, never a raw UPDATE
from app.tools.log_step import log_step  # steps-trace writer — every decision and every refusal lands in the trace (Golden Rule)
from app.write_gate import commit as write_gate_commit  # THE core-table write path — decision rows, revisions, suppressions

# ── Constants ────────────────────────────────────────────────────────────────

# The kind of principal every review write declares: the human operator.
# The registered "operator" agent (app/agents_registry.py) is the
# corresponding agent_id, so every review write is attributable to the
# operator in write_log and every decision row's actor column says
# "operator" — the write gate's actor allowlist is {"system", "operator"}.
REVIEW_ACTOR = "operator"

# The steps.tool_name every review step row carries (decisions AND
# refusals) — distinct from every pipeline tool so the trace log shows "a
# human review event happened here" at a glance.
REVIEW_TOOL_NAME = "review_decision"

# The five decision strings, verbatim from docs/human-review.md §3.  An
# allowlist, not a hint: anything not in here is refused before any write,
# so an unknown decision can never fall through to a default action
# (ticket §5: "never fall through to a default action").
VALID_DECISIONS = (
    "approve",
    "approve_with_edits",
    "reject",
    "reject_and_suppress",
    "escalate",
)

# The two decisions that AUTHORIZE an outbound action — the set the kill
# switch refuses while engaged (see the asymmetry comment at the check
# site).  approve_with_edits authorizes exactly as much as approve: the
# edited text is still an outbound email that will be sent.
_OUTBOUND_AUTHORIZING_DECISIONS = ("approve", "approve_with_edits")

# decision -> (to_state, transition_reason).  The to_state column matches
# the ticket's table exactly; the reason strings are this module's
# vocabulary (escalate's is PINNED verbatim by docs/state-machine.md §7 —
# reason=research_escalation).  The other four follow the same
# operator_* pattern so a reader of state_transitions can tell every
# review-driven hop from pipeline-driven hops at a glance.
_DECISION_TRANSITIONS = {
    "approve": ("approved", "operator_approval"),
    "approve_with_edits": ("approved", "operator_approval_with_edits"),
    "reject": ("not_target", "operator_rejection"),
    "reject_and_suppress": ("suppressed", "operator_rejection_with_suppression"),
    "escalate": ("researched", "research_escalation"),
}

# ── Structured I/O ───────────────────────────────────────────────────────────


class ReviewDecisionRequest(BaseModel):
    """One review decision as the UI (or any future caller) submits it.

    ``edited_subject``/``edited_body`` are read ONLY for
    approve_with_edits — for every other decision they are ignored form
    artifacts (the HTML form always submits the fields; a strict refusal
    on a stray non-empty field would punish a legitimate reject whose form
    still carried text, and the decision string itself is never silently
    altered).  ``research_note`` is the escalate decision's operator note,
    stored in review_decisions.reason so it travels with the decision
    (docs/state-machine.md §7: the escalation "carries the operator's
    research_note forward").
    """

    target_id: str  # which target the decision is about
    decision: str  # one of VALID_DECISIONS — anything else is refused
    reason: str = ""  # the operator's reasoning, stored in review_decisions.reason
    edited_subject: str | None = None  # approve_with_edits: the replacement subject line
    edited_body: str | None = None  # approve_with_edits: the replacement body text
    research_note: str | None = None  # escalate: what the follow-up research should look into


class ReviewOutcome(BaseModel):
    """What a decision attempt produced — the success AND the refusal shape.

    Refusals are first-class outcomes (refused=True + refusal_reason), not
    exceptions: the UI renders them, the API returns them as JSON, and the
    operator always sees WHY nothing happened.
    """

    target_id: str  # echoed so an API caller can match outcome to request without bookkeeping
    decision: str  # echoed for the same reason
    new_state: str | None  # the target's state after a successful decision; None on refusal
    review_decision_id: str | None  # the review_decisions row's id; None on refusal
    refused: bool  # True = nothing was written, nothing transitioned; refusal_reason says why
    refusal_reason: str  # "" on success — the human-readable reason otherwise


# ── The refusal path ─────────────────────────────────────────────────────────


def _record_refusal(
    conn, *, request: ReviewDecisionRequest, run_id: str, step_id: str, refusal_reason: str
) -> ReviewOutcome:
    """Log a refused decision to the steps trace and return the refusal
    outcome.  Writes NOTHING else: no review_decisions row (a refusal is
    not a decision), no transition, no suppression — the target's state is
    untouched, exactly as ticket §5 requires ("refuse ... do not
    transition").  The failed steps row is what makes the refusal
    observable rather than a silent no-op (Golden Rule: never skip logs).
    """
    log_step(
        conn,
        run_id=run_id,
        step_id=step_id,
        target_id=request.target_id,
        tool_name=REVIEW_TOOL_NAME,  # same tool_name as successful decisions: one greppable review vocabulary
        agent_id=REVIEW_ACTOR,  # the operator attempted this decision
        input_data={"stage": "review_decision", "decision": request.decision},
        output_data={"refusal_reason": refusal_reason},  # the why, in the trace — never a bare "failed"
        status="failed",  # the steps vocabulary's honest refusal status (no "refused" value exists)
    )
    return ReviewOutcome(
        target_id=request.target_id,
        decision=request.decision,
        new_state=None,
        review_decision_id=None,
        refused=True,
        refusal_reason=refusal_reason,
    )


# ── The gate ─────────────────────────────────────────────────────────────────


def record_review_decision(
    conn, *, request: ReviewDecisionRequest, run_id: str
) -> ReviewOutcome:
    """Record one operator review decision, execute its transition and its
    extra effects, and log every step.

    Checks, in order (each refusal is logged and returns refused=True):
    1. decision is one of VALID_DECISIONS (never a default fall-through);
    2. the target exists and is in ``awaiting_review`` (a double-submitted
       form must not double-decide — the second attempt finds the target
       already moved on and is refused);
    3. the kill switch (read UNCACHED, fail-closed — the same reader every
       guard uses) is not engaged, UNLESS the decision de-escalates (see
       the asymmetry comment at the check site);
    4. the target has at least one draft revision to decide on (a target
       in awaiting_review without one is a corrupt state — B3 guarantees
       awaiting_review only after a revision persisted — and refusing
       loudly beats recording a decision with no draft reference).

    On success it writes, in order: the edited revision (if any), the
    review_decisions row, the state transition, and the success step row —
    see the module docstring's WRITE ORDER for why the decision row
    precedes the transition.

    ``run_id`` groups this decision into the run that produced the draft
    (the console passes the target's latest state_transitions run_id).
    """
    # One fresh step id shared by the transition, the gated writes, and the
    # trace row — the repo's established pattern (draft.py's persist node):
    # the decision's audit entries hang together under one step.
    step_id = new_id("step")

    # ── Check 1: the decision string ──────────────────────────────────────
    # Allowlist membership, not a mapping lookup with a default: an unknown
    # string ("send_it", "", "approve ") must be REFUSED, never guessed and
    # never mapped to the nearest known decision (ticket §5).
    if request.decision not in VALID_DECISIONS:
        return _record_refusal(
            conn, request=request, run_id=run_id, step_id=step_id,
            refusal_reason=f"unknown decision {request.decision!r} — valid decisions are "
                           f"{', '.join(VALID_DECISIONS)}",
        )

    # ── Check 2: target exists and is awaiting_review ─────────────────────
    # The state read is the gate: awaiting_review is the ONLY state a
    # review decision may act on (docs/state-machine.md §3 — its three
    # outbound edges all start there).  Reading it fresh (not trusting the
    # caller) is what makes a double-submitted form a refusal: the first
    # decision moved the target, so the second read sees the new state.
    target_row = conn.execute(
        "SELECT state FROM targets WHERE target_id=?;", (request.target_id,)
    ).fetchone()
    if target_row is None:
        return _record_refusal(
            conn, request=request, run_id=run_id, step_id=step_id,
            refusal_reason=f"unknown target {request.target_id!r}",
        )
    if target_row["state"] != "awaiting_review":
        return _record_refusal(
            conn, request=request, run_id=run_id, step_id=step_id,
            refusal_reason=(
                f"target {request.target_id!r} is in state {target_row['state']!r}, "
                f"not awaiting_review — refusing to decide an already-decided target"
            ),
        )

    # ── Check 3: the kill switch (deliberate asymmetry) ───────────────────
    # Read UNCACHED at decision time (runbook.md §1 — a cached read cannot
    # see a flip, and the moment the operator decides is exactly when the
    # switch must be current).  Fail-closed: a missing/unreadable file
    # reads engaged and therefore refuses approvals — the safe direction.
    kill_state = read_kill_switch()
    # THE ASYMMETRY — deliberate, do NOT "tidy" into one blanket refusal:
    # P6 (docs/policy-matrix.md) denies all OUTBOUND actions while the
    # switch is engaged.  approve and approve_with_edits both authorize an
    # outbound send, so both are refused.  reject, reject_and_suppress and
    # escalate DE-ESCALATE — they move the target away from sending or
    # send it back to research — and an operator who has just hit the
    # emergency stop must still be able to reject and suppress the very
    # targets that caused them to hit it.  A kill switch that blocks the
    # brakes as well as the accelerator is a bug.  (The switch's own
    # reason is surfaced in the refusal so the operator sees WHY, not just
    # that.)
    if kill_state.engaged and request.decision in _OUTBOUND_AUTHORIZING_DECISIONS:
        return _record_refusal(
            conn, request=request, run_id=run_id, step_id=step_id,
            refusal_reason=(
                f"kill switch engaged — refusing {request.decision!r} "
                f"(P6: all outbound actions denied unconditionally). "
                f"Switch reason: {kill_state.reason}"
            ),
        )

    # ── Check 4: a draft revision to decide on ────────────────────────────
    # The latest revision by revision_number, with insert_seq DESC as the
    # deterministic tiebreaker (created_at is second-precision and two
    # same-second rows ordered arbitrarily — ticket B5 made that hazard
    # operational in the send gate, so this read orders on the sequence
    # column too).  It supplies review_decisions.draft_message_id (the
    # revision the decision is about — see the db-schema contract) and, for
    # approve_with_edits, the base to increment from and the footer to
    # carry over.
    latest = conn.execute(
        "SELECT draft_version_id, revision_number, footer FROM message_draft_versions "
        "WHERE target_id=? "
        "ORDER BY revision_number DESC, insert_seq DESC, created_at DESC LIMIT 1;",
        (request.target_id,),
    ).fetchone()
    if latest is None:
        return _record_refusal(
            conn, request=request, run_id=run_id, step_id=step_id,
            refusal_reason=(
                f"target {request.target_id!r} has no draft revision to decide on "
                f"— refusing to record a decision with no draft reference"
            ),
        )

    # ── Per-decision extra effects ────────────────────────────────────────
    # decision_draft_ref is what review_decisions.draft_message_id stores:
    # the NEW revision id for approve_with_edits (the operator approved
    # the edited text), the latest existing revision id otherwise.
    decision_draft_ref = latest["draft_version_id"]
    edited_flag = 0  # review_decisions.edited: 1 only for approve_with_edits
    suppression_skipped = False  # reject_and_suppress on an already-suppressed email (see below)

    if request.decision == "approve_with_edits":
        # ── §2.1: the edit is a NEW revision, never an in-place overwrite ─
        # docs/human-review.md §5 / resolved open-question 13: the original
        # draft is preserved untouched, and the edit lands as a new
        # message_draft_versions row with an incremented revision_number
        # and edited_by="operator".  The edit must independently re-pass
        # policy, the injection scan, and the send gate before B5 may send
        # it — B4b does not run those (they are B5's), so the three gate
        # columns are written NULL here, EXACTLY as B3 writes them (the
        # B3-Z3 invariant).  Both edit fields are required non-empty: an
        # edit with no new text is not an edit, and refusing beats
        # silently approving the un-edited draft.
        edited_subject = (request.edited_subject or "").strip()
        edited_body = (request.edited_body or "").strip()
        if not edited_subject or not edited_body:
            return _record_refusal(
                conn, request=request, run_id=run_id, step_id=step_id,
                refusal_reason=(
                    "approve_with_edits requires a non-empty edited subject and "
                    "edited body — to approve the draft as written, use approve"
                ),
            )
        new_revision_id = new_id("dv")  # same "dv" prefix as B3's revisions: one id vocabulary for the table
        write_gate_commit(
            conn,
            action="insert_message_draft_version",  # the same action B3 uses — the write is a revision append, audited distinctly in write_log
            table_name="message_draft_versions",
            record_id=new_revision_id,
            payload={
                "revision_number": latest["revision_number"] + 1,
                "edited_by": REVIEW_ACTOR,
                # B3-Z3, made visible in the audit row itself: this write
                # does not run policy / injection / send gates.
                "gate_columns_written": None,
            },
            run_id=run_id,
            step_id=step_id,
            actor=REVIEW_ACTOR,  # the operator performs this write
            agent_id=REVIEW_ACTOR,  # attributed to the operator principal (B3's rows attribute to draft_writer; operator edits attribute to operator)
            sql="""
                INSERT INTO message_draft_versions
                    (draft_version_id, target_id, message_id, revision_number,
                     subject, body, footer, edited_by, policy_check_passed,
                     injection_scan_passed, send_gate_passed, critique_passed,
                     critique_json, insert_seq, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,
                        (SELECT COALESCE(MAX(insert_seq),0)+1 FROM message_draft_versions),
                        datetime('now'))
            """,
            # The insert_seq item above is computed IN the INSERT (a scalar
            # subquery, no parameter): one statement, so the sequence number
            # is assigned atomically inside the write gate's transaction —
            # no SELECT-then-INSERT race, and identical SQL on both dialects.
            # It is the column that makes "which revision is latest?"
            # deterministic (created_at is second-precision and two same-second
            # rows ordered arbitrarily — ticket B5's send-gate bug).
            params=(
                new_revision_id,
                request.target_id,
                None,  # message_id: no messages row exists until B5 sends — NULL is the honest "not a message yet" (same as B3)
                latest["revision_number"] + 1,  # the incremented revision number — the edit's place in the history
                edited_subject,  # the operator's replacement subject
                edited_body,  # the operator's replacement body
                latest["footer"],  # the deterministic compliance footer, carried over UNCHANGED (B3-Z1: the footer is never operator- or LLM-authored; the edit touches subject/body only)
                REVIEW_ACTOR,  # edited_by: the operator — human edits record the human, agent drafts record the agent (B3's db-schema correction)
                None,  # policy_check_passed — B3-Z3: the G2 draft gate runner owns this; the edited revision re-passes it below
                None,  # injection_scan_passed — B3-Z3, same ownership
                None,  # send_gate_passed — B3-Z3: the send gate owns this
                None,  # critique_passed: no critic ran on the operator's edit — NULL is the honest "no verdict exists"
                None,  # critique_json: same — no critic, no critique
            ),
        )
        # ── G2: the edited revision must independently re-pass its gates ─
        # human-review.md §5 / gates.md §2.2's last item: the edit is a NEW
        # revision, so B4b wrote its gate columns NULL (B3-Z3).  This
        # SEPARATE deterministic runner now evaluates the edited text and
        # writes policy_check_passed / injection_scan_passed.  If it refuses
        # (writes 0), the send gate names that reason; if it crashes, the
        # columns stay NULL and the send gate still fails closed — the review
        # decision itself proceeds either way, because approving an edit is
        # the operator's call, and the gate is what keeps a bad edit from
        # ever sending.
        run_draft_gate(conn, draft_version_id=new_revision_id, run_id=run_id)
        decision_draft_ref = new_revision_id  # the decision approves the EDITED revision
        edited_flag = 1
    elif request.decision == "reject_and_suppress":
        # ── §2.2: the suppression row ─────────────────────────────────────
        # suppressions.email_normalized is UNIQUE (ticket F1b/H4b) and the
        # table's CHECK constraints pin reason to the documented vocabulary
        # and added_by to system/operator.  Since H4b email is no longer the
        # primary key — it stays as the address-AS-WRITTEN audit record.
        # An operator rejection is reason="manual", added_by="operator" —
        # recorded here through the write gate.
        # Two explicit edge cases, neither left to raise IntegrityError:
        #  - no contact / no email on the contact: there is nothing to
        #    suppress.  REFUSE the whole decision (never silently downgrade
        #    to reject — that would be changing the operator's decision)
        #    and tell the operator to use reject instead.  The target
        #    stays in awaiting_review.
        #  - email already suppressed: the operator's goal (this email can
        #    never be mailed) is ALREADY true, so the decision proceeds
        #    and the INSERT is skipped — idempotent, recorded in the
        #    payload so the audit trail shows the suppression predated the
        #    decision.
        email_row = conn.execute(
            "SELECT c.email FROM targets t LEFT JOIN contacts c "
            "ON t.contact_id = c.contact_id WHERE t.target_id=?;",
            (request.target_id,),
        ).fetchone()
        email = email_row["email"] if email_row is not None else None
        if not email:
            return _record_refusal(
                conn, request=request, run_id=run_id, step_id=step_id,
                refusal_reason=(
                    f"reject_and_suppress requires a contact email to suppress, "
                    f"and target {request.target_id!r} has none recorded — "
                    f"use reject instead"
                ),
            )
        existing = conn.execute(
            "SELECT 1 FROM suppressions WHERE email_normalized=?;",
            (normalize_email(email),),
        ).fetchone()
        if existing is None:
            # The INSERT — through the write gate, never a raw INSERT (the
            # audit-trail test catches the raw path).  record_id is the
            # email itself: the address AS WRITTEN is the row's natural
            # identity in write_log (the audit record — not the primary key
            # since ticket H4b; email_normalized is the UNIQUE matching key).
            write_gate_commit(
                conn,
                action="insert_suppression",  # B4b's new action — suppression writes are audited distinctly in write_log
                table_name="suppressions",
                record_id=email,  # the email AS WRITTEN is the row's natural identity in write_log (the audit record)
                payload={"reason": "manual", "added_by": REVIEW_ACTOR},
                run_id=run_id,
                step_id=step_id,
                actor=REVIEW_ACTOR,
                agent_id=REVIEW_ACTOR,
                sql="""
                    INSERT INTO suppressions (email, email_normalized, domain, reason, added_at, added_by, notes)
                    VALUES (?,?,?,?,datetime('now'),?,?)
                """,
                params=(
                    email,  # the address as written — preserved, never overwritten (ticket §2)
                    normalize_email(email),  # F1b: the matching key, folded by the ONE shared helper
                    None,  # domain: NULL — a manual operator suppression suppresses the address, not the whole domain
                    "manual",  # the CHECK-constrained reason vocabulary: an operator rejection is "manual"
                    REVIEW_ACTOR,  # the CHECK-constrained added_by vocabulary: the operator added it
                    None,  # notes: no extra context was supplied — NULL is the honest "nothing recorded"
                ),
            )
        else:
            # Already suppressed: skip the INSERT (a second one would raise
            # IntegrityError on the email_normalized UNIQUE — the matching
            # key since F1b/H4b) and mark the skip so the review_decisions
            # payload records the idempotent no-op.
            suppression_skipped = True

    # ── Write 1 of 2: the review_decisions row ─────────────────────────────
    # The PRIMARY record of the operator's decision — written BEFORE the
    # transition (see the module docstring's WRITE ORDER).  reason carries
    # the operator's reasoning; for escalate it carries the research note
    # (docs/state-machine.md §7: the escalation "carries the operator's
    # research_note forward" — this row is where it is stored).
    review_decision_id = new_id("rev")  # "rev" prefix, matching the table's PK name review_decision_id
    if request.decision == "escalate":
        # The note travels with the decision; fall back to the generic
        # reason field when no note was given (both are operator text —
        # nothing is invented).
        stored_reason = (request.research_note or request.reason or "").strip()
    else:
        stored_reason = request.reason
    write_gate_commit(
        conn,
        action="insert_review_decision",  # B4b's new action — the decision write is audited distinctly in write_log
        table_name="review_decisions",
        record_id=review_decision_id,
        payload={
            "decision": request.decision,
            "edited": bool(edited_flag),
            # The draft-reference contract, made visible in the audit row:
            # draft_message_id holds a draft_version_id (no messages row
            # exists until B5) — see docs/db-schema.md §review_decisions.
            "draft_version_id": decision_draft_ref,
            "kill_switch_active": kill_state.engaged,
            "suppression_skipped": suppression_skipped,
        },
        run_id=run_id,
        step_id=step_id,
        actor=REVIEW_ACTOR,
        agent_id=REVIEW_ACTOR,
        sql="""
            INSERT INTO review_decisions
                (review_decision_id, run_id, target_id, draft_message_id,
                 decision, edited, reason, actor, kill_switch_active,
                 insert_seq, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,
                    (SELECT COALESCE(MAX(insert_seq),0)+1 FROM review_decisions),
                    datetime('now'))
        """,
        # insert_seq computed inside the INSERT (see the edited-revision
        # write's comment): the atomic, dialect-neutral monotonic sequence
        # that makes the send gate's "latest decision" read deterministic.
        params=(
            review_decision_id,
            run_id,
            request.target_id,
            decision_draft_ref,  # draft_message_id: the draft_version_id being decided on (the db-schema contract)
            request.decision,
            edited_flag,
            stored_reason,
            REVIEW_ACTOR,  # actor: the operator decided
            1 if kill_state.engaged else 0,  # kill_switch_active: the switch state AT DECISION TIME, recorded so the audit trail answers "was the switch on?" without re-reading the (mutable) file
        ),
    )

    # ── Write 2 of 2: the state transition ────────────────────────────────
    # The hop, through THE state-change gate (never a raw UPDATE).
    # from_state="awaiting_review" is safe to hardcode: check 2 read it
    # fresh and refused anything else.  escalate's reason is the
    # state-machine-pinned "research_escalation" verbatim (the re-entry
    # point — back into fetch_sources — is the pipeline's next run; B4b
    # does not trigger a run, it only moves the target to researched).
    to_state, transition_reason = _DECISION_TRANSITIONS[request.decision]
    transition(
        conn,
        target_id=request.target_id,
        from_state="awaiting_review",
        to_state=to_state,
        reason=transition_reason,
        actor=REVIEW_ACTOR,
        run_id=run_id,
        step_id=step_id,
        # agent_id defaults to actor ("operator") inside transition — the
        # decision's transition is attributed to the operator principal.
    )

    # ── The success trace row (never skip logs) ───────────────────────────
    log_step(
        conn,
        run_id=run_id,
        step_id=step_id,
        target_id=request.target_id,
        tool_name=REVIEW_TOOL_NAME,
        agent_id=REVIEW_ACTOR,
        input_data={"stage": "review_decision", "decision": request.decision},
        output_data={
            "new_state": to_state,
            "review_decision_id": review_decision_id,
            "draft_version_id": decision_draft_ref,
            "edited": bool(edited_flag),
            "kill_switch_active": kill_state.engaged,
        },
        status="success",
    )
    return ReviewOutcome(
        target_id=request.target_id,
        decision=request.decision,
        new_state=to_state,
        review_decision_id=review_decision_id,
        refused=False,
        refusal_reason="",
    )
