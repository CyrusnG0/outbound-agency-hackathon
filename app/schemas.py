"""
Pydantic models for Phase 1 typed I/O.

Every cross-task data structure in the pipeline uses these models — no tool
returns a loose dict where a schema is defined.  This file is imported by
Task 3+, but it consumes nothing (no DB, no network, no side effects).

Produced by: Task 2 (this file itself — consumed by nothing internally)
Consumed by: summarize_company, detect_signals, score_lead, policy_check,
             graph wiring, phase1_cli, and every test that touches structured
             pipeline data.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# -- Type aliases -----------------------------------------------------------
# Each Literal constrains a field to exactly the values the rest of the
# pipeline expects — no free-form strings that would silently diverge from
# downstream match/if-elif chains.

# Constrains signal_type to the four dimensions defined in
# docs/scoring-rules.md §1.  No other signal types exist in v1, so no
# "other" catch-all — detecting a fifth signal type means the scoring
# formula has an undefined input, which must fail loudly.
SignalType = Literal[
    "hiring_relevant_role",
    "product_or_ops_change",
    "recent_launch_or_expansion",
    "workflow_complexity_evidence",
]

# Constrains fit_label to the four tiers used by the lead-scoring formula
# (docs/scoring-rules.md).  "maybe_fit" or other hand-wavy labels are
# refused at the schema layer — an LLM that returns one has produced
# malformed output, and the structured-output validator catches it before
# anything downstream sees it.
FitLabel = Literal["strong_fit", "good_fit", "watchlist", "not_target"]

# Constrains policy-gate decisions to exactly the three actions
# docs/policy-matrix.md defines: allow (proceed), deny (block), and
# review_required (pause for human).  No "maybe" or "warn" — those are
# ambiguous and would require every consumer to guess what they mean.
PolicyDecisionValue = Literal["allow", "deny", "review_required"]

# Risk-level buckets used by the policy gate (docs/policy-matrix.md P3a).
# Three discrete tiers — no numeric 0-100 score because risk tolerance is
# a human judgment, not a continuous function.
RiskLevel = Literal["low", "medium", "high"]

# Constrains reply classifications to the nine classes defined in
# docs/reply-routing.md §1 — enumerated VERBATIM in the classifier's prompt
# because the Literal refuses anything else and an invented class ("spam",
# "out_of_office") would fail schema validation and waste an LLM attempt
# (the judge_icp.py precedent).  The router's fixed class→action table keys
# on exactly these strings, so a tenth class would be an undefined routing
# input — refused at the schema layer, never guessed downstream.
ReplyClass = Literal[
    "positive",
    "not_now",
    "negative",
    "unsubscribe",
    "wrong_person",
    "objection",
    "meeting_request",
    "risky",
    "unclear",
]


# -- Database-record models -------------------------------------------------
# These mirror the SQLite tables from docs/db-schema.md.  They're used
# when reading rows back into typed Python objects — every row returned by
# a SELECT maps into one of these before any business logic touches it.

class AccountRecord(BaseModel):
    """Produced by: get_targets (CSV import), research nodes (updates).
    Consumed by: every scoring and policy node that reads account state."""

    account_id: str
    company_name: str
    domain: str
    normalized_domain: str
    industry: Optional[str] = None
    estimated_size: Optional[str] = None
    geo: Optional[str] = None
    company_summary: Optional[str] = None
    icp_fit_label: Optional[FitLabel] = None
    # Clamped 0-100: mirrors the ICP scoring formula range
    # (docs/scoring-rules.md).  Anything outside [0,100] is a computation
    # bug, not a valid score.
    icp_fit_score: Optional[int] = Field(default=None, ge=0, le=100)
    created_at: datetime
    updated_at: datetime


class ContactRecord(BaseModel):
    """Produced by: get_targets (optional contact row during CSV import).
    Consumed by: scoring (persona fit), policy check, and Phase 1b send."""

    contact_id: str
    account_id: str
    full_name: Optional[str] = None
    title: Optional[str] = None
    seniority: Optional[str] = None
    department: Optional[str] = None
    email: Optional[str] = None
    email_verified: bool = False
    linkedin_url: Optional[str] = None
    # Clamped 0-100: same range as icp_fit_score — persona scoring uses
    # the same 0-100 integer scale for consistency across all scores.
    persona_fit_score: Optional[int] = Field(default=None, ge=0, le=100)
    created_at: datetime
    updated_at: datetime


class TargetRecord(BaseModel):
    """Produced by: get_targets (CSV import).  Updated by: state machine
    transitions and scoring nodes.
    Consumed by: every node that needs the current target state and score."""

    target_id: str
    account_id: str
    contact_id: Optional[str] = None
    offer_id: str
    source: str
    state: str
    # Composite lead score, 0-100.  None means "not yet scored" — the
    # scoring formula hasn't run or couldn't produce a result.
    score: Optional[int] = Field(default=None, ge=0, le=100)
    final_recommendation: Optional[str] = None
    last_signal_refresh_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# -- Pipeline-output models -------------------------------------------------
# These are the structured outputs each node produces.  Every LLM node
# uses Pydantic's structured-output mode (via app.llm.call_structured) so
# the model must return JSON matching one of these shapes — malformed or
# out-of-range values are rejected before any downstream consumer sees them.

class CompanyProfile(BaseModel):
    """Produced by: summarize_company (LLM node — Task 10).
    Consumed by: score_lead (as one input to the deterministic formula)."""

    one_line_summary: str
    industry: Optional[str] = None
    estimated_size: Optional[str] = None
    geo: Optional[str] = None
    # Confidence must be a probability in [0.0, 1.0] — the LLM estimates
    # how certain it is about this summary.  A value outside [0,1] is
    # nonsensical (negative confidence or >100% certainty) and indicates
    # a malformed LLM response that must be retried, not silently clamped.
    confidence: float = Field(ge=0.0, le=1.0)


class Signal(BaseModel):
    """Produced by: detect_signals (LLM node — Task 11, extended in plan task B2a).
    Consumed by: score_lead (each signal's strength feeds into the
    weighted scoring formula in docs/scoring-rules.md)."""

    signal_type: SignalType
    signal_value: str
    # evidence_quote is a span copied VERBATIM from the source text that
    # supports this signal — the checkable backing for what would otherwise
    # be an unattributed assertion (plan task B2a).  Required, with no
    # default: a signal with no evidence is exactly what B2a exists to
    # prevent, and an optional field would let the model quietly omit it.
    # min_length=20 stops the model satisfying the field with a token
    # fragment like "admin" that proves nothing — a quote short enough to be
    # coincidental carries no information.  Whether the quote actually
    # appears in the source text is verified downstream by detect_signals'
    # deterministic containment check and recorded as evidence_verified in
    # the signals table; this schema only guarantees the field exists.
    evidence_quote: str = Field(min_length=20)
    # signal_strength is a probability: how strong is this signal relative
    # to a perfect indicator?  0.0 = irrelevant noise, 1.0 = definitive.
    # Clamped to [0,1] because the scoring formula multiplies by a weight
    # in [0,1] — anything outside this range would break the formula.
    signal_strength: float = Field(ge=0.0, le=1.0)
    source_url: Optional[str] = None
    # How confident the LLM is about the source it pulled this signal from.
    # Separate from signal_strength: a weak signal from a highly-trusted
    # source still has low signal_strength but high source_confidence.
    source_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ICPAssessment(BaseModel):
    """Produced by: score_lead (deterministic formula — Task 12).
    Consumed by: policy_check (Phase 1 scoped, P3a — Task 13) and judge_icp
    (as the EVIDENCE the judge weighs, plan task B2c).

    B2c status change: this model is the deterministic evidence, no longer
    the final verdict.  judge_icp consumes it and issues the final
    ``fit_label`` (ICPVerdict below); policy_check_phase1 still reads THIS
    model's ``fit_score`` for P4 — the judge cannot touch that number
    (see ICPVerdict's docstring for why that is by construction)."""

    fit_label: FitLabel
    # Integer 0-100: the ICP-fit dimension of the composite lead score.
    # Bounded at both ends because the formula maps every input into a
    # bounded range — a score outside [0,100] means the formula itself
    # has a bug.
    fit_score: int = Field(ge=0, le=100)
    fit_reasons: list[str]
    non_fit_reasons: list[str]


class ICPVerdict(BaseModel):
    """Produced by: judge_icp (the LLM ICP judge — plan task B2c).
    Consumed by: ScoreNode (the judge's label is the target's final state)
    and the operator console (persisted to accounts.judge_* columns).

    This is the judge's verdict ON TOP of the deterministic ICPAssessment —
    real judgement, bounded by construction:

    - The judge MAY set the final ``fit_label``: that is the point of B2c.
    - The judge CANNOT produce, alter, or influence the numeric fit_score
      that policy rule P4 reads.  This model has NO score field at all, so a
      judge cannot emit one even if its prompt is subverted — the absence of
      the field is the enforcement, not an instruction in a prompt.  Do not
      "simplify" a score field in here later: P4 must keep reading
      ICPAssessment.fit_score only.
    - ``deterministic_fit_label`` is the label the judge was TOLD the
      formula produced (it must echo it back).  The caller (judge_icp)
      verifies the echo against the real assessment before trusting the
      divergence fields — see the validator's contract below.
    - The divergence contract is enforced HERE, in the model, not in a
      comment: a judge that diverges without justifying fails Pydantic
      validation (LLMSchemaValidationError) instead of slipping through.
    """

    # extra='forbid' follows the ticket's requirement and the repo's
    # structured-I/O discipline: a judge emitting unknown fields (e.g. a
    # sneaky "fit_score") is rejected, not silently ignored — this is the
    # second half of "the judge cannot influence P4 by construction".
    model_config = ConfigDict(extra="forbid")

    # The deterministic label the judge was given — echoed back so the
    # divergence validator (and the caller's echo check) can compare the
    # judge's label against the real one inside the model itself.
    deterministic_fit_label: FitLabel
    # The judge's final label — the one the target ends up in (reusing the
    # existing FitLabel vocabulary; no new labels or states exist for this).
    fit_label: FitLabel
    # The written rationale.  min_length=120 is the "real minimum length":
    # it forces an actual argument (tier weighting, ICP comparison), not a
    # restatement of thresholds — the exact boilerplate failure this ticket
    # exists to fix.
    rationale: str = Field(min_length=120)
    # Required when the judge diverges, forbidden otherwise — the
    # conditional is enforced by _divergence_requires_justification below,
    # not left to prompt discipline.
    divergence_justification: Optional[str] = None

    @model_validator(mode="after")
    def _divergence_requires_justification(self) -> "ICPVerdict":
        """Enforce the divergence contract in the model (ticket B2c).

        Runs after field validation on every construction, so BOTH the
        retry-then-fail path in judge_icp and any direct test construction
        get the same enforcement.  Two branches, both hard errors:

        - Judge AGREES with the deterministic label: a non-empty
          justification is contradictory (nothing was overridden) and is
          refused — it would poison the audit trail with a fake divergence.
        - Judge DIVERGES: the justification is mandatory and must be a real
          argument (>=40 chars), not a token gesture.  A diverging judge
          with no justification fails validation — that is the ticket's
          "must fail validation, not slip through".
        """
        # Strip before checking so a whitespace-only string counts as absent
        # (an LLM emitting "\n\n" has not justified anything) — but the
        # ORIGINAL value is stored, never rewritten.
        justification = (self.divergence_justification or "").strip()
        if self.fit_label == self.deterministic_fit_label:
            # Agreement branch: any non-empty justification is a lying
            # divergence record and is refused here.
            if justification:
                raise ValueError(
                    "divergence_justification must be empty when the judge "
                    "agrees with the deterministic label"
                )
        else:
            # Divergence branch: missing or token justification is refused —
            # the audit trail must carry the judge's reason for overriding
            # the deterministic score.
            if not justification:
                raise ValueError(
                    "a judge that diverges from the deterministic label must "
                    "provide divergence_justification"
                )
            if len(justification) < 40:
                raise ValueError(
                    "divergence_justification must be at least 40 characters "
                    "to be a real justification"
                )
        # Return self unchanged — after-validation validators mutate nothing;
        # they exist only to refuse invalid combinations.
        return self


class MeetingProposal(BaseModel):
    """Produced by: schedule_meeting's LLM verdict (app/tools/schedule_meeting.py,
    demo 2026-08-30). Consumed by: schedule_meeting's deterministic wiring
    code, which re-validates the chosen slot before ever writing it — the
    model's judgement is a VERDICT, not an action, exactly the same split
    ICPVerdict and DraftCritique use elsewhere in this repo.

    The model is handed a real, already-computed list of open calendar
    slots (never invents one) and must pick exactly one BY LABEL — the
    label is free text, but the caller only trusts the pick if it matches
    one of the labels it actually offered; a hallucinated label is treated
    exactly like a schema-invalid response (retry, then degrade to the
    deterministic earliest-available slot, same as a judge that cannot
    produce a valid verdict never fails the target).
    """

    # extra='forbid' for the same reason ICPVerdict uses it: a model that
    # tries to smuggle an unrecognized field (e.g. its own invented
    # scheduled_at, bypassing the label-based re-validation) is rejected
    # outright rather than silently ignored.
    model_config = ConfigDict(extra="forbid")

    # Which offered slot the model picked, BY LABEL — never a raw datetime
    # the model computed itself. schedule_meeting.py looks this label up in
    # the SAME candidate list it built and offered; a label that doesn't
    # match any candidate is refused before anything is persisted.
    chosen_slot_label: str = Field(min_length=1)
    # The company name the model was given, echoed back — the same
    # anti-drift check ICPVerdict's deterministic_fit_label echo performs:
    # a verdict that names the wrong company is a sign of a misrouted or
    # corrupted call, caught here rather than trusted blind.
    company_name: str = Field(min_length=1)
    # A real (if short) justification for the pick — min_length is low on
    # purpose: picking a meeting slot has much less to reason about than an
    # ICP verdict, and a boilerplate-length floor here would just train the
    # model to pad, not to reason harder.
    reasoning: str = Field(min_length=20)


class ScheduledMeeting(BaseModel):
    """The result of a real, committed calendar reservation
    (app/tools/schedule_meeting.py, demo 2026-08-30) — what the caller
    (the draft footer composer, the console) actually gets back once a
    slot is reserved. Distinct from MeetingProposal: this is the
    POST-write, re-validated fact (mirrors ``meetings`` table columns),
    never the model's raw, not-yet-trusted verdict.
    """

    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    company_name: str
    contact_name: Optional[str] = None
    # ISO 8601 with the fixed +08:00 offset (see schedule_meeting.py's
    # _HKT constant for why this is a literal offset, not zoneinfo).
    scheduled_at: str
    # The SAME human-readable rendering of scheduled_at every caller sees
    # (e.g. "Tuesday, Sep 1 at 10:30 HKT") — carried as its own field so a
    # footer/console caller never re-implements the format string and
    # risks it drifting from schedule_meeting.py's own _label_for.
    slot_label: str
    duration_minutes: int
    # None only on the deterministic-fallback path (the LLM verdict failed
    # twice and code picked the earliest open slot itself) — the SAME
    # never-fail-the-target degradation judge_icp uses, so a meeting is
    # still always produced.
    reasoning: Optional[str] = None


class EmailDraft(BaseModel):
    """Produced by: the draft writer agent (plan task B3 — the writer half of
    the writer⇄critic LoopAgent).  Consumed by: DraftPersistAndDecideNode
    (persisted to message_draft_versions) and the operator console (B4).

    B3-Z1 — THE ABSENT FOOTER FIELD IS A SECURITY CONTROL, NOT AN OMISSION:
    this model deliberately has NO ``footer`` field.  A compliance footer
    authored by an LLM is a footer that can be silently omitted or mangled;
    the unsubscribe line is therefore composed by deterministic code
    (_compose_footer in app/agents/draft.py) from the offer config, and
    message_draft_versions.footer is NOT NULL so every persisted version
    carries that deterministic, code-generated footer.  A future reader must
    not "helpfully" add a footer field here — the field's absence is the
    enforcement that keeps the model from authoring compliance text.
    """

    # extra='forbid' follows the repo's structured-I/O discipline (same as
    # ICPVerdict): a writer emitting unknown fields is rejected, not
    # silently ignored.
    model_config = ConfigDict(extra="forbid")

    # min_length=3 stops a degenerate empty/one-letter subject; max_length=120
    # keeps it a cold-outreach subject line, not a paragraph.
    subject: str = Field(min_length=3, max_length=120)
    # min_length=80 forces a real email body — the boilerplate failure this
    # ticket exists to fix is a two-sentence "we help you" placeholder.
    body: str = Field(min_length=80)
    # The written rationale for the HUMAN reviewer (the console, B4): why
    # this angle was chosen.  min_length=60 forces an actual argument, not a
    # restatement of the offer pitch.
    rationale: str = Field(min_length=60)
    # The writer's self-assessed confidence, a probability in [0.0, 1.0] —
    # same clamp as CompanyProfile.confidence; it feeds the operator's
    # review (and policy rule P4's draft_email.confidence floor), never a
    # send decision by itself.
    confidence: float = Field(ge=0.0, le=1.0)


class DraftCritique(BaseModel):
    """Produced by: the draft critic agent (plan task B3 — the critic half of
    the writer⇄critic LoopAgent).  Consumed by: DraftPersistAndDecideNode
    (persisted to message_draft_versions.critique_json so the console can
    show WHY the agent rewrote) and, via the published critique_feedback
    state key, the writer's next iteration.

    B3-Z2 — ``passed`` gates LOOP EXIT ONLY.  A passing critique makes the
    loop stop early; it does not send, does not approve, and does not skip
    human review — the target lands in ``awaiting_review`` on every path.
    The coupling between ``passed`` and the other fields is enforced by the
    model validator below, so an LLM cannot emit a "passed" verdict that
    still carries issues, or a "failed" verdict with no instructions."""

    # extra='forbid': same discipline as ICPVerdict — unknown fields refused.
    model_config = ConfigDict(extra="forbid")

    # The critic's verdict on the draft.  True only when the critique is
    # clean (see _passed_couples_to_evidence below).
    passed: bool
    # What is wrong with the draft; empty iff passed.
    issues: list[str]
    # Concrete instructions for the next revision; "" iff passed.  When the
    # critic does not pass, this string is published as critique_feedback
    # and templated into the writer's next instruction ({critique_feedback?}).
    required_changes: str
    # Severity vocabulary — enumerated verbatim in the critic's prompt, and
    # the Literal refuses any other string (same reasoning as judge_icp's
    # prompt item 4: an invented value would fail schema validation and burn
    # one of the three bounded iterations).
    severity: Literal["none", "minor", "major"]

    @model_validator(mode="after")
    def _passed_couples_to_evidence(self) -> "DraftCritique":
        """Enforce the passed↔evidence coupling in the model (ticket B3).

        Runs after field validation on every construction, so BOTH the
        persist node's re-validation and any direct test construction get
        the same enforcement (same shape as ICPVerdict's
        _divergence_requires_justification).  Two branches, both hard errors:

        - ``passed=True`` requires a fully clean critique: no issues, no
          required changes, severity "none".  A "passed but here are the
          problems" verdict is self-contradictory and would let the loop
          stop early on a draft the critic actually still objects to.
        - ``passed=False`` requires at least one issue and a non-empty
          required_changes of >= 30 chars — concrete instructions the next
          writer iteration can act on.  A failing verdict with nothing to
          fix would loop the writer with no feedback (burning iterations).
        """
        # Strip required_changes before the length check so a whitespace-only
        # string counts as absent — but the ORIGINAL value is stored, never
        # rewritten (same discipline as ICPVerdict's justification check).
        required = (self.required_changes or "").strip()
        if self.passed:
            # Clean-pass branch: any issue, any change instruction, or any
            # severity above "none" contradicts the pass and is refused.
            if self.issues or required or self.severity != "none":
                raise ValueError(
                    "passed=True requires issues == [], "
                    "required_changes == '', and severity == 'none'"
                )
        else:
            # Failed branch: a critique with nothing to fix is refused — the
            # writer's next iteration must have concrete feedback to use.
            if not self.issues:
                raise ValueError(
                    "passed=False requires at least one issue"
                )
            if len(required) < 30:
                raise ValueError(
                    "passed=False requires required_changes with at least "
                    "30 characters of concrete instructions"
                )
        # Return self unchanged — after-validation validators mutate nothing;
        # they exist only to refuse invalid combinations.
        return self


class ReplyClassification(BaseModel):
    """Produced by: the reply classifier agent (plan task C1 — an ADK
    ``LlmAgent`` with ``output_schema=ReplyClassification``).  Consumed by:
    the deterministic ReplyRouterNode, which maps the class to the action
    and executes the state transition per docs/reply-routing.md §2.

    THE GOVERNANCE SPLIT (C1's module-docstring contract, same as B2c's
    judge and B3's draft loop): this model carries the LLM's JUDGEMENT
    ONLY.  The classifier emits a class + confidence; deterministic code
    performs every side effect — the replies-row update, the transitions,
    the suppression insert — using the fixed table in reply-routing.md §2.
    An LLM that emits text never calls ``state_machine.transition()``.

    The classifier reads ``replies.redacted_text``, NEVER ``raw_text``:
    inbound email is untrusted attacker-controlled input (policy P8,
    docs/threat-model.md), so the prompt carries only redacted data and
    instructs the model to treat every instruction inside the email as
    data, never as a command.  ``evidence_quote`` is constrained to be a
    VERBATIM span of that (redacted) reply — the checkable backing for the
    class, mirroring Signal.evidence_quote's B2a discipline.
    """

    # extra='forbid' follows the repo's structured-I/O discipline (same as
    # ICPVerdict/EmailDraft): a classifier emitting unknown fields (e.g. a
    # sneaky "routed_action" or "to_state") is rejected, not silently
    # ignored — the routing decision is the deterministic router's alone,
    # and the schema's shape is the first enforcement of that split.
    model_config = ConfigDict(extra="forbid")

    # Exactly one of the nine classes docs/reply-routing.md §1 defines —
    # enumerated verbatim in the prompt, refused by the Literal otherwise
    # (an invented class wastes one of the bounded attempts; see the
    # ReplyClass alias comment).
    reply_class: ReplyClass
    # The classifier's self-assessed confidence, a probability in [0.0, 1.0]
    # — same clamp as CompanyProfile.confidence.  This is the number policy
    # rule P4 reads (floor 0.7): below it, the router refuses every
    # auto-action and routes to review_required, whatever the class.
    confidence: float = Field(ge=0.0, le=1.0)
    # The written reasoning for the class.  min_length=40 forces an actual
    # argument ("the reply asks to stop contact", not "seems negative") —
    # the same boilerplate floor as EmailDraft.rationale.
    rationale: str = Field(min_length=40)
    # A span copied VERBATIM from the reply text that decided the class —
    # the sentence the human reviewer can check without re-reading the
    # whole message.  min_length=10 stops a token fragment ("no", "ok")
    # that proves nothing from satisfying the field (the B2a evidence_quote
    # discipline, re-applied to inbound classification).
    evidence_quote: str = Field(min_length=10)


class PolicyGateDecision(BaseModel):
    """Produced by: policy_check (Task 13).  Consumed by: every downstream
    node that needs to know whether it's allowed to proceed — phase1_cli
    checks decision=='allow' before advancing to the next pipeline stage,
    and every LLM node checks it before spending tokens."""

    action: str
    decision: PolicyDecisionValue
    reasons: list[str]
    matched_rules: list[str]
    required_fields_missing: list[str]
    risk_level: RiskLevel
