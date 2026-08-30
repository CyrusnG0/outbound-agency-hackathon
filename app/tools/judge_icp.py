"""
judge_icp — the ICP judge (plan task B2c): the deterministic score becomes
evidence, not the verdict.

``score_lead`` computes a 0-100 number and a label by fixed thresholds.  B2c
demotes that number from verdict to EVIDENCE: it stays — auditable, cheap,
regression-testable — and this module's LLM judge weighs it (plus the
signals' B2b evidence tiers, the company profile, and the offer's icp block
+ pitch) and issues the FINAL fit_label with a written rationale.  A judge
that diverges from the deterministic label must justify the divergence, and
the model (ICPVerdict in app/schemas.py) refuses a divergence without one.

THE GOVERNANCE SPLIT — real judgement, bounded by construction (the
submission's Twist in miniature; do NOT "simplify" it away):

- The judge MAY set the final ``fit_label``.  That is the point of B2c.
- The judge MUST NOT be able to talk its way past the policy floor.  The
  deterministic ``fit_score`` remains the number policy_check_phase1
  evaluates for P4 (app/policy.py), and the judge's output schema
  (ICPVerdict) has NO score field at all — so the judge cannot produce,
  alter, or influence the number P4 reads even if its prompt is subverted.
  Keeping a score field OUT of this model is the enforcement, not a
  prompt instruction.
- Both labels are persisted, always — the deterministic one (by
  score_lead, to accounts.icp_fit_label) and the judge's (by this module,
  to accounts.judge_fit_label / judge_rationale /
  judge_divergence_justification) — so a divergence is visible in the
  audit trail without reading code.
- Every write here goes through write_gate.commit with
  agent_id=JUDGE_AGENT_ID ("icp_judge"), the judge's own registered
  principal — never "system" — so its writes and the judge-driven routing
  transition are attributable to it in write_log.

FAILURE PATH — a broken judge degrades, it never fails the target: after
the same bounded two-attempt retry as summarize_company/detect_signals, a
judge that still cannot produce a valid verdict returns None and the caller
(ScoreNode) routes the target with the DETERMINISTIC label — exactly
today's pre-B2c behaviour.  The judge failure is logged (never skipped),
but there is NO transition to "failed": the target is still scored, because
a judge outage is not a research failure.  This asymmetry is deliberate and
must not be "tidied" into the summarize/detect pattern, whose final failure
DOES transition the target to failed.

The judge also does not perform the routing transition itself: it only
produces a verdict and persists it.  Deterministic wiring code
(app/tools/score_lead.apply_final_fit_label) executes the scored→{label}
hop through state_machine.transition(), attributing it to the judge when
the label came from the judge.  An LLM that emits text never touches the
state machine.
"""

import json
import time

from app.llm import (
    call_structured,
    hash_call,
    LLMEmptyResponseError,
    LLMSchemaValidationError,
    LLMTransportError,
    TRANSPORT_RETRY_SLEEP_SECONDS,
)
from app.schemas import CompanyProfile, ICPAssessment, ICPVerdict, Signal
from app.tools.log_step import log_step
from app.write_gate import commit as write_gate_commit

# ── The judge's identity ─────────────────────────────────────────────────────
# The judge's OWN registered agent_id (app/agents_registry.py seeds the
# matching row with model_alias="judge_model").  Imported by
# app/agents/phase1.py (ScoreNode) and app/tools/score_lead.py's callers so
# every judge-attributed write and transition names the same principal — the
# id lives here, next to the agent it names, so the two can never drift.
JUDGE_AGENT_ID = "icp_judge"

# The config/models.yaml role alias the judge's call_structured resolves.
# Its own role (not research_model) so the operator can pin a different
# model for judgement than for extraction without touching either.
JUDGE_MODEL_ALIAS = "judge_model"

# The steps.tool_name every judge step row carries — distinct from
# score_lead's rows so the trace log can tell "the formula scored" apart
# from "the judge weighed the score".
JUDGE_TOOL_NAME = "judge_icp"

# ── System prompt ─────────────────────────────────────────────────────────────
# The prompt does four jobs, each load-bearing:
# 1. States the tier-weighting contract (the B2b payoff): source-tier
#    signals are attributable to a persisted page we actually fetched —
#    weight them most; findings-tier is the research agent's own prose
#    (plausibly from a server-side search we can never capture) — treat as
#    an uncorroborated assertion; unverified-tier means the quote appears
#    in NO source it was given — treat with explicit suspicion.  A judge
#    that ignores tiers wastes B2b; this paragraph exists so it cannot
#    claim it was not told.
# 2. States the judge's power and its limit: it sets the final fit_label;
#    the numeric score is fixed evidence it cannot change (the output
#    schema has no score field — saying so makes the absence a stated
#    contract rather than a hidden trap).
# 3. States the divergence contract: diverging requires
#    divergence_justification; agreeing requires it empty.  The model
#    validator enforces this — the prompt states it so an honest judge
#    complies on the first attempt instead of burning the retry.
# 4. Enumerates the four FitLabel values verbatim, because the response
#    schema's Literal refuses any other string and the judge must know the
#    vocabulary (an invented "maybe_fit" fails validation and wastes an
#    attempt).
_SYSTEM_PROMPT = (
    "You are the ICP judge for an outbound campaign. You issue the FINAL "
    "fit_label for a researched company. A deterministic scoring formula "
    "has already produced a 0-100 fit_score and a fit_label; treat them as "
    "EVIDENCE, not as the verdict — you may agree or disagree, but you must "
    "reason from the evidence. Weight the signals by their evidence tier: "
    "signals with tier 'source' are backed by a quote found in a fetched "
    "page we stored — weight them most; tier 'findings' means the quote "
    "appears only in the research agent's own prose — an uncorroborated "
    "assertion; tier 'unverified' means the quote appears in NO source it "
    "was given — treat it with explicit suspicion. Compare the company "
    "against the offer's ICP block and pitch: a company matching an ICP "
    "disqualifier must not be labelled strong_fit or good_fit. Valid "
    "fit_label values are exactly: strong_fit, good_fit, watchlist, "
    "not_target. You cannot change the numeric score — your output has no "
    "score field, and the score remains what policy reads. Echo the "
    "deterministic fit_label you were given into deterministic_fit_label. "
    "If your fit_label differs from it, you MUST provide a "
    "divergence_justification explaining why; if it matches, leave "
    "divergence_justification empty."
)

def _load_signal_tiers(conn, target_id: str, run_id: str) -> dict[tuple[str, str], str]:
    """Load each persisted signal's B2b evidence tier for (target, run).

    The tiers are computed ONCE by detect_signals and persisted to
    signals.evidence_tier — this judge does not recompute or second-guess
    them; it reads the same verdict the operator sees in the console.  The
    key is (signal_type, signal_value), which is UNIQUE per (target, run)
    in the signals table (app/db.py DDL), so the map cannot collide.  A
    signal with no row (or a legacy NULL tier) maps to "unknown" — the
    prompt treats unknown like unverified (suspicion), the safe direction
    for evidence the pipeline never assessed.
    """
    rows = conn.execute(
        "SELECT signal_type, signal_value, evidence_tier FROM signals "
        "WHERE target_id=? AND run_id=?;",
        (target_id, run_id),
    ).fetchall()
    # tier or "unknown": NULL is the migration accommodation for pre-B2b
    # rows — honest as "never assessed", never silently upgraded to a
    # stronger tier.
    return {
        (row["signal_type"], row["signal_value"]): (row["evidence_tier"] or "unknown")
        for row in rows
    }


def _build_user_content(
    company_profile: CompanyProfile,
    signals: list[Signal],
    tiers: dict[tuple[str, str], str],
    icp_assessment: ICPAssessment,
    offer_icp,
    offer_pitch: str | None,
) -> str:
    """Assemble the judge's single structured input blob.

    Everything the judge may reason from is here, as one JSON text: the
    company profile, the deterministic assessment (score + label + the
    formula's reasons — the evidence being weighed), each signal WITH its
    persisted evidence tier attached (the B2b payoff), and the offer's ICP
    block + pitch (what the company is compared against).  A missing icp
    block or pitch is passed as null — the judge still works with less to
    go on (an offer without an icp definition is a legitimate, supported
    configuration, not an error).
    """
    # Each signal is serialized with its tier INLINE (not in a parallel
    # list) so the model sees "this signal, this tier" as one fact — a
    # parallel structure invites the model to misalign the two.
    signals_with_tiers = [
        {
            **sig.model_dump(),
            "evidence_tier": tiers.get((sig.signal_type, sig.signal_value), "unknown"),
        }
        for sig in signals
    ]
    payload = {
        "company_profile": company_profile.model_dump(),
        # The deterministic evidence: the score is FIXED INPUT the judge
        # cannot edit — including it in the input (and excluding any score
        # from the output schema) is the P4 boundary made explicit.
        "deterministic_assessment": icp_assessment.model_dump(),
        "signals_with_evidence_tiers": signals_with_tiers,
        "offer": {
            "icp": offer_icp,  # may be None — optional offer configuration
            "pitch": offer_pitch,  # may be None — same
        },
    }
    # json.dumps (not a hand-rolled string): structured input for a
    # structured-output call, and json quoting keeps any hostile text in
    # the profile/signals inert data rather than prompt syntax.
    return json.dumps(payload, ensure_ascii=False)


def _call_judge_llm(system_prompt: str, user_content: str) -> ICPVerdict:
    """Call the LLM and return a validated ICPVerdict.

    A separate function (not inlined in judge_icp) for the same reason
    summarize_company keeps call_structured mockable and detect_signals
    keeps _call_detect_signals: tests patch THIS seam to stay offline
    (tests/conftest.py's autouse live-client guard refuses any unmocked
    model boundary).  The production path is exactly call_structured with
    the judge's model alias — no hand-rolled LLM path, no ADK LlmAgent:
    this is a structured single-shot judgement, and call_structured already
    carries A4c's transport handling and B1g's timeouts.
    """
    return call_structured(
        model_alias=JUDGE_MODEL_ALIAS,
        system_prompt=system_prompt,
        user_content=user_content,
        response_schema=ICPVerdict,
    )


def _verify_echo(verdict: ICPVerdict, icp_assessment: ICPAssessment) -> None:
    """Refuse a verdict whose echoed deterministic label is wrong.

    The divergence validator inside ICPVerdict compares the judge's label
    against the deterministic label the judge ECHOED — so a judge that lies
    about what the deterministic label was could smuggle a divergence past
    the validator (or fake an agreement).  This deterministic post-check
    closes that: the echo must equal the real assessment's label, or the
    verdict is treated exactly like schema-invalid output (retry, then
    degrade to the deterministic label).  LLMSchemaValidationError is the
    honest category — the model produced structured output that fails our
    contract.
    """
    if verdict.deterministic_fit_label != icp_assessment.fit_label:
        raise LLMSchemaValidationError(
            f"judge echoed deterministic_fit_label={verdict.deterministic_fit_label!r} "
            f"but the real deterministic label is {icp_assessment.fit_label!r}"
        )


def judge_icp(
    conn,
    *,
    company_profile: CompanyProfile,
    signals: list[Signal],
    icp_assessment: ICPAssessment,
    offer_icp,
    offer_pitch: str | None,
    target_id: str,
    run_id: str,
    step_id: str,
) -> ICPVerdict | None:
    """Run the ICP judge and persist its verdict to accounts.judge_*.

    Returns the validated ICPVerdict on success, or None when the judge
    failed after its bounded retries.  None is NOT an error for the target:
    the caller routes with the deterministic label and the target is still
    scored — a broken judge degrades to today's pre-B2c behaviour, never
    fails the target (see the module docstring for why this must not be
    "tidied" into the summarize/detect failure pattern).

    Every attempt is logged; the success row and the persisted verdict both
    carry the divergence flag so an operator can grep the trace for
    "diverged": true.
    """
    # The tiers are loaded from the signals table — the SAME persisted
    # verdict detect_signals computed (B2b) — so the judge reasons over
    # the evidence the operator can audit, never a fresh recomputation.
    tiers = _load_signal_tiers(conn, target_id, run_id)
    # The structured input blob, built once: it does not change between
    # attempts (the stricter re-prompt changes the SYSTEM prompt, not the
    # facts — same discipline as summarize/detect).
    user_content = _build_user_content(
        company_profile, signals, tiers, icp_assessment, offer_icp, offer_pitch
    )
    # Compute the call hash once, from the ORIGINAL prompt, so the success
    # and retry rows share one hash identifying "this evidence was judged"
    # — the same discipline as the other two LLM nodes.
    call_hash = hash_call(_SYSTEM_PROMPT, user_content)

    # The stricter re-prompt for the retry attempt — mirrors
    # summarize_company/detect_signals exactly.
    stricter_prompt = _SYSTEM_PROMPT + " Return ONLY valid structured output matching the schema exactly."

    # Bounded two-attempt retry, identical shape to the other LLM nodes:
    # first attempt with the standard prompt, second with the stricter one,
    # then give up.  No third attempt, no exponential backoff.
    for attempt, prompt in enumerate([_SYSTEM_PROMPT, stricter_prompt]):
        # Per-attempt step_id suffix so the failure row (attempt 0) and the
        # success/retry row (attempt 1) can coexist in the steps table
        # (step_id is its PK) — the same pattern as summarize/detect.
        attempt_step_id = f"{step_id}_a{attempt}"
        try:
            verdict = _call_judge_llm(prompt, user_content)
            # Post-validate the judge's echo of the deterministic label
            # BEFORE anything is persisted: a wrong echo means the
            # divergence fields are computed against a lie, so the verdict
            # is refused here (this raises LLMSchemaValidationError, which
            # the except clause below routes through the normal retry).
            _verify_echo(verdict, icp_assessment)
            # Did the judge override the deterministic label?  Computed once
            # and used in BOTH the step log and the persisted payload so
            # the two views of "did the judge diverge" can never disagree.
            diverged = verdict.fit_label != icp_assessment.fit_label
            # Resolve the account_id ONCE, before the gate call — the
            # verdict lives on accounts (like the deterministic icp_fit_*
            # columns), and looking it up twice invites the two uses to
            # drift.
            account_id = conn.execute(
                "SELECT account_id FROM targets WHERE target_id=?;", (target_id,)
            ).fetchone()["account_id"]
            # ── Persist the verdict to accounts (write 1 of 1 here) ──────
            # Both labels are persisted, always: the deterministic one was
            # already written by score_lead; this write adds the judge's
            # label + rationale + (only when diverged) its justification.
            # The gate's per-agent check runs against agent_id=icp_judge —
            # the judge's own registered principal — so this write is
            # attributable to the judge, never to "system".  actor stays
            # "system" because it is deterministic pipeline code that
            # executes the write; agent_id records WHOSE verdict it is.
            write_gate_commit(
                conn,
                action="update_account_icp_verdict",  # B2c's new KNOWN_ACTION — the verdict write is audited distinctly from the score write.
                table_name="accounts",
                record_id=account_id,
                payload={
                    "fit_label": verdict.fit_label,
                    "rationale": verdict.rationale,
                    "divergence_justification": verdict.divergence_justification,
                    "diverged": diverged,
                },
                run_id=run_id,
                step_id=attempt_step_id,
                actor="system",  # the deterministic wiring code performs the write
                agent_id=JUDGE_AGENT_ID,  # the judge principal owns the verdict
                sql="""
                    UPDATE accounts SET judge_fit_label=?, judge_rationale=?,
                        judge_divergence_justification=?, updated_at=datetime('now')
                    WHERE account_id=?
                """,
                params=(
                    verdict.fit_label,
                    verdict.rationale,
                    verdict.divergence_justification,
                    account_id,
                ),
            )
            # Log the successful call.  output_data carries the divergence
            # flag AND both labels so "judge overrode the deterministic
            # score" is greppable in the steps trace without joining
            # accounts — the ticket's "the divergence must be greppable".
            # status="success" vs "retried" distinguishes first-attempt from
            # retry success, same as the other two LLM nodes.
            log_step(
                conn, run_id=run_id, step_id=attempt_step_id, target_id=target_id,
                tool_name=JUDGE_TOOL_NAME,
                agent_id=JUDGE_AGENT_ID,  # the judge's step, attributed to the judge
                input_data={
                    "deterministic_fit_label": icp_assessment.fit_label,
                    "deterministic_fit_score": icp_assessment.fit_score,
                    "signal_count": len(signals),
                },
                output_data={
                    "fit_label": verdict.fit_label,
                    "deterministic_fit_label": icp_assessment.fit_label,
                    "diverged": diverged,
                    "divergence_justification": verdict.divergence_justification,
                },
                status="success" if attempt == 0 else "retried",
                model_call_hash=call_hash,
            )
            return verdict
        except (LLMEmptyResponseError, LLMSchemaValidationError, LLMTransportError) as exc:
            # Only the three documented call_structured failure types are
            # caught — anything else is a genuine bug and propagates (the
            # pipeline's crash guard records it; the judge does not pretend
            # to rescue arbitrary exceptions).
            output_data = {"error": str(exc), "error_type": type(exc).__name__}
            if isinstance(exc, LLMTransportError):
                # Same transport diagnostics as the other nodes: retryable
                # and status_code make "429, retried" distinguishable from
                # "401, fatal" in the trace.
                output_data["retryable"] = exc.retryable
                output_data["status_code"] = exc.status_code
            # Log the failed attempt (Golden Rule: never skip logging) —
            # the row carries the attempt number and the failure category.
            log_step(
                conn, run_id=run_id, step_id=attempt_step_id, target_id=target_id,
                tool_name=JUDGE_TOOL_NAME,
                agent_id=JUDGE_AGENT_ID,
                input_data={"deterministic_fit_label": icp_assessment.fit_label},
                output_data=output_data,
                status="failed",
                model_call_hash=call_hash,
            )
            if isinstance(exc, LLMTransportError):
                # Non-retryable transport errors (400/401/403/404) cannot be
                # fixed by an identical second call — break immediately
                # rather than burning the doomed attempt (the log above runs
                # first, the break second — same order as detect_signals).
                if not exc.retryable:
                    break
                if attempt < 1:
                    # Fixed pause before the retry — a 429 retried with zero
                    # delay is almost certain to 429 again; the sleep is
                    # bounded at the same TRANSPORT_RETRY_SLEEP_SECONDS the
                    # other nodes use.
                    time.sleep(TRANSPORT_RETRY_SLEEP_SECONDS)
            # Output-empty / schema-invalid failures: no sleep needed — the
            # stricter re-prompt on the next attempt is the fix.

    # Both attempts failed — the judge produced no usable verdict.  Return
    # None: the CALLER routes the target with the deterministic label and
    # the target is still scored.  DELIBERATELY no transition() call here:
    # a judge outage is not a research failure, and "any → failed" would
    # wrongly kill an otherwise fully-researched target — the degradation
    # to today's behaviour is the whole safety property.  The failed
    # attempts are already in the steps trace above, so the outage is
    # visible without a state change.
    return None
