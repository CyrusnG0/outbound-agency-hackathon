import json

from app.schemas import CompanyProfile, ICPAssessment, Signal
from app.state_machine import transition
from app.tools.log_step import log_step
from app.write_gate import commit as write_gate_commit

# ── _SIGNAL_WEIGHTS ───────────────────────────────────────────────────────────
# Copied verbatim from docs/scoring-rules.md §3.C. Each signal_type has a
# specific weight because different kinds of signals indicate different
# strengths of buying intent:
#   - hiring_relevant_role (8): hiring for a role related to what we sell is a
#     strong direct buying indicator — the company is actively investing in
#     headcount to solve the problem.
#   - product_or_ops_change (7): operational or product shifts suggest a
#     near-term need, but the signal is slightly less direct than a dedicated
#     hire — it could be internal-only.
#   - recent_launch_or_expansion (5): a launch or expansion often correlates
#     with tooling need, but it's circumstantial — the company may already have
#     a solution.
#   - workflow_complexity_evidence (5): complex workflows mean there's a
#     problem to solve, but complexity alone doesn't guarantee they're shopping.
# These are the only adjustable weights in the formula; feedback-loop.md can
# re-weight them over time from real outcomes.
_SIGNAL_WEIGHTS = {
    "hiring_relevant_role": 8,
    "product_or_ops_change": 7,
    "recent_launch_or_expansion": 5,
    "workflow_complexity_evidence": 5,
}


def _company_fit_score(profile: CompanyProfile) -> int:
    """Compute the company-fit sub-score (0–30).

    Phase 1 approximates this from profile completeness + LLM confidence rather
    than a real industry/process/size classifier (that's Phase 1b/2 scope —
    those phases will have a dedicated enrichment node that actually maps
    industry to our ICP definition). This keeps the formula honest about what
    Phase 1 actually knows, rather than pretending to a precision it doesn't
    have.

    The cap is at 30, matching the max defined in docs/scoring-rules.md §3.A.
    """
    # Start from zero — every point is earned, never assumed.
    score = 0
    # industry presence is worth 10pts: if the LLM could identify an industry
    # from the company's web content, that's partial evidence of fit.
    if profile.industry:
        score += 10
    # estimated_size presence is worth 10pts: if the LLM identified company
    # size, the target is in a recognizable size bracket.
    if profile.estimated_size:
        score += 10
    # confidence * 10 maps [0.0, 1.0] to [0, 10]: higher LLM confidence in
    # the profile summary means the company data is more reliable, so we give
    # it partial credit toward fit.
    score += round(profile.confidence * 10)
    # Cap at 30: the formula may sum above 30 with all fields present + high
    # confidence, but the scoring spec's max for this sub-component is 30, so
    # any excess is discarded — the sub-score can never exceed its defined
    # ceiling.
    return min(score, 30)


def _persona_fit_score(has_contact_data: bool) -> int:
    """Compute the persona-fit sub-score (0–20).

    Returns 15 if the target has contact data, 0 otherwise. The max is 15
    (not 20) because the "seniority sufficient" 5pts sub-factor cannot be
    determined from raw contact presence alone — that requires title parsing
    and a seniority classifier (Phase 1b scope).

    This being 0 without contact data is EXPECTED, not a bug: Phase 1 has no
    contact-enrichment node (no email finder, no LinkedIn lookup, no Apollo/
    Clearbit enrichment), so `has_contact_data` is only true when the original
    CSV import happened to carry a contact_name/contact_title — a rare case in
    the default workflow where targets arrive as domain-only rows. The formula
    does not penalize this; it just reflects reality.
    """
    # 15 points out of the 0–20 sub-range: contact identified (10pts) +
    # "directly relevant to the pain hypothesis" (5pts). The remaining 5pts
    # ("seniority sufficient") is not computable in Phase 1 because we have
    # no title parsing or seniority classifier yet.
    return 15 if has_contact_data else 0


def _signal_score(signals: list[Signal]) -> int:
    """Compute the weighted signal sub-score (0–25).

    Each signal's strength (0.0–1.0) is multiplied by its type's weight and
    summed. The result is capped at 25 so a target with 10 weak signals
    doesn't outscore one with a single genuinely strong signal at the cap —
    the ceiling ensures signal quality matters more than signal quantity.
    """
    # Sum each signal's contribution: signal_strength * assigned_weight.
    # Unknown signal types (not in _SIGNAL_WEIGHTS) contribute 0 — they are
    # data points the formula doesn't know how to interpret.
    total = sum(sig.signal_strength * _SIGNAL_WEIGHTS.get(sig.signal_type, 0) for sig in signals)
    # Round to nearest integer (the scoring formula produces integer scores),
    # then clamp to the defined max of 25. A target with signals summing above
    # the cap still gets 25 — the cap is a ceiling, not a penalty.
    return min(round(total), 25)


def _data_completeness_score(profile: CompanyProfile, has_contact_data: bool) -> int:
    """Compute the data-completeness sub-score (0–15).

    Per docs/scoring-rules.md §3.D, completeness is measured across three
    dimensions: website content (5pts), contact identified (5pts), and
    verified email (5pts). Phase 1 can only evaluate the first two:

    - "Website with readable content" is proxied by profile.one_line_summary
      being present — if summarize_company produced a summary, it means the
      company's web content was readable enough to extract meaning from.
    - "Clear contact/persona identified" is proxied by has_contact_data —
      only true when the CSV import carried contact columns.

    The "verified email exists" 5pts is NOT implemented here and always
    contributes 0 in Phase 1. Email verification (Phase 1b's verify_email
    + NeverBounce integration) doesn't exist yet, so there's no way to know
    whether the email is valid. The score deliberately does not guess.
    """
    # Proxy for "Website with readable content exists" (5pts per spec):
    # if summarize_company produced a meaningful summary, the site was
    # readable enough to extract a one-line description.
    score = 0
    if profile.one_line_summary:
        score += 5
    # Proxy for "Clear contact/persona identified" (5pts per spec):
    # only true when the CSV import carried a contact_name/contact_title —
    # Phase 1 has no enrichment to discover contacts from domain alone.
    if has_contact_data:
        score += 5
    # "Verified email exists" (5pts) intentionally always 0 here — no email
    # verification exists in Phase 1. The spec's max for this sub-score is
    # 15, so the reachable max in the common no-contact case is 10.
    return score


def _evidence_quality_score(signals: list[Signal], profile: CompanyProfile) -> int:
    """Compute the evidence-quality sub-score (0–10).

    Per docs/scoring-rules.md §3.E, this is measured across two dimensions:
    enough independent sources (5pts) and no contradictions (5pts).

    Phase 1 implementation notes:
    - The "at least 1 signal + profile confidence ≥0.5" condition is a proxy
      for "at least 2 independent sources." A single low-confidence signal
      shouldn't count as strong evidence — we require both a signal (web
      content produced it) AND sufficient LLM confidence (the summary is
      reliable) before awarding these 5 points.
    - The "no contradiction tracking" 5 points is unconditional in Phase 1
      because there is no multi-source enrichment yet to actually detect
      contradictions. When multiple enrichment providers exist (Phase 1b/2),
      this will become conditional on actual contradiction flags — but for
      now, there is no contradictory evidence to detect, so the points are
      always awarded to avoid unfairly penalizing Phase 1 for infra it
      doesn't have yet.
    """
    score = 0
    # Proxy for "At least 2 independent sources support the assessment"
    # (5pts per spec): we require both at least one signal (evidence from
    # web research) AND profile confidence ≥0.5 (the LLM is more than
    # 50% sure) before counting this as corroborated evidence.
    if len(signals) >= 1 and profile.confidence >= 0.5:
        score += 5
    # "No unresolved contradiction" (5pts per spec): always awarded in
    # Phase 1 because there is no multi-source enrichment yet to actually
    # produce contradictory candidate fields. This keeps the score honest
    # rather than docking points for conflicts the system can't detect.
    score += 5  # no contradiction tracking in Phase 1 (no multi-source enrichment yet)
    return score


# ── Fit-label thresholds ─────────────────────────────────────────────────────
# Named module-level constants (2026-08-30 — pulled out of _fit_label's body
# so a reader elsewhere in the repo, e.g. the console's /rules page, can
# import the real numbers instead of hand-copying them into prose that can
# silently drift out of sync with the code). Values unchanged from
# docs/scoring-rules.md §4 — this is a naming refactor, not a policy change.
STRONG_FIT_THRESHOLD = 80  # >= this: strong_fit — high-confidence ICP match, ready for drafting
GOOD_FIT_THRESHOLD = 60  # >= this: good_fit — meets the policy floor (P4) for draft consideration
WATCHLIST_THRESHOLD = 40  # >= this: watchlist — interesting but insufficient signal; re-check later
# < WATCHLIST_THRESHOLD: not_target — below the bar; drop from active pipeline


def _fit_label(score: int) -> str:
    """Map a 0–100 fit_score to one of four fit labels.

    Thresholds are the module-level constants above, copied verbatim from
    docs/scoring-rules.md §4.

    These thresholds are the routing decision: strong/good stay at scored
    (later phases will draft from there), watchlist and not_target route the
    target away from the active pipeline.
    """
    if score >= STRONG_FIT_THRESHOLD:
        return "strong_fit"
    if score >= GOOD_FIT_THRESHOLD:
        return "good_fit"
    if score >= WATCHLIST_THRESHOLD:
        return "watchlist"
    return "not_target"


def score_lead(
    conn,
    *,
    company_profile: CompanyProfile,
    signals: list[Signal],
    has_contact_data: bool,
    target_id: str,
    run_id: str,
    step_id: str,
    actor: str = "system",
) -> ICPAssessment:
    """Compute a deterministic ICP fit score and persist it as EVIDENCE.

    This is the first non-LLM node in the research pipeline — it takes the
    structured outputs of summarize_company (Task 10) and detect_signals
    (Task 11) and computes a 0–100 fit score via a fixed formula defined in
    docs/scoring-rules.md §3.

    B2c status change: this function's assessment is now the EVIDENCE the
    ICP judge weighs, not the final verdict.  It still transitions
    researched → scored (scoring did complete — that fact is unchanged) and
    still persists the deterministic label/score/reasons, but it NO LONGER
    performs the scored → watchlist/not_target routing hop: the final label
    is decided after the judge runs, and the routing happens exactly once
    in ``apply_final_fit_label`` below, so the audit trail never shows a
    misleading "route to the deterministic label, then immediately re-route
    to the judge's label" double-hop.

    The function never raises for missing-but-optional profile fields — Phase
    1 has no persona/email data by design, and the formula treats their
    absence as "0 points for that sub-factor," not an error.
    """

    # ── Compute each sub-score independently ──────────────────────────────
    # Each _xxx_score function is a pure function of its inputs with no side
    # effects — they can be tested, reviewed, and tuned in isolation.
    # company: 0–30, proxied from profile completeness + LLM confidence.
    company = _company_fit_score(company_profile)
    # persona: 0 or 15 — 0 if no contact data, 15 if the CSV import carried
    # contact fields. Phase 1 has no enrichment to discover contacts from
    # domain alone, so 0 is the common case.
    persona = _persona_fit_score(has_contact_data)
    # signal: 0–25, weighted sum of signal strengths capped at the ceiling.
    signal = _signal_score(signals)
    # completeness: 0–15, but max 10 in the common no-contact case because
    # "verified email exists" is always 0 in Phase 1 (no email verification).
    completeness = _data_completeness_score(company_profile, has_contact_data)
    # evidence: 0–10, with the contradiction point always awarded in Phase 1
    # (no multi-source enrichment exists to detect contradictions yet).
    evidence = _evidence_quality_score(signals, company_profile)

    # The final fit_score is the sum of all five sub-scores, bounded 0–100.
    # The sub-score caps (30+20+25+15+10=100) guarantee the sum never exceeds
    # 100 without an additional clamp — the caps are the clamp.
    fit_score = company + persona + signal + completeness + evidence

    # Map the numeric score to a discrete label that drives routing.
    # Thresholds: ≥80 → strong_fit, ≥60 → good_fit, ≥40 → watchlist, else
    # not_target. Copied verbatim from docs/scoring-rules.md §4.
    label = _fit_label(fit_score)

    # ── Build human-readable reason lists ─────────────────────────────────
    # These are persisted to accounts.icp_fit_reasons / icp_non_fit_reasons
    # so the operator can see at a glance WHY a target scored the way it
    # did, without re-computing the formula by hand.
    # Each condition was chosen as a meaningful threshold — not just
    # restating a number, but surfacing the cause: e.g. "company >= 20"
    # means the company-fit sub-score is at least 2/3 of its max, which
    # is materially significant; "signal >= 15" means the signal component
    # is at least 60% of its ceiling.
    fit_reasons = []
    non_fit_reasons = []

    # Company fit ≥20 (out of 30) means the profile has at least two of its
    # three dimensions present (industry, size) or high confidence + one —
    # this is a meaningful threshold, not arbitrary.
    if company >= 20:
        fit_reasons.append("Strong company-profile match")

    # Signal ≥15 (out of 25) means at least ~60% of the signal ceiling is
    # hit — two strong signals or multiple moderate ones have fired. This is
    # the threshold at which signals materially affect the fit label.
    if signal >= 15:
        fit_reasons.append("Strong buying signals detected")

    # No contact data is the single most common non-fit reason in Phase 1:
    # the pipeline was designed to enrich contacts, but that enrichment
    # doesn't exist yet, so we surface this explicitly rather than leaving
    # the operator wondering why fit_score is capped.
    if not has_contact_data:
        non_fit_reasons.append("No contact data available (Phase 1 has no enrichment)")

    # Zero signal score means no buying signals were found at all — the
    # target is indistinguishable from noise. This is the strongest non-fit
    # indicator short of a mismatched industry.
    if signal == 0:
        non_fit_reasons.append("No buying signals detected")

    # Package the assessment into a typed model — no loose dicts leave this
    # function. ICPAssessment is a Pydantic model with validated fields so
    # downstream consumers (policy_check, phase1_cli) can rely on the shape
    # without defensive checks.
    assessment = ICPAssessment(
        fit_label=label, fit_score=fit_score, fit_reasons=fit_reasons, non_fit_reasons=non_fit_reasons,
    )

    # ── Look up the account_id from targets ───────────────────────────────
    # The icp_fit_* columns live on accounts, not targets, because a target's
    # fit assessment is really an assessment of the company — two targets at
    # the same company can in principle share an account in a later phase.
    # We resolve account_id from targets here rather than requiring the
    # caller to pass it (redundancy invites mismatch).
    account_id = conn.execute(
        "SELECT account_id FROM targets WHERE target_id=?;", (target_id,)
    ).fetchone()["account_id"]

    # ── Write 1 of 2: update accounts.icp_fit_* columns ───────────────────
    # This is the account-scoped write: stores the fit label, score, and
    # human-readable reasons on the account row so the operator can see the
    # full assessment context without joining across tables.
    # fit_reasons and non_fit_reasons are JSON-serialized before storage
    # because they're Python lists and the icp_fit_reasons /
    # icp_non_fit_reasons columns are TEXT — SQLite doesn't have a native
    # array type, so JSON is the standard storage format.
    write_gate_commit(
        conn,
        action="update_account_score",  # Pre-existing action — already in KNOWN_ACTIONS (Task 12 context: added for score_lead's account-column update).
        table_name="accounts",
        record_id=account_id,
        payload=assessment.model_dump(),
        run_id=run_id,
        step_id=step_id,
        actor=actor,
        # agent_id mirrors actor: the two seeded principals are 1:1 with the
        # actor allowlist today. Plan A4's agents will pass their own id.
        agent_id=actor,
        sql="""
            UPDATE accounts SET icp_fit_label=?, icp_fit_score=?,
                icp_fit_reasons=?, icp_non_fit_reasons=?, updated_at=datetime('now')
            WHERE account_id=?
        """,
        params=(label, fit_score, json.dumps(fit_reasons), json.dumps(non_fit_reasons), account_id),
    )

    # ── Write 2 of 2: update targets.score ────────────────────────────────
    # This is a NEW write path (added in this task): the action
    # "update_target_score" had to be added to write_gate.KNOWN_ACTIONS
    # before this call would be accepted — a raw conn.execute here would
    # bypass the write gate and violate the "never skip the write gate for
    # core-table writes" rule.
    # targets.score is a separate write from accounts.icp_fit_score even
    # though they hold the same number: the target's score column is the
    # routing/query field the rest of the pipeline reads (e.g. "SELECT *
    # FROM targets WHERE score >= 60"), while accounts.icp_fit_score is the
    # same number but account-scoped, stored alongside label and reasons for
    # audit and reporting. Both get set from this one computation so they
    # can never diverge.
    write_gate_commit(
        conn,
        action="update_target_score",  # New action — requires the one-line addition to KNOWN_ACTIONS made in this task.
        table_name="targets",
        record_id=target_id,
        payload={"score": fit_score},
        run_id=run_id,
        step_id=step_id,
        actor=actor,
        # agent_id mirrors actor — same 1:1 mapping as the accounts write above.
        agent_id=actor,
        sql="UPDATE targets SET score=? WHERE target_id=?;",
        params=(fit_score, target_id),
    )

    # ── Transition: researched → scored (ALWAYS runs) ─────────────────────
    # This is the only valid first transition out of "researched" — per
    # state_machine.VALID_TRANSITIONS (Task 5), ("researched", "scored") is
    # the sole path from the researched state. There is no direct
    # ("researched", "watchlist") or ("researched", "not_target") pair in
    # the transition table — every scored target passes through "scored"
    # first, even ones immediately routed onward.
    # If this transition is omitted or replaced with a direct hop to
    # watchlist/not_target, state_machine.transition raises
    # StateTransitionRefused because the pair isn't in VALID_TRANSITIONS.
    # B2c: the scored → watchlist/not_target routing hop that USED to live
    # here is gone — it moved to apply_final_fit_label (below), which runs
    # AFTER the ICP judge, so the target is routed exactly once with the
    # FINAL label (the judge's, or the deterministic one when the judge
    # failed).  Keeping both hops here would produce the misleading
    # "route to the deterministic label, then re-route to the judge's"
    # trail the ticket forbids.
    transition(
        conn, target_id=target_id, from_state="researched", to_state="scored",
        reason="scoring_complete", actor=actor, run_id=run_id, step_id=step_id,
    )

    # ── Log the step ──────────────────────────────────────────────────────
    # Following the same pattern as prior tasks: persist a lightweight trace
    # signal to the steps table. The full assessment is already persisted to
    # accounts (icp_fit_* columns) and targets (score), so this row serves
    # as the audit-trail entry, not the primary data store.
    log_step(
        conn, run_id=run_id, step_id=step_id, target_id=target_id,
        tool_name="score_lead",
        agent_id=actor,  # Which registered agent ran this step — mirrors actor (see the write_gate_commit calls above).
        input_data={"has_contact_data": has_contact_data},
        output_data=assessment.model_dump(), status="success",
    )
    return assessment


def apply_final_fit_label(
    conn,
    *,
    target_id: str,
    run_id: str,
    step_id: str,
    final_label: str,
    deterministic_label: str,
    agent_id: str = "system",
) -> None:
    """Route a scored target to its FINAL Phase 1 terminal state (ticket B2c).

    Runs AFTER the ICP judge: the caller (ScoreNode in app/agents/phase1.py)
    passes the judge's label when the judge produced a verdict, or the
    deterministic label when the judge failed (the documented degradation —
    today's pre-B2c behaviour).  This is the ONLY place the
    scored → watchlist / scored → not_target hop happens, so the audit
    trail never shows a route-then-reroute double-hop around the judge.

    The transition reason is always truthful about who decided:

    - final == deterministic: ``fit_label=<label>`` — the vocabulary
      unchanged since Task 12, because the judge merely agreed.
    - final != deterministic: ``fit_label=<final>
      judge_overrode_deterministic=<deterministic>`` — the divergence is
      greppable in state_transitions.reason alone, without reading the
      accounts columns or the steps JSON (the ticket's "a divergence must
      be visible in the audit trail").

    Attribution: the caller passes agent_id="icp_judge" when the label came
    from the judge, so both write_log rows of the transition name the judge
    principal; the deterministic fallback passes "system".  actor is
    "system" either way — it records that deterministic code performed the
    write, while agent_id records whose decision it applies.

    strong_fit / good_fit need no second hop: the target simply stays at
    "scored", exactly as before B2c.
    """
    # No routing hop for strong_fit/good_fit — the target remains at
    # "scored", ready for later phases' drafting (unchanged from Task 12).
    if final_label not in ("watchlist", "not_target"):
        return
    # The reason string: the agree path keeps the pre-B2c vocabulary; the
    # diverge path names both labels so the override is greppable and
    # self-explanatory in state_transitions.
    if final_label == deterministic_label:
        reason = f"fit_label={final_label}"
    else:
        reason = (
            f"fit_label={final_label} "
            f"judge_overrode_deterministic={deterministic_label}"
        )
    # from_state="scored" is guaranteed: score_lead transitioned
    # researched → scored immediately before the judge ran, so the target
    # is at "scored" when this routing hop fires.  to_state is the
    # final_label itself — both ("scored", "watchlist") and ("scored",
    # "not_target") are in VALID_TRANSITIONS.
    transition(
        conn, target_id=target_id, from_state="scored", to_state=final_label,
        reason=reason, actor="system", agent_id=agent_id,
        run_id=run_id, step_id=step_id,
    )
