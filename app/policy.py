"""
policy_check — Phase 1 scoped policy gate (Task 13).

This module implements the single choke point for every action that could
produce an external effect or a state transition, per docs/policy-matrix.md.
In Phase 1, three rules are relevant (the harness doesn't draft or send yet):

- P6: the kill switch — if config/kill_switch.json is engaged (or
  unreadable, which fails closed), the decision is deny unconditionally.
  P6 dominates: it outranks a passing P3a/P4 (ticket B4a).
- P3a: data completeness — requires company_profile.one_line_summary,
  icp_assessment.fit_label, and (if signals exist) each signal has a signal_type.
  Explicitly does NOT require verified_email or contact_candidates[].name
  since Phase 1 has no contact-enrichment node.
- P4: confidence/score floor — any icp_assessment.fit_score below 60 is an
  automatic deny, regardless of data completeness.

The function never raises for a "bad" input — a low score or missing field
is a normal decision outcome (deny/review_required), not an exception.
It never fails silently — every path produces a PolicyGateDecision and
persists it to policy_decisions via the write gate.
"""

import json

from app.ids import new_id
from app.kill_switch import read_kill_switch
from app.schemas import CompanyProfile, ICPAssessment, PolicyGateDecision, Signal
from app.write_gate import commit as write_gate_commit


def policy_check_phase1(
    conn,
    *,
    company_profile: CompanyProfile,
    icp_assessment: ICPAssessment,
    signals: list[Signal],
    target_id: str,
    run_id: str,
    step_id: str,
    actor: str = "system",
) -> PolicyGateDecision:
    """Phase 1 scoped policy check — P6 (kill switch, added ticket B4a),
    P3a (data completeness, excluding verified_email and
    contact_candidates[].name) and P4 (confidence floor). Full
    P1/P2/P3/P5/P7/P8/P9 apply starting Phase 1b, per docs/policy-matrix.md
    P3a and docs/data-flow.md §7's phase boundary. Never fails silently —
    always produces a decision."""

    # ── Accumulator lists: these are built across the P6/P3a/P4 checks and
    # fed into the PolicyGateDecision at the end. We use lists (not
    # mutable globals) so every branch can append without coordination.
    missing_fields: list[str] = []
    matched_rules: list[str] = []
    reasons: list[str] = []

    # ── P6: the kill switch (ticket B4a) — read FIRST because it
    # dominates. Per docs/policy-matrix.md P6: "If the kill switch is on,
    # all outbound actions → deny, unconditionally." Unconditionally means
    # P6 outranks a PASSING P3a/P4 — an engaged switch denies even a
    # complete, high-scoring target. The read is UNCACHED and FAIL-CLOSED
    # (app/kill_switch.py): a missing/unreadable/malformed switch file
    # also reads engaged, so the deny cannot be defeated by deleting the
    # file. When P6 fires, the P3a/P4 checks below are deliberately NOT
    # evaluated — their verdicts are irrelevant while the switch is on,
    # and reporting their fields alongside a kill-switch deny would dilute
    # the one reason an operator needs to see (why everything stopped).
    kill_switch_state = read_kill_switch()
    if kill_switch_state.engaged:
        # Record the P6 match immediately; the shared decision-resolution
        # and persistence blocks at the bottom of this function handle it
        # exactly like a P4 deny (single persistence path — no duplicated
        # SQL, no second write route).
        matched_rules.append("P6")
        reasons.append(kill_switch_state.reason)
    else:
        # ── P3a + P4: evaluated ONLY while the switch is disengaged — a
        # disengaged switch must leave the P3a/P4 behaviour byte-identical
        # to before B4a (the existing tests assert exactly that).

        # ── P3a: Phase 1 scoped data-completeness check ─────────────────────
        # Per docs/policy-matrix.md P3a (corrected 2026-08-02): Phase 1
        # checks only company_profile.one_line_summary,
        # icp_assessment.fit_label, and signals[].signal_type.
        # verified_email and contact_candidates[].name are intentionally
        # NOT checked here — Phase 1 has no enrichment node that could
        # populate them, so requiring them would deny every lead.  They
        # become required in Phase 1b when the full P3 rule applies.

        # one_line_summary is the core output of summarize_company (Task
        # 10).  An empty or missing summary means the LLM research node
        # produced no usable summary — the lead is incomplete, so we flag
        # it.
        if not company_profile.one_line_summary:
            missing_fields.append("company_profile.one_line_summary")

        # fit_label is the output of score_lead (Task 12) and is the
        # primary ICP classification. Without it, downstream nodes can't
        # decide whether to proceed — this is a required Phase 1 field per
        # P3a.
        if not icp_assessment.fit_label:
            missing_fields.append("icp_assessment.fit_label")

        # Signals that exist but lack a signal_type are malformed — the
        # detect_signals node (Task 11) should always produce a
        # signal_type.  We only flag this if signals is non-empty: an
        # empty signal list is valid (it means no signals were detected),
        # but a non-empty list where every signal has a blank signal_type
        # is a data-quality problem.
        if not any(sig.signal_type for sig in signals) and signals:
            missing_fields.append("signals[].signal_type")

        # If any P3a field is missing, record the rule match and the
        # reason.  Policy rules fire when their conditions are met — P3a
        # fires exactly when required Phase 1 fields are absent.
        if missing_fields:
            matched_rules.append("P3a")
            reasons.append(f"missing required Phase 1 fields: {missing_fields}")

        # ── P4: confidence/score floor ──────────────────────────────────────
        # Per docs/policy-matrix.md P4: any icp_assessment.fit_score below
        # 60 is an automatic deny, regardless of data completeness. This
        # is a hard floor — the scoring formula (Task 12) is designed to
        # produce scores in [0,100], and 60 is the minimum confidence
        # threshold for proceeding.  A score of exactly 60 is allowed
        # (>= 60, not > 60).
        if icp_assessment.fit_score < 60:
            matched_rules.append("P4")
            reasons.append(f"fit_score {icp_assessment.fit_score} is below the 60-point floor")

    # ── Decision resolution: priority-ordered ───────────────────────────────
    # P6 (deny) outranks everything — an engaged kill switch denies even a
    # complete, high-scoring target; P4 (deny) takes priority over P3a
    # (review_required). Per docs/policy-matrix.md: a deny decision is a
    # hard block — the lead cannot proceed, period. If P6 or P4 fires, the
    # lead is dead regardless of data completeness. If neither fires but
    # P3a does, the lead needs human review (review_required) because it's
    # incomplete but not necessarily a bad fit. If none fires, the lead is
    # allowed through.
    if "P6" in matched_rules or "P4" in matched_rules:
        # P6/P4 deny: the kill switch or the score floor blocks the lead.
        # This is the highest-priority decision in Phase 1.
        decision_value = "deny"
    elif missing_fields:
        # P3a fired without P6/P4: data is incomplete but the score is ok.
        # The lead needs human review to fill in missing fields before
        # it can proceed.
        decision_value = "review_required"
    else:
        # All rules clear: the switch is disengaged, all required fields
        # present and score clears the floor. The lead is allowed to
        # proceed to the next pipeline stage.
        decision_value = "allow"
        reasons.append("all Phase 1 required fields present, score clears P4 floor")

    # ── Assemble the decision object ────────────────────────────────────────
    # risk_level maps to the decision: deny is high risk (blocking a lead),
    # review_required is medium (needs human attention), allow is low
    # (everything checks out).
    decision = PolicyGateDecision(
        action="score_lead",  # The action being evaluated — in Phase 1, the
                              # only action that reaches the policy gate is
                              # score_lead (the output of Task 12).
        decision=decision_value,
        reasons=reasons,
        matched_rules=matched_rules,
        required_fields_missing=missing_fields,
        risk_level=(
            "high" if decision_value == "deny"
            else "medium" if decision_value == "review_required"
            else "low"
        ),
    )

    # ── Persist the decision to policy_decisions via the write gate ─────────
    # Per CLAUDE.md §3 ("no hidden side effects"): every core-table write
    # must go through write_gate.commit, never a raw conn.execute. The
    # write gate logs the write atomically — no separate log_step call is
    # needed because the write gate's audit row IS the log for this action.

    # Generate a unique id for this policy decision row. "pol" prefix makes
    # it self-describing in logs and join output.
    policy_decision_id = new_id("pol")

    # Persist through the write gate. This is the ONLY write path — if it
    # fails, the entire transaction rolls back and the caller gets the
    # exception. The decision payload (PolicyGateDecision.model_dump()) is
    # serialized to JSON in the write_log audit row.
    write_gate_commit(
        conn,
        action="insert_policy_decision",  # Must be in KNOWN_ACTIONS (added in
                                          # Task 13 to app/write_gate.py).
        table_name="policy_decisions",
        record_id=policy_decision_id,
        payload=decision.model_dump(),  # Full decision as a dict — audit row
                                        # captures exactly what was decided.
        run_id=run_id,
        step_id=step_id,
        actor=actor,
        # agent_id mirrors actor: the two seeded principals are 1:1 with the
        # actor allowlist today ("system"/"operator" both registered with the
        # full KNOWN_ACTIONS set). Plan A4's agents will pass their own id.
        agent_id=actor,
        sql="""
            INSERT INTO policy_decisions
                (policy_decision_id, run_id, step_id, target_id, action, decision,
                 risk_level, reasons_json, matched_rules_json, missing_fields_json,
                 insert_seq, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,
                    (SELECT COALESCE(MAX(insert_seq),0)+1 FROM policy_decisions),
                    datetime('now'))
        """,
        # insert_seq computed inside the INSERT (scalar subquery, no
        # parameter): the monotonic sequence assigned atomically inside the
        # write gate's transaction.  It makes the send gate's and the draft
        # precondition's "LATEST policy decision" reads deterministic —
        # created_at alone is second-precision and two same-second rows
        # order arbitrarily (ticket B5 made that hazard operational).
        params=(
            policy_decision_id,
            run_id,
            step_id,
            target_id,
            decision.action,
            decision.decision,
            decision.risk_level,
            json.dumps(decision.reasons),
            json.dumps(decision.matched_rules),
            json.dumps(decision.required_fields_missing),
        ),
    )

    # Return the decision so the caller (phase1_cli or the graph wiring)
    # can inspect it and decide whether to advance to the next stage or
    # stop and surface the denial/review to the operator.
    return decision
