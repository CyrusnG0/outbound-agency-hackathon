"""
State machine — the single function allowed to change a target's state.

## Why this module exists

Every target in the pipeline moves through a fixed set of states (new →
researched → scored → …). No component may transition state directly — every
transition goes through `transition()`, which validates the requested
(from_state, to_state) pair against a fixed allow-list before writing.
This module is the source of truth for which transitions exist; if a transition
isn't here, it isn't valid, period. Per `docs/state-machine.md` §3—4.

## Dependencies

- `app.ids.new_id("trn")` — generates primary keys for `state_transitions` rows.
- `app.write_gate.commit()` — the sole write path for every core-table mutation.
  Two separate calls: one UPDATE to `targets.state`, one INSERT to
  `state_transitions`. Each write is independently auditable in `write_log`.

## Interfaces (later tasks depend on these exact names/types)

- `transition(conn, *, target_id, from_state, to_state, reason, actor,
  run_id, step_id, policy_decision_id=None) -> str` — returns the transition_id
  of the new `state_transitions` audit row.
- `StateTransitionRefused(Exception)` — raised when a transition is invalid
  (unknown pair, or actor not "system"/"operator").
- `VALID_TRANSITIONS: set[tuple[str, str]]` — the full valid-transition table,
  copied verbatim from `docs/state-machine.md` §3.
- `PHASE_1_REACHABLE_STATES: set[str]` — the subset of states reachable in
  Phase 1 (before enrichment/drafting/sending exists). Used by Task 12
  (score_lead) and Task 16 (checkpoint test) to assert Phase 1 never reaches a
  Phase 1b/2-only state.
"""

from app.ids import new_id
from app.write_gate import commit as write_gate_commit

# ── VALID_TRANSITIONS ────────────────────────────────────────────────────────
# This set is the complete valid-transition table, copied verbatim from
# docs/state-machine.md §3.  It covers both Phase 1 (e.g. new→researched,
# researched→scored) and Phase 1b/2 transitions (e.g. scored→drafted,
# approved→sent).  Phase 1 code only exercises the Phase 1 subset, but the
# full table is defined here so this file is the single source of truth for
# every allowed transition — no reader should have to open another doc to know
# what's valid.  Every (from_state, to_state) pair:
#   - new → enriched: enrichment succeeded (Phase 1b/2)
#   - new → researched: research succeeded without enrichment (Phase 1 direct
#     path, per §7a — resolves the gap that Phase 1 had no enrichment node)
#   - enriched → researched: research succeeded after enrichment
#   - researched → scored: scoring succeeded
#   - scored → drafted: policy allows drafting (Phase 1b/2)
#   - scored → watchlist: insufficient signal for an outbound attempt
#   - scored → not_target: non-ICP or missing critical fields
#   - drafted → awaiting_review: draft is complete, needs operator review
#   - awaiting_review → approved: operator approves the draft
#   - awaiting_review → not_target: operator rejects outright
#   - awaiting_review → researched: operator escalates for further research (§7)
#   - awaiting_review → failed: review timeout expired (§7)
#   - watchlist → scored: periodic re-check or new signal detected (§7)
#   - approved → sent: send gate passed in LIVE mode
#   - approved → dry_run_sent: send gate passed in DRY_RUN mode (§7e)
#   - approved → failed: SMTP or provider failure
#   - sent → replied: inbound message linked to this target
#   - sent → bounced: SMTP or provider reports a bounce (§7)
#   - bounced → suppressed: hard bounce — auto-suppression (§7)
#   - dry_run_sent → replied: inbound SIMULATED message linked to this
#     target (ticket C1, §7j) — the DRY_RUN mirror of sent → replied, so a
#     simulated reply can attach to a target B5 left in dry_run_sent (which
#     previously had no outbound edges at all).  Deliberately a SEPARATE
#     edge rather than collapsing DRY_RUN into "sent": dry_run_sent must
#     never inherit LIVE-mode semantics (rate limits, cooldowns, §7e), and
#     a distinct edge keeps the two send paths distinguishable in
#     state_transitions forever.
#   - replied → routed: classifier + routing rule applied
#   - routed → suppressed: classification was "unsubscribe"
#   - routed → drafted: a positive reply queued a follow-up draft (ticket
#     E1, §7k) — the ONLY exit from routed besides suppressed, so a target
#     classified positive can re-enter the draft stage and hold a
#     conversation instead of sitting in routed with nowhere to go.
VALID_TRANSITIONS: set[tuple[str, str]] = {
    ("new", "enriched"),
    ("new", "researched"),
    ("enriched", "researched"),
    ("researched", "scored"),
    ("scored", "drafted"),
    ("scored", "watchlist"),
    ("scored", "not_target"),
    ("drafted", "awaiting_review"),
    ("awaiting_review", "approved"),
    ("awaiting_review", "not_target"),
    ("awaiting_review", "researched"),
    ("awaiting_review", "failed"),
    ("watchlist", "scored"),
    ("approved", "sent"),
    ("approved", "dry_run_sent"),
    ("approved", "failed"),
    ("sent", "replied"),
    ("sent", "bounced"),
    ("bounced", "suppressed"),
    ("dry_run_sent", "replied"),
    ("replied", "routed"),
    ("routed", "suppressed"),
    ("routed", "drafted"),
}

# ── ANY_TARGET_TRANSITIONS ───────────────────────────────────────────────────
# These two states are reachable from *any* other state (per §3 "any → …"
# rows).  A target can fail (error path, no sources, invalid LLM output) or get
# suppressed (email lands in suppression list) at any point in the pipeline —
# there's no prerequisite state for these.  The `is_valid` check below ORs
# `to_state in ANY_TARGET_TRANSITIONS` with the exact-pair lookup so these are
# always allowed without listing every possible (from, to) pair.
ANY_TARGET_TRANSITIONS: set[str] = {"suppressed", "failed"}

# ── PHASE_1_REACHABLE_STATES ─────────────────────────────────────────────────
# Phase 1 of the build (research + scoring only, no enrichment/draft/send) can
# reach exactly these seven states.  "enriched" is included even though Phase 1
# doesn't run enrichment — it's valid in the table and could be reached if the
# operator manually sets it.  "drafted", "sent", "awaiting_review", etc. are NOT
# in this set because they require Phase 1b/2 nodes that don't exist yet.
# Task 12 (score_lead) and Task 16 (checkpoint test) import this set to assert
# that Phase 1 never accidentally reaches a Phase 1b-only state.
PHASE_1_REACHABLE_STATES: set[str] = {
    "new", "enriched", "researched", "scored", "watchlist", "not_target", "failed",
}


class StateTransitionRefused(Exception):
    """Raised when `transition()` refuses a state change.

    Two scenarios produce this:
    1. The (from_state, to_state) pair is not in VALID_TRANSITIONS and to_state
       is not in ANY_TARGET_TRANSITIONS — the requested path doesn't exist in
       the state machine.
    2. The actor is not "system" or "operator" — an untrusted caller attempted
       a state change.
    """


def transition(
    conn,
    *,
    target_id: str,
    from_state: str,
    to_state: str,
    reason: str,
    actor: str,
    run_id: str,
    step_id: str,
    policy_decision_id: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Validate and execute a target state transition, logging every step.

    This is the ONLY function in the codebase allowed to change a target's
    state.  Every other module calls this — never a raw SQL UPDATE.  It
    performs two writes through the write gate (each independently auditable):
    1. UPDATE targets SET state = ? WHERE target_id = ?
    2. INSERT INTO state_transitions (one audit row)

    Args:
        conn: An app.db.Conn from app.db.connect() (sqlite or postgres).
        target_id: The primary key of the target row to update.
        from_state: The state the target is currently in (caller's belief).
        to_state: The state to transition into.
        reason: Human-readable reason for the transition (e.g. "research success").
        actor: Must be "system" or "operator" — any other value is refused.
        run_id: The pipeline run this transition belongs to.
        step_id: The specific step within the run that triggered this.
        policy_decision_id: Optional reference to a policy_decisions row.
        agent_id: WHICH registered agent this transition is attributed to in
            write_log (ticket B2c).  Defaults to ``actor`` so every pre-B2c
            caller is unchanged.  The LLM judge never calls this itself —
            deterministic wiring code executes the transition and passes
            agent_id="icp_judge" so the write_log rows for a judge-driven
            routing hop name the judge principal, not "system".  ``actor``
            still records what KIND of principal performed the write
            (deterministic code), which is why it stays "system" there.

    Returns:
        The transition_id of the new state_transitions audit row (e.g. "trn_3f9a2b1c").

    Raises:
        StateTransitionRefused: If the transition pair is invalid or the actor
            is not "system"/"operator".
    """

    # agent_id defaults to actor so every existing caller (which passes only
    # actor) keeps its current attribution byte-for-byte — the param is purely
    # additive for callers that want to name a different registered principal
    # (B2c's judge-driven routing hop).  The write gate validates agent_id
    # against agent_registry downstream, so a bogus id is refused there.
    write_agent_id = agent_id if agent_id is not None else actor

    # ── Validate the transition pair ────────────────────────────────────────
    # is_valid is True if either:
    #   (a) the exact (from, to) pair is in the VALID_TRANSITIONS allow-list, OR
    #   (b) to_state is in ANY_TARGET_TRANSITIONS (suppressed/failed are
    #       reachable from anywhere, per §3 "any → …" rows).
    # This is an OR, not an AND — either condition alone is sufficient.  Any
    # transition that matches neither is immediately refused.
    is_valid = (from_state, to_state) in VALID_TRANSITIONS or to_state in ANY_TARGET_TRANSITIONS
    if not is_valid:
        raise StateTransitionRefused(
            f"{from_state} -> {to_state} is not a valid transition"
        )

    # ── Validate the actor ──────────────────────────────────────────────────
    # Only "system" (deterministic pipeline code) and "operator" (the human
    # running the harness) are allowed to change state.  An LLM-generated
    # string, empty string, or any other value is refused — no ambiguous
    # or unauthenticated state changes.  This check is separate from the
    # write_gate's own actor check because the state machine enforces its own
    # actor policy independently of the write gate.
    if actor not in ("system", "operator"):
        raise StateTransitionRefused(f"invalid actor: {actor}")

    # ── Write 1 of 2: UPDATE targets.state ──────────────────────────────────
    # This is the actual data mutation — the target row's state column is
    # changed and updated_at is bumped.  It flows through write_gate.commit()
    # so it produces a write_log audit row with action="state_transition".
    # The payload records the from/to/reason so the write_log row is
    # self-describing even without joining state_transitions.
    write_gate_commit(
        conn,
        action="state_transition",
        table_name="targets",
        record_id=target_id,
        payload={"from": from_state, "to": to_state, "reason": reason},
        run_id=run_id,
        step_id=step_id,
        actor=actor,
        # write_agent_id names the attributed principal: actor by default
        # (pre-B2c behaviour), or the caller-supplied agent_id — e.g.
        # "icp_judge" when a judge-driven routing hop is recorded (B2c).
        # The write gate refuses an unregistered/disabled/unauthorized
        # agent_id, so this cannot forge attribution to a phantom principal.
        agent_id=write_agent_id,
        policy_decision_id=policy_decision_id,
        sql="UPDATE targets SET state = ?, updated_at = datetime('now') WHERE target_id = ?",
        params=(to_state, target_id),
    )

    # ── Write 2 of 2: INSERT state_transitions audit row ────────────────────
    # A separate write_gate.commit() call from the UPDATE above — each write
    # is independently auditable in write_log.  This is deliberate: if the
    # UPDATE succeeds but the audit INSERT fails (or vice versa), each failure
    # is isolated and traceable rather than collapsed into one multi-statement
    # write that might succeed partially.  The transition_id is generated here
    # with the "trn" prefix so it's self-identifying in logs and join output.
    transition_id = new_id("trn")  # "trn" prefix — self-describing in write_log and join output.
    write_gate_commit(
        conn,
        action="state_transition",
        table_name="state_transitions",
        record_id=transition_id,
        payload={"from": from_state, "to": to_state, "reason": reason},
        run_id=run_id,
        step_id=step_id,
        actor=actor,
        # write_agent_id — the attributed principal (see write 1 of 2; the
        # two write_log rows for one transition must agree on who to blame).
        agent_id=write_agent_id,
        policy_decision_id=policy_decision_id,
        # One comment per SQL block (not per line — this is a literal, not logic).
        # INSERT a row recording the previous state, new state, why the
        # transition happened, and who triggered it.  datetime('now') uses
        # SQLite's built-in UTC timestamp — single source of truth for created_at.
        # insert_seq (ticket C1, extending B5's fix) is computed IN the
        # INSERT via the scalar MAX+1 subquery — the same atomic,
        # dialect-neutral monotonic sequence B5 gave review_decisions /
        # policy_decisions / message_draft_versions, now on the state
        # machine's own audit log.  created_at is second-precision TEXT, so
        # two hops inside one second (replied → routed → suppressed, one
        # classify_and_route_reply call) previously ordered arbitrarily;
        # the sequence column makes the audit trail's order deterministic.
        sql="""
            INSERT INTO state_transitions
                (transition_id, run_id, step_id, target_id, previous_state,
                 new_state, reason, actor, matched_policy_id, insert_seq, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,
                    (SELECT COALESCE(MAX(insert_seq),0)+1 FROM state_transitions),
                    datetime('now'))
        """,
        params=(
            transition_id,
            run_id,
            step_id,
            target_id,
            from_state,
            to_state,
            reason,
            actor,
            policy_decision_id,
        ),
    )

    # Return the transition_id so the caller can reference this specific
    # transition later (e.g. to link a policy decision or error object).
    return transition_id
