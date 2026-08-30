"""
Agent registry bootstrap seeder (plan task A3; third principal added by
ticket B2c; fourth and fifth by ticket B3; sixth by ticket C1; seventh by
ticket C4; eighth by the 2026-08-30 real-scheduling demo).

The ``agent_registry`` table is the write gate's per-agent capability store:
every ``write_gate.commit()`` and ``log_step()`` call names the agent that
made it, and the gate refuses writes from agents that are not registered
here, are disabled (``enabled=0``), or lack the attempted action in their
``allowed_actions`` JSON array.

This module seeds the eight principals that exist today:

- ``system``           — deterministic pipeline code (no LLM involved)
- ``operator``         — the human running the harness
- ``icp_judge``        — the LLM ICP judge (ticket B2c): issues the final
  fit_label on top of the deterministic score.  An LLM principal, so it has
  a model_alias — the ``judge_model`` role from config/models.yaml.
- ``draft_writer``     — the LLM draft writer (ticket B3): one half of the
  writer⇄critic LoopAgent.  An LLM principal with model_alias
  ``draft_model``.
- ``draft_critic``     — the LLM draft critic (ticket B3): the other half of
  the LoopAgent.  An LLM principal with model_alias ``draft_model``.
- ``reply_classifier`` — the LLM reply classifier (ticket C1): classifies
  inbound replies; the deterministic router applies its verdict.  An LLM
  principal with model_alias ``reply_classifier_model``.
- ``taskmaster``       — the natural-language root agent (ticket C4): plans
  and dispatches the pipeline stages through FunctionTools.  An LLM
  principal with model_alias ``taskmaster_model`` — and the ONE principal
  whose allowed_actions is deliberately EMPTY (see the row's comment).
- ``meeting_scheduler`` — the LLM meeting scheduler (demo, 2026-08-30):
  picks a real open slot from the computed calendar for a follow-up
  draft's proposed meeting.  An LLM principal with model_alias
  ``scheduling_model``.

THE ROUTER DELIBERATELY HAS NO PRINCIPAL OF ITS OWN (ticket C1's §3.5
decision, argued here): ``reply_router`` is deterministic code that emits
no judgement of its own — its every write (the verdict update, the
transitions, the unsubscribe suppression) is wholly determined by the
classifier's verdict, so those writes are attributed to
``reply_classifier`` (the B2c pattern: "agent_id records whose decision
it applies", state-machine.md §4).  Registering a router principal would
imply a capability the router does not independently hold and would add
a sixth full-allowlist row for no audit benefit.  The B3 draft_critic
precedent does not apply: the critic is an LLM principal whose verdicts
ARE its own judgement; the router has none.

Seven of the eight get the full KNOWN_ACTIONS allowlist and unrestricted
transitions (``allowed_transitions="*"``), because every existing write path
belongs to these principals — narrowing any would break live pipeline
writes.

THE SEVENTH IS DIFFERENT — and deliberately so (ticket C4's §3.3 decision,
argued here): ``taskmaster`` gets ``allowed_actions: []`` — the EMPTY
allowlist.  The Taskmaster performs no gated writes of its own: every write
its tools trigger is made by deterministic inner code attributed to the
stage principals (import/research/transition writes as ``system``, draft
persists as ``draft_writer``, reply verdicts as ``reply_classifier``), and
the Taskmaster's own rows are steps-trace rows, which the write gate does
not govern.  The empty allowlist is therefore HONEST — it records exactly
the capability the agent holds — and it is a structural backstop for the
C4-Z1/Z3 zero-trust boundary: even a hypothetical future tool that
attributed a gated write (an approval, a switch flip, anything) to
``taskmaster`` would be refused by the gate before any SQL ran, because no
action is in the empty set.  Narrowing here breaks nothing today (nothing
writes as ``taskmaster``), so the correct capability set is implemented
rather than guessed around.  docs/agents.md and docs/policy-matrix.md §3b
restate this as the enforcement fact it is.
"""

import json

from app.write_gate import KNOWN_ACTIONS, commit as write_gate_commit

# ── Seed rows ─────────────────────────────────────────────────────────────
# The rows this module owns. allowed_actions is NOT hand-written here:
# it is serialized from the live KNOWN_ACTIONS set at seed time, so the
# registry can never drift out of sync with the gate's code-level allowlist.
# model_alias is NULL for the deterministic principals (system/operator) —
# NULL is the marker for "no LLM role alias" per docs/db-schema.md — and
# set for icp_judge, which calls call_structured through the judge_model
# role alias (B2c).  Giving the judge its OWN agent_id (rather than reusing
# "system") is what makes its verdict writes and its routing transitions
# attributable to it in write_log — the operator can see exactly which
# principal set the final label.
_SEED_ROWS = (
    {
        "agent_id": "system",
        "display_name": "Deterministic pipeline code",
        "description": "Non-LLM pipeline code (import, research plumbing, scoring, state machine) — the deterministic principal.",
        "model_alias": None,
        "allowed_transitions": "*",  # Unrestricted; transition enforcement lands in a later task.
    },
    {
        "agent_id": "operator",
        "display_name": "Human operator",
        "description": "The human running the harness — the only principal who can approve sends and review drafts.",
        "model_alias": None,
        "allowed_transitions": "*",
    },
    {
        # B2c: the ICP judge — an LLM principal, registered so its writes
        # (accounts.judge_* columns via action "update_account_icp_verdict")
        # and its judge-driven routing transitions are attributable to it.
        # It MAY set the final fit_label; it MUST NOT touch the numeric
        # fit_score policy P4 reads — enforced by construction (its output
        # schema, ICPVerdict, has no score field), not by this row.  The
        # kill switch (enabled) and capability narrowing apply to it like
        # any other agent: an operator can disable the judge and the
        # pipeline degrades to the deterministic label.
        "agent_id": "icp_judge",
        "display_name": "ICP judge agent",
        "description": "LLM judge that weighs the deterministic ICP score (evidence) plus signal evidence tiers against the offer's ICP and issues the final fit_label with a written rationale.",
        "model_alias": "judge_model",  # Non-NULL: the config/models.yaml role alias the judge's call_structured resolves (B2c).
        "allowed_transitions": "*",
    },
    {
        # B3: the draft writer — an LLM principal, registered so the draft
        # versions it authors (message_draft_versions rows via action
        # "insert_message_draft_version") and the draft-phase transitions
        # (scored→drafted, drafted→awaiting_review) are attributable to it
        # in write_log.  It PRODUCES TEXT and owns NO governed writes of its
        # own: deterministic wiring code (DraftPersistAndDecideNode) executes
        # every write and transition, attributing them to this principal —
        # the same split the judge uses.  The kill switch (enabled) applies
        # to it like any other agent: disabling the writer refuses its
        # attributed writes, which stops the drafting phase.
        "agent_id": "draft_writer",
        "display_name": "Draft writer agent",
        "description": "LLM writer that drafts the outreach email from the target's research brief, the offer's ICP/pitch/persona, and (from iteration 2 on) the critic's required changes.",
        "model_alias": "draft_model",  # Non-NULL: the config/models.yaml role alias the writer's LlmAgent resolves (B3).
        "allowed_transitions": "*",
    },
    {
        # B3: the draft critic — the writer's counterpart inside the
        # LoopAgent.  Its only product is the DraftCritique verdict; it has
        # no write capability of its own and every write attributed to it is
        # made by deterministic code on its behalf (none today — its verdicts
        # are persisted by the persist node under the writer principal, since
        # the row being written IS the writer's revision).  Kept as its own
        # registered principal so the audit vocabulary already has the name
        # when a future task attributes critic-side writes.
        "agent_id": "draft_critic",
        "display_name": "Draft critic agent",
        "description": "LLM critic that checks the writer's draft against the offer's ICP/pitch, evidence-tier discipline, cold-outreach tone, and compliance rules, and issues a passed/issues/required_changes verdict.",
        "model_alias": "draft_model",  # Non-NULL: the critic's LlmAgent resolves the same draft_model role (B3).
        "allowed_transitions": "*",
    },
    {
        # C1: the reply classifier — an LLM principal, registered so the
        # verdict it emits (persisted by the deterministic router via
        # action "update_reply_classification") and the router's
        # verdict-driven transitions/suppressions are attributable to it
        # in write_log.  It EMITS A CLASS and owns NO governed writes of
        # its own: the router executes every write and transition,
        # attributing them to this principal — the same split the judge
        # and the draft writer use.  The kill switch (enabled) applies to
        # it like any other agent: disabling the classifier refuses its
        # attributed writes, and the B4a guardrail refuses the whole
        # reply invocation at entry before a single model token is spent.
        "agent_id": "reply_classifier",
        "display_name": "Reply classifier agent",
        "description": "LLM classifier that reads one redacted inbound reply and assigns exactly one of the nine reply classes with a confidence and rationale; the deterministic reply router applies the verdict.",
        "model_alias": "reply_classifier_model",  # Non-NULL: the config/models.yaml role alias the classifier's LlmAgent resolves (C1).
        "allowed_transitions": "*",
    },
    {
        # C4: the Taskmaster root agent — an LLM principal, registered so
        # the guardrail's per-agent check can refuse it at entry
        # (agent_registry.enabled=0) and so its own step rows carry a
        # registered name.  IT PERFORMS NO GATED WRITES OF ITS OWN: its
        # tools dispatch the stage runners, whose inner deterministic code
        # writes under the stage principals (system / draft_writer /
        # reply_classifier) — the B2c/C1 attribution pattern, one level
        # up.  allowed_actions is therefore the EMPTY list: the honest
        # capability set, and the structural backstop that refuses any
        # gated write attributed to this agent (C4-Z1/Z3 — an approval or
        # a kill-switch flip attributed to the Taskmaster is refused by
        # the gate before any SQL runs, because no action is in the empty
        # set).  See the module docstring's seventh-principal note.
        "agent_id": "taskmaster",
        "display_name": "Taskmaster root agent",
        "description": "Natural-language root agent that plans and dispatches the pipeline stages (research, draft, DRY_RUN send, reply classification) through tools over the existing stage runners; stops and reports at the human approval gate, which it is structurally incapable of opening.",
        "model_alias": "taskmaster_model",  # Non-NULL: the config/models.yaml role alias the Taskmaster's LlmAgent resolves (C4).
        "allowed_transitions": "*",  # Transitions still go through state_machine.transition(), which does not enforce this column yet (policy-matrix.md §3a); the toolset's only transition is -> failed (Z1).
        "allowed_actions": [],  # THE NARROWED CAPABILITY (C4 §3.3): no gated write may ever be attributed to the Taskmaster.
    },
    {
        # Demo, 2026-08-30: the meeting scheduler — an LLM principal,
        # registered so the real slot reservation it produces (a
        # ``meetings`` row via action "insert_meeting") is attributable to
        # it in write_log, the same B2c/C1 split every other LLM principal
        # here uses. It picks WHICH open slot to propose from a REAL
        # computed calendar (a fixed weekly template projected forward from
        # "now", filtered against every already-reserved row so two targets
        # can never collide) and states why; app/tools/schedule_meeting.py
        # is the deterministic wiring code that performs the write and
        # re-validates the chosen slot is still free before committing it —
        # the model's judgement is never trusted blind. Fires only on the
        # follow-up-draft path (a positive reply already queued one), never
        # unattended: the resulting draft still needs the SAME human
        # approval as every other send, exactly like the booking_url link
        # it replaces did. The kill switch (enabled) applies to it like any
        # other agent: disabling it makes schedule_meeting degrade to a
        # deterministic earliest-available pick, never fail the target —
        # the same never-fail-the-target rule judge_icp uses.
        "agent_id": "meeting_scheduler",
        "display_name": "Meeting scheduler agent",
        "description": "LLM agent that picks a real open slot from the computed calendar for a follow-up draft's proposed meeting and states its reasoning; deterministic code re-validates the slot and performs the reservation write.",
        "model_alias": "scheduling_model",  # Non-NULL: the config/models.yaml role alias the scheduler's call_structured resolves.
        "allowed_transitions": "*",
    },
)


def seed_agent_registry(conn, *, run_id: str, step_id: str) -> None:
    """Upsert the five existing principals into agent_registry. Idempotent.

    Safe to call on every startup: rows that already exist are updated to
    match the seed definitions (so a code change to a capability propagates
    on the next run), EXCEPT ``enabled`` — re-seeding never re-enables an
    agent the operator has disabled, because the per-agent kill switch is
    operational state, not configuration.

    Writes go through the write gate (action ``insert_agent_registry``), so
    each registration produces a write_log audit row like every other
    core-table mutation. The writing principal is ``agent_id="system"``:
    the seeder is deterministic pipeline code, even when it writes the
    ``operator`` row — write_log.agent_id records who wrote, and the seeder
    is the system agent.
    """
    for row in _SEED_ROWS:
        # Serialize the agent's allowlist into the JSON array the gate
        # parses back at enforcement time.  The DEFAULT is the full current
        # KNOWN_ACTIONS set — serialized live so the registry can never
        # drift out of sync with the gate's code-level allowlist — and a
        # row may override it with its own capability set (C4: the
        # taskmaster row overrides with the EMPTY list — no gated write
        # may ever be attributed to it).  Sorted for stable,
        # human-readable diffs between seed runs.
        row_allowed = row.get("allowed_actions", KNOWN_ACTIONS)
        allowed_actions = json.dumps(sorted(row_allowed))
        write_gate_commit(
            conn,
            action="insert_agent_registry",  # Registered in KNOWN_ACTIONS by task A3.
            table_name="agent_registry",
            # The registry's primary key doubles as the audit row's record_id.
            record_id=row["agent_id"],
            # The audit payload carries the same capability facts the row
            # gets, so the write_log row is self-describing without a join.
            # allowed_actions is the SAME per-row set the SQL gets (the
            # override-aware row_allowed above) — the payload must never
            # claim a wider allowlist than the row actually received.
            payload={**row, "allowed_actions": sorted(row_allowed)},
            run_id=run_id,
            step_id=step_id,
            actor="system",     # Deterministic code — passes the actor allowlist.
            agent_id="system",  # The seeder itself is the system agent.
            sql="""
                INSERT INTO agent_registry
                    (agent_id, display_name, description, model_alias,
                     allowed_actions, allowed_transitions, enabled, created_at)
                VALUES (?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(agent_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    description=excluded.description,
                    model_alias=excluded.model_alias,
                    allowed_actions=excluded.allowed_actions,
                    allowed_transitions=excluded.allowed_transitions
                -- enabled and created_at are deliberately NOT updated on
                -- conflict: the kill switch must survive re-seeds (an
                -- operator's disable is operational state), and created_at
                -- records first registration, not the latest seed.
            """,
            params=(
                row["agent_id"],
                row["display_name"],
                row["description"],
                row["model_alias"],  # None → SQL NULL: no LLM role alias.
                allowed_actions,
                row["allowed_transitions"],
                1,  # enabled=1 on first insert only (see comment in the SQL above).
            ),
        )
