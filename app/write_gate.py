"""
Write gate — the sole entry point for every core-table mutation.

## Why this module exists

Every INSERT, UPDATE, or DELETE against a "core" business table (offers, accounts,
contacts, targets, signals, etc.) must flow through `commit()`. There is no other
path. This single choke point enforces the following invariants per docs/gates.md §1.2:

1. **Unknown action types are refused.** If an action name isn't in KNOWN_ACTIONS,
   the write is rejected before it ever touches the database — no tool can silently
   introduce a new write path without updating this allowlist.
2. **Only trusted actors may write.** The only allowed actor values are "system"
   (deterministic pipeline code) and "operator" (the human running the harness).
   An LLM-generated actor string like "the_llm_decided" is rejected — no
   ambiguous or unauthenticated writes.
3. **The write and its audit row are one atomic transaction.** On success, both
   the data row INSERT/UPDATE and the corresponding write_log row are committed
   together. On failure, both are rolled back — there is never a dangling audit
   row with no corresponding data, or a data mutation that wasn't logged.
4. **Only registered agents may write, and only their allowed actions.** Every
   commit() carries an `agent_id` naming WHICH principal wrote (`actor` says
   what KIND of principal). The gate looks the agent up in `agent_registry`
   and refuses — via WriteGateRefused, before any SQL runs — writes from an
   unregistered agent, a disabled agent (enabled=0, the per-agent kill
   switch), or an action outside that agent's allowed_actions. This turns the
   global KNOWN_ACTIONS allowlist into a per-agent capability set (plan A3).
5. **Suppression removal requires the operator flag (ticket H8).** A
   `delete_suppression` write is refused unless BOTH hold:
   `operator_confirmed=True` AND `actor="operator"`. The refusal fires before
   any SQL runs, so a removal can never be attributed to system code, or to
   a caller that forgot the flag, and leave a partial or unlogged delete.
   docs/gates.md §1.2's "any suppression removal without an operator flag"
   is enforced HERE, in the gate — not in the goodwill of whichever caller
   happens to remove a suppression.

## Dependencies

- Uses `app.ids.new_id("wr")` to generate the primary key for the write_log row.
- Consumes `app.db.Conn` objects created by `app.db.connect()` (sqlite or postgres).
- Does NOT import or depend on any other app module (no circular imports).

## Interfaces (later tasks depend on these)

- `WriteGateRefused` — exception class raised when a gate check fails.
- `KNOWN_ACTIONS: set[str]` — the fixed allowlist of permitted action names.
- `commit(conn, *, ...) -> str` — the sole write function; returns the write_id
  of the newly-created audit row.
"""

import json

from app.ids import new_id

# ── KNOWN_ACTIONS ─────────────────────────────────────────────────────────────
# This is a FIXED allowlist, not an open set, because every new write path must
# be explicitly registered here before it can touch any core table. If a tool
# invents a new action string without adding it to this set, commit() raises
# WriteGateRefused. Per docs/gates.md §1.2: "refuses any action type it doesn't
# recognize — no silent new write paths." The actions are named as verb_table
# so a reader can tell at a glance what's being written and to which table.
KNOWN_ACTIONS = {
    "insert_offer",          # Creating a new offer row (the product being pitched).
    "insert_account",        # Creating a new account/company row.
    "insert_contact",        # Creating a new contact (person at an account).
    "insert_target",         # Creating a new target (account + contact + offer link).
    "update_account_score",  # Updating the score field on an existing account row.
    "update_account_profile",  # Updating the researched profile fields (company_summary/industry/estimated_size/geo) on an existing account row.
    # Persisting the ICP judge's verdict (judge_fit_label / judge_rationale /
    # judge_divergence_justification) on an account row — ticket B2c.  A
    # SEPARATE action from update_account_score so the write_log trail can
    # tell "the deterministic formula wrote the score" apart from "the LLM
    # judge wrote its verdict", and so a future capability narrowing of the
    # judge agent can revoke verdict writes without touching score writes.
    "update_account_icp_verdict",
    "update_target_score",   # Updating the score field on an existing target row.
    "insert_signal",         # Creating a new enrichment signal row.
    "insert_source",         # Persisting one evidence source's raw text (ticket B2b: fetched pages + research findings).
    "state_transition",      # Writing a state transition (moving a target between states).
    "insert_policy_decision",  # Writing a policy_check decision row (Task 13).
    "insert_agent_registry",   # Writing an agent_registry row (A3's bootstrap seeder).
    # Persisting one writer⇄critic iteration to message_draft_versions
    # (ticket B3).  Its OWN action — not folded into any existing one — for
    # the same reason update_account_icp_verdict is separate from
    # update_account_score: so the write_log trail distinguishes "the draft
    # agent produced a revision" from every other write, and so a future
    # capability narrowing can revoke draft writes without touching
    # score/verdict writes.
    "insert_message_draft_version",
    # Persisting one human review decision (ticket B4b).  Its OWN action —
    # not folded into state_transition or any existing one — for the same
    # reason as the entries above: the write_log trail must distinguish
    # "the operator approved/rejected/escalated this target" from every
    # other write, and a future capability narrowing (e.g. removing review
    # writes from a delegated principal) can revoke it without touching
    # transition or suppression writes.  app/review.py is the only caller.
    "insert_review_decision",
    # Persisting one suppression row (ticket B4b, the reject_and_suppress
    # decision).  Its OWN action for the same audit-trail reason: write_log
    # must tell "the OPERATOR manually suppressed this email" (added_by /
    # actor both record the operator) apart from a reply-classifier
    # suppression or a hard-bounce auto-suppression (added_by=system), so
    # the provenance of every suppression is attributable from the audit
    # log alone.  app/review.py is the only caller.
    "insert_suppression",
    # Removing one suppression row (ticket H4b, scripts/add_suppression.py;
    # enforcement moved INTO the gate by ticket H8).  Its OWN action for the
    # same audit-trail reason as insert_suppression: write_log must tell
    # "the operator REMOVED a suppression" apart from an insertion.  The
    # gate REFUSES this action unless BOTH hold: operator_confirmed=True
    # AND actor="operator" (docs/gates.md §1.2) — enforced in commit()
    # before any SQL runs, so a removal can never be attributed to system
    # code or to a caller that forgot the flag.  scripts/add_suppression.py
    # is the only caller.
    "delete_suppression",
    # Persisting one outbound message row (ticket B5, the DRY_RUN send).
    # Its OWN action for the same audit-trail reason: write_log must tell
    # "the send stage recorded a message" apart from every other write.
    # In B5 the ONLY rows written carry status='dry_run_sent' and
    # sent_at=NULL — no LIVE send path exists anywhere in the repo (an AST
    # test in tests/test_send_gate.py enforces that no mail transport can
    # even be imported), so no code can produce a real 'sent' row today.
    # app/tools/send_email.py is the only caller.
    "insert_message",
    # Persisting one send-gate preflight verdict (ticket B5).  Its OWN
    # action for the same audit-trail reason: write_log must tell "the
    # send gate evaluated a send" apart from every other write, and every
    # evaluation — allow OR refuse — writes exactly one row
    # (docs/gates.md §2.2/§2.3a: a refused send that leaves no record is
    # the failure mode the gate exists to prevent).  app/send_gate.py is
    # the only caller.
    "insert_send_gate_decision",
    # Persisting one inbound reply row (ticket C1, the simulated inbox).
    # Its OWN action for the same audit-trail reason: write_log must tell
    # "a reply ARRIVED" (the fetch step) apart from "the classifier judged
    # it" (update_reply_classification below) and from every other write.
    # The payload deliberately carries only REDACTED forms — never
    # raw_text — because write_log is a trace log and docs/threat-model.md
    # item 18 forbids raw PII in any trace payload (raw_text lives in the
    # master table replies alone, which is allowed to hold real data).
    # app/tools/fetch_inbox.py is the only caller.
    "insert_reply",
    # Persisting the classifier's verdict back onto an existing replies
    # row (ticket C1).  Its OWN action rather than being folded into
    # insert_reply for the same audit-trail reason as
    # update_account_icp_verdict: write_log must distinguish "the reply
    # arrived" from "the LLM classified it", so a future capability
    # narrowing of the reply_classifier agent can revoke verdict writes
    # without touching the inbox fetch.  app/agents/reply.py is the only
    # caller.
    "update_reply_classification",
    # Writing the two draft-gate verdict columns (ticket G2) onto one
    # persisted message_draft_versions revision.  Its OWN action for the
    # same audit-trail reason: write_log must distinguish "the draft agent
    # persisted a revision" (insert_message_draft_version) from "the
    # deterministic draft gate runner evaluated it" — and a future
    # capability narrowing of the drafting principal can revoke
    # insert_message_draft_version without revoking this system-owned
    # runner write (and vice versa).  app/draft_gate.py is the only caller.
    "update_draft_gate_columns",
    # Persisting one real scheduling reservation (demo, 2026-08-30): a slot
    # from an actual computed calendar (app/tools/schedule_meeting.py),
    # reserved for a target on the follow-up-draft path. Its OWN action for
    # the same audit-trail reason as every action above: write_log must
    # tell "the scheduling agent reserved a real slot" apart from every
    # other write, and a future capability narrowing of the scheduling
    # principal can revoke it without touching draft/verdict writes.
    # app/tools/schedule_meeting.py is the only caller.
    "insert_meeting",
}


class WriteGateRefused(Exception):
    """Raised when the write gate rejects a commit attempt.

    Four scenarios produce this:
    1. The action is not in KNOWN_ACTIONS — a tool is trying to introduce a write
       path without registering it.
    2. The actor is not "system" or "operator" — an untrusted or ambiguous caller
       is attempting a core-table mutation.
    3. The agent_id is not registered in agent_registry, or is disabled
       (enabled=0), or lacks the action in its allowed_actions — per-agent
       capability enforcement (plan A3). Each message names the agent and the
       action so a refused write is observable, never silent."""


def commit(
    conn,
    *,
    action: str,
    table_name: str,
    record_id: str,
    payload: dict,
    run_id: str,
    step_id: str,
    actor: str,
    agent_id: str,
    sql: str,
    params: tuple,
    policy_decision_id: str | None = None,
    operator_confirmed: bool = False,
) -> str:
    """Execute a core-table mutation and log it atomically.

    Args:
        conn: An app.db.Conn from app.db.connect() — sqlite or postgres; the
            wrapper makes the two dialects indistinguishable to this function.
        action: Must be one of KNOWN_ACTIONS (e.g. "insert_offer").
        table_name: The target table being written to (e.g. "offers").
        record_id: The primary key of the row being written (e.g. "off_abc123").
        payload: A dict of the data being written — serialized to JSON for the audit row.
        run_id: The pipeline run this write belongs to (groups steps together).
        step_id: The specific step within the run that triggered this write.
        actor: Must be "system" or "operator" — any other value is refused.
        agent_id: Which registered agent is writing (e.g. "system", "operator",
            or a later task's "research_agent"). Required — every caller must
            declare its identity. Refused unless registered, enabled, and
            authorized for this action in agent_registry.
        sql: The parameterized SQL statement to execute (no f-strings, no formatting).
        params: The tuple of parameter values matching the ? placeholders in sql.
        policy_decision_id: Optional reference to a policy_decisions row (for writes
            that were gated by a policy check).
        operator_confirmed: The explicit operator flag docs/gates.md §1.2
            demands for a suppression removal.  Defaults to False — the safe
            value — so a caller that does not pass it cannot remove a
            suppression.  Meaningless for every action EXCEPT
            delete_suppression; inserts are never gated by it.  A
            delete_suppression write is refused unless BOTH
            operator_confirmed=True AND actor="operator".

    Returns:
        The write_id of the new write_log row (e.g. "wr_3f9a2b1c").

    Raises:
        WriteGateRefused: If action is not in KNOWN_ACTIONS, actor is not
            allowed, the agent is unregistered/disabled/unauthorized for
            the action, or (ticket H8) the action is delete_suppression
            without the operator flag / without actor="operator" — each
            message names the agent and the action.
        Exception: Any database error propagates after rollback (the caller must handle it)."""

    # ── Gate check 1: refuse unknown action types ─────────────────────────────
    # Per docs/gates.md §1.2: "refuses any action type it doesn't recognize."
    # This is an allowlist, not a denylist — if it's not here, it's not allowed.
    # Every new write path must update this set before it can call commit().
    if action not in KNOWN_ACTIONS:
        raise WriteGateRefused(f"unknown action type: {action}")

    # ── Gate check 2: refuse invalid actors ───────────────────────────────────
    # Per docs/gates.md §1.2: only "system" (deterministic pipeline code) and
    # "operator" (the human) are allowed to write. LLM-generated strings,
    # empty strings, "admin", "auto", etc. are all rejected — no exceptions.
    # This ensures every write can be traced back to a trusted caller.
    if actor not in ("system", "operator"):
        raise WriteGateRefused(f"invalid actor: {actor}")

    # ── Gate check 3: per-agent capability enforcement (plan task A3) ─────────
    # `actor` answers what KIND of principal wrote; `agent_id` answers WHICH
    # registered principal wrote. The agent_registry row carries that agent's
    # allowed_actions, so the KNOWN_ACTIONS check above is now per-agent.
    # Three refusals, each naming the agent and the action so a rejected
    # write is observable in logs and stack traces, never silent:
    #   - no registry row for agent_id (unregistered principal)
    #   - enabled=0 (that agent's kill switch is thrown)
    #   - action not in the agent's allowed_actions JSON array
    #
    # Bootstrap exemption — writes to agent_registry ITSELF skip this check.
    # The registry is the gate's own configuration table: it cannot gate its
    # own first population, because no agent exists yet when the seeder runs.
    # The ONLY caller that writes agent_registry is
    # app/agents_registry.seed_agent_registry(), and its writes are still
    # subject to the KNOWN_ACTIONS and actor checks above, so this exemption
    # opens no path around those layers.
    if table_name != "agent_registry":
        # Read the agent's row BEFORE begin_write — this is a read-only
        # capability check, and refusing here means no transaction was ever
        # opened and no SQL below ever ran.
        row = conn.execute(
            "SELECT enabled, allowed_actions FROM agent_registry WHERE agent_id=?;",
            (agent_id,),
        ).fetchone()
        if row is None:
            # Unregistered principal: no row in agent_registry for this id.
            raise WriteGateRefused(
                f"agent {agent_id!r} is not registered in agent_registry; "
                f"refusing action {action!r}"
            )
        if row["enabled"] == 0:
            # Disabled agent: the per-agent kill switch. This fires even for
            # an otherwise fully-authorized agent, so the operator can stop
            # one agent (e.g. the drafter) without touching any other.
            raise WriteGateRefused(
                f"agent {agent_id!r} is disabled (enabled=0); "
                f"refusing action {action!r}"
            )
        # allowed_actions is a JSON array of KNOWN_ACTIONS names — parse it
        # and test membership, exactly like gate check 1 but scoped to this
        # one agent rather than the global set.
        allowed_actions = json.loads(row["allowed_actions"])
        if action not in allowed_actions:
            raise WriteGateRefused(
                f"action {action!r} is not in agent {agent_id!r}'s "
                f"allowed_actions; refusing"
            )

    # ── Gate check 4: suppression removal requires the operator flag ──────────
    # Per docs/gates.md §1.2: "any suppression removal without an operator
    # flag" is refused.  Ticket H8 moved this rule INTO the gate — it used
    # to live only in scripts/add_suppression.py, so the guarantee held for
    # callers that chose to go through that script and silently died for any
    # future caller of the delete_suppression action.  A gate whose
    # guarantee depends on every caller's goodwill is not a gate.
    #
    # BOTH conditions must hold for a removal:
    #   - operator_confirmed=True — the caller explicitly asserts the
    #     operator decided to remove this suppression.  Defaults to False
    #     (the safe value), so an existing caller that does not pass it
    #     cannot perform a removal.
    #   - actor="operator" — the write is attributed to the human, never to
    #     system/deterministic code.  An LLM or pipeline principal must not
    #     be able to silently lift a suppression.
    #
    # The check is scoped to delete_suppression ONLY — operator_confirmed is
    # meaningless for every insert/update action and must not gate them (if
    # you find yourself touching an insert path here, stop: that is a bug).
    # It fires before begin_write(), matching how the other refusals order
    # themselves, so a refused removal never opens a transaction, never
    # executes SQL, and leaves no write_log row claiming the removal
    # happened.
    if action == "delete_suppression":
        if actor != "operator":
            raise WriteGateRefused(
                f"delete_suppression requires actor='operator' "
                f"(got {actor!r}); refusing suppression removal"
            )
        if not operator_confirmed:
            raise WriteGateRefused(
                "delete_suppression requires operator_confirmed=True; "
                "refusing suppression removal without the operator flag"
            )

    # Generate the primary key for the audit row — this becomes the return value.
    # "wr" prefix makes write_log ids self-describing in logs and join output.
    write_id = new_id("wr")

    # ── begin_write(): take the write lock now, not later ─────────────────────
    # Using begin_write() instead of a literal BEGIN keeps this function
    # dialect-agnostic (see app/db.py's Conn.begin_write for the per-engine
    # spelling). On SQLite it issues BEGIN IMMEDIATE, which prevents a specific
    # race condition: plain BEGIN starts as a read transaction and only
    # upgrades to a write lock on the first INSERT/UPDATE/DELETE. If two
    # connections both BEGIN, then both try to upgrade, SQLite must abort one
    # with SQLITE_BUSY. But if they upgrade simultaneously, one can silently
    # "succeed" with a corrupted write. BEGIN IMMEDIATE takes the write lock
    # upfront — either it succeeds (we have exclusive write access) or it fails
    # cleanly with SQLITE_BUSY (we can retry or fail). Per docs/gates.md §1.2:
    # this repo requires immediate write-locking, never deferred.
    # On Postgres begin_write() issues plain BEGIN — Postgres has no BEGIN
    # IMMEDIATE and needs none, because row-level locking + MVCC arbitrate
    # concurrent writers instead of a global write lock (see app/db.py).
    conn.begin_write()
    try:
        # Execute the caller's parameterized SQL — this is the actual data mutation.
        # Any error here (bad SQL, constraint violation, etc.) jumps to the except
        # block, which rolls back the entire transaction including this statement.
        conn.execute(sql, params)

        # Insert the audit row: exactly one write_log row per successful write.
        # This is the "never skip logging" guarantee — every core-table mutation
        # produces an append-only log record with who wrote what and when.
        # agent_id is persisted alongside actor so the audit trail answers
        # both "what kind of principal" and "which specific agent".
        # datetime('now') is SQLite's UTC timestamp function; on the postgres
        # dialect the db layer's Conn.execute() translates it to
        # CURRENT_TIMESTAMP, so this statement is written once and works on
        # both engines.
        conn.execute(
            """
            INSERT INTO write_log
                (write_id, run_id, step_id, action, table_name, record_id,
                 actor, agent_id, matched_policy_id, payload_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?, datetime('now'))
            """,
            (
                write_id,
                run_id,
                step_id,
                action,
                table_name,
                record_id,
                actor,
                agent_id,
                policy_decision_id,
                json.dumps(payload),  # Serialize the payload dict to a JSON string for the audit row.
            ),
        )

        # Both the data write and the audit row succeeded — commit the transaction.
        # Until this point, neither row is visible to any other connection.
        conn.execute("COMMIT")
    except Exception:
        # The data write or the audit INSERT failed. Roll back the entire
        # transaction so neither the data row NOR the audit row persists.
        # This is the atomicity guarantee: a failed write must not leave a
        # dangling write_log row that claims a write happened when it didn't.
        # The original error is then re-raised so the caller can handle it.
        conn.execute("ROLLBACK")
        raise

    # Return the write_id so the caller can reference the audit row later
    # (e.g. to link a policy decision or state transition back to its write).
    return write_id
