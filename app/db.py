"""
Database connection and schema for the outbound-agency pipeline.

This module is the single source of truth for how this system talks to its
database. Every other module that needs the database imports connect() and
apply_schema() from here.

As of the Google port (plan 2026-08-17-google-agentic-port, Task A2) this
module speaks TWO dialects behind ONE interface:

- **sqlite** — the original engine. Target is a plain file path or
  ":memory:". Every existing test still runs against this.
- **postgres** — Cloud SQL for the deployed hackathon app, because SQLite on
  Cloud Run is ephemeral (the container's disk resets on cold start and a
  judge hitting the hosted URL would get an empty database). Reached via a
  ``postgresql://`` URL or the ``cloudsql://<instance-connection-name>/<db>``
  sentinel.

connect() inspects its argument and returns a Conn whose behavior is identical
from the caller's point of view: same execute()/commit()/rollback() call
shapes, same dict-like rows (``row["column_name"]`` works on both), same ``?``
placeholders in the SQL the caller writes. The dialect differences — ``?`` vs
``%s`` placeholders, ``datetime('now')`` vs ``CURRENT_TIMESTAMP``, ``BEGIN
IMMEDIATE`` vs ``BEGIN``, exception classes — are all absorbed inside this
module, NOT pushed onto the ~13 modules that write SQL.

Two hard requirements baked into the sqlite path (unchanged since Task 1):

1. WAL mode (Write-Ahead Logging) — without this, any reader blocks all writers
   and vice versa. In a pipeline where the state machine reads one target while
   the send gate writes another, WAL is the difference between "works" and
   "SQLITE_BUSY everywhere." See docs/gates.md §1.2.

2. BEGIN IMMEDIATE — every write transaction in this codebase (starting in Task 4
   with write_gate.py) uses BEGIN IMMEDIATE, never plain BEGIN. Plain BEGIN starts
   as a read transaction and upgrades to write on the first INSERT/UPDATE/DELETE.
   If two connections upgrade simultaneously, one can get a silently-corrupted
   write. BEGIN IMMEDIATE takes the write lock up front or fails cleanly — the
   test test_write_transaction_uses_begin_immediate_semantics proves this.

TABLE-LEVEL COMMENTS below cross-reference the docs and later tasks so a reader
can see, without opening 5 files, which tool or task populates each table.
"""

import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse

# pg8000 is the postgres driver for BOTH postgres paths: the Cloud SQL Python
# Connector's pg8000 driver (google.cloud.sql.connector) and direct
# ``postgresql://`` URLs. The dbapi submodule gives us the DB-API surface
# (cursor.execute, commit/rollback) and the IntegrityError class we re-raise
# normalized constraint violations as; the exceptions submodule is what pg8000
# actually raises (it never raises its own dbapi error subclasses on its own —
# see _re_raise_pg_integrity).
from pg8000 import dbapi as pg8000_dbapi
from pg8000 import exceptions as pg8000_exceptions

# ── IntegrityError ─────────────────────────────────────────────────────────────
# A tuple of BOTH dialects' integrity-violation exception classes, so callers
# (e.g. app/tools/detect_signals.py, which dedupes signals against a UNIQUE
# constraint) write ONE ``except IntegrityError:`` and get the right class per
# dialect. sqlite raises sqlite3.IntegrityError natively; pg8000 raises a raw
# pg8000.exceptions.DatabaseError, which Conn.execute() re-raises as
# pg8000_dbapi.IntegrityError when the SQLSTATE is class 23 (integrity
# constraint violation — unique/fk/not-null/check). This tuple is load-bearing:
# if it regressed, duplicate signals would crash the detect_signals node
# instead of being skipped.
IntegrityError = (sqlite3.IntegrityError, pg8000_dbapi.IntegrityError)


# ── Suppression address normalisation (ticket F1b) ─────────────────────────
# The suppression table stores the address AS WRITTEN (its audit record of
# what actually arrived) and a separate `email_normalized` matching key.
# EVERY producer and consumer of suppressions must call these two functions
# — duplicated folding logic is exactly how the C2 suppression-evasion bug
# came back, so there is one definition here and nowhere else.


def normalize_email(email: str) -> str:
    """Return the canonical matching key for one email address.

    Three folds, each a deliberate choice with a stated reason:
    - domain -> lowercase, always (RFC 1035: domain names are
      case-insensitive, so ``SERENITY-CLINIC.TEST`` and
      ``serenity-clinic.test`` are the SAME mailbox by the standard);
    - local part -> lowercase (the RFC permits local-part case-sensitivity,
      but effectively no real provider uses it — Gmail and Outlook both
      fold it — and the ticket biases toward OVER-suppression);
    - plus-tag -> strip ``+suffix`` from the local part (Gmail and Outlook
      both treat ``user`` and ``user+tag`` as the same mailbox).

    The stored `email` column is never overwritten; this value is written
    beside it and used only for matching.  A malformed string with no `@`
    is not an address — the only safe fold is lowercasing it, so the caller
    keeps its guard rather than crash here.
    """
    if not email:
        return email  # Empty string is its own key; no fold is meaningful.
    local, sep, domain = email.partition("@")  # one split: local/domain halves
    if not sep:
        # No @ means no local/domain split — nothing to plus-strip, and
        # lowercasing the whole string is the only defensible fold.
        return email.lower()
    local = local.lower().split("+", 1)[0]  # lowercase FIRST, then strip the +tag from the local part (order matters: a tag is matched after case folding)
    domain = domain.lower()  # RFC 1035: domains are case-insensitive, always fold
    return f"{local}@{domain}"


def email_syntax_valid(email: str) -> str | None:
    """Return None if ``email`` is syntactically well-formed, else a reason.

    This is the G1 operator-assertion validator: the operator can vouch for
    a real address's DELIVERABILITY, but not for a malformed string, so a
    CSV ``email_verified`` assertion is only honoured when the address also
    passes this purely-syntactic gate.  No network, no DNS, no dependency —
    the checks are all local string rules (ticket G1 §2.2).

    WHY IT REUSES ``normalize_email`` (the F1b lesson): there is exactly ONE
    address parser in this codebase, ``normalize_email``, which folds domain
    case, local-part case, and plus-tags.  This validator does NOT re-split
    or re-fold the address itself; it runs the one shared fold first and
    then checks the already-canonical halves.  Duplicated address logic is
    exactly how the C2 suppression-evasion bug came back, so any new email
    rule must build on the same canonical form every other consumer sees.

    The rejected forms (each returned as its own reason so a reader can tell
    a deliberate rule from an accident):
      - empty address (nothing to validate);
      - whitespace anywhere inside (an address is one atom, no spaces);
      - anything but exactly one ``@`` (zero or several ``@``s is not a
        mailbox address);
      - empty local part after folding (``@domain`` — nothing before the @);
      - empty domain (``user@`` — nothing after the @);
      - a domain with no dot (``user@localhost`` has no registrable split;
        reserved test domains like ``.test``/``.invalid`` all carry a dot);
      - a leading/trailing dot in the local part or the domain (those are
        never valid in an address).
    """
    address = email.strip()  # Trim copy-paste padding BEFORE judging — leading/trailing whitespace is not a syntax failure, interior whitespace is.
    if not address:
        return "empty address"  # Nothing to validate — a blank value can never be asserted verified.
    if any(ch.isspace() for ch in address):
        return "contains whitespace"  # An address is a single token; a space inside means it was pasted wrong, so refuse loudly.
    if address.count("@") != 1:
        return "must contain exactly one @"  # Exactly one separator; zero @ is a username, two @s is never a mailbox.
    # Reuse THE shared parser (normalize_email) so case and plus-tags are
    # folded by the same code every other module uses, then split the single
    # @ that normalization is guaranteed to have produced (count == 1 above).
    local, _sep, domain = normalize_email(address).partition("@")
    if not local:
        # The local part is empty after folding — ``@domain``, or a local
        # that is only a plus-tag (``+tag@domain`` folds to ``@domain``).
        return "empty local part"
    if not domain:
        return "empty domain"  # ``user@`` — there is no host to deliver to.
    if "." not in domain:
        return "domain has no dot"  # ``user@localhost`` cannot be a deliverable public mailbox; reserved test domains all carry a dot.
    if local.startswith(".") or local.endswith("."):
        return "leading or trailing dot in local part"  # ``.user@d.test`` / ``user.@d.test`` are not valid local forms.
    if domain.startswith(".") or domain.endswith("."):
        return "leading or trailing dot in domain"  # ``user@.d.test`` / ``user@d.test.`` are not valid hosts.
    return None  # Every rule passed — the address is syntactically sound.


def normalize_domain(domain: str) -> str:
    """Return the canonical (lowercased) form of a domain.

    Domain names are case-insensitive per RFC 1035, so case is not data and
    lowercasing on both write and read is the matching key.  A None/empty
    domain is returned as "" so callers can compare without a separate null
    branch (the send gate already guards ``domain is not None``).
    """
    return (domain or "").lower()


# ── DDL ──────────────────────────────────────────────────────────────────────
# Copied from docs/db-schema.md (plan-reviewed, 2026-08-01). Each CREATE TABLE
# uses IF NOT EXISTS so apply_schema() is idempotent — safe to call at startup
# without checking whether tables already exist.
#
# NOTE ON ORDERING: statements are ordered so every table a FOREIGN KEY
# REFERENCES appears EARLIER in the script. SQLite never needed this — it
# resolves forward references lazily inside executescript() — but Postgres
# validates "REFERENCES targets(target_id)" at CREATE TABLE time and refuses a
# forward reference ("relation targets does not exist"). Reordering satisfies
# both engines with one DDL, which is why the table order below no longer
# matches the section order in docs/db-schema.md exactly.

# ── The suppressions table DDL, defined ONCE (ticket H4b) ────────────────────
# This constant is the single source of the suppressions table shape for BOTH
# the fresh-database path (it is spliced into _DDL below) and the in-place
# H4b migration (_migrate_suppressions), so the rebuild can never drift from
# what apply_schema creates from scratch.  Why each line is what it is:
#   - `email` is NOT the primary key any more.  It stays as the audit record
#     of the address AS WRITTEN (suppression-policy.md §1a) — but a
#     domain-only suppression has no address, and `email TEXT PRIMARY KEY`
#     made that row impossible on Postgres (SQLite tolerated a NULL PK, a
#     long-standing quirk; Postgres rejects it).  Dropping the PK lets a
#     domain-only row carry email = NULL.
#   - `email_normalized TEXT UNIQUE` is the matching key every reader and
#     writer already uses (ticket F1b).  SQL treats NULLs as distinct in a
#     UNIQUE column on BOTH dialects, so any number of domain-only rows
#     (email_normalized = NULL) can coexist — the property that makes the
#     many-domain-only-rows case legal.
#   - `domain TEXT UNIQUE` is the lowercased domain (normalize_domain()); a
#     domain-only row's uniqueness key, so re-adding the same domain is
#     caught by the constraint (and by the writer's check-then-insert).
#   - The table-level CHECK stops a row that suppresses NOTHING (email NULL
#     AND domain NULL) — the one row shape that would silently match nothing
#     on every read path.
_SUPPRESSIONS_DDL = """CREATE TABLE IF NOT EXISTS suppressions (
  email TEXT,
  email_normalized TEXT UNIQUE,
  domain TEXT UNIQUE,
  reason TEXT NOT NULL CHECK (reason IN ('unsubscribe','bounce','complaint','manual','legal','risky_reply')),
  added_at TEXT NOT NULL,
  added_by TEXT NOT NULL CHECK (added_by IN ('system','operator')),
  notes TEXT,
  CHECK (email IS NOT NULL OR domain IS NOT NULL)
);"""

_DDL = ("""
CREATE TABLE IF NOT EXISTS accounts (
  account_id TEXT PRIMARY KEY,
  company_name TEXT NOT NULL,
  domain TEXT NOT NULL,
  normalized_domain TEXT NOT NULL UNIQUE,
  industry TEXT,
  estimated_size TEXT,
  geo TEXT,
  company_summary TEXT,
  icp_fit_label TEXT,
  icp_fit_score INTEGER,
  icp_fit_reasons TEXT,
  icp_non_fit_reasons TEXT,
  -- The ICP judge's verdict (plan task B2c): the deterministic icp_fit_*
  -- columns above are the EVIDENCE (score + label + reasons the formula
  -- produced); these three are the judge's FINAL verdict — its label, its
  -- written rationale, and (only when the judge diverged from the
  -- deterministic label) its divergence justification.  All three are
  -- nullable: NULL means the judge never produced a verdict for this
  -- account (it failed after its bounded retries, or has not run yet) —
  -- in which case the deterministic label stands.  A divergence is visible
  -- in the audit trail as judge_fit_label != icp_fit_label without reading
  -- any code.  The judge has NO score column here by construction: the
  -- numeric fit_score policy P4 reads stays icp_fit_score, which only
  -- score_lead's deterministic formula writes.
  judge_fit_label TEXT,
  judge_rationale TEXT,
  judge_divergence_justification TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Offers are the products/services being pitched. Targets are linked to an
-- offer so the system knows what's being sold. One active offer per workspace
-- is the v1 default, but the schema supports multiple for future expansion.
-- Written manually or via admin tooling, not auto-populated.
CREATE TABLE IF NOT EXISTS offers (
  offer_id TEXT PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);

-- Contacts are people at an account. One account can have many contacts.
-- Populated by target import and enrichment tools. email_verified is a boolean
-- (0/1) defaulting to 0 — later verification steps set it to 1.
CREATE TABLE IF NOT EXISTS contacts (
  contact_id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(account_id),
  full_name TEXT,
  title TEXT,
  seniority TEXT,
  department TEXT,
  email TEXT,
  email_verified INTEGER NOT NULL DEFAULT 0,
  linkedin_url TEXT,
  persona_fit_score INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Targets are the execution units of the pipeline: an account + contact + offer
-- moving through the state machine. state is driven by docs/state-machine.md.
-- final_recommendation is set by the scoring/classification step before human
-- review. Populated initially by the target import tool, then updated at each
-- pipeline stage.
CREATE TABLE IF NOT EXISTS targets (
  target_id TEXT PRIMARY KEY,
  account_id TEXT NOT NULL REFERENCES accounts(account_id),
  contact_id TEXT REFERENCES contacts(contact_id),
  offer_id TEXT NOT NULL REFERENCES offers(offer_id),
  source TEXT NOT NULL,
  state TEXT NOT NULL,
  score INTEGER,
  final_recommendation TEXT,
  last_signal_refresh_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Signals are enrichment data points attached to a target (e.g. "funding_round",
-- "hiring_surge", "tech_stack_change"). Written by the enrichment pipeline
-- (Task TBD: research/enrich tools). Each signal is scoped to a (target, run,
-- type, value) tuple so the same signal isn't scored twice in one run.
-- evidence_quote / evidence_verified (plan task B2a) back every signal with a
-- verbatim quote from the text detect_signals was given.  evidence_tier (plan
-- task B2b) replaces that single boolean with the three-way verdict
-- detect_signals now computes: 'source' = the quote appears in a persisted raw
-- source text we actually fetched; 'findings' = it appears only in the
-- research agent's prose (plausibly from a server-side search we cannot
-- capture); 'unverified' = it appears in neither — the fabrication signal.
-- INVARIANT (B2b): evidence_verified = 1 if and only if evidence_tier =
-- 'source' — "verified" now means verified against persisted raw text, and
-- the two columns are written from the same computation so they can never
-- disagree.  All three columns are nullable ONLY as a migration accommodation
-- for signals rows written before B2a/B2b existed — detect_signals always
-- populates them on new rows, so new data is never actually NULL.
CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  target_id TEXT NOT NULL REFERENCES targets(target_id),
  signal_type TEXT NOT NULL,
  signal_value TEXT NOT NULL,
  signal_strength REAL NOT NULL,
  source_url TEXT,
  source_confidence REAL,
  evidence_quote TEXT,
  evidence_verified INTEGER,
  evidence_tier TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (target_id, run_id, signal_type, signal_value)
);

-- sources persist the raw evidence the research stage actually saw, so an
-- unverified signal can be triaged AFTER the run and a fact-checker can run
-- retroactively (ticket B2b — before this table both strings died with the
-- process).  Two kinds of rows share it: raw fetched pages
-- (source_type='company_website', written by fetch_sources on every
-- successful fetch) and the research agent's consolidated findings
-- (source_type='research_findings', written by ResearchBookkeepingNode);
-- detect_signals checks each evidence_quote against these texts, raw rows
-- FIRST.  The columns mirror app/tools/fetch_sources.py's NormalizedSource
-- dataclass one-for-one (plus source_id/run_id/target_id/created_at) so the
-- in-memory shape and the persisted shape cannot drift.  target_id
-- deliberately has NO REFERENCES targets(target_id): this table belongs to
-- the audit-trail family (steps, state_transitions), whose rows are written
-- at pipeline stages where the target row is real but FK enforcement is not
-- wanted — the DDL mirrors steps/state_transitions, not signals.
-- research_findings rows carry NULL source_url/source_confidence/
-- source_priority because agent prose has no single URL, no measured
-- confidence, and no normalization priority; extraction_method='agent'
-- marks its provenance.
CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_url TEXT,
  extracted_text TEXT NOT NULL,
  extracted_at TEXT NOT NULL,
  source_confidence REAL,
  source_priority INTEGER,
  extraction_method TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Messages are the actual emails sent or received. direction is constrained to
-- 'outbound' or 'inbound'. body holds the plain text; body_redacted is a
-- version with PII/private info stripped for logging. status tracks the email
-- lifecycle (draft, sent, bounced, etc.).
CREATE TABLE IF NOT EXISTS messages (
  message_id TEXT PRIMARY KEY,
  target_id TEXT NOT NULL REFERENCES targets(target_id),
  contact_id TEXT NOT NULL REFERENCES contacts(contact_id),
  direction TEXT NOT NULL CHECK (direction IN ('outbound', 'inbound')),
  provider_message_id TEXT,
  thread_id TEXT,
  subject TEXT,
  body TEXT,
  body_redacted TEXT,
  status TEXT NOT NULL,
  sent_at TEXT,
  created_at TEXT NOT NULL
);

-- Each draft version is a snapshot of what was proposed before human review.
-- revision_number increments per edit. policy_check_passed, injection_scan_passed,
-- and send_gate_passed are set by the review/send-gate pipeline and determine
-- whether the draft can become an outbound message.  edited_by records WHO
-- authored the revision: the agent id for agent-authored versions (ticket B3
-- writes draft_writer here), or the operator for human edits.  critique_passed
-- / critique_json (ticket B3) carry the critic's verdict on THIS revision —
-- the console (B4) reads them to show why the agent rewrote; both are
-- nullable because rows written before B3 have no critique.
-- insert_seq (ticket B5) is the monotonic insertion-order column that makes
-- "which revision/decision is the latest?" deterministic: created_at is
-- second-precision TEXT, so two rows written in the same second ordered
-- arbitrarily — an operational bug the send gate hit (see _MIGRATION_COLUMNS
-- and app/send_gate.py).  Nullable only as the migration accommodation for
-- rows written before B5; every writer populates it on new rows.
CREATE TABLE IF NOT EXISTS message_draft_versions (
  draft_version_id TEXT PRIMARY KEY,
  target_id TEXT NOT NULL REFERENCES targets(target_id),
  message_id TEXT REFERENCES messages(message_id),
  revision_number INTEGER NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL,
  footer TEXT NOT NULL,
  edited_by TEXT NOT NULL,
  policy_check_passed INTEGER,
  injection_scan_passed INTEGER,
  send_gate_passed INTEGER,
  critique_passed INTEGER,
  critique_json TEXT,
  insert_seq INTEGER,
  created_at TEXT NOT NULL
);

-- Replies are inbound messages parsed from the email provider, classified by
-- the reply classifier, and routed to the next action. classification is the
-- label (e.g. "positive", "negative", "unsubscribe", "out_of_office").
-- routed_action is what the state machine should do next.
CREATE TABLE IF NOT EXISTS replies (
  reply_id TEXT PRIMARY KEY,
  message_id TEXT NOT NULL REFERENCES messages(message_id),
  thread_id TEXT,
  from_email TEXT NOT NULL,
  raw_text TEXT NOT NULL,
  redacted_text TEXT NOT NULL,
  classification TEXT,
  confidence REAL,
  routed_action TEXT,
  insert_seq INTEGER,
  created_at TEXT NOT NULL
);

-- Meetings (demo, 2026-08-30): a REAL scheduling record, not a link to a
-- static page. When a positive reply queues a follow-up draft, the
-- scheduling agent (app/tools/schedule_meeting.py) proposes ONE slot from
-- an actual computed calendar (a fixed weekly template projected forward
-- from "now", filtered against every already-scheduled row here so two
-- targets can never collide on the same slot) and this table is where that
-- reservation actually lives — company_name/contact_name are denormalized
-- (not just joined through target_id) so a row is fully self-describing in
-- an audit read, matching the denormalization precedent replies.from_email
-- already sets. reasoning is the LLM's own stated justification for the
-- slot it picked (or "earliest available slot" when the LLM verdict failed
-- and deterministic code degraded to the fallback — see schedule_meeting.py
-- for the never-fail-the-target rule this mirrors from judge_icp). status
-- stays 'proposed' until a real human confirms it by some future channel —
-- nothing here ever claims a meeting is CONFIRMED, only that a slot was
-- reserved and offered.
CREATE TABLE IF NOT EXISTS meetings (
  meeting_id TEXT PRIMARY KEY,
  target_id TEXT NOT NULL REFERENCES targets(target_id),
  account_id TEXT NOT NULL REFERENCES accounts(account_id),
  contact_id TEXT REFERENCES contacts(contact_id),
  company_name TEXT NOT NULL,
  contact_name TEXT,
  scheduled_at TEXT NOT NULL,
  duration_minutes INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('proposed', 'cancelled')),
  reasoning TEXT,
  proposed_by TEXT NOT NULL,
  run_id TEXT,
  step_id TEXT,
  created_at TEXT NOT NULL
);

-- Steps are the audit trail: every tool invocation, its inputs, its outputs,
-- and whether it succeeded. This is the most important table for debugging —
-- every pipeline action (research, score, draft, send, classify) writes exactly
-- one row here. run_id ties steps together into a single pipeline execution.
-- Written by each tool via log_step() (Task 6).
CREATE TABLE IF NOT EXISTS steps (
  step_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  target_id TEXT,
  tool_name TEXT NOT NULL,
  input_json TEXT NOT NULL,
  output_json TEXT,
  model_call_hash TEXT,
  -- agent_id records WHICH registered agent performed this step (plan task A3).
  -- Nullable ONLY as a migration accommodation for steps rows written before
  -- agent attribution existed — log_step() always populates it on new rows.
  agent_id TEXT,
  status TEXT NOT NULL CHECK (status IN ('success','failed','retried')),
  created_at TEXT NOT NULL
);

-- Suppressions block all future outreach to an email address (and optionally
-- its domain). reason is constrained to the legal/operational categories the
-- operator can document. checked_by send_gate_decisions on every send attempt.
-- Written by the reply classifier or operator manual action
-- (scripts/add_suppression.py), or the review/reply gates.
-- `email` is the address AS WRITTEN (the audit record of what arrived) —
-- NOT the primary key since ticket H4b (a domain-only row has no address);
-- `email_normalized` is the canonical matching key (ticket F1b), computed by
-- app.db.normalize_email() so every read/write folds local-part case,
-- domain case, and plus-tags identically.  `domain` is stored lowercased
-- (app.db.normalize_domain()); domains are case-insensitive per RFC 1035.
"""
    + _SUPPRESSIONS_DDL
    + """
-- agent_registry is the write gate's per-agent capability table (plan task A3).
-- Every write_gate.commit() carries an agent_id; the gate refuses writes from
-- agents that are not registered here, are disabled (enabled=0), or whose
-- allowed_actions JSON does not contain the attempted action — turning the
-- global KNOWN_ACTIONS allowlist into a per-agent capability set.
-- allowed_transitions is STORED BUT NOT YET ENFORCED — enforcement lands in a
-- later task; do not assume it is live. Populated by
-- app/agents_registry.seed_agent_registry() (only `system` and `operator`
-- exist today); later agent tasks (A4, B1-B3, C1, C4) append their own rows.
CREATE TABLE IF NOT EXISTS agent_registry (
  agent_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  description TEXT NOT NULL,
  model_alias TEXT,
  allowed_actions TEXT NOT NULL,
  allowed_transitions TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);

-- write_log is the append-only WAL of every state-changing write. Each row
-- records who wrote what to which table, with a copy of the payload. Used by
-- write_gate.py (Task 4) to enforce that every db mutation is logged and
-- traceable. Never updated — only INSERT.
CREATE TABLE IF NOT EXISTS write_log (
  write_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  step_id TEXT NOT NULL,
  action TEXT NOT NULL,
  table_name TEXT NOT NULL,
  record_id TEXT NOT NULL,
  actor TEXT NOT NULL,
  -- agent_id records WHICH registered agent performed the write (plan task A3).
  -- Nullable ONLY as a migration accommodation for rows written before agent
  -- attribution existed — write_gate.commit() always populates it on new rows.
  agent_id TEXT,
  matched_policy_id TEXT,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Policy decisions record the outcome of every policy check run against a
-- target. action is what was being evaluated (e.g. "send_email", "classify").
-- decision is allow/deny/flag_for_review. Written by the policy engine
-- (Task TBD: policy_check tool) and referenced by send_gate_decisions.
-- insert_seq (ticket B5): the monotonic ordering column — see the
-- message_draft_versions comment for the second-precision created_at bug it
-- fixes.  Nullable only as the migration accommodation for pre-B5 rows.
CREATE TABLE IF NOT EXISTS policy_decisions (
  policy_decision_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  step_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  action TEXT NOT NULL,
  decision TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  reasons_json TEXT NOT NULL,
  matched_rules_json TEXT NOT NULL,
  missing_fields_json TEXT NOT NULL,
  insert_seq INTEGER,
  created_at TEXT NOT NULL
);

-- state_transitions is the state machine's audit log. Every time a target
-- changes state, one row is INSERTed here recording previous_state, new_state,
-- why the transition happened, and who (actor) triggered it. Used for
-- debugging "how did this target get to 'sent'?" questions. Written by the
-- state machine engine (Task 5).
-- insert_seq (ticket C1, extending B5's fix): the monotonic insertion-order
-- column — see the message_draft_versions comment for the second-precision
-- created_at bug it fixes.  C1 extended it to THIS table because
-- state_transitions is the state machine's own audit log: two hops landing
-- in the same second (e.g. replied → routed → suppressed inside one
-- classify_and_route_reply call) ordered arbitrarily under ORDER BY
-- created_at, so the trail could not answer "what happened, in what order"
-- — the property this whole project is pitched on.  B5 left this table out
-- of its fix; C1 closes the gap.  Nullable only as the migration
-- accommodation for rows written before C1; every writer populates it on
-- new rows.
CREATE TABLE IF NOT EXISTS state_transitions (
  transition_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  step_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  previous_state TEXT NOT NULL,
  new_state TEXT NOT NULL,
  reason TEXT NOT NULL,
  actor TEXT NOT NULL,
  matched_policy_id TEXT,
  insert_seq INTEGER,
  created_at TEXT NOT NULL
);

-- send_gate_decisions is the final pre-send safety check. Before any email goes
-- out, the send gate checks: suppression lists, policy decisions, kill switch,
-- and human approval. Exactly one row is written per send attempt. allowed=0
-- means the send was blocked. Written by the send gate (Task TBD).
-- matched_rules_json (ticket H6): the policy-matrix rule IDs (P1-P9) behind
-- each refusal, recorded ALONGSIDE the prose reasons_json — the same shape and
-- semantics as policy_decisions.matched_rules_json, so a query can answer
-- "every send P2 refused" without parsing prose.  Empty ([]) on allow.
-- NULL semantics: NULL means "this decision predates rule attribution"
-- (legacy pre-H6 rows — never backfilled), deliberately distinguishable from
-- [] which means "evaluated, no rule matched" (an allow).
CREATE TABLE IF NOT EXISTS send_gate_decisions (
  send_gate_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  step_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  contact_id TEXT NOT NULL,
  allowed INTEGER NOT NULL,
  reasons_json TEXT NOT NULL,
  missing_requirements_json TEXT NOT NULL,
  matched_rules_json TEXT NOT NULL,
  suppression_hit INTEGER NOT NULL,
  approval_verified INTEGER NOT NULL,
  kill_switch_active INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

-- review_decisions record the human operator's verdict on a draft message
-- (ticket B4b — the console approval gate, written ONLY by app/review.py).
-- decision is one of the five review actions (docs/human-review.md §3):
-- approve / approve_with_edits / reject / reject_and_suppress / escalate.
-- approve_with_edits sets edited=1 and appends a NEW message_draft_versions
-- revision (never an in-place overwrite — docs/human-review.md §5).
-- draft_message_id CONTRACT (ticket B4b): the column name implies a
-- messages row, but no messages row exists until B5 sends — it actually
-- holds the draft_version_id of the revision the decision is about (see
-- docs/db-schema.md §review_decisions).  reason carries the operator's
-- reasoning (for escalate: the operator's research note).
-- kill_switch_active (ticket B4b) records the switch state AT DECISION
-- TIME, read by app/review.py from read_kill_switch().engaged — so the
-- audit trail can answer "was the switch engaged when the operator made
-- this call?" without re-reading the (mutable) file later.  Nullable only
-- as the established migration accommodation for rows written before B4b;
-- app/review.py always populates it on new rows.
-- insert_seq (ticket B5): the monotonic ordering column — THE fix for the
-- send gate's same-second created_at tie (see the message_draft_versions
-- comment).  An operator who approves and then approves-with-edits inside
-- the same second must have the gate resolve the LATEST decision; created_at
-- alone cannot guarantee that, insert_seq can.  Nullable only as the
-- migration accommodation for pre-B5 rows; app/review.py always populates
-- it on new rows.
CREATE TABLE IF NOT EXISTS review_decisions (
  review_decision_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  draft_message_id TEXT NOT NULL,
  decision TEXT NOT NULL,
  edited INTEGER NOT NULL DEFAULT 0,
  reason TEXT,
  actor TEXT NOT NULL,
  kill_switch_active INTEGER,
  insert_seq INTEGER,
  created_at TEXT NOT NULL
);

-- signal_outcome_link closes the feedback loop: for each signal that
-- contributed to a send decision, this records what the actual outcome was
-- (reply, bounce, no response). Used later to tune signal_weights.
-- Written by the reply classifier when it processes an inbound reply.
CREATE TABLE IF NOT EXISTS signal_outcome_link (
  link_id TEXT PRIMARY KEY,
  target_id TEXT NOT NULL,
  message_id TEXT,
  signal_type TEXT NOT NULL,
  signal_strength REAL NOT NULL,
  signal_weight_at_send REAL NOT NULL,
  outcome_type TEXT NOT NULL,
  outcome_value REAL,
  pipeline_stage TEXT,
  recorded_at TEXT NOT NULL
);

-- signal_weights are the dynamic scoring weights that the lead scorer uses.
-- Each signal_type gets a current_weight, clamped between min_weight and
-- max_weight. These can be tuned over time based on signal_outcome_link data.
-- Initial weights are set by the operator or a seed script.
CREATE TABLE IF NOT EXISTS signal_weights (
  signal_type TEXT PRIMARY KEY,
  current_weight REAL NOT NULL,
  min_weight REAL NOT NULL,
  max_weight REAL NOT NULL,
  updated_at TEXT NOT NULL
);

-- candidate_fields are enrichment data points discovered by research tools
-- before being accepted into the master entity tables. verification_status
-- tracks whether the field has been human-verified, auto-accepted, or is
-- contradictory (two providers disagree). Written by enrichment tools.
CREATE TABLE IF NOT EXISTS candidate_fields (
  candidate_id TEXT PRIMARY KEY,
  target_id TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_key TEXT NOT NULL,
  field_name TEXT NOT NULL,
  field_value TEXT,
  confidence REAL NOT NULL,
  source_provider TEXT NOT NULL,
  source_type TEXT NOT NULL,
  verification_status TEXT NOT NULL,
  contradiction_flag INTEGER NOT NULL DEFAULT 0,
  observed_at TEXT NOT NULL,
  accepted INTEGER,
  accepted_at TEXT
);

-- enrichment_runs track each research/enrichment execution. provider_chain_json
-- records which tools were used in what order. completion_score measures how
-- complete the enrichment was (did we fill all fields we wanted?).
-- contradictions_found counts candidate_fields rows that conflict.
CREATE TABLE IF NOT EXISTS enrichment_runs (
  enrichment_run_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_key TEXT NOT NULL,
  provider_chain_json TEXT NOT NULL,
  completion_score REAL NOT NULL,
  contradictions_found INTEGER NOT NULL,
  needs_manual_review INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
""")


# ── SQL translation (sqlite dialect → postgres dialect) ─────────────────────
# Every query in this repo is written with SQLite's ``?`` placeholders and
# SQLite's ``datetime('now')`` timestamp function. The postgres side of the
# Conn wrapper runs each statement through _translate_sql() before sending it,
# which rewrites both constructs WITHOUT touching quoted regions — a literal
# ``?`` inside a string (or a datetime('now') that is itself data) must survive
# byte-for-byte. A blind str.replace() would corrupt such queries, so this is a
# small character scanner instead.

# Matches the start of a Postgres dollar-quoted string: $$...$$ or $tag$...$tag$.
# The tag may be empty or an identifier ([A-Za-z_][A-Za-z0-9_]*). Dollar quotes
# are the one Postgres quoting form sqlite never had, but the scanner honors
# them anyway so postgres-authored SQL passes through untranslated.
_DOLLAR_QUOTE_START = re.compile(r"\$\$|\$[A-Za-z_][A-Za-z0-9_]*\$")

# Matches SQLite's datetime('now') call — the exact spelling used by every
# write in this repo (write_gate, log_step, state_machine, get_targets, ...).
# Postgres has no datetime() function at all, so it must be rewritten.
_DATETIME_NOW = re.compile(r"datetime\s*\(\s*'now'\s*\)", re.IGNORECASE)

# The postgres spelling that produces a byte-identical string to SQLite's
# datetime('now'): 'YYYY-MM-DD HH:MM:SS', UTC, second precision, no offset.
#
# CURRENT_TIMESTAMP is the obvious translation and is what this started as, but
# it is WRONG for this schema. Every timestamp column here is TEXT, and
# Postgres assignment-casts CURRENT_TIMESTAMP (a timestamptz) into TEXT as
# '2026-08-19 07:39:38.620064+00' — microseconds and a UTC offset that SQLite
# never writes. That silently gives dev and prod two different formats in the
# same logical column. Nothing parses these timestamps today, but the cooldown
# windows and send rate limits in docs/state-machine.md §7e do arithmetic on
# them, so the divergence would surface later as a wrong-by-a-timezone bug in
# the send gate rather than as an obvious crash here.
_PG_DATETIME_NOW = "to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')"


def _translate_sql(sql: str) -> str:
    """Rewrite a SQLite-dialect statement into a postgres-dialect one.

    Two rewrites, both applied ONLY outside quoted/comment regions:
    - ``?`` placeholder → ``%s`` (pg8000's paramstyle is "format")
    - ``datetime('now')`` → ``CURRENT_TIMESTAMP``

    Quoted regions skipped: 'single-quoted strings' (with '' escapes),
    "double-quoted identifiers" (with "" escapes), $dollar$ $quoted$ strings,
    -- line comments, and /* block comments */ (nested, as Postgres allows).
    Anything inside them is copied verbatim — a question mark inside a string
    literal is data, not a placeholder.
    """
    out: list[str] = []  # Translated pieces are collected here and joined once at the end.
    i, n = 0, len(sql)  # i is the scan position; n the statement length.
    while i < n:
        ch = sql[i]

        # 'single-quoted string' — copy verbatim to the closing quote, honoring
        # the SQL convention that a doubled quote ('') is an escaped quote, not
        # the end of the string.
        if ch == "'":
            out.append(ch)
            i += 1
            while i < n:
                out.append(sql[i])
                if sql[i] == "'":
                    # If the quote is doubled, it's an escaped quote INSIDE the
                    # string — consume both and keep scanning.
                    if i + 1 < n and sql[i + 1] == "'":
                        out.append(sql[i + 1])
                        i += 2
                        continue
                    # Otherwise it closes the string — resume normal scanning.
                    i += 1
                    break
                i += 1
            continue

        # "double-quoted identifier" — same logic as single quotes; a doubled
        # double-quote is an escaped quote inside the identifier.
        if ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                out.append(sql[i])
                if sql[i] == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        out.append(sql[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue

        # $dollar-quoted$ string — find the matching closing tag and copy the
        # whole span verbatim. If the tag never closes, treat the $ as an
        # ordinary character (the server will report the real syntax error).
        if ch == "$":
            tag_match = _DOLLAR_QUOTE_START.match(sql, i)
            if tag_match:
                tag = tag_match.group(0)
                end = sql.find(tag, tag_match.end())
                if end != -1:
                    out.append(sql[i : end + len(tag)])
                    i = end + len(tag)
                    continue
            out.append(ch)
            i += 1
            continue

        # -- line comment — copy verbatim up to (not including) the newline;
        # the newline itself is handled by the normal path on the next loop.
        if sql.startswith("--", i):
            eol = sql.find("\n", i)
            if eol == -1:  # Comment runs to end of statement — done.
                out.append(sql[i:])
                break
            out.append(sql[i:eol])
            i = eol
            continue

        # /* block comment */ — Postgres nests these, so count depth rather
        # than stopping at the first */. Copy the whole comment verbatim.
        if sql.startswith("/*", i):
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if sql.startswith("/*", j):
                    depth += 1
                    j += 2
                elif sql.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            out.append(sql[i:j])
            i = j
            continue

        # datetime('now') — a function call in code position, so it is only
        # rewritten here in the normal (unquoted) scan state. Rewritten to the
        # to_char(...) form rather than CURRENT_TIMESTAMP so the stored string
        # is byte-identical to SQLite's — see _PG_DATETIME_NOW for why that
        # matters more than it looks.
        dt_match = _DATETIME_NOW.match(sql, i)
        if dt_match:
            out.append(_PG_DATETIME_NOW)
            i = dt_match.end()
            continue

        # ? positional placeholder → %s, pg8000's native placeholder spelling.
        if ch == "?":
            out.append("%s")
            i += 1
            continue

        # Anything else is ordinary SQL text — copy it through unchanged.
        out.append(ch)
        i += 1
    return "".join(out)


def _tx_keyword(sql: str) -> str | None:
    """Return the transaction-control keyword of a statement, or None.

    Used by the postgres path of Conn.execute() to spot the one SQLite-only
    transaction spelling (BEGIN IMMEDIATE) and downgrade it to plain BEGIN.
    The comparison is case-insensitive and tolerates a trailing semicolon,
    because both spellings appear in real code.
    """
    stripped = sql.strip().rstrip(";").strip().lower()
    if stripped in ("begin", "begin transaction", "begin work", "begin immediate"):
        return "begin"
    if stripped in ("commit", "commit transaction", "end"):
        return "commit"
    if stripped in ("rollback", "rollback transaction"):
        return "rollback"
    return None


def _re_raise_pg_integrity(exc: Exception) -> None:
    """Re-raise a pg8000 error as pg8000_dbapi.IntegrityError when it is one.

    pg8000 raises a raw pg8000.exceptions.DatabaseError for EVERY server error
    and never maps it to its DB-API exception subclasses itself. Server errors
    carry the structured fields dict as args[0], including the SQLSTATE code
    under key "C". Class 23 is integrity_constraint_violation — the class that
    contains unique_violation (23505), foreign_key_violation (23503),
    not_null_violation (23502) and check_violation (23514). When the SQLSTATE
    says class 23, re-raise as pg8000_dbapi.IntegrityError (preserving the
    structured args) so callers can catch app.db.IntegrityError — one tuple,
    both dialects. Anything else returns without raising and the original
    exception propagates untouched.
    """
    if (
        isinstance(exc, pg8000_exceptions.DatabaseError)
        and exc.args
        and isinstance(exc.args[0], dict)
    ):
        if str(exc.args[0].get("C", "")).startswith("23"):
            raise pg8000_dbapi.IntegrityError(*exc.args) from exc


def _split_statements(script: str) -> list[str]:
    """Split a DDL script into individual statements on ';' boundaries.

    Quote-aware ('', "", $$) AND comment-aware (-- line, /* block */) so a
    semicolon inside a string, identifier or comment never splits a statement
    — the DDL's own table comments contain a real one ("...plain text;
    body_redacted is a..."). Only needed by the postgres path — sqlite has
    native executescript() — because pg8000 executes one statement per call.
    """
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(script)
    while i < n:
        ch = script[i]
        # Quoted regions — copy verbatim to the matching close, exactly like
        # _translate_sql, so a ';' inside them is data, not a separator.
        if ch in ("'", '"'):
            buf.append(ch)
            i += 1
            while i < n:
                buf.append(script[i])
                if script[i] == ch:
                    if i + 1 < n and script[i + 1] == ch:
                        buf.append(script[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        # Dollar-quoted strings — copy verbatim to the matching closing tag.
        if ch == "$":
            tag_match = _DOLLAR_QUOTE_START.match(script, i)
            if tag_match:
                tag = tag_match.group(0)
                end = script.find(tag, tag_match.end())
                if end != -1:
                    buf.append(script[i : end + len(tag)])
                    i = end + len(tag)
                    continue
        # -- line comment — copy verbatim to the newline so a ';' inside the
        # comment text can't terminate a statement.
        if script.startswith("--", i):
            eol = script.find("\n", i)
            if eol == -1:  # Comment runs to end of script.
                buf.append(script[i:])
                break
            buf.append(script[i:eol])
            i = eol
            continue
        # /* block comment */ — copy verbatim (nesting counted, same as
        # _translate_sql); a ';' inside it can't terminate a statement either.
        if script.startswith("/*", i):
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if script.startswith("/*", j):
                    depth += 1
                    j += 2
                elif script.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            buf.append(script[i:j])
            i = j
            continue
        # ';' outside quotes and comments terminates a statement — flush the
        # buffer.
        if ch == ";":
            statement = "".join(buf).strip()
            if statement:  # Skip empty fragments (e.g. trailing ";;").
                statements.append(statement)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    # Flush any trailing statement that has no closing semicolon.
    statement = "".join(buf).strip()
    if statement:
        statements.append(statement)
    return statements


# ── Rows and cursors ────────────────────────────────────────────────────────


class _PGMappingRow:
    """A dict-like row over pg8000's tuple rows, mirroring sqlite3.Row.

    pg8000 returns plain tuples; sqlite3.Row supports row["column_name"] and
    row[0]. Every query in this repo uses row["column_name"], so the postgres
    path wraps each fetched tuple in this class to give the same access. The
    name→value index is built once per row (rows here are ~10 columns, so the
    dict build is trivial).
    """

    __slots__ = ("_values", "_by_name")  # Slots keep per-row overhead minimal.

    def __init__(self, names: list[str], values: tuple):
        self._values = values  # The raw tuple, for integer indexing.
        # zip collapses duplicate column names to the last occurrence — the
        # same caveat sqlite3.Row has for name lookups on duplicate columns.
        self._by_name = dict(zip(names, values))

    def __getitem__(self, key):
        # int → positional access (row[0]); str → name access (row["column"]).
        # A missing name raises KeyError, mirroring dict (sqlite3.Row raises
        # IndexError there — close enough; no caller relies on either).
        if isinstance(key, int):
            return self._values[key]
        return self._by_name[key]

    def keys(self):
        return self._by_name.keys()

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __repr__(self):
        return f"_PGMappingRow({self._by_name!r})"


class _PGCursor:
    """Cursor wrapper that adds dict-like rows over a pg8000 cursor.

    Conn.execute() returns one of these for every postgres statement so the
    caller gets .fetchone()/.fetchall()/.rowcount/iteration exactly like the
    sqlite cursor it is used to. Column names are captured from the cursor's
    description at construction time (i.e. right after execute) — for
    non-row statements (INSERT/UPDATE) there is no description and the fetch
    methods return None/[] to match sqlite's behavior.
    """

    def __init__(self, native):
        self._native = native  # The underlying pg8000 dbapi cursor.
        description = native.description  # None when the statement returns no rows.
        self._names = [col[0] for col in description] if description else None

    def fetchone(self):
        # A non-row statement has no description — return None without asking
        # the driver, which would raise "attempting to use unexecuted cursor".
        if self._names is None:
            return None
        row = self._native.fetchone()
        if row is None:
            return None
        return _PGMappingRow(self._names, row)

    def fetchall(self):
        # Non-row statement — sqlite returns [] here, so the postgres path
        # matches rather than asking the driver for rows that don't exist.
        if self._names is None:
            return []
        return [_PGMappingRow(self._names, row) for row in self._native.fetchall()]

    @property
    def rowcount(self):
        # pg8000 reports the affected-row count for INSERT/UPDATE/DELETE
        # (and SELECTs once the result set is exhausted) — same shape as
        # sqlite's cursor.rowcount.
        return self._native.rowcount

    def __iter__(self):
        # Iteration yields mapping rows, one per result tuple, so
        # ``for row in conn.execute(...)`` works identically on both dialects.
        for row in self._native:
            yield _PGMappingRow(self._names, row)


# ── Conn — the dialect-agnostic connection ──────────────────────────────────


class Conn:
    """Dialect-agnostic database connection — a drop-in for sqlite3.Connection.

    Every module in this repo takes ``conn`` as its first argument and calls
    conn.execute(...)/commit()/rollback()/close() on it. Conn preserves that
    exact call shape for BOTH dialects:

    - sqlite: a thin pass-through over the raw sqlite3.Connection (WAL mode,
      Row factory) — byte-for-byte the behavior that existed before this task.
    - postgres: translates the SQL (``?``→``%s``, datetime('now')→
      CURRENT_TIMESTAMP), maps BEGIN IMMEDIATE to plain BEGIN, wraps tuple
      rows as dict-like, and normalizes integrity errors.

    The wrapper is deliberately dumb about transactions: explicit BEGIN /
    COMMIT / ROLLBACK statements flow through as-is (they are valid SQL on
    both engines), so the transaction discipline stays in the caller — exactly
    where write_gate.py puts it today.
    """

    def __init__(self, native, dialect: str, connector=None):
        self._native = native  # sqlite3.Connection, or a pg8000 dbapi Connection.
        self._dialect = dialect  # "sqlite" or "postgres" — set once, never mutated.
        # The Cloud SQL Python Connector instance, present only for
        # cloudsql:// targets. Kept so close() can shut down its background
        # refresh thread — without that, every cloudsql connection leaks a
        # thread until process exit.
        self._connector = connector

    @property
    def dialect(self) -> str:
        # Exposed so callers that must branch (begin_write is the one in-repo
        # example) can check the engine explicitly instead of guessing.
        return self._dialect

    def execute(self, sql: str, params=()):
        """Execute one statement; returns a cursor with fetch/iterate methods.

        The returned cursor supports .fetchone(), .fetchall(), .rowcount and
        iteration, and rows are dict-like (row["column_name"]) on BOTH
        dialects. Statements are executed exactly as the caller wrote them on
        sqlite; on postgres they pass through _translate_sql() first.
        """
        if self._dialect == "sqlite":
            # Pass-through — the native cursor already returns sqlite3.Row
            # (dict-like) and supports every cursor method callers use.
            return self._native.execute(sql, params)

        # postgres path: rewrite sqlite dialect constructs first.
        sql = _translate_sql(sql)
        # BEGIN IMMEDIATE is SQLite-only syntax — Postgres rejects it. Map the
        # full statement to plain BEGIN (see begin_write() for why plain BEGIN
        # is safe on Postgres). The mapping only fires for the exact statement
        # keyword, never for a larger statement containing the words.
        if _tx_keyword(sql) == "begin" and "immediate" in sql.strip().lower():
            sql = "BEGIN"
        try:
            # A fresh cursor per execute() mirrors sqlite3.Connection.execute,
            # which also returns a new cursor object each call.
            cursor = self._native.cursor()
            cursor.execute(sql, params)
        except Exception as exc:
            # Normalize class-23 SQLSTATEs to pg8000_dbapi.IntegrityError so
            # callers can catch app.db.IntegrityError for both dialects.
            _re_raise_pg_integrity(exc)
            raise
        return _PGCursor(cursor)

    def executemany(self, sql: str, seq) -> _PGCursor:
        """Execute one parameterized statement once per parameter set."""
        if self._dialect == "sqlite":
            return self._native.executemany(sql, seq)
        # postgres: same translation as execute(); pg8000's dbapi cursor
        # implements executemany by looping execute() per parameter set.
        sql = _translate_sql(sql)
        cursor = self._native.cursor()
        try:
            cursor.executemany(sql, seq)
        except Exception as exc:
            _re_raise_pg_integrity(exc)
            raise
        return _PGCursor(cursor)

    def executescript(self, script: str) -> None:
        """Run a multi-statement DDL script (apply_schema's entry point).

        sqlite has native executescript(); pg8000 executes one statement per
        call, so the postgres path splits the script on ';' (quote-aware) and
        executes each statement through the normal translated execute().
        """
        if self._dialect == "sqlite":
            self._native.executescript(script)
            return
        for statement in _split_statements(script):
            self.execute(statement)

    def begin_write(self):
        """Open a write transaction using this dialect's correct spelling.

        sqlite: BEGIN IMMEDIATE — takes the write lock up front, preventing
        the lock-upgrade race documented at the top of this module (and proven
        by test_write_transaction_uses_begin_immediate_semantics).

        postgres: plain BEGIN. Postgres HAS no BEGIN IMMEDIATE, and needs none:
        the race BEGIN IMMEDIATE prevents is an artifact of SQLite's
        single-writer, lock-upgrade-on-first-write model. Postgres uses
        row-level locking + MVCC instead — writers don't take a global write
        lock at all, and a conflicting concurrent write blocks or fails on the
        specific rows involved, with the server (not the client) arbitrating.
        These two are therefore NOT identical semantics — SQLite serializes
        writers up front; Postgres lets them proceed until a real row
        conflict. This is a deliberate, documented difference, not an
        oversight: on each engine begin_write() means "start a write
        transaction with that engine's correct locking behavior."
        """
        if self._dialect == "sqlite":
            return self.execute("BEGIN IMMEDIATE")
        return self.execute("BEGIN")

    def commit(self) -> None:
        # sqlite (isolation_level=None): CPython's commit() only issues COMMIT
        # when a transaction is actually open — a no-op in autocommit mode,
        # matching the pre-wrapper behavior of log_step's commit-after-insert.
        # postgres (autocommit=True): _in_transaction is the driver-tracked
        # server transaction status — skip the round-trip when idle so
        # log_step's commit() doesn't send a pointless (warning-producing)
        # COMMIT after every already-autocommitted insert.
        if self._dialect == "postgres":
            if getattr(self._native, "_in_transaction", True):
                self._native.commit()
            return
        self._native.commit()

    def rollback(self) -> None:
        # pg8000's own rollback() already no-ops when the server reports no
        # open transaction (it checks the same driver-tracked status), which
        # mirrors SQLite's behavior of ROLLBACK outside a transaction being
        # a harmless no-op.
        self._native.rollback()

    def close(self) -> None:
        # Close the underlying connection first, then the Cloud SQL connector
        # (if any) so its background refresh thread and sockets are released —
        # connections here are short-lived by design (open, do work, close).
        self._native.close()
        if self._connector is not None:
            self._connector.close()


# ── connect() — dialect detection ───────────────────────────────────────────


def connect(target: str) -> Conn:
    """Open a connection to the database described by ``target``.

    Dialect is decided by the argument, never by guessing:

    - ``postgresql://...`` (or ``postgres://...``) → postgres, direct TCP
      connection to the URL's host/port (works for a Cloud SQL public IP).
    - ``cloudsql://<instance-connection-name>/<database>`` → postgres via
      Google's Cloud SQL Python Connector, which authenticates with
      Application Default Credentials (local ADC or the Cloud Run service
      account) — no IP allowlisting, no password in the container. The DB
      user/password come from the OUTBOUND_DB_USER (default "outbound_app")
      and OUTBOUND_DB_PASSWORD env vars.
    - anything else (e.g. ``data/outbound.db``, ``:memory:``) → sqlite,
      behaving exactly as before this task on a writable file: WAL,
      busy_timeout, foreign_keys ON, isolation_level=None, Row factory. On a
      read-only database the WAL pragma is skipped (it needs to write the
      file header) — see the sqlite branch below for why that is safe.

    A misconfigured DSN raises immediately with a message naming what was
    wrong — there is NO silent fallback to SQLite.
    """
    if target.startswith("cloudsql://"):
        return _connect_cloudsql(target)
    if target.startswith("postgresql://") or target.startswith("postgres://"):
        return _connect_postgres_url(target)
    # ── sqlite — the original path, plus the read-only fallback ───────────────
    conn = sqlite3.connect(target, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        # WAL mode is persisted in the database file's header, so turning it
        # ON is a file write: on a read-only database (file and/or directory
        # chmod'd read-only, or a read-only container mount) this raises
        # OperationalError("attempt to write a readonly database") and would
        # fail the whole connect() before any query could run.
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        # A read-only database is a legitimate configuration, not a
        # misconfiguration: the read-only operator console (plan task A5b)
        # mounts the operator's database :ro, and any read-only inspection
        # tooling wants the same. So stay in the file's existing journal mode
        # and continue. What is given up is concurrency only — WAL lets
        # readers avoid blocking the single writer — never correctness: a
        # read-only connection has no writers to be concurrent with, and all
        # reads still see committed data in any journal mode.
        # Only this pragma is guarded, deliberately: busy_timeout and
        # foreign_keys are per-connection runtime settings that never touch
        # the file (verified against a read-only database), so they must
        # still be applied below. A genuinely corrupt file is NOT hidden by
        # this: its error still surfaces as an OperationalError on the first
        # real statement executed against the file, and a bare Exception is
        # deliberately not caught — unrelated failures must still propagate.
        pass
    # Same two pragmas as before, same order, same values, unconditional: on
    # a writable database this branch is byte-identical to the pre-fallback
    # behaviour, and on a read-only one both statements succeed (no file
    # write) so the connection still gets its runtime settings.
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return Conn(conn, "sqlite", None)


def scratch_target_violation(target: str) -> str | None:
    """Return a refusal message when ``target`` must NOT be destructively
    reset, else None when a reset is safe.

    This is the dialect-aware counterpart to demo_seed._guard_violation, which
    answers "is this the operator's REAL database?" but cannot answer "is this
    a SCRATCH database I may wipe?" for a URL.  A URL cannot be path-resolved,
    so _guard_violation passes every URL through untouched (it says so at
    app/demo_seed.py line ~154); without the marker check here, pointing the
    sim at the operator's real Cloud SQL instance would drop the production
    schema.  Fail closed: an unrecognised database name is refused, never
    assumed to be scratch.
    """
    # URL-shaped targets are the dangerous case — a DROP SCHEMA on a URL reaches
    # whatever database that URL names.  Branch first on the three URL schemes
    # connect() accepts, and extract the database name the SAME way the connect
    # helpers do so this predicate and connect() can never disagree.
    if target.startswith(("postgresql://", "postgres://")):
        # Same extraction as _connect_postgres_url: the URL path after the
        # leading slash is the database name.
        database = urlparse(target).path.lstrip("/")
    elif target.startswith("cloudsql://"):
        # Same split as _connect_cloudsql: cloudsql://<instance>/<database>,
        # so the database is the part after the FIRST '/' following the scheme.
        database = target[len("cloudsql://") :].partition("/")[2]
    else:
        # File-shaped target (data/demo.db, :memory:, any non-URL): safe to
        # destructively reset.  File targets already have their own guard —
        # app/demo_seed.py::_guard_violation refuses data/outbound.db and is
        # applied by every caller — so it is NOT duplicated here.
        return None
    # A URL with no database name names nothing to reset — refusing here keeps
    # a malformed DSN from ever reaching connect() with a destructive intent.
    if not database:
        return (
            f"refusing to reset URL target {target!r}: it has no database name. "
            f"Point the sim at a scratch/test-named database instead."
        )
    # The marker requirement, case-insensitive on purpose: database names are
    # case-folded on the server, so 'OUTBOUND_SCRATCH' is just as scratch as
    # 'outbound_scratch' and must not be refused (or allowed) by accident.
    if "scratch" not in database.lower() and "test" not in database.lower():
        return (
            f"refusing to reset URL target {target!r}: its database name "
            f"{database!r} does not contain 'scratch' or 'test' "
            f"(case-insensitive). Point the sim at a scratch/test-named "
            f"database instead."
        )
    return None


def reset_scratch_database(target: str) -> None:
    """Reset ``target`` to an empty database.

    The single wipe path the adversarial harness goes through before every
    attack, so a URL-shaped target is emptied with Postgres DDL instead of the
    SQLite-only ``Path.unlink`` that silently no-ops on a URL.
    """
    # The refusal must be provable to happen with no I/O: run the pure
    # predicate BEFORE any connection is opened, and raise before connect().
    violation = scratch_target_violation(target)
    if violation is not None:
        raise ValueError(violation)
    # URL target: empty the whole `public` schema and recreate it.  We
    # deliberately reset `public` rather than use a per-test schema with a
    # search_path: app/db.py's migration check hardcodes table_schema='public',
    # so a non-public schema would break the migration path.  Resetting
    # `public` keeps every existing assumption true.
    if target.startswith(("postgresql://", "postgres://", "cloudsql://")):
        conn = connect(target)
        try:
            # DROP ... IF EXISTS tolerates a database that has no public schema
            # yet (or a previous reset already dropped it); CASCADE removes the
            # schema's tables in dependency order so the drop cannot fail on
            # foreign keys.
            conn.execute("DROP SCHEMA IF EXISTS public CASCADE;")
            conn.execute("CREATE SCHEMA public;")
        finally:
            conn.close()
        return
    # File-shaped target: exactly today's behaviour, unchanged — unlink the
    # file so the next connect() recreates it fresh.  missing_ok=True keeps
    # this a no-op for :memory: and for a not-yet-created scratch file.
    Path(target).unlink(missing_ok=True)


def _connect_cloudsql(target: str) -> Conn:
    """Connect to Cloud SQL through the Python Connector (ADC auth)."""
    # cloudsql://<instance-connection-name>/<database> — split on the FIRST
    # slash; the instance name contains no slashes and the database name none.
    rest = target[len("cloudsql://") :]
    instance, sep, database = rest.partition("/")
    if not instance or not sep or not database:
        raise ValueError(
            f"cloudsql:// target {target!r} is malformed — expected "
            "cloudsql://<instance-connection-name>/<database>"
        )
    # The DB user defaults to the app user created in docs/gcp-setup.md step 6;
    # allow an override via env for operator flexibility.
    user = os.environ.get("OUTBOUND_DB_USER", "outbound_app")
    # The password must come from the environment (populated from Secret
    # Manager). Refuse loudly rather than half-connecting: a missing password
    # is a configuration error, and silently falling back to SQLite would
    # strand data in two places without anyone noticing.
    password = os.environ.get("OUTBOUND_DB_PASSWORD")
    if password is None:
        raise ValueError(
            "cloudsql:// connection requires the OUTBOUND_DB_PASSWORD env var "
            "(set it from `gcloud secrets versions access latest "
            "--secret=outbound-db-password`); refusing to fall back to SQLite"
        )
    # Imported lazily on purpose: only the Cloud SQL path needs the google
    # auth stack, and the sqlite path (every test, the CLI) must not pay its
    # import cost or require it to be importable.
    from google.cloud.sql.connector import Connector

    connector = Connector()  # Owns the refresh thread + socket for this connection.
    try:
        native = connector.connect(
            instance,
            "pg8000",  # Driver selection is explicit — never inferred.
            user=user,
            password=password,
            database=database,
        )
    except Exception:
        # If the connector half-initialized before failing, release it so we
        # don't leak its thread; the original error still propagates.
        connector.close()
        raise
    # Autocommit=True mirrors the sqlite path's isolation_level=None: the
    # driver never opens implicit transactions, and explicit BEGIN/COMMIT
    # statements (write_gate's discipline) are the only transactions.
    native.autocommit = True
    return Conn(native, "postgres", connector)


def _connect_postgres_url(target: str) -> Conn:
    """Connect directly to a postgresql:// URL (public-IP fallback path)."""
    parsed = urlparse(target)
    # Collect every missing piece BEFORE connecting so one error message
    # names them all, rather than failing on the first one and sending the
    # operator on a fix-one-at-a-time loop.
    missing = []
    database = parsed.path.lstrip("/")
    if not database:
        missing.append("database name (postgresql://user:pass@host:port/db)")
    if not parsed.hostname:
        missing.append("host")
    if not parsed.username:
        missing.append("username")
    if missing:
        raise ValueError(
            f"postgresql:// target {target!r} is missing: {', '.join(missing)}"
        )
    # Percent-encoded credentials are legal in URLs — decode them so the
    # driver receives the real values.
    native = pg8000_dbapi.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,  # 5432 is Postgres's default port.
        user=unquote(parsed.username),
        password=unquote(parsed.password) if parsed.password else None,
        database=unquote(database),
    )
    # Same autocommit mirror as the cloudsql path — see the comment there.
    native.autocommit = True
    return Conn(native, "postgres", None)


# Columns added to tables AFTER those tables first shipped (plan tasks A3
# and B2a). CREATE TABLE IF NOT EXISTS only creates missing TABLES — it
# never adds columns to a table that already exists, so the
# already-provisioned databases (the operator's sqlite dev file AND the live
# Cloud SQL instance, which has pre-migration tables) would silently keep
# running without these columns. Each entry is (table_name, column_name,
# column_declaration); the table/column names are code literals, never user
# input, so interpolating them into DDL below is safe.
#
# The B2a/B2b entries are declared without NOT NULL on purpose: ALTER TABLE
# ADD COLUMN cannot add a NOT NULL column without a DEFAULT on SQLite (and
# the pre-existing rows they must coexist with have no value to default to),
# so they are nullable exactly like the A3 agent_id precedent — a migration
# accommodation for rows written before the columns existed, NOT a design
# choice. detect_signals populates all three on every new row. The new
# `sources` table needs no migration entry: it is a NEW table, and
# CREATE TABLE IF NOT EXISTS creates missing tables on provisioned databases.
_MIGRATION_COLUMNS = (
    ("steps", "agent_id", "TEXT"),
    ("write_log", "agent_id", "TEXT"),
    ("signals", "evidence_quote", "TEXT"),
    ("signals", "evidence_verified", "INTEGER"),
    ("signals", "evidence_tier", "TEXT"),  # B2b: the three-way verdict column, same migration pattern as its B2a siblings
    # B2c: the ICP judge's verdict columns on accounts — same in-place
    # migration pattern as the A3/B2a/B2b precedents, so the operator's
    # already-provisioned sqlite dev file and Cloud SQL instance pick them up
    # on the next apply_schema without a destructive rebuild.  Nullable for
    # the same reason: rows written before B2c have no judge verdict, and
    # NULL is their honest value ("the judge never produced one").
    ("accounts", "judge_fit_label", "TEXT"),
    ("accounts", "judge_rationale", "TEXT"),
    ("accounts", "judge_divergence_justification", "TEXT"),
    # B3: the critique columns on message_draft_versions — same in-place
    # migration pattern as the B2b/B2c precedents above, so the operator's
    # already-provisioned sqlite dev file and Cloud SQL instance pick them up
    # on the next apply_schema without a destructive rebuild.  Nullable for
    # the same reason: rows written before B3 have no critique, and NULL is
    # their honest value ("no critic ran for this revision").
    ("message_draft_versions", "critique_passed", "INTEGER"),
    ("message_draft_versions", "critique_json", "TEXT"),
    # B4b: kill_switch_active on review_decisions — same in-place migration
    # pattern as the entries above, so the operator's already-provisioned
    # sqlite dev file and Cloud SQL instance pick it up on the next
    # apply_schema.  NOTE (a lead ticket error B4a already corrected): the
    # pre-existing kill_switch_active column lives on send_gate_decisions,
    # NOT on review_decisions — this entry adds a GENUINELY NEW column to a
    # different table; it does not move or duplicate the send-gate one.
    # Nullable for the same reason: rows written before B4b have no value,
    # and NULL is their honest value ("no switch state was recorded");
    # app/review.py populates it on every new row.
    ("review_decisions", "kill_switch_active", "INTEGER"),
    # B5: insert_seq on the three ordering-dependent tables — the monotonic
    # insertion-order column that makes "which row is the LATEST?" a
    # deterministic question.  created_at is second-precision TEXT, so two
    # rows written in the same second (an operator approving and then
    # approving-with-edits inside one second, or any scripted flow) ordered
    # arbitrarily under ORDER BY created_at — the send gate demonstrably
    # resolved to the OLDER review decision and produced "the approved
    # revision is #1 but the latest revision is #2" on correct data.  Same
    # in-place migration pattern as the precedents above, so the operator's
    # provisioned sqlite file and Cloud SQL instance pick the columns up on
    # the next apply_schema without a destructive rebuild.  Nullable for
    # the same reason as every entry above: rows written before B5 have no
    # sequence value, and NULL is their honest value.  Reads that pick "the
    # latest row" order by insert_seq DESC, created_at DESC — NULL seq
    # (legacy rows) sorts LAST in DESC on both SQLite and Postgres, which
    # is chronologically correct: legacy rows predate seq-carrying rows.
    ("review_decisions", "insert_seq", "INTEGER"),
    ("policy_decisions", "insert_seq", "INTEGER"),
    ("message_draft_versions", "insert_seq", "INTEGER"),
    # C1: insert_seq on state_transitions — B5's fix extended to the state
    # machine's own audit log.  B5 covered the three tables whose "latest
    # row" reads the send/review path resolves; C1 hit the same
    # second-precision created_at tie one table further down the audit
    # chain: two hops written inside one classify_and_route_reply call
    # (replied → routed → suppressed) shared a created_at second and the
    # history read ordered them arbitrarily.  Same in-place migration
    # pattern and same NULL accommodation as the B5 entries above.
    ("state_transitions", "insert_seq", "INTEGER"),
    # E1: insert_seq on replies — the same fix one table further still.
    # The follow-up path (ticket E1) must resolve "which reply on this
    # thread is the LATEST?" to decide whether the target is eligible for
    # a follow-up draft, and replies.  Two replies fetched (or classified)
    # in the same second previously ordered arbitrarily under ORDER BY
    # created_at alone.  Same in-place migration pattern and same NULL
    # accommodation: pre-E1 rows have no sequence value, and NULL is
    # their honest value (NULL seq sorts LAST in DESC on both dialects,
    # chronologically correct).
    ("replies", "insert_seq", "INTEGER"),
    # F1b: email_normalized on suppressions — the canonical matching key
    # added BESIDE the preserved email column.  Same in-place migration
    # pattern as the precedents above, so a suppression written before F1b
    # still suppresses after apply_schema.  Nullable ONLY as the migration
    # accommodation for rows written before the column existed; the
    # backfill pass (see _backfill_suppressions) fills it for every legacy
    # row, and every new writer (review.py / reply.py) populates it.
    ("suppressions", "email_normalized", "TEXT"),
    # H6: matched_rules_json on send_gate_decisions — the policy-matrix rule
    # IDs behind each refusal, mirroring policy_decisions.matched_rules_json
    # so an audit query can surface "every send P2 refused".  Same in-place
    # migration pattern as the precedents above, so the operator's
    # already-provisioned sqlite dev file and Cloud SQL instance pick the
    # column up on the next apply_schema without a destructive rebuild.
    # Declared WITHOUT NOT NULL on purpose (the established accommodation):
    # SQLite's ALTER TABLE ADD COLUMN cannot add a NOT NULL column without a
    # DEFAULT, and legacy rows written before H6 have no rule list — NULL is
    # their honest value.  NULL semantics: NULL = "predates rule attribution"
    # (a pre-H6 decision — never backfilled, because a backfilled [] would
    # falsely claim the decision was evaluated for rules), deliberately
    # distinguishable from [] = "evaluated, no rule matched" (an allow).
    # Every send-gate writer populates it on new rows.
    ("send_gate_decisions", "matched_rules_json", "TEXT"),
)


def _ensure_column(conn: Conn, table: str, column: str, decl: str) -> None:
    """Add ``column`` to ``table`` if it doesn't have one yet. Idempotent.

    Dialect-branched on purpose: Postgres supports ``ADD COLUMN IF NOT
    EXISTS``, but SQLite does NOT (its ALTER TABLE has no IF NOT EXISTS form
    and raises "duplicate column name" on a second run). So the sqlite path
    checks existence via PRAGMA table_info first and only then issues a
    plain ADD COLUMN; the postgres path checks information_schema and then
    issues the IF NOT EXISTS form as belt-and-braces. Both engines end up in
    the same place: the column exists, and re-running is a no-op.
    """
    if conn.dialect == "sqlite":
        # PRAGMA table_info returns one row per column (cid, name, type, ...)
        # — the row factory makes them dict-like, so row["name"] is the
        # column name. A set lookup is O(1) and makes the idempotency check
        # exact rather than exception-driven.
        names = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
        }
        if column in names:
            return  # Column already present — nothing to migrate.
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl};")
        return
    # postgres: information_schema.columns is the canonical column catalog.
    # The table_schema='public' filter avoids matching same-named columns in
    # other schemas. fetchone() returning None means the column is absent.
    found = conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=? AND column_name=?;",
        (table, column),
    ).fetchone()
    if found is None:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {decl};")


def _backfill_suppressions(conn: Conn) -> None:
    """Fill the F1b suppression matching keys for rows written before F1b.

    Two in-place fixes, both idempotent (re-running apply_schema is a no-op):

    1. ``email_normalized`` — legacy rows have NULL because the column did
       not exist when they were written.  For every such row the key is
       computed in Python by the ONE helper (normalize_email) and written
       back, so a pre-F1b suppression of ``Dr.Chan@serenity-clinic.test``
       still matches a lowercase/plus-tag probe afterwards.

    2. ``domain`` — domain names are case-insensitive (RFC 1035), so a
       legacy mixed-case domain row would miss a lowercase probe.  Fold it
       to lowercase in place (there is no audit reason to preserve domain
       casing; unlike ``email``, the domain is not "the address as
       written").

    These are schema/data-migration writes performed at apply_schema time —
    the same provisioning category as the ALTER TABLE pass above, NOT a
    business write (there is no run_id/step_id/actor to attribute here).

    WHY THIS IS DELIBERATELY NOT ONE TRANSACTION — unlike the H4b rebuild
    (_migrate_suppressions), this pass is self-healing and never
    destructive: every statement is an idempotent per-row fill (the UPDATEs
    are scoped to NULL keys, so a crash mid-loop just leaves the remaining
    rows for the next apply_schema to pick up, and `lower(lower(x))` is a
    no-op), and nothing here deletes or drops anything.  A crash cannot
    lose data, so the per-statement autocommit is safe — and wrapping it in
    a transaction would only make a crash leave MORE work undone, not less.
    """
    # Backfill email_normalized for every legacy address row whose key is
    # still NULL.  The fetch is scoped to NULL keys so re-runs do nothing.
    for row in conn.execute(
        "SELECT email FROM suppressions "
        "WHERE email_normalized IS NULL AND email IS NOT NULL;"
    ).fetchall():
        # The email value identifies the row (every address row's email is
        # distinct: it was the old primary key, and in the new shape the
        # UNIQUE email_normalized is derived from it); the normalized key is
        # the shared helper's product — never a second, drift-prone copy.
        conn.execute(
            "UPDATE suppressions SET email_normalized=? WHERE email=?;",
            (normalize_email(row["email"]), row["email"]),
        )
    # Fold legacy domain rows to lowercase in place.  `lower()` is the one
    # fold a domain ever needs (no plus-tags, no local part).
    conn.execute(
        "UPDATE suppressions SET domain = lower(domain) WHERE domain IS NOT NULL;"
    )


def _suppressions_is_old_shape(conn: Conn) -> bool:
    """Return True when the suppressions table is still the pre-H4b shape:
    ``email`` is the PRIMARY KEY.

    The H4b shape (see _SUPPRESSIONS_DDL) has NO primary key at all —
    uniqueness lives on email_normalized and domain — so a fresh or
    already-migrated table returns False and the rebuild below is a no-op.
    Dialect-branched on purpose: PRAGMA is SQLite-only and
    information_schema is the postgres catalog.
    """
    if conn.dialect == "sqlite":
        # PRAGMA table_info lists one row per column; the `pk` field is 0
        # for a non-PK column and the 1-based position INSIDE the primary
        # key.  The old shape made email the PK, so pk=1 on email is the
        # exact old-shape signal (the new shape has pk=0 on every column).
        for row in conn.execute("PRAGMA table_info(suppressions);").fetchall():
            if row["name"] == "email" and row["pk"]:
                return True
        return False
    # postgres: any PRIMARY KEY on the table means the old shape (the new
    # shape declares none).  information_schema.table_constraints is the
    # canonical constraint catalog; table_schema='public' is the app schema
    # every other migration read in this module already pins.
    row = conn.execute(
        "SELECT 1 FROM information_schema.table_constraints "
        "WHERE constraint_type='PRIMARY KEY' "
        "AND table_schema='public' AND table_name='suppressions' LIMIT 1;"
    ).fetchone()
    return row is not None


def _migrate_suppressions(conn: Conn) -> None:
    """In-place rebuild of a pre-H4b suppressions table to the H4b shape.

    WHY THIS EXISTS — CREATE TABLE IF NOT EXISTS never alters an existing
    table, and neither dialect can drop a PRIMARY KEY with a simple ALTER
    in the way this needs, so the established rebuild pattern is used:
    create the new-shape table under a temp name, copy every row, drop the
    old table, rename the temp into place.  This is what makes the
    operator's already-provisioned database (the sqlite dev file and the
    live Cloud SQL instance) pick up the H4b shape on the next apply_schema
    without a destructive manual rebuild.

    ORDERING REQUIREMENT — must run AFTER _backfill_suppressions (and the
    _MIGRATION_COLUMNS pass it depends on): the copy SELECT below reads the
    ``email_normalized`` column, which a pre-F1b table does not have until
    _ensure_column adds it.  The backfill also fills that column for legacy
    rows first, so the copy carries the canonical keys into the new table.

    ATOMIC (the H4b review-gate blocker) — the whole rebuild is wrapped in
    ONE transaction via conn.begin_write(), the same dialect-correct
    spelling write_gate.commit() uses (sqlite BEGIN IMMEDIATE, postgres
    plain BEGIN).  Both dialects have transactional DDL, so a failure at
    ANY point — including the worst case, a crash between ``DROP TABLE
    suppressions`` and the RENAME — rolls the rebuild back and the old
    table survives with every row.  Before this fix the five statements ran
    in autocommit, so a failure in that gap unrecoverably destroyed the
    suppression list, and the next apply_schema silently recreated an EMPTY
    table (CLAUDE.md §9: that loss means mailing people who opted out — the
    worst available failure).  The COMMIT is the single point of no return;
    everything before it is undoable.

    IDEMPOTENT — the old-shape detection at the top returns early once the
    table is new shape, so a second apply_schema is a no-op.  A crash
    BEFORE the COMMIT rolls back to the untouched old table, and the next
    run rebuilds it fresh (the temp name is dropped IF EXISTS first); a
    crash AFTER the COMMIT leaves the new shape, which the detection skips.
    There is no remaining window in which the suppression list can be lost:
    the old table is only dropped inside the transaction, and the
    transaction either commits the complete rebuild or rolls back to the
    old table.

    PRESERVES EVERY ROW — the INSERT...SELECT copies the whole table.  The
    H4b table-level CHECK (email IS NOT NULL OR domain IS NOT NULL) is
    satisfied by construction: every legacy row has a non-NULL email (the
    old primary key) or is a SQLite-quirk domain-only row whose domain is
    non-NULL.  One real data anomaly surfaces loudly rather than silently
    losing a row: two legacy rows whose emails normalise to the SAME key
    (e.g. 'Dr.Chan@x.test' and 'dr.chan@x.test', both legal under the old
    email PK) collide on the new email_normalized UNIQUE and the copy
    raises IntegrityError — which now ROLLS BACK the whole rebuild, so the
    old table (with both rows) survives and the operator must resolve the
    genuine duplicate before the migration can complete.
    """
    if not _suppressions_is_old_shape(conn):
        return  # Fresh or already-migrated — nothing to do.
    # Open ONE transaction for the whole rebuild (see the ATOMIC note): the
    # write-gate precedent — begin_write() spells the transaction correctly
    # per dialect, and every DDL/INSERT below participates, so the DROP of
    # the old table is invisible to every other reader until the COMMIT.  A
    # crash at any point rolls back to the untouched old table.
    conn.begin_write()
    try:
        # A leftover temp table from a crashed prior run must not break the
        # rebuild; drop it (IF EXISTS) before recreating it.  This drop is
        # inside the same transaction, so it rolls back with everything else.
        conn.execute("DROP TABLE IF EXISTS suppressions_h4b_mig;")
        # 1. Create the new-shape table under the temp name.  The DDL is THE
        #    same constant the fresh-database path splices into _DDL, so the
        #    rebuild can never drift from what apply_schema creates from scratch.
        conn.execute(
            _SUPPRESSIONS_DDL.replace(
                "CREATE TABLE IF NOT EXISTS suppressions",
                "CREATE TABLE IF NOT EXISTS suppressions_h4b_mig",
                1,
            )
        )
        # 2. Copy every existing row.  email_normalized is guaranteed present
        #    and filled by the _ensure_column / _backfill_suppressions passes
        #    that run before this one.
        conn.execute(
            """
            INSERT INTO suppressions_h4b_mig
                (email, email_normalized, domain, reason, added_at, added_by, notes)
            SELECT email, email_normalized, domain, reason, added_at, added_by, notes
            FROM suppressions;
            """
        )
        # 3. Drop the old-shape table, then 4. rename the new-shape table
        #    into place.  Nothing references suppressions via a foreign key
        #    (verified against the DDL), so the drop is safe on both dialects.
        #    Both statements are inside the transaction, so the gap between
        #    them is safe: a crash here rolls the DROP back, not the list.
        conn.execute("DROP TABLE suppressions;")
        conn.execute("ALTER TABLE suppressions_h4b_mig RENAME TO suppressions;")
        # 5. The COMMIT is the single point of no return — only after this
        #    is the new shape visible to any other reader.
        conn.execute("COMMIT")
    except Exception:
        # Any failure (a constraint collision, a crash stand-in, a real
        # connection drop) rolls the ENTIRE rebuild back — the old table is
        # restored and the suppression list survives.  Re-raise so the
        # caller knows the migration did not complete (and the next
        # apply_schema will retry it).
        conn.execute("ROLLBACK")
        raise


def apply_schema(conn: Conn) -> None:
    """Create every table from docs/db-schema.md if not already present.

    Idempotent: uses CREATE TABLE IF NOT EXISTS so it's safe to call at
    startup every time without checking whether tables already exist — on
    BOTH dialects (sqlite executescript, postgres split-and-execute).

    Then runs the _MIGRATION_COLUMNS pass: tables that already existed
    before a column was added to the DDL (e.g. agent_id on steps and
    write_log, plan task A3) are ALTERed in place, because CREATE TABLE
    IF NOT EXISTS alone would leave them missing on provisioned databases.

    Finally runs the F1b suppression backfill and the H4b suppression
    rebuild, so the operator's provisioned databases converge on the
    current suppressions shape on every startup.

    Called once at the beginning of each pipeline run (or at import time
    in tests) to guarantee the schema is in place before any tool writes.
    """
    conn.executescript(_DDL)
    # Apply the in-place column migrations AFTER the CREATE pass — on a
    # fresh database the columns already exist from the CREATE statements,
    # so _ensure_column no-ops; on a pre-existing database it adds them.
    for table, column, decl in _MIGRATION_COLUMNS:
        _ensure_column(conn, table, column, decl)
    # F1b: backfill the suppression matching keys on provisioned databases
    # (fresh databases have no legacy rows, so this is a no-op there).
    _backfill_suppressions(conn)
    # H4b: rebuild a pre-H4b suppressions table (email PRIMARY KEY) into the
    # H4b shape (no PK; uniqueness on email_normalized / domain).  Runs
    # AFTER the backfill so the copy can read email_normalized (see the
    # ORDERING REQUIREMENT note in _migrate_suppressions).
    _migrate_suppressions(conn)
