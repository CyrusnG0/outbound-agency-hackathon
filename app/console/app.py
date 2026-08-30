"""
Operator console (ticket A5a: read-only audit views; ticket B4b: the review
gate) — FastAPI + Jinja2.

This module is the hosted surface for the hackathon demo. Its job is to make
the pipeline's audit trail visible — targets, scores, ICP verdicts, signals,
policy decisions, state transitions, the full steps trace log — and, since
B4b, to host the operator's approval UI: the review queue, the per-target
draft diff with the critic's verdicts, the five review decisions, and the
kill-switch toggle.

Hard structural guarantees (enforced by tests/test_console.py, not just
promised here):

- THE CONSOLE'S OWN CODE ISSUES ONLY SELECTs. Every mutating SQL statement —
  write_gate commits, state_machine.transition, the kill-switch check,
  log_step — lives in app/review.py (the review gate) and
  app/kill_switch.py (the toggle).  This module calls them; it contains no
  write path of its own, not even an unused one.  The import set is an
  ALLOWLIST (app.db, app.review, app.kill_switch, app.console.auth) — adding
  any other app.* import here is a deliberate test edit (ticket B4b narrowed
  A5a's guarantee to exactly one door instead of reopening the wall; ticket
  H11 added app.console.auth, a PURE credential check that touches no
  database, so the wall is unchanged).
- AUTHENTICATION (ticket H11).  Every request — including every one of the
  three write routes — must prove it is the operator before the handler runs.
  The whole decision lives in app/console/auth.py (a pure module: one env
  var, hmac.compare_digest, no database); this module only wires it as a
  GLOBAL dependency so a future route is protected automatically.
- NO DDL. The console reads a database the CLI created; apply_schema() is
  deliberately never called here.
- One connection per request, ALWAYS closed (see get_conn — the close is a
  background-thread-leak fix for the cloudsql:// path, not hygiene).
- /_health never touches the database, so Cloud Run's health check (ticket
  A5b) reports process-alive even during a transient database blip.  It is
  named /_health, NOT /healthz, because Google's Cloud Run frontend
  intercepts the exact, case-sensitive path /healthz and answers it itself
  before the request ever reaches the container (ticket H16; measured
  2026-08-28 by the absent `server: Google Frontend` header on a /healthz
  404).  The unconventional name is deliberate, not a typo.
- Autoescape stays ON everywhere (draft bodies and critiques are LLM output
  derived from scraped third-party pages — no |safe anywhere, for any
  reason; tests/test_console.py proves the review page escapes them).
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode  # query-string building for the post-decision redirects (refusal reason + outcome banner)

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# The console's app imports — the ALLOWLIST (see the module docstring and
# tests/test_console.py's strengthened import test):
# - app.db: the dialect-agnostic connection wrapper (the original A5a import).
# - app.review: the review gate (B4b) — the console CALLS it for decisions;
#   every mutating statement lives in that module, not here.
# - app.kill_switch: the switch reader (for the always-visible indicator)
#   and writer (for the toggle).  The writer is the one non-SELECT action
#   the console may trigger, and it lives entirely in that module.
# - app.console.auth: the authentication dependency (H11).  It is on the
#   allowlist because it is a PURE credential check — no database, no SQL,
#   no write path of any kind — so importing it does not open the wall the
#   allowlist protects.  tests/test_console_auth.py proves it stays pure.
from app.console.auth import require_operator
from app.db import Conn, connect, normalize_email  # normalize_email: F1b — the console's suppression-status read folds identically to the gate and the writers
from app.kill_switch import read_kill_switch, write_kill_switch
from app.review import ReviewDecisionRequest, ReviewOutcome, VALID_DECISIONS, record_review_decision

# ── Templates ─────────────────────────────────────────────────────────────────
# Templates are resolved relative to THIS file, not the process cwd, so the
# console works no matter where uvicorn is started from.
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# fastapi.templating.Jinja2Templates builds its Environment with
# autoescape=jinja2.select_autoescape() (verified against the installed
# starlette 1.6.0 / jinja2 3.1.6 source), whose defaults enable escaping for
# .html templates. That is a SECURITY CONTROL, not a styling default: several
# fields rendered here (accounts.company_summary, signals.signal_value, the
# steps input/output JSON) contain LLM output derived from scraped
# third-party web pages — untrusted text by definition (docs/threat-model.md;
# policy rule P8 exists precisely because scraped input is hostile). With
# autoescape, a stored "<script>" renders as inert text; without it, the
# operator's own console is a stored-XSS vector. tests/test_console.py proves
# the escaping behaviour end to end.
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ── Database access ───────────────────────────────────────────────────────────


def _db_target() -> str:
    # OUTBOUND_DB_TARGET is the repo-wide convention for where the database
    # lives (docs/gcp-setup.md §6, tests/test_db_postgres.py): a sqlite file
    # path locally, a cloudsql:// (or postgresql://) URL in deployment. Read
    # per request (not at import time) so tests can repoint it via
    # monkeypatch and Cloud Run can change it without a code change.
    return os.environ.get("OUTBOUND_DB_TARGET", "data/outbound.db")


def get_conn() -> Iterator[Conn]:
    # One connection per request, opened when the request starts and closed
    # when it ends — a yield-based FastAPI dependency, so FastAPI guarantees
    # the teardown runs even if the route raises.
    #
    # The finally: conn.close() is LOAD-BEARING, not hygiene: app.db's
    # cloudsql:// path creates a Cloud SQL Python Connector whose background
    # refresh thread is released only by Conn.close() (see the comment on
    # _connect_cloudsql in app/db.py). A console that skipped close() would
    # leak one live thread per page view until process exit — on Cloud Run
    # (ticket A5b) that is the difference between a stable revision and one
    # that dies of thread exhaustion. For sqlite it is cheap correctness.
    conn = connect(_db_target())
    try:
        yield conn
    finally:
        conn.close()


# ── Data loading (shared by the HTML and JSON routes) ─────────────────────────


def _fetch_target_detail(conn: Conn, target_id: str) -> dict[str, Any] | None:
    """Load every section of one target's audit trail, or None if unknown.

    Both the HTML detail route and the JSON API route consume this one
    function, so the two views can never disagree about what the data is.
    All rows are converted to plain dicts (sqlite3.Row and the postgres
    wrapper's mapping rows both support dict(row)) so they serialize to JSON
    and render in Jinja identically.
    """
    # Section 1 (Target) + the offer slug: the target row itself, joined to
    # offers so the page shows the human-readable slug instead of an id.
    target_row = conn.execute(
        """
        SELECT t.target_id, t.account_id, t.contact_id, t.offer_id, t.source,
               t.state, t.score, t.final_recommendation,
               t.last_signal_refresh_at, t.created_at, t.updated_at,
               o.slug AS offer_slug
        FROM targets t
        JOIN offers o ON t.offer_id = o.offer_id
        WHERE t.target_id = ?
        """,
        (target_id,),
    ).fetchone()
    if target_row is None:
        # Unknown target id — the routes turn this into a 404 with a readable
        # message, never a blank page or a stack trace.
        return None

    # Section 2 (Company): the account's research summary and ICP verdict.
    # targets.account_id is NOT NULL, so this row always exists for a target.
    # B2c: the query also selects the judge columns (judge_fit_label /
    # judge_rationale / judge_divergence_justification) so the template can
    # render the judge's verdict next to the deterministic one — a
    # divergence is the most interesting thing on the page and must be
    # legible, never hidden behind a join or a code read.
    account_row = conn.execute(
        """
        SELECT account_id, company_name, domain, normalized_domain, industry,
               estimated_size, geo, company_summary, icp_fit_label,
               icp_fit_score, icp_fit_reasons, icp_non_fit_reasons,
               judge_fit_label, judge_rationale, judge_divergence_justification,
               created_at, updated_at
        FROM accounts
        WHERE account_id = ?
        """,
        (target_row["account_id"],),
    ).fetchone()

    # Section 3 (Contact): targets.contact_id is NULLABLE — company-only
    # leads are explicitly allowed by CSV import — so the query runs only
    # when a contact actually exists and the section renders "no contact
    # data" otherwise.
    contact_row = None
    if target_row["contact_id"] is not None:
        contact_row = conn.execute(
            """
            SELECT contact_id, account_id, full_name, title, seniority,
                   department, email, email_verified, linkedin_url,
                   persona_fit_score, created_at, updated_at
            FROM contacts
            WHERE contact_id = ?
            """,
            (target_row["contact_id"],),
        ).fetchone()

    # Section 4 (Signals): enrichment data points found for this target,
    # oldest first so the page reads in the pipeline's discovery order.
    # B2b: the query now also selects the evidence columns (evidence_quote,
    # evidence_verified, evidence_tier) so the template can render each
    # signal's backing quote and its three-way verdict — the whole point of
    # B2a/B2b is that an operator can SEE what a signal is backed by and
    # whether it checks out, not just that it exists.
    signal_rows = conn.execute(
        """
        SELECT signal_id, run_id, target_id, signal_type, signal_value,
               signal_strength, source_url, source_confidence,
               evidence_quote, evidence_verified, evidence_tier, created_at
        FROM signals
        WHERE target_id = ?
        ORDER BY created_at
        """,
        (target_id,),
    ).fetchall()

    # Section 5 (Policy decisions): every policy check recorded against this
    # target, oldest first.
    policy_rows = conn.execute(
        """
        SELECT policy_decision_id, run_id, step_id, target_id, action,
               decision, risk_level, reasons_json, matched_rules_json,
               missing_fields_json, created_at
        FROM policy_decisions
        WHERE target_id = ?
        ORDER BY created_at
        """,
        (target_id,),
    ).fetchall()

    # Section 6 (State transitions): the state machine's audit log for this
    # target, chronological — this is how "how did this target get here?" is
    # answered.  The order key is insert_seq (ticket C1 extended B5's fix
    # to this table): created_at is second-precision TEXT, so two hops in
    # the same second (e.g. replied → routed → suppressed inside one reply
    # classification) previously ordered arbitrarily.  The `(insert_seq IS
    # NULL) DESC` prefix is load-bearing for the legacy rows: in ASC
    # ordering SQLite sorts NULLs FIRST and Postgres sorts them LAST, so a
    # bare `ORDER BY insert_seq` would disagree across dialects about
    # whether pre-C1 rows (NULL seq — and therefore older than every
    # seq-carrying row) come first.  The explicit prefix puts legacy rows
    # first on BOTH dialects, then seq ascending = chronological.
    transition_rows = conn.execute(
        """
        SELECT transition_id, run_id, step_id, target_id, previous_state,
               new_state, reason, actor, matched_policy_id, created_at
        FROM state_transitions
        WHERE target_id = ?
        ORDER BY (insert_seq IS NULL) DESC, insert_seq, created_at
        """,
        (target_id,),
    ).fetchall()

    # Section 7 (Trace log): every pipeline step for this target, oldest
    # first — the full input/output JSON of each tool invocation.
    step_rows = conn.execute(
        """
        SELECT step_id, run_id, target_id, tool_name, input_json,
               output_json, model_call_hash, agent_id, status, created_at
        FROM steps
        WHERE target_id = ?
        ORDER BY created_at
        """,
        (target_id,),
    ).fetchall()

    return {
        "target": dict(target_row),
        "company": dict(account_row) if account_row is not None else None,
        "contact": dict(contact_row) if contact_row is not None else None,
        "signals": [dict(row) for row in signal_rows],
        "policy_decisions": [dict(row) for row in policy_rows],
        "state_transitions": [dict(row) for row in transition_rows],
        "steps": [dict(row) for row in step_rows],
    }


# The four showcase moments from the real 2026-08-29/30 batch (docs/current_status.md,
# docs/demo-script.md), named by company so the row still resolves after a reseed —
# a hardcoded target_id would silently 404 the moment the demo database is rebuilt.
# "why" is fixed narration text written once here, not read from the database: it is
# the demo page's own commentary on real, already-verified rows, not a claim about
# what the pipeline computed.
#
# "Mark Boyden Associates" removed 2026-08-30: its real follow-up draft was written
# BEFORE the real scheduling feature existed, when the therapy-app offer still carried
# the old static `booking_url` (a claude.ai link) — its stored footer still states that
# link, and the footer column is deliberately immutable outside a fresh drafting pass
# (B3-Z1: never operator- or LLM-edited). The target can't be redrafted either — it's
# state="awaiting_review", and run_target_through_draft only accepts "scored" or
# "routed". Delisting here (so nothing links to /review/{its target_id} from this page)
# plus rejecting the stale draft in the review console (an ordinary operator decision,
# recorded the normal way) is the fix; Solacetree already covers the same "positive
# reply -> follow-up draft" story with a footer built by the real scheduler.
_SHOWCASE_TARGETS = [
    (
        "MindnLife",
        "The first real, un-preselected live run: researched cold, scored, drafted, "
        "approved through the real console API, sent (DRY_RUN), replied positive, "
        "classified 0.98 — then unsubscribed and permanently suppressed.",
    ),
    (
        "Psychotherapy Counselling Clinic",
        "Scored strong_fit by the deterministic formula. The LLM judge overruled it: "
        "the company is in Victoria, Australia, and the ICP is Hong Kong only — a "
        "disqualifier the formula's fields never checked.",
    ),
    (
        "Focus2 Intelligent Therapy",
        "A real objection reply. The router held it (draft_hold) instead of "
        "auto-drafting a rebuttal — the one case in this batch where the "
        "correct action is to do nothing without a human.",
    ),
    (
        "Solacetree Counselling Limited",
        "A real positive reply routed to a real follow-up draft — whose footer "
        "below carries an ACTUAL reserved calendar slot (app/tools/schedule_meeting.py), "
        "picked and justified by an LLM from a real computed calendar, "
        "re-validated by code, and written through the same write gate as "
        "every other core-table row in this system.",
    ),
]


def _fetch_demo_showcase(conn: Conn) -> dict[str, Any]:
    """Assemble the one-screen demo page: the four showcase targets from the
    real batch (each linked to its real console page, never re-rendered
    here) plus the most recently reserved real meeting and the follow-up
    draft whose footer states it.

    SELECT-only, like every other console read — this function adds no new
    write path and no new app.* import. Every row here already exists and
    is independently visible on /targets/{id}, /review/{id}, or the
    ``meetings`` table; this page only saves the operator from hunting for
    it live on camera.
    """
    showcase = []
    for company_name, why in _SHOWCASE_TARGETS:
        # ILIKE would be the postgres-idiomatic match, but the repo's own
        # `?`-placeholder rewrite (app/db.py) targets both sqlite and
        # postgres — LIKE on an exact, already-known company name is
        # dialect-portable and exact enough for five fixed names.
        row = conn.execute(
            """
            SELECT t.target_id, t.state, a.company_name
            FROM targets t
            JOIN accounts a ON t.account_id = a.account_id
            WHERE a.company_name = ?
            """,
            (company_name,),
        ).fetchone()
        # A showcase name that no longer resolves (reseeded database, a
        # renamed fixture) renders as an explicit "not available in this
        # database" row rather than crashing the whole demo page —
        # mark-don't-drop, the same rule the signals table already follows.
        showcase.append(
            {
                "company_name": company_name,
                "why": why,
                "target_id": row["target_id"] if row is not None else None,
                "state": row["state"] if row is not None else None,
            }
        )

    # The most recently reserved REAL meeting — found by "the newest
    # non-cancelled row in meetings", never a hardcoded company/target_id,
    # so this page keeps working across whichever target the demo batch
    # actually produced a reservation for.
    meeting_row = conn.execute(
        """
        SELECT meeting_id, target_id, company_name, contact_name, scheduled_at,
               duration_minutes, reasoning
        FROM meetings
        WHERE status != 'cancelled'
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()
    meeting = dict(meeting_row) if meeting_row is not None else None

    # The follow-up draft's footer that actually STATES the reservation
    # above — found by "the latest revision on that same target", not a
    # footer-text search, now that the reservation itself is a real row
    # rather than a substring to grep for.
    draft = None
    if meeting is not None:
        draft_row = conn.execute(
            """
            SELECT subject, body, footer, revision_number, created_at
            FROM message_draft_versions
            WHERE target_id = ?
            ORDER BY revision_number DESC
            LIMIT 1
            """,
            (meeting["target_id"],),
        ).fetchone()
        draft = dict(draft_row) if draft_row is not None else None

    return {
        "showcase": showcase,
        "meeting": meeting,
        "meeting_draft": draft,
    }


def _is_demo_database(conn: Conn) -> bool:
    """Detect a database seeded by the demo seed (ticket D3a) — SELECT-only,
    exactly one cheap query, no new import.

    The marker is a steps row with tool_name='demo_seed' (the trace row
    app/demo_seed.py logs on every seed run).  A SELECT is all this may
    cost: the steps table is the trace log itself, the scan is a single
    indexed-by-nothing equality on a tiny table, and the banner below is
    the honesty surface — the operator (or a judge) must never mistake
    seeded placeholder contacts and gate results for real verified data.
    A database error (e.g. a not-yet-created steps table) degrades to
    False: the banner is an indicator, never a reason to break a page.
    """
    try:
        return (
            conn.execute(
                "SELECT 1 FROM steps WHERE tool_name='demo_seed' LIMIT 1;"
            ).fetchone()
            is not None
        )
    except Exception:
        # No steps table (schema never applied) or a transient DB blip —
        # the banner is cosmetic; the page must still render.
        return False


def _replay_mode() -> bool:
    """Detect an OFFLINE REPLAY of a real, previously recorded run (ticket D1)
    — a pure env-var read, NO database query, computed per request.

    OUTBOUND_REPLAY_MODE=1 is exported by scripts/demo_replay.sh when it serves
    a disposable copy of data/e2e_run2.db.  It is NOT an auth bypass and NOT a
    behavior-changing flag anywhere except the honest replay banner in
    base.html: it must never touch _db_target(), _is_demo_database(), or any
    query.  It is a sibling of demo_data (the DEMO DATA banner flag), not a
    substitute: the two flags are independent and must never suppress each
    other — a database could in principle be both demo-seeded and replayed, and
    both banners should render.
    """
    return os.environ.get("OUTBOUND_REPLAY_MODE") == "1"


def _pretty_json(raw: str | None) -> str:
    # Steps store input/output as single-line JSON TEXT; in a <pre> block
    # that is one unreadable line. Pretty-print for the HTML view only — the
    # API route returns the stored text untouched, because raw data is what a
    # debugging surface should show. If the stored text is not valid JSON
    # (a tool wrote a plain string), fall back to the raw text rather than
    # crashing the page.
    if raw is None:
        return ""  # No output was produced — render an empty block, not "None".
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw  # Not JSON after all — show it verbatim.


class TargetDetail(BaseModel):
    # Explicit output model for the JSON API route (CLAUDE.md §7: explicit
    # Pydantic models for structured I/O). Its fields mirror the seven
    # console sections; the dicts carry each table's exact column set
    # (ticket A5a appendix), so adding a column is a deliberate change here.
    target: dict[str, Any]
    company: dict[str, Any] | None
    contact: dict[str, Any] | None
    signals: list[dict[str, Any]]
    policy_decisions: list[dict[str, Any]]
    state_transitions: list[dict[str, Any]]
    steps: list[dict[str, Any]]


# ── Review payload loading (ticket B4b — the approval UI's data) ─────────────


def _pending_review_targets(conn: Conn) -> list[dict[str, Any]]:
    """The review queue: every target in ``awaiting_review``, newest-updated
    first — the same shape of summary row the index page shows, so the
    operator can go straight from the queue into a decision."""
    rows = conn.execute(
        """
        SELECT t.target_id, a.company_name, t.updated_at
        FROM targets t
        JOIN accounts a ON t.account_id = a.account_id
        WHERE t.state = 'awaiting_review'
        ORDER BY t.updated_at DESC
        """
    ).fetchall()
    # Plain dicts so the rows serialize and template identically (the A5a
    # convention).
    return [dict(row) for row in rows]


def _fetch_review_payload(conn: Conn, target_id: str) -> dict[str, Any] | None:
    """Load everything the operator needs to decide one target
    (docs/human-review.md §2: research summary, ICP assessment, signals,
    the draft — ALL revisions with the critique that produced each — the
    policy decision, risk flags, suppression status), or None if unknown.

    Read-only: every statement here is a SELECT, and the payload dicts are
    what both the HTML review page and the JSON queue API render — the two
    views can never disagree about what the data is (the A5a pattern).
    """
    # The target row + offer slug (the review page header needs the state
    # to show "already decided" for a double-submitted form).
    target_row = conn.execute(
        """
        SELECT t.target_id, t.account_id, t.contact_id, t.offer_id, t.state,
               t.score, o.slug AS offer_slug
        FROM targets t
        JOIN offers o ON t.offer_id = o.offer_id
        WHERE t.target_id = ?
        """,
        (target_id,),
    ).fetchone()
    if target_row is None:
        # Unknown target — the routes turn this into a 404, same contract
        # as the A5a detail routes.
        return None

    # The research summary + ICP verdicts (the same columns the A5a detail
    # page renders — the reviewer sees exactly what the pipeline learned).
    account_row = conn.execute(
        """
        SELECT account_id, company_name, domain, normalized_domain, industry,
               estimated_size, geo, company_summary, icp_fit_label,
               icp_fit_score, icp_fit_reasons, icp_non_fit_reasons,
               judge_fit_label, judge_rationale, judge_divergence_justification
        FROM accounts
        WHERE account_id = ?
        """,
        (target_row["account_id"],),
    ).fetchone()

    # The contact, when one exists (company-only leads have none — the
    # suppression status must say so rather than crash).
    contact_row = None
    if target_row["contact_id"] is not None:
        contact_row = conn.execute(
            """
            SELECT contact_id, full_name, title, seniority, department,
                   email, email_verified, linkedin_url
            FROM contacts
            WHERE contact_id = ?
            """,
            (target_row["contact_id"],),
        ).fetchone()

    # The signals used, oldest first — the evidence the draft was written from.
    signal_rows = conn.execute(
        """
        SELECT signal_type, signal_value, signal_strength, source_url,
               evidence_quote, evidence_verified, evidence_tier
        FROM signals
        WHERE target_id = ?
        ORDER BY created_at
        """,
        (target_id,),
    ).fetchall()

    # The LATEST policy decision — the risk picture the reviewer is shown.
    # Second-precision created_at tie is theoretical (the same caveat the
    # draft runner documents): the policy gate writes at most one row per
    # target per run.
    policy_row = conn.execute(
        """
        SELECT policy_decision_id, action, decision, risk_level,
               reasons_json, matched_rules_json, created_at
        FROM policy_decisions
        WHERE target_id = ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (target_id,),
    ).fetchone()

    # EVERY draft revision in order — the "draft diff across iterations"
    # the plan row asks for: B3 stored critique_passed/critique_json on
    # each revision precisely so this page can show the agent improving its
    # own work.  critique_json is parsed here (read-only code) so the
    # template renders the issues list and severity directly; a revision
    # with no critique (operator edits, pre-B3 rows) keeps None.
    draft_rows = conn.execute(
        """
        SELECT draft_version_id, revision_number, subject, body, footer,
               edited_by, policy_check_passed, injection_scan_passed,
               send_gate_passed, critique_passed, critique_json, created_at
        FROM message_draft_versions
        WHERE target_id = ?
        ORDER BY revision_number, created_at
        """,
        (target_id,),
    ).fetchall()
    draft_versions = []
    for row in draft_rows:
        display = dict(row)
        # Parse the critic's verdict for display; an unparseable stored
        # string degrades to None (the raw text stays visible in the A5a
        # trace page if an operator needs to see the mangled JSON).
        try:
            display["critique"] = json.loads(row["critique_json"]) if row["critique_json"] else None
        except json.JSONDecodeError:
            display["critique"] = None
        draft_versions.append(display)

    # The run_id the decision will be recorded under: the run that
    # produced the current state (the latest transition's run).  "" when a
    # target has no transitions (a hand-seeded fixture) — the decision
    # write still records, just without a run grouping.
    run_row = conn.execute(
        "SELECT run_id FROM state_transitions WHERE target_id=? "
        "ORDER BY created_at DESC LIMIT 1;",
        (target_id,),
    ).fetchone()

    # Suppression status (human-review.md §2's "suppression status"): is
    # the contact's email already suppressed?  A missing contact or email
    # is its own honest status, not an error.
    email = contact_row["email"] if contact_row is not None else None
    suppressed = False
    if email:
        suppressed = (
            conn.execute(
                "SELECT 1 FROM suppressions WHERE email_normalized=?;",
                (normalize_email(email),),
            ).fetchone()
            is not None
        )

    # Risk flags, built deterministically (never invented): a
    # non-allow latest policy decision, and the absence of a contact email
    # (which also disables reject_and_suppress — the page says so).
    risk_flags: list[str] = []
    if policy_row is not None and policy_row["decision"] != "allow":
        risk_flags.append(
            f"policy:{policy_row['decision']}({policy_row['risk_level']})"
        )
    if not email:
        risk_flags.append("no_contact_email")

    return {
        "target": dict(target_row),
        "company": dict(account_row) if account_row is not None else None,
        "contact": dict(contact_row) if contact_row is not None else None,
        "signals": [dict(row) for row in signal_rows],
        "policy_decision": dict(policy_row) if policy_row is not None else None,
        "draft_versions": draft_versions,
        "suppressed": suppressed,
        "contact_email": email,
        "run_id": run_row["run_id"] if run_row is not None else "",
        "risk_flags": risk_flags,
    }


class ReviewDecisionApiRequest(ReviewDecisionRequest):
    """The JSON body of POST /review/decision (docs/api.md §4): the review
    request plus the run_id the decision is recorded under.  Kept as a
    subclass so the console's wire shape and the service's shape can never
    disagree on the decision fields."""

    run_id: str = ""  # the run the decision is grouped under (the queue payload carries it; "" is legal)


class ReviewQueueApiResponse(BaseModel):
    """The JSON shape of GET /api/review/queue (docs/api.md §4): the full
    review payload per pending target, plus the kill-switch state so an API
    caller sees the same indicator the HTML page shows."""

    targets: list[dict[str, Any]]
    kill_switch: dict[str, Any]


class ReviewQueueCountApiResponse(BaseModel):
    """The JSON shape of GET /api/review/queue/count (ticket U3): how many
    targets are awaiting review right now — a single cheap integer, nothing
    else.

    Deliberately NOT the full review payload the /api/review/queue route
    returns: this endpoint feeds a background poll running on EVERY page
    (base.html's header badge), so pulling LLM-derived research content (the
    summaries, drafts and critiques the heavy endpoint carries) into client JS
    just to render a count would waste bandwidth on a background poll for no
    benefit.  The count is enough to render "(N)" next to the Review queue
    link and in the tab title."""

    count: int


class RunStepSummary(BaseModel):
    """One trace-log row in a live run (ticket U2): everything the live view
    renders per step.  Deliberately NOT the input/output JSON — those can
    contain LLM output derived from scraped third-party pages (policy rule
    P8), the exact content class review_target.html / target_detail.html are
    careful about; the live view's rows LINK to the target-detail trace-log
    page for the full payload (see the U2 scope decision)."""

    step_id: str
    run_id: str
    target_id: str | None
    tool_name: str
    status: str
    created_at: str


class RunStepsApiResponse(BaseModel):
    """The JSON shape of GET /api/run/{run_id}/steps (ticket U2): every step
    logged under one run_id, the CURRENT state of each target the run touched,
    and whether the run has reached a stop condition so the client-side poll
    loop knows when to stop."""

    run_id: str
    steps: list[RunStepSummary]
    target_states: dict[str, str]
    complete: bool


# The states that mean "this run's work on a target is done — stop watching."
# The single most important demo beat is `awaiting_review` (the human gate —
# the pipeline stops there and WILL NOT proceed; the demo plan §5 calls it
# "the single most important demo beat").  The terminal states are from
# docs/state-machine.md §5 (`suppressed` / `not_target` / `failed` are terminal
# for outbound; `dry_run_sent` is terminal for a DRY_RUN run; `sent` / `bounced`
# are the LIVE-mode send outcomes).  States NOT listed (`new` / `enriched` /
# `researched` / `scored` / `drafted` / `watchlist` / `approved` / `replied` /
# `routed`) are mid-pipeline, operator-paused, or still-advancing: the page
# keeps polling for them, because the run may still produce steps — until the
# client-side max-poll safety valve stops the loop (run.html).
#
# The reviewer confirmed `awaiting_review` / `dry_run_sent` / `sent` / `bounced`
# / `suppressed` / `not_target` / `failed` are correctly chosen.  Three states
# were deliberately REMOVED from the pre-U2-fix set, all for the same root
# cause: the check must not declare "complete" from the ABSENCE of evidence
# that the run is still working, only from POSITIVE evidence that it has
# finished (ticket U2-fix Findings 2 & 3):
#   - `approved`: a send run STARTS there.  app/send_cli.py selects targets in
#     `approved` and moves them to `dry_run_sent`; opening the live view at the
#     moment a send run begins — first send_email-family step logged, targets.state
#     still `approved` — would falsely declare complete before a single send
#     renders.
#   - `routed` / `replied`: docs/state-machine.md §7j/§7k (and its line 97)
#     documents both as TRANSIENT within a single invocation — the reply router
#     fires `replied → routed` and then, in the SAME invocation, `routed →
#     suppressed` (a high-confidence unsubscribe) or `routed → drafted` (a
#     follow-up).  A poll landing between those writes sees `replied` or
#     `routed`, declares complete, and stops — hiding the unsubscribe
#     suppression, a headline safety beat.
# Do not add a new stop state without checking docs/state-machine.md yourself
# first and justifying it.
_RUN_STOP_STATES = frozenset({
    "awaiting_review",  # the human gate — THE demo beat (a decision is waiting)
    "dry_run_sent",     # terminal for a DRY_RUN send run (state-machine.md §5, §7e)
    "sent",             # post-gate LIVE-mode send outcome
    "bounced",          # post-gate LIVE-mode send outcome
    "suppressed",       # terminal for outbound (state-machine.md §5)
    "not_target",       # terminal for outbound (state-machine.md §5)
    "failed",           # error path — terminal (state-machine.md §5)
})


# ── Batch manifests: the four run types that declare their full batch up front ──
# Every CLI-originated run type logs ONE batch-manifest step (target_id NULL)
# whose output_json lists the run's FULL intended target set under
# ``target_ids`` — known up front, before any per-item step exists:
#   - ``get_targets``            phase1_cli (app/tools/get_targets.py)
#   - ``draft_batch_manifest``   draft_cli
#   - ``send_batch_manifest``    send_cli
#   - ``reply_batch_manifest``   reply_cli
# _get_manifest_target_ids() reads these names; keep this set the SINGLE list
# of manifest tool names so the query and any other consumer cannot drift.
_MANIFEST_TOOL_NAMES: tuple[str, ...] = (
    "get_targets",
    "draft_batch_manifest",
    "send_batch_manifest",
    "reply_batch_manifest",
)


# ── Quiet-period fallback for runs WITHOUT any batch manifest (U2-fix Finding 1) ──

# The client polls every 2s (run.html's POLL_INTERVAL_MS); LLM-call latency in
# this pipeline can run several seconds per step.  12s is ~6 poll intervals —
# comfortably above per-step latency (so a just-logged step cannot falsely
# declare complete while the pipeline is mid-LLM-call on the NEXT target), yet
# far below the client's 120s max-poll safety valve (so a genuinely-finished
# run reports complete within a few polls).  This is an HONESTLY-DISCLOSED
# heuristic, not a structural guarantee.
#
# Since U2-fix2 every one of the four CLI-originated run types above logs a
# real batch manifest, so this fallback fires only for a run with NO
# manifest step at all — a direct API/programmatic run, or a future
# run-originating path that has not been given a manifest yet.
#
# CORRECTION (round-3 review): a taskmaster-agent run is NOT in that "no
# manifest" set — app/agents/taskmaster.py's import stage calls
# get_targets.import_csv() under the run's own run_id, which DOES log a
# get_targets manifest, so a taskmaster run takes the AUTHORITATIVE branch
# above, not this fallback.  The catch: that manifest only names the
# targets the IMPORT stage saw.  Draft/send/reply run as inner functions
# under the SAME run_id without logging their own manifest step, so a
# multi-stage taskmaster run can report complete once the import-stage
# targets reach a stop state, while later stages are still working on
# targets the manifest never named (a pre-existing gap, not introduced
# here — flagged for its own follow-up ticket, out of U2's CLI scope).
#
# For a run with genuinely NO manifest, this fallback deliberately trades a
# few extra seconds of polling for not stopping the live view mid-run, and
# it is NOT equivalent to the manifest-backed check: against this
# pipeline's own configured worst-case per-node latency (~605s, app/llm.py's
# 300s request timeout × 2 retries), 12s is too short to be safe — which is
# exactly why the manifest, not a longer quiet period, is the real fix.
_QUIET_PERIOD_SECONDS = 12


def _quiet_period_elapsed(conn: Conn, run_id: str) -> bool:
    """True when the run's most recent step is older than _QUIET_PERIOD_SECONDS.

    A run with no steps at all has nothing in flight, so it counts as elapsed
    (the per-target stop-state check alone decides).  steps.created_at is
    second-precision UTC TEXT (datetime('now')); it is parsed here in Python
    rather than compared in SQL so the logic is dialect-agnostic.  An
    unparseable timestamp is treated as elapsed (fail-open to the per-target
    check) — it is not evidence of activity, and hanging the live poll on a
    corrupt timestamp would be worse than the heuristic it guards.
    """
    row = conn.execute(
        "SELECT MAX(created_at) AS max_created FROM steps WHERE run_id = ?;",
        (run_id,),
    ).fetchone()
    if row is None or row["max_created"] is None:
        return True  # No steps — nothing can be in flight.
    try:
        last_step_at = datetime.strptime(
            row["max_created"], "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return True  # Unparseable — not evidence of activity; fail open.
    return (datetime.now(timezone.utc) - last_step_at) >= timedelta(
        seconds=_QUIET_PERIOD_SECONDS
    )


def _get_manifest_target_ids(conn: Conn, run_id: str) -> set[str] | None:
    """The run's full intended target set from its batch-manifest step(s), or
    None when the run has no usable manifest.

    Every CLI-originated run type logs exactly one manifest step (target_id
    NULL — the batch bookkeeping step) whose output_json carries the run's
    complete target list under ``target_ids``: phase1_cli's ``get_targets``
    (app/tools/get_targets.py) and, since U2-fix2, draft_cli's
    ``draft_batch_manifest`` / send_cli's ``send_batch_manifest`` /
    reply_cli's ``reply_batch_manifest`` (see _MANIFEST_TOOL_NAMES).  When
    such a step exists, its target list is the authoritative "this run is
    responsible for these targets" set — known UP FRONT, before any target
    has a per-item step.  This is what closes the U2-fix Finding 1 bug: a
    target the run has not reached yet has ZERO steps/transitions and is
    invisible to a rows-derived set, so a multi-target run used to report
    complete after target #1 while the pipeline was still working on target
    #2.

    Returns None when there is no manifest step of any of the four names, or
    when none of their output_json can be parsed into a non-empty target_ids
    list (an anomaly — the caller then falls back to the derived set +
    quiet-period heuristic rather than hanging or guessing).  Multiple
    manifest steps (all names, or repeated rows) are UNIONED.  SELECT-only,
    like every console query.
    """
    rows = conn.execute(
        """
        SELECT output_json FROM steps
        WHERE run_id = ? AND tool_name IN (?, ?, ?, ?) AND output_json IS NOT NULL
        ORDER BY created_at, step_id;
        """,
        (run_id, *_MANIFEST_TOOL_NAMES),
    ).fetchall()
    target_ids: set[str] = set()
    for row in rows:
        try:
            data = json.loads(row["output_json"])
        except (TypeError, json.JSONDecodeError):
            continue  # Unparseable manifest row — not usable; fall through.
        if not isinstance(data, dict):
            continue
        ids = data.get("target_ids")
        if isinstance(ids, list):
            target_ids.update(t for t in ids if isinstance(t, str))
    return target_ids if target_ids else None


def _fetch_run_steps(conn: Conn, run_id: str) -> RunStepsApiResponse:
    """Load every step logged under one run_id, the current state of each
    target the run touched, and whether the run has reached a stop condition
    (ticket U2).

    SELECT-only, like every console query (the module docstring and
    tests/test_console.py).  An UNKNOWN / not-yet-started run_id is NOT an
    error: a live run legitimately starts with zero steps logged, so this
    returns an empty list with complete=False and the page polls from the very
    beginning without erroring.
    """
    step_rows = conn.execute(
        """
        SELECT step_id, run_id, target_id, tool_name, status, created_at
        FROM steps
        WHERE run_id = ?
        ORDER BY created_at, step_id
        """,
        (run_id,),
    ).fetchall()

    # The targets this run is responsible for.  Two cases (ticket U2-fix
    # Finding 1):
    #
    # 1. The run has a batch-manifest step — one of the four in
    #    _MANIFEST_TOOL_NAMES (phase1_cli's `get_targets`, or U2-fix2's
    #    draft/send/reply `*_batch_manifest` steps): that step's output_json
    #    lists the run's FULL intended target set, known up front.  Use it as
    #    the AUTHORITATIVE set.  This closes the Finding-1 bug — a target the
    #    run has not reached yet has ZERO steps/transitions and was previously
    #    invisible, so a multi-target run reported complete after target #1
    #    while the pipeline was still working on target #2.
    # 2. No manifest (a run-originating path that does not log one yet —
    #    direct API/programmatic use, or a future CLI that has not been given
    #    a manifest step; NOT a taskmaster-agent run — see the CORRECTION
    #    comment on _quiet_period_elapsed's caller below): fall back to
    #    deriving the set
    #    from the rows the run has already written (its steps, and its
    #    state_transitions — the same table _fetch_review_payload reads for a
    #    run's grouping), and rely on the quiet-period fallback below so a
    #    mid-step poll cannot falsely declare complete.  At the very start of
    #    a run both can be empty — that is the "waiting for the run to start"
    #    state, never an error.
    manifest_target_ids = _get_manifest_target_ids(conn, run_id)
    if manifest_target_ids is not None:
        target_ids = manifest_target_ids
    else:
        target_ids = set()
        for row in step_rows:
            if row["target_id"] is not None:
                target_ids.add(row["target_id"])
        for row in conn.execute(
            "SELECT DISTINCT target_id FROM state_transitions WHERE run_id = ?;",
            (run_id,),
        ).fetchall():
            target_ids.add(row["target_id"])

    # The CURRENT state of each target: targets.state is the source of truth
    # for where a target stands right now (it reflects every run that has
    # touched the target, not just this one — the draft stage's targets are
    # `scored` until this run's first hop, which is exactly why `scored` is
    # NOT a stop state above).
    target_states: dict[str, str] = {}
    for target_id in sorted(target_ids):
        row = conn.execute(
            "SELECT state FROM targets WHERE target_id = ?;", (target_id,)
        ).fetchone()
        if row is not None:
            target_states[target_id] = row["state"]

    # Complete: every target this run is responsible for is in a stop state.
    # An empty target set (run not started) is NOT complete — keep polling.  A
    # target that vanished from targets (no row) is excluded from target_states
    # and forces complete False, because we cannot prove the run is done.
    complete = (
        bool(target_ids)
        and len(target_states) == len(target_ids)
        and all(state in _RUN_STOP_STATES for state in target_states.values())
    )

    # Quiet-period fallback (ticket U2-fix Finding 1): for runs WITHOUT any
    # batch manifest, the per-target check above only proves the targets the
    # run has ALREADY touched are done — it cannot see an untouched target, so
    # it can still declare complete while the pipeline is working on the next
    # one.  As an honestly-disclosed heuristic (NOT a structural guarantee,
    # and not equivalent to the manifest-backed check), additionally require
    # the run's most recent step to be older than _QUIET_PERIOD_SECONDS before
    # declaring complete.  This is now the EXCEPTION path: all four
    # CLI-originated run types log a real manifest (see _MANIFEST_TOOL_NAMES)
    # and skip this entirely — their check is structural.  The fallback stays
    # only as a defensive last resort for a run type that does not log a
    # manifest yet (direct API use, a future CLI).  A taskmaster-agent run
    # does NOT hit this path — it inherits a manifest from its import stage
    # (see the CORRECTION comment two blocks up) and so is NOT covered by
    # this fallback either; its later-stage completeness gap is a separate,
    # already-flagged, pre-existing issue.
    if manifest_target_ids is None:
        complete = complete and _quiet_period_elapsed(conn, run_id)

    return RunStepsApiResponse(
        run_id=run_id,
        steps=[RunStepSummary(**dict(row)) for row in step_rows],
        target_states=target_states,
        complete=complete,
    )


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build the console app. Kept as a factory so uvicorn points at the
    module-level instance below while tests could build a fresh one."""
    # docs_url/redoc_url/openapi_url disabled so the app exposes exactly the
    # routes written here and nothing else — no auto-generated surface that
    # might imply capabilities the console does not have.  (B4b: the title
    # no longer claims read-only — the review gate is one deliberate write
    # door, everything else is still SELECT-only.)
    # Auth is wired GLOBALLY, not per-route (ticket H11).  The property this
    # buys is the one that matters: a route added to this file in six months
    # is protected AUTOMATICALLY, because the dependency runs for every route
    # on the app.  Per-route decoration would fail open for anything a future
    # edit forgets.  The one carve-out (/_health, for Cloud Run's health
    # check) lives INSIDE require_operator — the dependency still runs, it
    # just allows that exact path — so the carve-out cannot be widened by
    # adding an unprotected route here.  The path is /_health, not /healthz:
    # Google's Cloud Run frontend intercepts the exact path /healthz before
    # it reaches the container (ticket H16), so the conventional name is
    # unreachable in production.
    app = FastAPI(
        title="Outbound Agency Console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        dependencies=[Depends(require_operator)],
    )

    @app.get("/_health")
    def health() -> dict[str, str]:
        # Deliberately NO database access on this route: Cloud Run's health
        # check hits /_health (ticket A5b), and a health endpoint that fails
        # whenever the database is briefly unreachable gets the whole
        # revision killed and replaced — turning a transient database blip
        # into a full outage. "Process alive" must be reportable independent
        # of database availability; database problems are degradation, not
        # death. tests/test_console.py proves this route never connects.
        #
        # WHY /_health and not /healthz (ticket H16): Google's Cloud Run
        # frontend intercepts the EXACT, case-sensitive path /healthz and
        # answers it itself (an HTML 404 with NO `server: Google Frontend`
        # header) before the request reaches the container — measured against
        # the live service on 2026-08-28.  /_health is one of the names
        # measured to reach the app (when absent it 404s WITH the `server:
        # Google Frontend` header, proving the container answered).  Do NOT
        # "restore the conventional name": it is unreachable in production
        # and the health check would silently stop working.  The guard in
        # tests/test_console_auth.py asserts no /healthz route exists.
        return {"status": "ok"}

    @app.get("/")
    def index(request: Request, conn: Conn = Depends(get_conn)) -> HTMLResponse:
        # Landing view: every target, newest-updated first. The joins pull
        # the company name and offer slug so each row is self-describing
        # without a query per row. `?` placeholders are the repo convention
        # (app/db.py rewrites them for postgres).
        rows = conn.execute(
            """
            SELECT t.target_id, a.company_name, o.slug AS offer_slug,
                   t.state, t.score, t.final_recommendation, t.updated_at
            FROM targets t
            JOIN accounts a ON t.account_id = a.account_id
            JOIN offers o ON t.offer_id = o.offer_id
            ORDER BY t.updated_at DESC
            """
        ).fetchall()
        # Convert rows to plain dicts — Jinja attribute access and JSON
        # serialization both need dicts, not sqlite3.Row objects.
        target_list = [dict(row) for row in rows]
        return templates.TemplateResponse(
            request, "targets.html", {
                "targets": target_list,
                # D3a: the DEMO DATA banner flag — True when the database
                # is a seeded demo one (one SELECT, see _is_demo_database).
                "demo_data": _is_demo_database(conn),
                # D1: the REPLAY MODE banner flag — a pure env read, no query
                # (see _replay_mode); never touches _is_demo_database.
                "replay_mode": _replay_mode(),
            }
        )

    @app.get("/demo")
    def demo_showcase(request: Request, conn: Conn = Depends(get_conn)) -> HTMLResponse:
        # One screen for the moments worth showing a judge: links to the four
        # real showcase targets (never re-queried in depth here — each links
        # out to its own real /targets or /review page for the full audit
        # trail) and the most recently reserved real meeting plus the
        # follow-up draft whose footer states it. Pure convenience for the
        # operator running a live demo; adds no write path, no new app.*
        # import, and no data that isn't already visible elsewhere in the
        # console.
        payload = _fetch_demo_showcase(conn)
        return templates.TemplateResponse(
            request, "demo.html", {
                "showcase": payload["showcase"],
                "meeting": payload["meeting"],
                "meeting_draft": payload["meeting_draft"],
                "demo_data": _is_demo_database(conn),
                "replay_mode": _replay_mode(),
            }
        )

    @app.get("/rules")
    def rules(request: Request) -> HTMLResponse:
        # The one-screen rulebook (operator request, 2026-08-30): the scoring
        # formula, the policy gate, the state machine, and the capability
        # boundaries, in one place instead of eight docs/*.md files a judge
        # is unlikely to open. NO conn dependency at all — this route opens
        # no database connection, because the page renders no query: every
        # number in rules.html was copied by hand from the real source and
        # cross-checked against it (see the template's own header comment
        # for exactly which files and why this isn't a live import instead —
        # short version: importing app.tools.score_lead or app.state_machine
        # here would be a real app.* import into console code, which is
        # precisely what the console's audited import allowlist test exists
        # to catch). No demo_data/replay_mode banners either — those flag
        # whether TARGET DATA on a page is seeded or replayed, and this page
        # shows no target data at all.
        return templates.TemplateResponse(request, "rules.html", {})

    @app.get("/targets/{target_id}")
    def target_detail(
        request: Request, target_id: str, conn: Conn = Depends(get_conn)
    ) -> HTMLResponse:
        # One target's full audit trail as HTML (sections 1-7).
        detail = _fetch_target_detail(conn, target_id)
        if detail is None:
            # Unknown target: an explicit 404 with a readable message —
            # never a 500, never a blank page (CLAUDE.md §7).
            raise HTTPException(
                status_code=404, detail=f"unknown target {target_id!r}"
            )
        # The trace section renders pretty-printed JSON; every other section
        # renders its stored values as-is.
        display_steps = [
            {
                **step,
                "input_json": _pretty_json(step["input_json"]),
                "output_json": _pretty_json(step["output_json"]),
            }
            for step in detail["steps"]
        ]
        return templates.TemplateResponse(
            request, "target_detail.html", {
                **detail,
                "steps": display_steps,
                # D3a: the DEMO DATA banner flag, same one-SELECT detection
                # as the index route.
                "demo_data": _is_demo_database(conn),
                # D1: the REPLAY MODE banner flag, same pure env read as the
                # index route.
                "replay_mode": _replay_mode(),
            }
        )

    @app.get("/api/targets/{target_id}", response_model=TargetDetail)
    def target_detail_api(
        target_id: str, conn: Conn = Depends(get_conn)
    ) -> TargetDetail:
        # The same data as the detail page, as JSON — the demo/debugging
        # surface. Stored JSON strings are passed through unparsed so the API
        # shows exactly what the database holds.
        detail = _fetch_target_detail(conn, target_id)
        if detail is None:
            # Same 404 contract as the HTML route.
            raise HTTPException(
                status_code=404, detail=f"unknown target {target_id!r}"
            )
        return TargetDetail(**detail)

    # ── Review routes (ticket B4b — the approval gate's surface) ───────────
    # The review UI is the SOLE approval channel in v1 (docs/human-review.md
    # §4): these routes are where the operator's decision is recorded.  The
    # routes themselves hold no write logic — every decision goes through
    # app.review.record_review_decision, and the toggle through
    # app.kill_switch.write_kill_switch.

    @app.get("/review/queue")
    def review_queue(request: Request, conn: Conn = Depends(get_conn)) -> HTMLResponse:
        # The queue page: everything awaiting review, plus the kill-switch
        # indicator and toggle (read UNCACHED on every page view — the
        # operator must see the switch's current state, not a cached one).
        targets = _pending_review_targets(conn)
        return templates.TemplateResponse(
            request,
            "review_queue.html",
            {
                "targets": targets,
                "kill_switch": read_kill_switch(),
                "valid_decisions": VALID_DECISIONS,
                # D3a: the DEMO DATA banner flag — the review surface is
                # exactly where the operator must see it (a seeded
                # approval is not a real one, however genuine the gate).
                "demo_data": _is_demo_database(conn),
                # D1: the REPLAY MODE banner flag, same pure env read.
                "replay_mode": _replay_mode(),
            },
        )

    @app.get("/api/review/queue", response_model=ReviewQueueApiResponse)
    def review_queue_api(conn: Conn = Depends(get_conn)) -> ReviewQueueApiResponse:
        # The JSON form of the queue (docs/api.md §4): the FULL review
        # payload per pending target — research summary, ICP assessment,
        # signals, every draft revision with its critique, policy decision,
        # risk flags, suppression status — so the API consumer needs no
        # second call to render a decision screen.
        targets = [
            _fetch_review_payload(conn, row["target_id"])
            for row in _pending_review_targets(conn)
        ]
        # _fetch_review_payload returns None only for a target that vanished
        # between the two reads — drop rather than crash (read-only view).
        return ReviewQueueApiResponse(
            targets=[payload for payload in targets if payload is not None],
            kill_switch=read_kill_switch().model_dump(),
        )

    @app.get("/api/review/queue/count", response_model=ReviewQueueCountApiResponse)
    def review_queue_count_api(conn: Conn = Depends(get_conn)) -> ReviewQueueCountApiResponse:
        # The U3 badge endpoint: how many targets are awaiting review RIGHT NOW.
        # SELECT-only, and deliberately CHEAP — it reuses _pending_review_targets
        # (the exact query the queue page runs; never duplicated) and returns
        # only len(...).  A background poll running on every page (base.html's
        # header badge) must not pull the FULL review payload the /api/review/queue
        # route returns (LLM-derived research summaries, draft revisions,
        # critiques) just to render an integer.  Rides the SAME global
        # require_operator dependency as every other route — no auth bypass.
        return ReviewQueueCountApiResponse(count=len(_pending_review_targets(conn)))

    # ── Live run view (ticket U2) ─────────────────────────────────────────
    # The judge-facing "watch the pipeline move in real time" surface.  Two
    # routes: a SELECT-only JSON endpoint the page polls, and the HTML shell
    # that hosts the (deliberately first-in-this-codebase) inline client-side
    # script.  Both go through the SAME global require_operator dependency as
    # every other route — U2 adds no auth bypass and no write door.

    @app.get("/api/run/{run_id}/steps", response_model=RunStepsApiResponse)
    def run_steps_api(run_id: str, conn: Conn = Depends(get_conn)) -> RunStepsApiResponse:
        # The polling endpoint (ticket U2): every step logged under one run_id,
        # the current state of each target the run touched, and whether the run
        # has reached a stop condition.  SELECT-only, like every console query.
        # An unknown run_id is an EMPTY list with 200, never a 404 — a live run
        # legitimately starts with zero steps, and the page must be able to
        # poll from the very beginning without erroring.
        return _fetch_run_steps(conn, run_id)

    @app.get("/run/{run_id}")
    def run_view(request: Request, run_id: str, conn: Conn = Depends(get_conn)) -> HTMLResponse:
        # The live-run HTML shell (ticket U2): a server-rendered page whose
        # inline <script> polls /api/run/{run_id}/steps on an interval and
        # renders each step as it lands.  The page must work correctly with
        # zero steps yet — "waiting for the run to start" is a legitimate,
        # non-error initial state (run.html renders that state, never a
        # loading spinner masking a bug).
        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "run_id": run_id,
                # D3a: the DEMO DATA banner flag, same one-SELECT detection as
                # every other route.
                "demo_data": _is_demo_database(conn),
                # D1: the REPLAY MODE banner flag, same pure env read.
                "replay_mode": _replay_mode(),
            },
        )

    @app.get("/review/{target_id}")
    def review_target(
        request: Request, target_id: str, conn: Conn = Depends(get_conn)
    ) -> HTMLResponse:
        # One target's review screen: the full payload (company, ICP,
        # signals, policy, suppression status) and EVERY draft revision in
        # order with the critique that produced each — the "draft diff
        # across iterations" the plan row asks for, made legible so the
        # operator can watch the agent improve its own work.
        payload = _fetch_review_payload(conn, target_id)
        if payload is None:
            # Unknown target: the same 404 contract as the A5a routes.
            raise HTTPException(
                status_code=404, detail=f"unknown target {target_id!r}"
            )
        # A target no longer in awaiting_review (a double-submitted form
        # redirects here) still renders its payload — the page shows the
        # "already decided" notice instead of the decision form, so the
        # second submit is a visible non-event, never a 500.
        # The post-decision redirect carries outcome/refused in the query
        # string; they are passed through so the page renders the result
        # banner (autoescaped like every other value).
        return templates.TemplateResponse(
            request,
            "review_target.html",
            {
                "review": payload,
                "kill_switch": read_kill_switch(),
                "valid_decisions": VALID_DECISIONS,
                "outcome": request.query_params.get("outcome"),
                "refused": request.query_params.get("refused"),
                # D3a: the DEMO DATA banner flag — the per-target review
                # screen is the approval surface itself.
                "demo_data": _is_demo_database(conn),
                # D1: the REPLAY MODE banner flag, same pure env read.
                "replay_mode": _replay_mode(),
            },
        )

    @app.post("/review/decision", response_model=ReviewOutcome)
    def review_decision_api(
        payload: ReviewDecisionApiRequest, conn: Conn = Depends(get_conn)
    ) -> ReviewOutcome:
        # The JSON decision endpoint (docs/api.md §4): records the decision
        # through the review gate and returns the outcome.  A refusal is a
        # 200 with refused=True (an observable outcome, never a 500) — the
        # HTTP status is reserved for transport problems, not gate verdicts.
        return record_review_decision(
            conn,
            request=ReviewDecisionRequest(
                target_id=payload.target_id,
                decision=payload.decision,
                reason=payload.reason,
                edited_subject=payload.edited_subject,
                edited_body=payload.edited_body,
                research_note=payload.research_note,
            ),
            run_id=payload.run_id,
        )

    @app.post("/review/{target_id}/decision")
    def review_decision_form(
        target_id: str,
        decision: str = Form(...),  # required: one of the five — the service refuses anything else
        reason: str = Form(""),
        run_id: str = Form(""),  # the hidden field the review page fills from the payload
        edited_subject: str = Form(""),
        edited_body: str = Form(""),
        research_note: str = Form(""),
        conn: Conn = Depends(get_conn),
    ):
        # The HTML form endpoint: the review page's five decision buttons
        # POST here.  The target_id comes from the PATH, not a form field —
        # a form can never decide a different target than the one on screen.
        # Redirect (303) back to the review page with the outcome in the
        # query string, so the operator sees the result banner — refusals
        # included, with the reason (never a silent no-op).
        outcome = record_review_decision(
            conn,
            request=ReviewDecisionRequest(
                target_id=target_id,
                decision=decision,
                reason=reason,
                # Empty form fields map to None: "no edits supplied" is not
                # the same as an empty-string edit (the service refuses an
                # approve_with_edits with no usable text).
                edited_subject=edited_subject or None,
                edited_body=edited_body or None,
                research_note=research_note or None,
            ),
            run_id=run_id,
        )
        if outcome.refused:
            return RedirectResponse(
                f"/review/{target_id}?{urlencode({'refused': outcome.refusal_reason})}",
                status_code=303,
            )
        return RedirectResponse(
            f"/review/{target_id}?{urlencode({'outcome': f'{outcome.decision} recorded — new state: {outcome.new_state}'})}",
            status_code=303,
        )

    @app.post("/kill-switch")
    def kill_switch_toggle(engaged: str = Form(...)):
        # The toggle (ticket B4b): flips the switch by REWRITING the file —
        # the file stays the single source of truth (runbook.md §1); the
        # console is just the second way to flip it.  updated_by is the
        # operator: the file's own metadata records who flipped it and
        # when, so the halt messages and the trace name the actor.
        # Strict parse of the form value — anything but "true"/"false" is
        # a 422, never a guessed state (the HTML radios only emit these
        # two, so a third value is a malformed request, not a decision).
        if engaged == "true":
            value = True
        elif engaged == "false":
            value = False
        else:
            raise HTTPException(
                status_code=422, detail="engaged must be 'true' or 'false'"
            )
        write_kill_switch(engaged=value, updated_by="operator")
        # Redirect back to the queue so the always-visible indicator shows
        # the post-toggle state on the next page view (read uncached).
        return RedirectResponse("/review/queue", status_code=303)

    return app


# Module-level instance for `uvicorn app.console.app:app` — the standard
# ASGI entry-point spelling for the A5b deploy.
app = create_app()
