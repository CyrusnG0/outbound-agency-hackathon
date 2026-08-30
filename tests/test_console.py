"""
Tests for the operator console (ticket A5a: read-only audit views; ticket
B4b: the review gate routes, the kill-switch toggle, the strengthened import
allowlist).

Two layers of guarantee are tested here:

1. Behaviour — the audit-trail routes serve the seeded data, return 404 for
   unknown targets, tolerate a NULL contact_id (company-only leads), and
   escape stored XSS payloads; the B4b review routes list the queue, show
   the draft diff, record decisions, and toggle the kill switch.
2. Structure — app/console/ is parsed with ``ast``: its SQL string literals
   must all be SELECT (unchanged since A5a), and its app.* imports must be
   an ALLOWLIST of exactly app.db / app.review / app.kill_switch (B4b —
   converted from a denylist, which would silently pass a future write
   module).  A behavioural test could pass a console that imports a write
   module and simply has not been asked to write yet; the structural tests
   fail the moment such an import appears, so the console is unable to write
   in its own code even in principle.

Seeding here goes through write_gate.commit / log_step on purpose: the
*console* may not write, but the test fixture is a normal pipeline write.
"""

import ast
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agents_registry import seed_agent_registry
from app.console.app import app
from app.db import apply_schema, connect
from app.state_machine import transition
from app.tools.log_step import log_step
from app.write_gate import commit

# The XSS payload used to prove autoescape is a live security control:
# company_summary holds LLM output derived from scraped third-party pages
# (docs/threat-model.md, policy rule P8), so a stored script tag must render
# as inert text in the operator's own console.
XSS_PAYLOAD = "<script>alert('xss')</script>"

# U1's new review-page fields (ticket U1b): three DISTINCT stored-XSS payloads
# — one per LLM-derived field — so a regression on any single field pinpoints
# WHICH one, instead of a combined check that could pass for the wrong reason.
# Each payload names its own field in the alert() so a failing assertion
# (or a peek at the rendered body) identifies the offender at a glance.
U1_ICP_FIT_REASONS_PAYLOAD = "<script>alert('icp_fit_reasons')</script>"
U1_JUDGE_RATIONALE_PAYLOAD = "<script>alert('judge_rationale')</script>"
U1_EVIDENCE_QUOTE_PAYLOAD = "<script>alert('evidence_quote')</script>"


# The secret the behavioural fixtures use.  It only needs to be a value the
# fixture's OWN authed client sends; the auth behaviour itself (wrong key,
# no key, 503, ...) is tested in tests/test_console_auth.py.
_CONSOLE_TEST_SECRET = "test-console-secret"


class _AuthedTestClient(TestClient):
    """A TestClient that sends a valid operator credential on EVERY request
    (ticket H11).

    The console now requires authentication, and these behavioural tests are
    about console behaviour, not about auth — so the fixture injects the
    documented X-Internal-API-Key header transparently instead of adding a
    headers= argument to every call site.  The auth layer itself is exercised
    on its own terms in tests/test_console_auth.py; here the credential is
    just the key that opens the door."""

    def request(self, method, url, **kwargs):
        # Ensure the request carries a valid credential regardless of how the
        # test called it (get/post with or without explicit headers).
        headers = kwargs.get("headers") or {}
        headers.setdefault("X-Internal-API-Key", _CONSOLE_TEST_SECRET)
        kwargs["headers"] = headers
        return super().request(method, url, **kwargs)


@pytest.fixture
def db_path(tmp_path):
    """Build a temp SQLite DB the same way the pipeline would (the fixture
    pattern from tests/test_score_lead.py): schema + agent registry + core
    rows through the write gate. Returns the path; the console connects to it
    per request via OUTBOUND_DB_TARGET."""
    path = str(tmp_path / "test.db")
    conn = connect(path)
    apply_schema(conn)
    # Register the system agent (plan A3) — commit() refuses unregistered agents.
    seed_agent_registry(conn, run_id="r0", step_id="s0")

    commit(
        conn, action="insert_offer", table_name="offers", record_id="off_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "therapy-app", 1),
    )
    commit(
        conn, action="insert_account", table_name="accounts", record_id="acc_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts
               (account_id, company_name, domain, normalized_domain, industry,
                estimated_size, geo, company_summary, icp_fit_label, icp_fit_score,
                icp_fit_reasons, icp_non_fit_reasons, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_1", "Acme Therapeutics", "acme.test", "acme.test", "Healthcare",
                "51-200", "US", "A healthcare company.", "strong_fit", 88,
                '["hiring_relevant_role present"]', '[]'),
    )
    commit(
        conn, action="insert_contact", table_name="contacts", record_id="con_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts
               (contact_id, account_id, full_name, title, seniority, department,
                email, email_verified, linkedin_url, persona_fit_score,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("con_1", "acc_1", "Jane Doe", "CTO", "executive", "Engineering",
                "jane@acme.test", 1, None, 15),
    )
    commit(
        conn, action="insert_target", table_name="targets", record_id="tgt_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets
               (target_id, account_id, contact_id, offer_id, source, state, score,
                final_recommendation, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("tgt_1", "acc_1", "con_1", "off_1", "csv", "scored", 78, "send"),
    )
    # Second target with contact_id NULL — company-only leads are explicitly
    # allowed by CSV import, and the console must render them without crashing.
    commit(
        conn, action="insert_target", table_name="targets", record_id="tgt_2",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets
               (target_id, account_id, contact_id, offer_id, source, state,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("tgt_2", "acc_1", None, "off_1", "csv", "researched"),
    )
    commit(
        conn, action="insert_signal", table_name="signals", record_id="sig_1",
        payload={}, run_id="r0", step_id="s1", actor="system", agent_id="system",
        # B2b: the evidence columns are seeded too — evidence_quote carries
        # the XSS payload on purpose, so the detail-page test can prove the
        # autoescape control holds on the new column (signal_value AND
        # evidence_quote are scraped third-party text / LLM output).
        sql="""INSERT INTO signals
               (signal_id, run_id, target_id, signal_type, signal_value,
                signal_strength, source_url, source_confidence,
                evidence_quote, evidence_verified, evidence_tier, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("sig_1", "r0", "tgt_1", "hiring_relevant_role", "Hiring 3 SDRs",
                1.0, "https://acme.test/careers", 0.9,
                XSS_PAYLOAD, 0, "findings"),
    )
    commit(
        conn, action="insert_policy_decision", table_name="policy_decisions",
        record_id="pd_1", payload={}, run_id="r0", step_id="s1",
        actor="system", agent_id="system",
        sql="""INSERT INTO policy_decisions
               (policy_decision_id, run_id, step_id, target_id, action, decision,
                risk_level, reasons_json, matched_rules_json, missing_fields_json,
                created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("pd_1", "r0", "s1", "tgt_1", "send_email", "allow", "low",
                '["target scored above threshold"]', '["P1"]', '[]'),
    )
    commit(
        conn, action="state_transition", table_name="state_transitions",
        record_id="tr_1", payload={}, run_id="r0", step_id="s1",
        actor="system", agent_id="system",
        sql="""INSERT INTO state_transitions
               (transition_id, run_id, step_id, target_id, previous_state,
                new_state, reason, actor, matched_policy_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("tr_1", "r0", "s1", "tgt_1", "researched", "scored",
                "score >= 60 after research", "system", None),
    )
    # Steps rows go through log_step, the trace log's normal writer (steps is
    # deliberately outside the write gate — see app/tools/log_step.py).
    log_step(
        conn, run_id="r0", step_id="s1", target_id="tgt_1",
        tool_name="score_lead", agent_id="system",
        input_data={"target_id": "tgt_1"},
        output_data={"fit_score": 78},
        status="success", model_call_hash="mc_hash_1",
    )
    conn.close()
    return path


@pytest.fixture
def client(db_path, monkeypatch):
    """Point the console at the seeded temp DB via the repo's env-var
    convention; app/console/app.py reads OUTBOUND_DB_TARGET per request.
    Also sets the auth secret (ticket H11): the console now FAILS CLOSED
    without OUTBOUND_CONSOLE_API_KEY (503 on every route except /_health),
    so the server must have a secret configured.  The _AuthedTestClient then
    sends that secret on every request.  The auth behaviour itself is tested
    in tests/test_console_auth.py; here the secret is just so these
    behavioural tests can get past the door."""
    monkeypatch.setenv("OUTBOUND_DB_TARGET", db_path)
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", _CONSOLE_TEST_SECRET)
    # D1: the replay banner must default to ABSENT even when the developer's
    # shell happens to export OUTBOUND_REPLAY_MODE — these behavioural tests
    # are about console behaviour, not about the operator's env.
    monkeypatch.delenv("OUTBOUND_REPLAY_MODE", raising=False)
    return _AuthedTestClient(app)


# ── Behaviour tests ───────────────────────────────────────────────────────────


def test_health_returns_ok_without_touching_database(tmp_path, monkeypatch):
    # Point the DB target at a path that does not exist AND whose parent
    # directory does not exist, so even sqlite's implicit file creation would
    # fail. If /_health touched the database at all, this request would blow
    # up or create the file — Cloud Run's health check (A5b) must report
    # process-alive independent of database availability.
    nonexistent = str(tmp_path / "no_such_dir" / "no_such.db")
    monkeypatch.setenv("OUTBOUND_DB_TARGET", nonexistent)
    client = TestClient(app)
    resp = client.get("/_health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    # Proving non-contact: a connection attempt would have created the file.
    assert not Path(nonexistent).exists()


def test_index_lists_targets_with_links(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    # The seeded company name, its state, and a link to its detail page.
    assert "Acme Therapeutics" in body
    assert "scored" in body
    assert "/targets/tgt_1" in body


def test_target_detail_shows_full_audit_trail(client):
    resp = client.get("/targets/tgt_1")
    assert resp.status_code == 200
    body = resp.text
    assert "78" in body                      # targets.score
    assert "strong_fit" in body              # accounts.icp_fit_label
    assert "hiring_relevant_role" in body    # signals.signal_type
    assert "Hiring 3 SDRs" in body           # signals.signal_value
    assert "allow" in body                   # policy_decisions.decision
    # B2b: the signal's three-way evidence verdict must be visible.
    assert "findings" in body                # signals.evidence_tier
    # B2b: evidence_quote is scraped/LLM text — the A5a autoescape control
    # must hold on the NEW column exactly as it does on company_summary: the
    # raw script tag must never render, only its escaped form.
    assert XSS_PAYLOAD not in body
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in body
    # state_transitions.reason — note ">" is autoescaped to "&gt;", which is
    # the security control doing its job, so assert the escaped form.
    assert "score &gt;= 60 after research" in body
    assert "score_lead" in body              # steps.tool_name
    assert "mc_hash_1" in body               # steps.model_call_hash


def test_target_detail_watch_run_link_skips_empty_run_id(client, db_path):
    # D2 follow-up regression: app/review.py's review_decision step is
    # legally logged with run_id="" (app/console/app.py:531 — the queue
    # payload may carry no run_id), so a target's steps can include one
    # falsy run_id alongside real ones. Before the fix, `unique` kept ""
    # as its own distinct entry and the template rendered a dead
    # `<a href="/run/">` link (a 404 on click). Seed exactly that shape —
    # tgt_1 already has a real run_id="r0" step from the client fixture —
    # and assert the empty-run_id step never produces its own link while
    # the real run_id's link still renders.
    conn = connect(db_path)
    log_step(
        conn, run_id="", step_id="s_review_decision", target_id="tgt_1",
        tool_name="review_decision", agent_id="operator",
        input_data={"stage": "review_decision"}, output_data={},
        status="success",
    )
    conn.close()

    resp = client.get("/targets/tgt_1")
    assert resp.status_code == 200
    body = resp.text
    # The real run still gets its link…
    assert 'href="/run/r0"' in body
    # …but the falsy run_id never produces a link with nothing after
    # "/run/" — this is the exact dead link the D2 live run surfaced.
    assert 'href="/run/"' not in body


def test_unknown_target_is_404_with_message(client):
    resp = client.get("/targets/no_such_target")
    assert resp.status_code == 404
    # A readable message naming the id — not a stack trace, not a blank page.
    assert "no_such_target" in resp.text


def test_target_without_contact_renders(client):
    # tgt_2 has contact_id NULL (company-only lead, allowed by CSV import).
    resp = client.get("/targets/tgt_2")
    assert resp.status_code == 200
    assert "No contact data for this target." in resp.text


def test_stored_xss_in_company_summary_is_escaped(client, db_path):
    # Seed an account whose company_summary is a script tag — the exact
    # stored-XSS shape Part 3 of the ticket is about — then assert the
    # console renders it as inert escaped text, never as executable markup.
    conn = connect(db_path)
    commit(
        conn, action="insert_account", table_name="accounts", record_id="acc_xss",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts
               (account_id, company_name, domain, normalized_domain, company_summary,
                created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_xss", "XSS Co", "xss.test", "xss.test", XSS_PAYLOAD),
    )
    commit(
        conn, action="insert_target", table_name="targets", record_id="tgt_xss",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets
               (target_id, account_id, offer_id, source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("tgt_xss", "acc_xss", "off_1", "csv", "researched"),
    )
    conn.close()

    resp = client.get("/targets/tgt_xss")
    assert resp.status_code == 200
    body = resp.text
    # The raw payload must not appear anywhere in the response body…
    assert XSS_PAYLOAD not in body
    # …while its HTML-escaped form (markupsafe's escaping, verified against
    # the installed jinja2) must appear in its place.
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in body


def test_judge_divergence_renders_with_escaped_rationale(client, db_path):
    """B2c: a judge verdict that DIVERGES from the deterministic label is
    the most interesting thing on the page — it must render both labels and
    the justification, and the judge's LLM output (rationale + justification)
    must stay autoescaped: no |safe anywhere near it."""
    # Seed the judge columns on the existing fixture account with a diverging
    # verdict, through the write gate (the fixture's own convention).
    conn = connect(db_path)
    commit(
        conn, action="update_account_icp_verdict", table_name="accounts",
        record_id="acc_1", payload={}, run_id="r0", step_id="s0",
        actor="system", agent_id="icp_judge",
        sql="""UPDATE accounts SET judge_fit_label=?, judge_rationale=?,
               judge_divergence_justification=?, updated_at=datetime('now')
               WHERE account_id=?""",
        # The rationale and justification carry the XSS payload on purpose:
        # they are LLM output and must render as inert text.
        params=("good_fit", XSS_PAYLOAD, XSS_PAYLOAD, "acc_1"),
    )
    conn.close()

    resp = client.get("/targets/tgt_1")
    assert resp.status_code == 200
    body = resp.text
    # The judge's label and the deterministic one are both visible…
    assert "good_fit" in body
    assert "strong_fit" in body  # the deterministic label, shown alongside
    # …and the divergence block is rendered (the ticket: "a divergence is
    # the most interesting thing on the page; make it legible").
    assert "Judge diverged from the deterministic score" in body
    # The LLM output is autoescaped: the raw payload never renders.
    assert XSS_PAYLOAD not in body
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in body


def test_api_target_returns_json(client):
    resp = client.get("/api/targets/tgt_1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target"]["state"] == "scored"
    assert data["target"]["score"] == 78
    assert data["company"]["company_name"] == "Acme Therapeutics"
    assert data["contact"]["email"] == "jane@acme.test"
    assert data["signals"][0]["signal_type"] == "hiring_relevant_role"
    assert data["policy_decisions"][0]["decision"] == "allow"
    assert data["state_transitions"][0]["new_state"] == "scored"
    # The API passes stored JSON strings through unparsed: raw debugging data.
    assert data["steps"][0]["tool_name"] == "score_lead"
    assert data["steps"][0]["input_json"] == '{"target_id": "tgt_1"}'


def test_api_unknown_target_is_404(client):
    resp = client.get("/api/targets/no_such_target")
    assert resp.status_code == 404


# ── /demo: the one-screen live-demo jumping-off page (operator request, 2026-08-30) ──


def test_demo_page_degrades_cleanly_when_no_showcase_names_match(client):
    # The base db_path fixture seeds "Acme Therapeutics", not any of the
    # five hardcoded showcase company names — this proves the
    # mark-don't-drop path: a showcase name absent from the database
    # renders an explicit "not available" cell instead of a KeyError/500,
    # and the meeting/draft sections render their own explicit empty
    # states when no meetings row exists at all.
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert resp.text.count("not available in this database") == 5
    assert "No meeting reserved in this database yet" in resp.text
    assert "No follow-up draft found for this meeting's target" in resp.text


def test_demo_page_shows_real_showcase_links_and_scheduled_meeting(client, db_path):
    # A second, targeted seed: one showcase name resolved to a target still
    # awaiting_review, a real meetings row reserved for it, and the
    # follow-up draft whose footer states that reservation.
    conn = connect(db_path)
    commit(
        conn, action="insert_account", table_name="accounts", record_id="acc_sol",
        payload={}, run_id="r1", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts
               (account_id, company_name, domain, normalized_domain, industry,
                created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_sol", "Solacetree Counselling Limited", "solacetree.test",
                "solacetree.test", "Healthcare"),
    )
    commit(
        conn, action="insert_target", table_name="targets", record_id="tgt_sol",
        payload={}, run_id="r1", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets
               (target_id, account_id, contact_id, offer_id, source, state,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("tgt_sol", "acc_sol", None, "off_1", "csv", "awaiting_review"),
    )
    # The real meetings row — the reservation the page's "The real
    # scheduled meeting" section reads.
    commit(
        conn, action="insert_meeting", table_name="meetings", record_id="mtg_1",
        payload={}, run_id="r1", step_id="s0", actor="system", agent_id="meeting_scheduler",
        sql="""INSERT INTO meetings
               (meeting_id, target_id, account_id, contact_id, company_name,
                contact_name, scheduled_at, duration_minutes, status,
                reasoning, proposed_by, run_id, step_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("mtg_1", "tgt_sol", "acc_sol", None, "Solacetree Counselling Limited",
                None, "2026-09-01T10:30:00+08:00", 15, "proposed",
                "earliest available slot", "meeting_scheduler", "r1", "s0"),
    )
    # Revision 1: no scheduling line — must NOT be the row the page picks.
    commit(
        conn, action="insert_message_draft_version", table_name="message_draft_versions",
        record_id="dv_sol1", payload={}, run_id="r1", step_id="s0",
        actor="system", agent_id="draft_writer",
        sql="""INSERT INTO message_draft_versions
               (draft_version_id, target_id, message_id, revision_number, subject,
                body, footer, edited_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("dv_sol1", "tgt_sol", None, 1, "First subject", "First body.",
                "[unsubscribe: {UNSUBSCRIBE_URL}]", "draft_writer"),
    )
    # Revision 2: the real follow-up shape — unsubscribe footer PLUS the
    # real scheduled-meeting line, exactly as _compose_footer builds it.
    commit(
        conn, action="insert_message_draft_version", table_name="message_draft_versions",
        record_id="dv_sol2", payload={}, run_id="r1", step_id="s0",
        actor="system", agent_id="draft_writer",
        sql="""INSERT INTO message_draft_versions
               (draft_version_id, target_id, message_id, revision_number, subject,
                body, footer, edited_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("dv_sol2", "tgt_sol", None, 2, "Second subject", "Second body.",
                "[unsubscribe: {UNSUBSCRIBE_URL}] We've held Tuesday, Sep 1 at "
                "10:30 HKT for a 15-min call — reply to confirm or suggest "
                "another time. (Reference: "
                "https://booking.outbound-agency.test/confirm/mtg_1)",
                "draft_writer"),
    )
    conn.close()

    # `client` already points OUTBOUND_DB_TARGET at this same db_path (the
    # fixture dependency chain: client depends on db_path) — the rows just
    # inserted above are visible to it without any extra wiring.
    resp = client.get("/demo")
    assert resp.status_code == 200
    # The awaiting_review showcase target links to the review/decision
    # screen, not the plain audit-trail page.
    assert "/review/tgt_sol" in resp.text
    # The real reservation's fields render on the meeting section.
    assert "2026-09-01T10:30:00+08:00" in resp.text
    assert "earliest available slot" in resp.text
    # The draft shown is the SECOND revision's subject (the one whose
    # footer actually states the reservation), not the first's.
    assert "Second subject" in resp.text
    assert "We&#39;ve held" in resp.text or "We've held" in resp.text
    # The removed artifact link must never resurface anywhere on this page.
    assert "claude.ai" not in resp.text
    # Four showcase names still don't resolve in this database.
    assert resp.text.count("not available in this database") == 4


# ── Structural tests: the console cannot write, even in principle ─────────────


# Write-path modules the console must never import. "app.agents" and
# "app.tools" are prefixes, so anything under them (app.agents.phase1,
# app.tools.log_step, ...) is covered too.
_FORBIDDEN_IMPORT_ROOTS = (
    "app.write_gate",
    "app.state_machine",
    "app.policy",
    "app.llm",
    "app.agents",
    "app.tools",
)

# The ALLOWLIST half of the import guarantee (ticket B4b): the complete set
# of app.* modules app/console/ may import.  Today that is exactly:
# - app.db — the dialect-agnostic connection wrapper (A5a's original import)
# - app.review — the review gate the console CALLS; every write statement
#   lives in that module, never in the console's own code
# - app.kill_switch — the switch reader (the always-visible indicator) and
#   writer (the toggle); the write lives in that module
# - app.console.auth — the authentication dependency (ticket H11).  It is a
#   PURE credential check: it reads one env var and compares strings, with
#   no database, no SQL and no write path of any kind.  It is on the list
#   precisely because it is pure — the wall this test protects (the console
#   cannot write in its own code) is unchanged.  test_console_auth.py keeps
#   it honest: it asserts auth.py itself imports nothing from app.* and
#   contains no SQL keyword, so this entry cannot later become a back door.
# Anything else under app.* fails the allowlist test.  A denylist alone is
# a hole — a future write path not on the denylist (exactly what app.review
# is) could be imported and the test would pass silently while the
# read-only guarantee quietly died.  With the allowlist, adding ANY future
# import becomes a deliberate test edit — the A5a property, preserved
# rather than weakened.
_ALLOWED_IMPORT_ROOTS = ("app.db", "app.review", "app.kill_switch", "app.console.auth")

# SQL keywords that mutate data or schema. Word boundaries keep "updated_at"
# (a column name) from tripping the UPDATE check.
_SQL_WRITE_KEYWORD = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|UPSERT)\b",
    re.IGNORECASE,
)


def _console_python_files() -> list[Path]:
    # Every .py file under app/console/ — templates (.html) are excluded
    # because ast.parse only handles Python source.
    console_dir = Path(__file__).resolve().parent.parent / "app" / "console"
    assert console_dir.is_dir(), "app/console/ missing — did the package move?"
    files = sorted(console_dir.rglob("*.py"))
    assert files, "no .py files found under app/console/"
    return files


def _docstring_constant_ids(tree: ast.Module) -> set[int]:
    # A module/class/function's first statement, when it is a bare string, is
    # its docstring. Collect those Constant nodes' ids so the SQL-keyword
    # scan below skips prose: only string literals that could actually reach
    # the database are checked, and docstrings stay free to discuss write
    # statements by name.
    containers: list[Any] = [tree]
    containers.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    ids: set[int] = set()
    for container in containers:
        body = getattr(container, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def test_console_cannot_import_any_write_path():
    # Structural guarantee, not behavioural: walk every Import/ImportFrom
    # node in every app/console file and fail on any forbidden module. A
    # behavioural test could pass a console that imports write_gate and
    # simply has not been asked to write yet; this test fails the moment a
    # write path is importable, so the console is UNABLE to write even in
    # principle. ImportFrom names are joined with their module (e.g.
    # `from app import write_gate` becomes "app.write_gate") so both import
    # spellings are caught.
    #
    # TWO layers (ticket B4b): the DENYLIST below keeps the specific,
    # readable failure message for the most dangerous modules; the
    # ALLOWLIST is the real guarantee — any app.* import not on it fails,
    # so a future write module (exactly the hole a denylist leaves) cannot
    # sneak in.  Adding a new import to the console is a deliberate edit to
    # _ALLOWED_IMPORT_ROOTS.
    for path in _console_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in imported:
                # Layer 1 — the denylist (belt): the most dangerous write
                # paths get their own specific message.
                for forbidden in _FORBIDDEN_IMPORT_ROOTS:
                    assert not (
                        name == forbidden or name.startswith(forbidden + ".")
                    ), (
                        f"{path.name} imports {name!r} — forbidden write path "
                        f"{forbidden!r}"
                    )
                # Layer 2 — the allowlist (braces): every app.* import must
                # be one of the three sanctioned modules.  "app" alone (a
                # bare `import app`) is refused too — no module here needs
                # the package itself.
                if name == "app" or name.startswith("app."):
                    assert any(
                        name == allowed or name.startswith(allowed + ".")
                        for allowed in _ALLOWED_IMPORT_ROOTS
                    ), (
                        f"{path.name} imports {name!r} — not on the console "
                        f"import allowlist {_ALLOWED_IMPORT_ROOTS}; adding a "
                        f"console import is a deliberate test edit"
                    )


def test_console_auth_is_pure_no_app_imports_no_sql():
    # The allowlist entry for app.console.auth exists ONLY because that
    # module is pure (ticket H11): a credential check with no database, no
    # SQL, no write path of any kind.  This test pins that purity so the
    # entry cannot later become a back door — if someone adds a database
    # import or a SQL statement to auth.py, this test fails and forces the
    # question of whether auth still belongs on the console's allowlist.
    auth_path = Path(__file__).resolve().parent.parent / "app" / "console" / "auth.py"
    assert auth_path.is_file(), "app/console/auth.py missing — did the package move?"
    tree = ast.parse(auth_path.read_text(encoding="utf-8"), filename=str(auth_path))
    # No app.* import of any kind — same Import/ImportFrom walk as the
    # console allowlist test, applied to auth.py itself.  A bare `import app`
    # is refused too.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        for name in imported:
            assert not (name == "app" or name.startswith("app.")), (
                f"app/console/auth.py imports {name!r} — auth must stay pure "
                f"(no app.* imports) so its console allowlist entry is not a "
                f"back door"
            )
    # No SQL write/DDL keyword in any string constant (docstrings skipped by
    # the same helper the SELECT-only test uses) — auth.py must never contain
    # a statement that could reach a database.
    skip = _docstring_constant_ids(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
        ):
            assert not _SQL_WRITE_KEYWORD.search(node.value), (
                f"app/console/auth.py contains a write/DDL SQL keyword in a "
                f"string literal: {node.value!r}"
            )


def test_console_sql_is_select_only():
    # Second structural half of the read-only guarantee: walk every string
    # constant in app/console/*.py and refuse any mutating/DDL SQL keyword.
    # Comments are not AST nodes and docstrings are skipped (see
    # _docstring_constant_ids), so prose mentioning these words is fine —
    # only string literals, i.e. code that could reach the database, are
    # checked.
    for path in _console_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        skip = _docstring_constant_ids(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in skip
            ):
                assert not _SQL_WRITE_KEYWORD.search(node.value), (
                    f"{path.name} contains a write/DDL SQL keyword in a string "
                    f"literal: {node.value!r}"
                )


# ── Review gate routes (ticket B4b) ──────────────────────────────────────────
# The console's write door: every decision and the toggle go through
# app/review.py / app/kill_switch.py — the routes hold no SQL of their own
# (the SELECT-only test above proves it).  These tests prove the door works
# and that the review surface escapes stored XSS.


@pytest.fixture
def review_db(tmp_path):
    """A DB shaped like a post-B3 pipeline: one target in awaiting_review
    with two draft revisions (whose body and critique carry the XSS payload
    on purpose), plus a second awaiting_review target — so the queue,
    review page, and decision routes have realistic data.  Seeding goes
    through write_gate.commit / log_step like the db_path fixture."""
    path = str(tmp_path / "review.db")
    conn = connect(path)
    apply_schema(conn)
    seed_agent_registry(conn, run_id="r0", step_id="s0")
    commit(
        conn, action="insert_offer", table_name="offers", record_id="off_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "therapy-app", 1),
    )
    commit(
        conn, action="insert_account", table_name="accounts", record_id="acc_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts
               (account_id, company_name, domain, normalized_domain, industry,
                estimated_size, geo, company_summary, icp_fit_label, icp_fit_score,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_1", "Acme Therapeutics", "acme.test", "acme.test", "Healthcare",
                "51-200", "US", "A healthcare company.", "strong_fit", 88),
    )
    commit(
        conn, action="insert_contact", table_name="contacts", record_id="con_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts
               (contact_id, account_id, full_name, email, email_verified,
                created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("con_1", "acc_1", "Jane Doe", "jane@acme.test", 1),
    )
    for target_id in ("tgt_r1", "tgt_r2"):
        commit(
            conn, action="insert_target", table_name="targets", record_id=target_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO targets
                   (target_id, account_id, contact_id, offer_id, source, state,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(target_id, "acc_1", "con_1", "off_1", "csv", "awaiting_review"),
        )
    # Two draft revisions for tgt_r1 — the second carries the XSS payload in
    # BOTH the body and the critique (LLM output derived from scraped pages:
    # the exact stored-XSS shape the review page must escape).
    critique = json.dumps({
        "passed": False,
        "issues": [XSS_PAYLOAD],
        "required_changes": "Remove the script tag claim.",
        "severity": "major",
    })
    commit(
        conn, action="insert_message_draft_version", table_name="message_draft_versions",
        record_id="dv_1", payload={}, run_id="r0", step_id="s0",
        actor="system", agent_id="draft_writer",
        sql="""INSERT INTO message_draft_versions
               (draft_version_id, target_id, message_id, revision_number, subject,
                body, footer, edited_by, policy_check_passed, injection_scan_passed,
                send_gate_passed, critique_passed, critique_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("dv_1", "tgt_r1", None, 1, "First subject", "First body.",
                "[unsubscribe: {UNSUBSCRIBE_URL}]", "draft_writer",
                None, None, None, 0, critique),
    )
    commit(
        conn, action="insert_message_draft_version", table_name="message_draft_versions",
        record_id="dv_2", payload={}, run_id="r0", step_id="s0",
        actor="system", agent_id="draft_writer",
        sql="""INSERT INTO message_draft_versions
               (draft_version_id, target_id, message_id, revision_number, subject,
                body, footer, edited_by, policy_check_passed, injection_scan_passed,
                send_gate_passed, critique_passed, critique_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        # The body carries the raw script tag — the review page must render
        # it as inert escaped text, never as markup.
        params=("dv_2", "tgt_r1", None, 2, "Second subject", XSS_PAYLOAD,
                "[unsubscribe: {UNSUBSCRIBE_URL}]", "draft_writer",
                None, None, None, 1, json.dumps({
                    "passed": True, "issues": [], "required_changes": "",
                    "severity": "none",
                })),
    )
    commit(
        conn, action="insert_message_draft_version", table_name="message_draft_versions",
        record_id="dv_3", payload={}, run_id="r0", step_id="s0",
        actor="system", agent_id="draft_writer",
        sql="""INSERT INTO message_draft_versions
               (draft_version_id, target_id, message_id, revision_number, subject,
                body, footer, edited_by, policy_check_passed, injection_scan_passed,
                send_gate_passed, critique_passed, critique_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("dv_3", "tgt_r2", None, 1, "Other subject", "Other body.",
                "[unsubscribe: {UNSUBSCRIBE_URL}]", "draft_writer",
                None, None, None, None, None),
    )
    # The run_id the payload loader reads from the latest transition.
    commit(
        conn, action="state_transition", table_name="state_transitions",
        record_id="tr_r1", payload={}, run_id="r0", step_id="s1",
        actor="system", agent_id="draft_writer",
        sql="""INSERT INTO state_transitions
               (transition_id, run_id, step_id, target_id, previous_state,
                new_state, reason, actor, matched_policy_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=("tr_r1", "r0", "s1", "tgt_r1", "drafted", "awaiting_review",
                "draft_complete", "system", None),
    )
    conn.close()
    return path


@pytest.fixture
def review_client(review_db, monkeypatch, tmp_path):
    """Point the console at the review DB, and point the kill-switch reader
    at a tmp disengaged file — the committed config/kill_switch.json stays
    untouched even by the toggle test.  Supplies the auth secret for the same
    fail-closed reason as the client fixture (ticket H11)."""
    monkeypatch.setenv("OUTBOUND_DB_TARGET", review_db)
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", _CONSOLE_TEST_SECRET)
    # D1: same hermetic default as the client fixture — the replay banner must
    # be absent unless a test explicitly sets OUTBOUND_REPLAY_MODE.
    monkeypatch.delenv("OUTBOUND_REPLAY_MODE", raising=False)
    switch_path = tmp_path / "console_switch.json"
    switch_path.write_text(
        json.dumps({"enabled": False, "updated_at": "2026-08-23T00:00:00Z", "updated_by": "test"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(switch_path))
    return _AuthedTestClient(app)


def _seed_review_target(
    review_db: str,
    *,
    target_id: str,
    account_id: str,
    company_name: str,
    domain: str,
    icp_fit_label: str | None = None,
    icp_fit_score: int | None = None,
    icp_fit_reasons: str | None = None,
    icp_non_fit_reasons: str | None = None,
    judge_fit_label: str | None = None,
    judge_rationale: str | None = None,
    judge_divergence_justification: str | None = None,
) -> None:
    """Seed one awaiting_review target plus its account for a review-page test.

    The account INSERT names every column (NULL where the caller gave
    nothing), the same shape the review_db fixture uses, so a test seeds only
    the U1 columns it cares about and the rest stay NULL.  The target links to
    the fixture's existing off_1 offer and carries a contact_id of NULL (a
    company-only lead — legal, and the review page must render it).  Both
    writes go through write_gate.commit: the review PAGE is read-only, but
    seeding is a normal pipeline write (the fixture's own convention)."""
    conn = connect(review_db)
    try:
        commit(
            conn, action="insert_account", table_name="accounts",
            record_id=account_id, payload={}, run_id="r0", step_id="s0",
            actor="system", agent_id="system",
            sql="""INSERT INTO accounts
                   (account_id, company_name, domain, normalized_domain, industry,
                    estimated_size, geo, company_summary, icp_fit_label, icp_fit_score,
                    icp_fit_reasons, icp_non_fit_reasons, judge_fit_label,
                    judge_rationale, judge_divergence_justification, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(account_id, company_name, domain, domain, None, None, None,
                    "A test company.", icp_fit_label, icp_fit_score, icp_fit_reasons,
                    icp_non_fit_reasons, judge_fit_label, judge_rationale,
                    judge_divergence_justification),
        )
        commit(
            conn, action="insert_target", table_name="targets",
            record_id=target_id, payload={}, run_id="r0", step_id="s0",
            actor="system", agent_id="system",
            sql="""INSERT INTO targets
                   (target_id, account_id, contact_id, offer_id, source, state,
                    score, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(target_id, account_id, None, "off_1", "csv", "awaiting_review", 78),
        )
    finally:
        conn.close()


def test_review_queue_lists_awaiting_review_targets(review_client):
    """The queue page lists the pending targets and shows the kill-switch
    indicator (always visible)."""
    resp = review_client.get("/review/queue")
    assert resp.status_code == 200
    body = resp.text
    assert "/review/tgt_r1" in body
    assert "/review/tgt_r2" in body
    assert "Acme Therapeutics" in body
    assert "Disengaged" in body  # the switch indicator renders its state


def test_review_queue_api_returns_full_payload(review_client):
    """The JSON queue returns each pending target's FULL review payload
    (docs/human-review.md §2): company summary, signals, every draft
    revision, policy decision, risk flags, suppression status — plus the
    kill-switch state."""
    resp = review_client.get("/api/review/queue")
    assert resp.status_code == 200
    data = resp.json()
    ids = [t["target"]["target_id"] for t in data["targets"]]
    assert set(ids) == {"tgt_r1", "tgt_r2"}
    tgt = next(t for t in data["targets"] if t["target"]["target_id"] == "tgt_r1")
    assert tgt["company"]["company_name"] == "Acme Therapeutics"
    assert tgt["contact"]["email"] == "jane@acme.test"
    assert [dv["revision_number"] for dv in tgt["draft_versions"]] == [1, 2]
    assert tgt["draft_versions"][0]["critique"]["severity"] == "major"
    assert tgt["run_id"] == "r0"
    assert tgt["suppressed"] is False
    assert data["kill_switch"]["engaged"] is False


def test_review_page_shows_draft_diff_with_critiques(review_client):
    """The review page shows every draft revision in order with the critique
    that produced each — the "draft diff across iterations" the plan row
    asks for."""
    resp = review_client.get("/review/tgt_r1")
    assert resp.status_code == 200
    body = resp.text
    assert "Revision 1" in body
    assert "Revision 2" in body
    assert "First subject" in body
    assert "Second subject" in body
    # The critique's verdict fields are legible on the page.
    assert "Severity: major" in body
    assert "Remove the script tag claim." in body
    # The five decision actions are all present.
    for decision in ("approve", "approve_with_edits", "reject", "reject_and_suppress", "escalate"):
        assert decision in body


def test_stored_xss_in_draft_body_and_critique_is_escaped(review_client):
    """The review page's new rendering paths (draft body + critique issues)
    are LLM output derived from scraped third-party pages — the A5a
    autoescape control must hold on them exactly as on company_summary:
    the raw script tag never renders, only its escaped form.  (The payload
    is seeded in BOTH the body and a critique issue.)"""
    resp = review_client.get("/review/tgt_r1")
    assert resp.status_code == 200
    body = resp.text
    assert XSS_PAYLOAD not in body
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in body


def test_review_escapes_u1_icp_fit_reasons_judge_rationale_and_evidence_quote(
    review_client, review_db
):
    """U1's new review-page fields — icp_fit_reasons, judge_rationale, and a
    signal's evidence_quote — are LLM output like every other rendered value,
    so the A5a autoescape control must hold on EACH of them independently.
    Three DISTINCT payloads, one per field: if a future template edit adds
    |safe to any single field, exactly that field's assertions fail (the raw
    script appears AND its escaped form vanishes) while the other two stay
    green — a combined check could not say WHICH field regressed."""
    _seed_review_target(
        review_db,
        target_id="tgt_u1", account_id="acc_u1", company_name="U1 XSS Co",
        domain="u1x.test",
        icp_fit_label="strong_fit", icp_fit_score=80,
        # The deterministic scorer's reasons are a JSON-encoded list in
        # storage (db-schema.md §accounts); the payload rides inside it the
        # way the pipeline would actually persist it.
        icp_fit_reasons=json.dumps([U1_ICP_FIT_REASONS_PAYLOAD]),
        icp_non_fit_reasons="[]",
        # judge_fit_label left NULL so no divergence block complicates the
        # page — this test isolates the three escaped fields.
        judge_rationale=U1_JUDGE_RATIONALE_PAYLOAD,
    )
    # One signal whose evidence_quote is the third distinct payload.
    conn = connect(review_db)
    try:
        commit(
            conn, action="insert_signal", table_name="signals",
            record_id="sig_u1", payload={}, run_id="r0", step_id="s1",
            actor="system", agent_id="system",
            sql="""INSERT INTO signals
                   (signal_id, run_id, target_id, signal_type, signal_value,
                    signal_strength, source_url, source_confidence,
                    evidence_quote, evidence_verified, evidence_tier, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
            params=("sig_u1", "r0", "tgt_u1", "hiring_relevant_role", "Hiring 3 SDRs",
                    1.0, "https://u1x.test/careers", 0.9,
                    U1_EVIDENCE_QUOTE_PAYLOAD, 0, "findings"),
        )
    finally:
        conn.close()

    resp = review_client.get("/review/tgt_u1")
    assert resp.status_code == 200
    body = resp.text
    # Field 1 — icp_fit_reasons: the raw script never renders, only its
    # escaped form (which appears inside the escaped JSON list).
    assert U1_ICP_FIT_REASONS_PAYLOAD not in body
    assert "&lt;script&gt;alert(&#39;icp_fit_reasons&#39;)&lt;/script&gt;" in body
    # Field 2 — judge_rationale: the same autoescape contract on the judge's
    # written reasoning.
    assert U1_JUDGE_RATIONALE_PAYLOAD not in body
    assert "&lt;script&gt;alert(&#39;judge_rationale&#39;)&lt;/script&gt;" in body
    # Field 3 — a signal's evidence_quote: the same contract on the evidence
    # row the reviewer reads before deciding.
    assert U1_EVIDENCE_QUOTE_PAYLOAD not in body
    assert "&lt;script&gt;alert(&#39;evidence_quote&#39;)&lt;/script&gt;" in body


def test_review_renders_divergence_block_when_judge_overrides_deterministic_label(
    review_client, review_db
):
    """When the judge's FINAL label differs from the deterministic one, the
    review page's divergence block (the U1 addition) must render BOTH labels
    and the judge's mandatory justification — the moment the operator needs
    at a glance.  The block's condition is judge_fit_label != icp_fit_label
    with both non-null, exactly as the template defines it; this test pins
    that contract so a future edit cannot silently drop the block."""
    _seed_review_target(
        review_db,
        target_id="tgt_div", account_id="acc_div", company_name="Divergence Co",
        domain="divergence.test",
        icp_fit_label="strong_fit", icp_fit_score=88,
        judge_fit_label="good_fit",
        judge_divergence_justification=(
            "The deterministic scorer missed the company's recent pivot to "
            "healthcare; the judge overrides based on direct evidence."
        ),
    )
    resp = review_client.get("/review/tgt_div")
    assert resp.status_code == 200
    body = resp.text
    # The block's marker text (verbatim from the template)…
    assert "Judge diverged from the deterministic score" in body
    assert "strong_fit" in body  # …the deterministic label the formula produced…
    assert "good_fit" in body    # …the judge's final label…
    assert "recent pivot to healthcare" in body  # …and the justification is legible.


def test_review_with_null_judge_and_zero_signals_renders_cleanly(review_client, review_db):
    """A target whose judge never produced a verdict (NULL judge_fit_label /
    judge_rationale — legal per db-schema.md) and that has zero signals must
    render 200 with no Jinja UndefinedError/AttributeError, and must NOT show
    the divergence block (its condition requires a non-null judge label that
    differs).  This is a real regression risk any template edit could
    reintroduce: a field added outside its null guard blows up the page."""
    _seed_review_target(
        review_db,
        target_id="tgt_null", account_id="acc_null", company_name="Null Safety Co",
        domain="nullsafety.test",
        icp_fit_label="strong_fit", icp_fit_score=80,
        # judge_fit_label / judge_rationale intentionally NULL: no divergence,
        # no judge-rationale section, and the page must still render.
    )
    resp = review_client.get("/review/tgt_null")
    assert resp.status_code == 200
    body = resp.text
    # The divergence marker never appears without a judge override…
    assert "Judge diverged from the deterministic score" not in body
    # …the judge-rationale section is skipped when no rationale exists.
    # (The bare substring "Judge rationale" DOES appear — in the template's
    # literal HTML comment — so assert on the heading element, which only
    # renders inside the {% if judge_rationale is not none %} guard.)
    assert "<h3>Judge rationale</h3>" not in body
    # …and zero signals render the empty-state, not a crash.
    assert "No signals recorded." in body


def test_review_decision_form_approves_and_redirects(review_client, review_db):
    """The HTML form's POST records the decision through the review gate and
    redirects with the outcome — the target actually moves to approved and
    a review_decisions row lands."""
    resp = review_client.post(
        "/review/tgt_r1/decision",
        data={"decision": "approve", "reason": "looks good", "run_id": "r0"},
        # follow_redirects=False: the 303 itself (not the followed page) is
        # the contract under test.
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "/review/tgt_r1" in resp.headers["location"]
    conn = connect(review_db)
    try:
        state = conn.execute(
            "SELECT state FROM targets WHERE target_id='tgt_r1';"
        ).fetchone()["state"]
        assert state == "approved"
        row = conn.execute(
            "SELECT * FROM review_decisions WHERE target_id='tgt_r1';"
        ).fetchone()
        assert row is not None
        assert row["decision"] == "approve"
        assert row["reason"] == "looks good"
        assert row["kill_switch_active"] == 0
    finally:
        conn.close()


def test_review_decision_json_api_returns_outcome(review_client):
    """The JSON endpoint (docs/api.md §4) records a decision and returns the
    ReviewOutcome shape."""
    resp = review_client.post(
        "/review/decision",
        json={"target_id": "tgt_r2", "decision": "escalate",
              "research_note": "re-check the careers page", "run_id": "r0"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is False
    assert data["new_state"] == "researched"
    assert data["review_decision_id"] is not None


def test_review_decision_json_api_refusal_is_an_outcome_not_a_500(review_client):
    """A refusal (double-submit here: decide the same target twice) is a
    200 with refused=True and a reason — an observable outcome, never a
    500, never a silent no-op."""
    first = review_client.post(
        "/review/decision",
        json={"target_id": "tgt_r1", "decision": "approve", "run_id": "r0"},
    )
    assert first.status_code == 200
    second = review_client.post(
        "/review/decision",
        json={"target_id": "tgt_r1", "decision": "approve", "run_id": "r0"},
    )
    assert second.status_code == 200
    data = second.json()
    assert data["refused"] is True
    assert "not awaiting_review" in data["refusal_reason"]


def test_unknown_review_target_is_404(review_client):
    """The review page keeps the console's 404 contract for unknown ids."""
    resp = review_client.get("/review/no_such_target")
    assert resp.status_code == 404


def test_kill_switch_toggle_from_console_flips_the_reader(review_client, tmp_path, monkeypatch):
    """The console toggle is the second way to flip the switch (runbook.md
    §1): POSTing the toggle rewrites the file, and a fresh read — through
    the console's own queue page — sees the new state."""
    from app.kill_switch import read_kill_switch  # imported locally: this test asserts the reader's view, and the env var is already pointed at the tmp file

    # Engage through the console.  follow_redirects=False: the 303 itself is
    # the contract under test (TestClient follows redirects by default and
    # would return the final 200 page).
    resp = review_client.post(
        "/kill-switch", data={"engaged": "true"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert read_kill_switch().engaged is True
    assert read_kill_switch().updated_by == "operator"  # the toggle records who flipped it
    # The queue page's indicator shows the engaged state.
    body = review_client.get("/review/queue").text
    assert "ENGAGED" in body
    # Disengage again — the file stays the single source of truth.
    review_client.post("/kill-switch", data={"engaged": "false"}, follow_redirects=False)
    assert read_kill_switch().engaged is False


def test_kill_switch_toggle_rejects_unknown_value(review_client):
    """A malformed toggle value is a 422, never a guessed state — the route
    only accepts exactly 'true'/'false'."""
    resp = review_client.post("/kill-switch", data={"engaged": "maybe"})
    assert resp.status_code == 422


# ── Replay-mode banner (ticket D1) ───────────────────────────────────────────
# OUTBOUND_REPLAY_MODE=1 marks an OFFLINE REPLAY of a real, previously recorded
# pipeline run (scripts/demo_replay.sh serves a disposable copy of
# data/e2e_run2.db). It is a PURE env read (app/console/app.py::_replay_mode) —
# no database query — and renders a DISTINCT blue banner alongside (never
# instead of) the amber demo_data banner. These tests pin both states and the
# independence of the two flags.

REPLAY_BANNER = "this is a REPLAY of a real, previously recorded pipeline run"
# The banner's distinctive clause, not the CSS comment (which also mentions
# "placeholder contacts and gate results").
DEMO_BANNER = "No real verification or injection/policy scan ran"


def test_replay_banner_absent_when_env_unset(client):
    """A real, non-replayed database (no demo_seed marker, OUTBOUND_REPLAY_MODE
    unset) renders a clean page — neither banner appears."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert REPLAY_BANNER not in resp.text
    assert DEMO_BANNER not in resp.text


def test_replay_banner_renders_on_console_routes_when_env_set(client, monkeypatch):
    """Every route that received replay_mode renders the honest banner: the
    index, a target's audit trail, and the review queue. (The review_target
    route is covered by the review_client test below, whose fixture has an
    awaiting_review target.)"""
    monkeypatch.setenv("OUTBOUND_REPLAY_MODE", "1")
    for path in ("/", "/targets/tgt_1", "/review/queue"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert REPLAY_BANNER in resp.text, path


def test_replay_banner_renders_on_review_target_route(review_client, monkeypatch):
    """The per-target review screen — the approval surface itself — also carries
    the honest replay banner."""
    monkeypatch.setenv("OUTBOUND_REPLAY_MODE", "1")
    resp = review_client.get("/review/tgt_r1")
    assert resp.status_code == 200
    assert REPLAY_BANNER in resp.text


def test_demo_banner_still_renders_when_seeded_and_replay_banner_absent(client, db_path):
    """D1 must NOT affect the existing D3a DEMO DATA banner: seed the
    demo_seed marker row, leave OUTBOUND_REPLAY_MODE unset, and the amber
    banner renders while the blue replay banner does not."""
    conn = connect(db_path)
    log_step(
        conn, run_id="r0", step_id="s_demo", target_id="tgt_1",
        tool_name="demo_seed", agent_id="system",
        input_data={}, output_data={}, status="success",
    )
    conn.close()
    resp = client.get("/")
    assert resp.status_code == 200
    assert DEMO_BANNER in resp.text
    assert REPLAY_BANNER not in resp.text


def test_both_banners_render_when_seeded_and_replay_mode_set(client, db_path, monkeypatch):
    """The two flags are INDEPENDENT: a database that is both demo-seeded and
    served in replay mode renders BOTH banners — D1 must never suppress D3a."""
    conn = connect(db_path)
    log_step(
        conn, run_id="r0", step_id="s_demo", target_id="tgt_1",
        tool_name="demo_seed", agent_id="system",
        input_data={}, output_data={}, status="success",
    )
    conn.close()
    monkeypatch.setenv("OUTBOUND_REPLAY_MODE", "1")
    resp = client.get("/")
    assert resp.status_code == 200
    assert DEMO_BANNER in resp.text
    assert REPLAY_BANNER in resp.text


# ── Live run view (ticket U2) ────────────────────────────────────────────────
# The judge watches the pipeline move in real time: an HTML shell (run.html)
# polls GET /api/run/{run_id}/steps.  These tests pin the API contract (empty
# unknown run, ordered steps, the stop condition), the auth contract (the new
# routes ride the SAME global require_operator as every other route), and the
# security posture of the first client-side JS this codebase has ever shipped
# (textContent-only rendering, no |safe, no string-assignment DOM setters).


def _read_console_template(name: str) -> str:
    # Templates live next to app/console/app.py; read the shipped file so the
    # security guards below pin the actual artifact, not a copy of it.
    return (
        Path(__file__).resolve().parent.parent
        / "app" / "console" / "templates" / name
    ).read_text(encoding="utf-8")


def _seed_run_target(conn, *, run_id: str, target_id: str, state: str) -> None:
    """Insert one target row (in the given state) plus one step under a run —
    the seeding convention of this module (through the write gate / log_step),
    so a test can build a run whose target has reached a terminal state."""
    commit(
        conn, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id=run_id, step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets
               (target_id, account_id, contact_id, offer_id, source, state,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, "acc_1", None, "off_1", "csv", state),
    )
    log_step(
        conn, run_id=run_id, step_id=f"s_{target_id}", target_id=target_id,
        tool_name="score_lead", agent_id="system",
        input_data={}, output_data={}, status="success",
    )


def _age_run_steps(conn, *, run_id: str, seconds: int) -> None:
    """Push every step under a run's created_at into the past.

    Direct UPDATE on steps (the trace table — deliberately outside the write
    gate, see app/tools/log_step.py), so tests can exercise the U2-fix
    quiet-period fallback without sleeping the real test for 10+ seconds:
    the fallback compares steps.created_at against _QUIET_PERIOD_SECONDS, and
    the console stores that column as second-precision UTC TEXT
    (datetime('now')), so this helper writes the same format back-dated.
    """
    old = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds)
    ).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE steps SET created_at=? WHERE run_id=?;", (old, run_id))
    conn.commit()


def test_run_api_unknown_run_id_returns_empty_list(client):
    """An UNKNOWN / not-yet-started run_id is 200 + an empty list + not
    complete — never a 404: a live run legitimately starts with zero steps
    logged, and the page must be able to poll from the very beginning without
    erroring (the U2 contract)."""
    resp = client.get("/api/run/run_does_not_exist/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run_does_not_exist"
    assert data["steps"] == []
    assert data["target_states"] == {}
    assert data["complete"] is False


def test_run_api_returns_ordered_steps_with_target_states(client, db_path):
    """A run with steps returns them in (created_at, step_id) order — step_id
    is the deterministic tiebreak on same-timestamp rows, matching the
    ordering fix the ticket calls for — plus the current state of each target
    the run touched.  A target in `scored` (mid-pipeline) is NOT complete."""
    conn = connect(db_path)
    # Two steps in the same run; step_a is logged first AND sorts before
    # step_b on step_id, so the (created_at, step_id) tiebreak is deterministic
    # whether or not the two log_step calls straddle a second boundary.
    log_step(
        conn, run_id="run_u2", step_id="step_a", target_id="tgt_1",
        tool_name="get_targets", agent_id="system",
        input_data={}, output_data={}, status="success",
    )
    log_step(
        conn, run_id="run_u2", step_id="step_b", target_id="tgt_1",
        tool_name="summarize_company", agent_id="system",
        input_data={}, output_data={}, status="success",
    )
    conn.close()

    resp = client.get("/api/run/run_u2/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert [s["step_id"] for s in data["steps"]] == ["step_a", "step_b"]
    assert [s["tool_name"] for s in data["steps"]] == ["get_targets", "summarize_company"]
    assert all(s["status"] == "success" for s in data["steps"])
    # tgt_1 is in `scored` (the db_path fixture's state) — a mid-pipeline
    # state, so the run must NOT be reported complete.
    assert data["target_states"] == {"tgt_1": "scored"}
    assert data["complete"] is False


def test_run_api_complete_when_target_awaiting_review(review_client):
    """THE demo beat: once a run's target reaches `awaiting_review`, the run is
    reported complete and the client-side poll loop stops.  review_db seeds a
    state_transition (drafted -> awaiting_review) for tgt_r1 under run r0, so
    the run's target set is {tgt_r1} at the human gate."""
    resp = review_client.get("/api/run/r0/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_r1": "awaiting_review"}
    assert data["complete"] is True


def test_run_api_complete_when_target_terminal(client, db_path):
    """A run whose target reached a TERMINAL state (suppressed here — terminal
    for outbound per state-machine.md §5) is reported complete.  This run has
    no get_targets manifest, so the quiet-period fallback applies: its single
    step is aged past _QUIET_PERIOD_SECONDS so the run reads as finished (a
    just-logged step keeps complete=False — proven by the dedicated
    quiet-period tests below)."""
    conn = connect(db_path)
    _seed_run_target(conn, run_id="run_term", target_id="tgt_supp", state="suppressed")
    _age_run_steps(conn, run_id="run_term", seconds=60)
    conn.close()
    resp = client.get("/api/run/run_term/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_supp": "suppressed"}
    assert data["complete"] is True


def _seed_untouched_target(conn, *, run_id: str, target_id: str) -> None:
    """Insert a targets row in `new` with NO steps and NO transitions under any
    run — the state of a target a phase1 run has imported but not yet reached
    (the pipeline processes targets SEQUENTIALLY; the write gate's insert is
    the fixture's normal seeding convention)."""
    commit(
        conn, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id=run_id, step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets
               (target_id, account_id, contact_id, offer_id, source, state,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, "acc_1", None, "off_1", "csv", "new"),
    )


def _seed_manifest_step(conn, *, run_id: str, target_ids: list[str]) -> None:
    """Log the phase1 get_targets manifest step naming the run's FULL intended
    target set — the batch-import step (target_id NULL) whose output_json
    carries target_ids (app/tools/get_targets.py)."""
    log_step(
        conn, run_id=run_id, step_id="s_manifest",
        target_id=None, tool_name="get_targets", agent_id="system",
        input_data={"csv_path": "prospects.csv", "cli_offer_slug": "therapy-app"},
        output_data={"target_ids": target_ids},
        status="success",
    )


def test_run_api_multi_target_manifest_not_complete_while_untouched_target_has_no_rows(client, db_path):
    """TICKET U2-fix Finding 1 (reproduction): a phase1 run whose get_targets
    manifest names BOTH targets — target A in a terminal stop state, target B
    imported but untouched (zero steps, zero transitions — the pipeline
    processes targets sequentially).  The old code derived its target set ONLY
    from rows the run had already written, so target B was invisible and the
    run reported complete after target #1 while still working on target #2.
    The manifest-backed check must keep complete=False until EVERY manifest
    target is done, and must make the untouched target visible."""
    conn = connect(db_path)
    _seed_manifest_step(conn, run_id="run_multi", target_ids=["tgt_a", "tgt_b"])
    _seed_run_target(conn, run_id="run_multi", target_id="tgt_a", state="failed")
    _seed_untouched_target(conn, run_id="run_multi", target_id="tgt_b")
    conn.close()

    resp = client.get("/api/run/run_multi/steps")
    assert resp.status_code == 200
    data = resp.json()
    # The manifest makes BOTH targets visible — including the untouched one.
    assert data["target_states"] == {"tgt_a": "failed", "tgt_b": "new"}
    assert data["complete"] is False  # tgt_b in `new` (mid-pipeline) — NOT complete


def test_run_api_multi_target_manifest_complete_only_when_all_done(client, db_path):
    """TICKET U2-fix Finding 1 (second half): with the get_targets manifest,
    complete flips True ONLY once EVERY manifest target has actually reached a
    stop state — no quiet-period heuristic is needed for manifest runs, the
    check is structural.  Target B advances to `failed` (valid from `new` via
    state-machine.md §3's any → failed) and the run completes."""
    conn = connect(db_path)
    _seed_manifest_step(conn, run_id="run_multi", target_ids=["tgt_a", "tgt_b"])
    _seed_run_target(conn, run_id="run_multi", target_id="tgt_a", state="failed")
    _seed_untouched_target(conn, run_id="run_multi", target_id="tgt_b")
    # Advance target B to a terminal stop state through the REAL state machine.
    transition(
        conn, target_id="tgt_b", from_state="new", to_state="failed",
        reason="no_sources_available", actor="system",
        run_id="run_multi", step_id="s_b",
    )
    conn.close()

    resp = client.get("/api/run/run_multi/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_a": "failed", "tgt_b": "failed"}
    assert data["complete"] is True


# ── U2-fix2: the three stage CLIs now log their own batch manifest ──────────
# The draft/send/reply CLIs now log a *_batch_manifest step (the same shape
# as get_targets) naming their FULL intended target set, so the live view's
# completeness check is structural for them too.  These parametrized tests
# reproduce the exact latency-race scenario the second review round proved
# against the real _fetch_run_steps: a >12s gap with no new step, one item
# finished, one manifest target still untouched — complete must stay False.
# One case per newly-manifested stage; each runs the SAME widened
# _get_manifest_target_ids query (the shared code path), and each is proven
# both ways (not-complete-while-untouched, complete-only-when-all-done).

_STAGE_MANIFEST_CASES = [
    # (tool_name, stop state for the already-finished target)
    ("draft_batch_manifest", "awaiting_review"),
    ("send_batch_manifest", "dry_run_sent"),
    ("reply_batch_manifest", "suppressed"),
]


@pytest.mark.parametrize("tool_name, stop_state", _STAGE_MANIFEST_CASES)
def test_run_api_stage_manifest_keeps_complete_false_while_a_manifest_target_is_untouched(
    client, db_path, tool_name, stop_state
):
    """U2-fix2 reproduction: a draft/send/reply run whose *_batch_manifest
    names BOTH targets — target A finished (its stop state), target B
    imported but untouched (zero steps, zero transitions), and target A's
    step aged past _QUIET_PERIOD_SECONDS (the exact >12s gap the review used
    to trigger a FALSE complete).  The old no-manifest path derived the
    target set from rows, saw only A in a stop state, found the quiet period
    elapsed, and declared complete while the pipeline was still working on B.
    The manifest makes B visible, so complete stays False."""
    conn = connect(db_path)
    log_step(
        conn, run_id="run_stage_manifest", step_id="s_manifest",
        target_id=None, tool_name=tool_name, agent_id="system",
        input_data={"stage": tool_name},
        output_data={"target_ids": ["tgt_a", "tgt_b"]},
        status="success",
    )
    _seed_run_target(conn, run_id="run_stage_manifest", target_id="tgt_a", state=stop_state)
    _age_run_steps(conn, run_id="run_stage_manifest", seconds=60)
    _seed_untouched_target(conn, run_id="run_stage_manifest", target_id="tgt_b")
    conn.close()

    resp = client.get("/api/run/run_stage_manifest/steps")
    assert resp.status_code == 200
    data = resp.json()
    # The manifest makes BOTH targets visible — including the untouched one.
    assert data["target_states"] == {"tgt_a": stop_state, "tgt_b": "new"}
    assert data["complete"] is False  # tgt_b in `new` (mid-pipeline) — NOT complete


@pytest.mark.parametrize("tool_name, stop_state", _STAGE_MANIFEST_CASES)
def test_run_api_stage_manifest_complete_only_when_all_manifest_targets_done(
    client, db_path, tool_name, stop_state
):
    """U2-fix2 second half: with the stage manifest, complete flips True ONLY
    once EVERY manifest target has actually reached a stop state — no
    quiet-period heuristic is needed.  Target B advances to `failed` (valid
    from `new` via state-machine.md §3's any → failed); its transition step
    is FRESH (not aged), which would keep a quiet-period run incomplete — so
    complete True here proves the manifest path is structural, not the
    heuristic."""
    conn = connect(db_path)
    log_step(
        conn, run_id="run_stage_manifest2", step_id="s_manifest",
        target_id=None, tool_name=tool_name, agent_id="system",
        input_data={"stage": tool_name},
        output_data={"target_ids": ["tgt_a", "tgt_b"]},
        status="success",
    )
    _seed_run_target(conn, run_id="run_stage_manifest2", target_id="tgt_a", state=stop_state)
    _seed_untouched_target(conn, run_id="run_stage_manifest2", target_id="tgt_b")
    # Advance B to a terminal stop state through the REAL state machine.
    transition(
        conn, target_id="tgt_b", from_state="new", to_state="failed",
        reason="no_sources_available", actor="system",
        run_id="run_stage_manifest2", step_id="s_b",
    )
    conn.close()

    resp = client.get("/api/run/run_stage_manifest2/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_a": stop_state, "tgt_b": "failed"}
    assert data["complete"] is True


def test_run_api_target_still_approved_mid_send_is_not_complete(client, db_path):
    """TICKET U2-fix Finding 2: a send run STARTS with its targets still in
    `approved` (app/send_cli.py selects targets in `approved` and moves them to
    `dry_run_sent`).  At the moment a send run begins — first send step logged,
    targets.state still `approved` — the old code declared complete and the
    live view stopped before a single send rendered.  `approved` is NOT a stop
    state: complete stays False until the target actually moves on."""
    conn = connect(db_path)
    _seed_run_target(conn, run_id="run_send", target_id="tgt_app", state="approved")
    conn.close()

    resp = client.get("/api/run/run_send/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_app": "approved"}
    assert data["complete"] is False


def test_run_api_target_passing_through_routed_is_not_complete_until_suppressed(client, db_path):
    """TICKET U2-fix Finding 3: `routed` (and `replied`) are TRANSIENT states —
    docs/state-machine.md §7j/§7k (line 97) document the reply router firing
    `replied → routed` and then, in the SAME invocation, `routed → suppressed`
    (a high-confidence unsubscribe) or `routed → drafted` (a follow-up).  A
    poll landing between those writes sees `routed` and must NOT declare
    complete — that would stop the view and hide the unsubscribe suppression.
    complete flips True only once the target actually lands in `suppressed`."""
    conn = connect(db_path)
    _seed_run_target(conn, run_id="run_reply", target_id="tgt_r", state="replied")
    # The router fires replied -> routed (classifier + routing rule).
    transition(
        conn, target_id="tgt_r", from_state="replied", to_state="routed",
        reason="classified_and_routed", actor="system",
        run_id="run_reply", step_id="s_2",
    )
    conn.close()

    resp = client.get("/api/run/run_reply/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_r": "routed"}
    assert data["complete"] is False  # routed is transient — keep polling

    # The SAME invocation then fires routed -> suppressed (unsubscribe).
    conn = connect(db_path)
    transition(
        conn, target_id="tgt_r", from_state="routed", to_state="suppressed",
        reason="unsubscribe_reply", actor="system",
        run_id="run_reply", step_id="s_3",
    )
    # No manifest: age the steps so the quiet-period fallback is satisfied.
    _age_run_steps(conn, run_id="run_reply", seconds=60)
    conn.close()

    resp = client.get("/api/run/run_reply/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_r": "suppressed"}
    assert data["complete"] is True


def test_run_api_no_manifest_recent_step_keeps_complete_false(client, db_path):
    """TICKET U2-fix Finding 1 (quiet-period fallback): a draft/send/reply run
    has no get_targets manifest, so complete additionally requires the run's
    most recent step to be OLDER than _QUIET_PERIOD_SECONDS.  A RECENT step
    (just logged) keeps complete=False even when every visible target is in a
    stop state — the pipeline may be about to log the next target's step."""
    conn = connect(db_path)
    _seed_run_target(conn, run_id="run_q", target_id="tgt_q", state="suppressed")
    # Step is just logged (created_at = now) — quiet period has NOT elapsed.
    conn.close()

    resp = client.get("/api/run/run_q/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_q": "suppressed"}
    assert data["complete"] is False


def test_run_api_no_manifest_old_step_allows_complete(client, db_path):
    """TICKET U2-fix Finding 1 (quiet-period fallback, elapsed): the same
    no-manifest run whose step is aged past _QUIET_PERIOD_SECONDS reads as
    finished — complete flips True."""
    conn = connect(db_path)
    _seed_run_target(conn, run_id="run_q", target_id="tgt_q", state="suppressed")
    _age_run_steps(conn, run_id="run_q", seconds=60)
    conn.close()

    resp = client.get("/api/run/run_q/steps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_states"] == {"tgt_q": "suppressed"}
    assert data["complete"] is True


def test_run_api_no_manifest_fallback_still_reachable_for_a_non_cli_run(client, db_path):
    """U2-fix2: the quiet-period fallback must still fire for a run with NO
    manifest step at all (e.g. a taskmaster-agent or direct-API run that only
    logs per-target steps) — now that every CLI-originated run type has a
    real manifest, this is the EXCEPTION path, and it must not rot
    un-exercised.  A run whose per-target step is aged past
    _QUIET_PERIOD_SECONDS reads as complete; a just-logged step keeps
    complete False.  Uses a draft-stage per-target tool_name to prove the
    fallback covers a stage-shaped run whose tool_name is NOT one of the four
    manifest names."""
    conn = connect(db_path)
    # A non-manifest, per-target-only run (the taskmaster-agent shape).
    _seed_run_target(conn, run_id="run_non_cli", target_id="tgt_q", state="suppressed")
    conn.execute(
        "UPDATE steps SET tool_name='draft_target_run' WHERE run_id='run_non_cli';"
    )
    conn.commit()
    conn.close()

    # A just-logged step -> quiet period has NOT elapsed -> not complete.
    resp = client.get("/api/run/run_non_cli/steps")
    assert resp.status_code == 200
    assert resp.json()["complete"] is False

    # Age the step -> the fallback now allows complete.
    conn = connect(db_path)
    _age_run_steps(conn, run_id="run_non_cli", seconds=60)
    conn.close()
    resp = client.get("/api/run/run_non_cli/steps")
    assert resp.status_code == 200
    assert resp.json()["complete"] is True


def test_run_view_renders_shell_with_waiting_state(client):
    """The HTML shell renders for ANY run_id — even one with zero steps — with
    the run_id server-rendered (autoescaped) and the "waiting for the run to
    start" initial state; it is never a loading spinner masking a bug."""
    resp = client.get("/run/run_abc123")
    assert resp.status_code == 200
    body = resp.text
    assert "Run <code>run_abc123</code>" in body
    assert 'data-run-id="run_abc123"' in body
    assert "/api/run/" in body          # the poll URL the script builds
    assert "No steps yet" in body       # the zero-steps initial state
    assert "waiting for the run to start" in body


def test_run_view_escapes_server_rendered_run_id(client):
    """The run_id is server-rendered into the HTML shell via Jinja — it must go
    through autoescape like every other value.  A run_id containing markup is
    an extreme edge (real ids are run_<hex> from app/ids.py), but the page
    must neutralise it the same way it neutralises a hostile company_summary."""
    # percent-encoded: run"><svg onload=alert(1)> — no "/", so the route param
    # decodes cleanly.  Jinja's autoescape turns < > and " into entities.
    resp = client.get("/run/run%22%3E%3Csvg%20onload%3Dalert(1)%3E")
    assert resp.status_code == 200
    body = resp.text
    assert "&lt;svg onload=alert(1)&gt;" in body   # the escaped form renders…
    assert '"><svg' not in body                     # …the raw payload never does
    assert "data-run-id=\"run&#34;&gt;" in body or 'data-run-id="run&quot;&gt;' in body


# The unsafe DOM-string-parsing APIs the live-view guard pins the ABSENCE of
# (ticket U2-fix Finding 4).  Every one of these parses a string as HTML and
# would bypass Jinja autoescape — a second, separate stored-XSS vector.  The
# list covers the two setters the original guard already named (innerHTML /
# insertAdjacentHTML) PLUS the four the adversarial review proved sailed
# through the old first-block-only, two-token guard (outerHTML,
# document.write, setHTMLUnsafe, createContextualFragment) PLUS the two
# other commonly-cited string-parsing entry points
# (Range.createContextualFragment is covered by the createContextualFragment
# token; DOMParser.parseFromString is covered by the parseFromString token).
# `document.write` also matches document.writeln (its substring).  This is a
# deliberate denylist of the well-known unsafe forms; textContent-only
# rendering is the safe path and the positive check below enforces it.
_UNSAFE_DOM_STRING_APIS = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "setHTMLUnsafe",
    "createContextualFragment",
    "parseFromString",
)


def _extract_script_blocks(text: str) -> list[str]:
    """Return the contents of EVERY inline <script>...</script> block in a
    template, HTML comments stripped first.

    Stripping comments matters because run.html's own prose mentions
    "<script>" by name (it is documentation, not executable code) — a naive
    first-match could capture a comment instead of the poll loop.  The U2-fix
    Finding 4 regression was exactly the reverse of that failure: the old
    regex anchored on the FIRST "<script>..." and scanned to the FIRST
    "</script>" only, so a SECOND <script> block anywhere in the template
    sailed through completely.  Returning every block closes that hole.
    """
    no_comments = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return re.findall(
        r"<script\b[^>]*>(.*?)</script>",
        no_comments,
        flags=re.DOTALL | re.IGNORECASE,
    )


@pytest.mark.parametrize("template_name", ["run.html", "base.html"])
def test_console_template_scripts_use_text_content_not_inner_html(template_name):
    """Every inline <script> block this console ships must render every
    API-derived value with textContent ONLY.  The string-assignment DOM
    setters / HTML-string parsers (innerHTML, outerHTML, insertAdjacentHTML,
    document.write, setHTMLUnsafe, createContextualFragment, DOMParser
    .parseFromString) bypass Jinja autoescape entirely — a second, separate
    stored-XSS vector — so their ABSENCE in EVERY <script> block of the
    shipped template is pinned statically.  A future editor who "simplifies" a
    textContent call into a string assignment fails this test, which is the
    point (the guard must name the exact setter).  The check reads the shipped
    file (not a copy) and scans EVERY script block, not just the first.

    Covers run.html (ticket U2, the FIRST client-side JS this codebase ever
    shipped) and base.html (ticket U3, the site-wide review-queue badge).
    base.html renders on EVERY page, so its script carries the same
    non-negotiable discipline even though it only renders an integer count
    today: a future editor extending the badge to show more must inherit a
    script that is already safe by construction, not one that got a pass
    because "it's just a number today."
    NOTE (review disclosure): this guard catches LITERAL denylisted forms
    only (e.g. ``element.innerHTML = ...``) and would not catch a deliberately
    obfuscated evasion like string-concatenation building the property name
    at runtime (``el["inner"+"HTML"]``) — that is a real but out-of-scope gap:
    the threat model here is an accidental regression by a future editor, not
    a malicious committer, and no accidental edit looks like that."""
    text = _read_console_template(template_name)
    script_blocks = _extract_script_blocks(text)
    assert script_blocks, (
        f"{template_name} has no <script> block — the page lost its poll loop?"
    )
    # Combine every block so a SECOND <script> block anywhere in the template
    # is scanned too (the reviewer's Finding 4 evasion: a second block with
    # unsafe rendering sailed through the old first-block-only regex).
    combined = "\n".join(script_blocks)
    assert "textContent" in combined, (
        f"{template_name}'s script never uses textContent — every API value "
        "must be inserted with textContent (or an equivalent safe API)"
    )
    for api in _UNSAFE_DOM_STRING_APIS:
        assert api not in combined, (
            f"{template_name}'s script uses unsafe DOM-string API {api!r} — "
            "that parses a string as HTML and would bypass Jinja autoescape "
            "(U2/U3). Every API value must be inserted with textContent."
        )


def test_run_template_has_no_safe_filter():
    """run_id / target_id are server-rendered via Jinja into the initial HTML
    shell — no `|safe` FILTER anywhere means they pass through autoescape like
    every other value in this console.  The check targets the actual Jinja
    filter usage (`{{ x | safe }}`), not prose comments that mention the
    filter by name (a comment saying "no |safe" is harmless documentation)."""
    text = _read_console_template("run.html")
    # A Jinja expression containing a |safe filter: {{ expr | safe }}.  The
    # [^}]* stops the expression match at the closing braces, so a comment
    # outside an expression can never trip this.
    assert not re.search(r"\{\{[^}]*\|\s*safe\b", text), (
        "run.html applies the |safe filter to a Jinja value — that disables "
        "autoescape on it and would reopen the stored-XSS class this console "
        "closes everywhere else (U2)"
    )


def test_run_api_fails_closed_without_secret(monkeypatch, tmp_path):
    """The new routes ride the SAME global require_operator as every other
    route: with no secret configured the console fails closed (503)."""
    monkeypatch.delenv("OUTBOUND_CONSOLE_API_KEY", raising=False)
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    resp = TestClient(app).get("/api/run/run_x/steps")
    assert resp.status_code == 503
    resp = TestClient(app).get("/run/run_x")
    assert resp.status_code == 503


def test_run_api_wrong_credential_is_401(monkeypatch, tmp_path):
    """A wrong credential is a 401, never a 500 and never a silent 200 — the
    fail-closed doctrine applied to the new routes exactly as to every other."""
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", "correct-secret")
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    resp = TestClient(app).get(
        "/api/run/run_x/steps", headers={"X-Internal-API-Key": "wrong-key"}
    )
    assert resp.status_code == 401


# ── Review-queue count badge (ticket U3) ─────────────────────────────────────
# The console tells the judge a decision is waiting: base.html's header badge
# polls GET /api/review/queue/count (a CHEAP, count-only SELECT that reuses
# _pending_review_targets — NOT the heavy /api/review/queue full-payload
# route) every ~10s, renders the count via textContent, and prefixes the tab
# title from a stored original.  These tests pin the API contract (0 / N
# pending), the auth contract (same global require_operator as every route),
# the SITE-WIDE render (base.html extends on every page — proven on a
# NON-review page so the badge is not accidentally scoped to
# review_queue.html), and the title-restore logic (never stacked prefixes).


def test_review_queue_count_api_zero_when_nothing_pending(client):
    """The count endpoint returns {"count": 0} when no target is awaiting
    review — the db_path fixture's targets are in scored/researched, so the
    badge must read as a plain link with no visible count."""
    resp = client.get("/api/review/queue/count")
    assert resp.status_code == 200
    assert resp.json() == {"count": 0}


def test_review_queue_count_api_returns_pending_count(review_client):
    """The count endpoint returns the number of awaiting_review targets —
    review_db seeds tgt_r1 and tgt_r2 in awaiting_review, so the badge must
    read 2.  (This is the CHEAP count-only shape — a single integer — never
    the full review payload the /api/review/queue route returns.)"""
    resp = review_client.get("/api/review/queue/count")
    assert resp.status_code == 200
    assert resp.json() == {"count": 2}


def test_review_queue_count_api_fails_closed_without_secret(monkeypatch, tmp_path):
    """The new route rides the SAME global require_operator as every other
    route: with no secret configured the console fails closed (503)."""
    monkeypatch.delenv("OUTBOUND_CONSOLE_API_KEY", raising=False)
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    resp = TestClient(app).get("/api/review/queue/count")
    assert resp.status_code == 503


def test_review_queue_count_api_wrong_credential_is_401(monkeypatch, tmp_path):
    """A wrong credential is a 401, never a 500 and never a silent 200 — the
    fail-closed doctrine applied to the new route exactly as to every other."""
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", "correct-secret")
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    resp = TestClient(app).get(
        "/api/review/queue/count", headers={"X-Internal-API-Key": "wrong-key"}
    )
    assert resp.status_code == 401


def test_review_queue_badge_renders_on_non_review_page(client):
    """U3 is SITE-WIDE: base.html renders on every page, so the badge element
    and its poll script must appear on a NON-review page (the index) — proving
    the notification is genuinely site-wide, not accidentally scoped to
    review_queue.html alone.  Also pins that the link still points at
    /review/queue and the poll targets the CHEAP count endpoint, never the
    heavy /api/review/queue full-payload route."""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    # The badge element and the (preserved) link live in the shared header.
    assert 'id="review-queue-count"' in body
    assert 'href="/review/queue"' in body
    # The poll script is present on this page and fetches the COUNT endpoint.
    script_blocks = _extract_script_blocks(body)
    assert script_blocks, (
        "the index page has no <script> block — base.html's U3 badge script "
        "is missing?"
    )
    combined = "\n".join(script_blocks)
    assert 'fetch("/api/review/queue/count")' in combined, (
        "the badge script does not fetch the cheap count endpoint — it must "
        "never poll the heavy /api/review/queue full-payload route for a badge"
    )
    # Belt-and-braces: the full-payload route is not what the script fetches
    # (fetch("/api/review/queue") would be an exact-match substring only if
    # the URL had no /count suffix).
    assert 'fetch("/api/review/queue")' not in combined


def test_base_badge_script_restores_original_title_without_stacking():
    """U3 title-restore logic: the badge script must capture the page's ORIGINAL
    title ONCE at load and compute every subsequent title from that stored
    baseline — never from document.title itself.  A naive
    `document.title = "(N) " + document.title` on every poll would stack
    prefixes across repeated polls ("(3) (3) (2) Outbound Agency Console").
    This pins the safe shape statically against the shipped base.html script:
    every document.title ASSIGNMENT's right-hand side must reference the
    stored ORIGINAL_TITLE and must never reference document.title.
    (The runtime behaviour — a fluctuating count like 2 → 0 → 3 → 0 ends
    correct at each step — is a direct consequence: each assignment replaces
    the title from a fixed baseline, so prefixes can never accumulate.)

    U3-fix guard: the checks above only look at document.title's RHS — they
    would sail past a line that reassigns the STORED BASELINE ITSELF
    (`ORIGINAL_TITLE = document.title;` after a title update — the
    adversarial finding that re-baselined the title mid-poll and re-opened
    the stacking bug).  A static count pins that ORIGINAL_TITLE is assigned
    EXACTLY ONCE, independently of how it is declared (var vs const), so a
    future editor who reverts const to var cannot silently reintroduce the
    gap."""
    text = _read_console_template("base.html")
    script_blocks = _extract_script_blocks(text)
    combined = "\n".join(script_blocks)
    # The baseline must be captured once, before any modification.
    assert re.search(r"ORIGINAL_TITLE\s*=\s*document\.title", combined), (
        "base.html's badge script never captures the ORIGINAL title once at "
        "load — the title restore cannot work without a stored baseline (U3)"
    )
    # …and it must be captured EXACTLY ONCE.  A reassignment of ORIGINAL_TITLE
    # (not of document.title) is invisible to the RHS checks below: the
    # reviewer's bypass added `ORIGINAL_TITLE = document.title;` after a title
    # update, keeping both document.title assignments textually compliant while
    # re-baselining the stored title each poll — the stacking bug again.  This
    # count catches that shape regardless of whether the declaration says var
    # or const (a const reassignment would throw at runtime, but the test must
    # hold even if someone reverts const for an unrelated reason).
    original_title_assignments = re.findall(r"ORIGINAL_TITLE\s*=", combined)
    assert len(original_title_assignments) == 1, (
        "ORIGINAL_TITLE must be assigned exactly once (the stored baseline), "
        f"found {len(original_title_assignments)}: a reassignment like "
        "ORIGINAL_TITLE = document.title after a title update would re-baseline "
        "the title mid-poll and re-open prefix stacking (U3-fix)"
    )
    # Every title assignment must be computed from the stored original.
    assignments = re.findall(r"document\.title\s*=\s*([^;]+);", combined)
    assert assignments, (
        "base.html's badge script has no document.title assignment — the U3 "
        "title-change feature is missing"
    )
    for rhs in assignments:
        assert "ORIGINAL_TITLE" in rhs, (
            "a document.title assignment does not derive from the stored "
            "ORIGINAL_TITLE — a naive document.title = '(N) ' + document.title "
            "would stack prefixes across polls (U3)"
        )
        assert "document.title" not in rhs, (
            "a document.title assignment reads document.title on the right — "
            "that stacks prefixes across polls (U3)"
        )
