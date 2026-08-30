import json
from unittest.mock import patch

import pytest

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.write_gate import commit
from app.llm import (
    LLMEmptyResponseError,
    LLMSchemaValidationError,
    LLMTransportError,
    TRANSPORT_RETRY_SLEEP_SECONDS,
)
from app.schemas import CompanyProfile
from app.tools.summarize_company import summarize_company


@pytest.fixture
def conn(scratch_db_target):
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # Register the system agent (plan A3) — commit() refuses unregistered agents.
    seed_agent_registry(c, run_id="r0", step_id="s0")
    commit(
        c, action="insert_offer", table_name="offers", record_id="off_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "acme", 1),
    )
    commit(
        c, action="insert_account", table_name="accounts", record_id="acc_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain, created_at, updated_at)
               VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_1", "Acme", "acme.test", "acme.test"),
    )
    commit(
        c, action="insert_target", table_name="targets", record_id="tgt_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, offer_id, source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("tgt_1", "acc_1", "off_1", "csv", "researched"),
    )
    yield c
    c.close()


def test_successful_call_returns_company_profile(conn):
    fake_profile = CompanyProfile(
        one_line_summary="B2B logistics software", industry="Logistics SaaS",
        estimated_size="51-200", geo="US", confidence=0.82,
    )
    with patch("app.tools.summarize_company.call_structured", return_value=fake_profile):
        result = summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    assert result == fake_profile
    step = conn.execute(
        "SELECT * FROM steps WHERE tool_name='summarize_company' AND target_id='tgt_1';"
    ).fetchone()
    assert step["status"] == "success"
    assert step["model_call_hash"] is not None


def test_malformed_output_retries_once_then_fails_target(conn):
    with patch(
        "app.tools.summarize_company.call_structured",
        side_effect=LLMSchemaValidationError("bad json"),
    ):
        result = summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    assert result is None
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "llm_output_invalid_phase1"


def test_succeeds_on_retry_after_one_failure(conn):
    fake_profile = CompanyProfile(
        one_line_summary="B2B logistics software", confidence=0.7,
    )
    with patch(
        "app.tools.summarize_company.call_structured",
        side_effect=[LLMEmptyResponseError("empty"), fake_profile],
    ):
        result = summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    assert result == fake_profile
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "researched"  # unchanged — only failure path transitions state here


# ── Transport error tests (LLMTransportError) ─────────────────────────────────
# These build LLMTransportError instances directly (the caller tests patch
# call_structured, so the SDK exception never appears here — the SDK-level
# construction is covered in tests/test_llm.py).  Every test patches
# time.sleep so the suite never actually pauses for TRANSPORT_RETRY_SLEEP_SECONDS.

def _transport_error(status_code: int | None, retryable: bool) -> LLMTransportError:
    """Build a transport error the way call_structured would raise it."""
    status_part = f"status {status_code}" if status_code is not None else "no HTTP response"
    return LLMTransportError(
        f"anthropic transport error ({status_part}): rate limited",
        provider="anthropic", status_code=status_code, retryable=retryable,
    )


def test_retryable_transport_error_retries_once_then_succeeds(conn):
    fake_profile = CompanyProfile(
        one_line_summary="B2B logistics software", confidence=0.7,
    )
    with patch(
        "app.tools.summarize_company.call_structured",
        side_effect=[_transport_error(429, True), fake_profile],
    ) as mock_call, patch("app.tools.summarize_company.time.sleep") as mock_sleep:
        result = summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    # A retryable transport error must consume the bounded retry and succeed
    # on the second attempt — exactly two calls, no more, no fewer.
    assert result == fake_profile
    assert mock_call.call_count == 2
    # The fixed pause fires once, before the retry, with the shared constant.
    mock_sleep.assert_called_once_with(TRANSPORT_RETRY_SLEEP_SECONDS)
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "researched"  # unchanged — success path never transitions


def test_non_retryable_transport_error_fails_without_second_attempt(conn):
    with patch(
        "app.tools.summarize_company.call_structured",
        side_effect=_transport_error(401, False),
    ) as mock_call, patch("app.tools.summarize_company.time.sleep") as mock_sleep:
        result = summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    assert result is None
    # THE assertion this test exists for: a 401 must NOT burn the second
    # attempt — the retry loop breaks after logging the first failure.
    assert mock_call.call_count == 1
    mock_sleep.assert_not_called()
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "llm_transport_error_phase1"
    # Log FIRST, break SECOND — the failed attempt still left a steps row
    # carrying the transport error details (machine-readable error_type plus
    # retryable/status_code for the operator).
    step = conn.execute(
        "SELECT * FROM steps WHERE tool_name='summarize_company' AND target_id='tgt_1';"
    ).fetchone()
    assert step["status"] == "failed"
    output = json.loads(step["output_json"])
    assert output["error_type"] == "LLMTransportError"
    assert output["retryable"] is False
    assert output["status_code"] == 401


def test_two_retryable_transport_errors_fail_target_with_transport_reason(conn):
    with patch(
        "app.tools.summarize_company.call_structured",
        side_effect=[_transport_error(429, True), _transport_error(503, True)],
    ) as mock_call, patch("app.tools.summarize_company.time.sleep") as mock_sleep:
        result = summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    assert result is None
    assert mock_call.call_count == 2
    # Sleep exactly once with the shared constant: after the FIRST failure,
    # and NOT after the final one — sleeping after the last attempt is pure
    # dead wall-clock and this assertion proves it doesn't happen.
    mock_sleep.assert_called_once_with(TRANSPORT_RETRY_SLEEP_SECONDS)
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "llm_transport_error_phase1"


def test_two_empty_responses_fail_target_with_output_invalid_reason(conn):
    # Regression: output errors (here: both attempts empty) must keep the
    # original reason string — the transport reason is reserved for
    # LLMTransportError, and the two categories must never bleed together.
    with patch(
        "app.tools.summarize_company.call_structured",
        side_effect=LLMEmptyResponseError("model returned no tool_use block"),
    ) as mock_call, patch("app.tools.summarize_company.time.sleep") as mock_sleep:
        result = summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    assert result is None
    assert mock_call.call_count == 2
    mock_sleep.assert_not_called()  # output errors never pause — no transport involved
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "llm_output_invalid_phase1"


# ── Persistence tests (ticket A8) ─────────────────────────────────────────────
# The profile used to be returned and then thrown away: accounts.company_summary
# stayed NULL on every real run (only score_lead's icp_fit_* writes persisted),
# so the operator console's Company section rendered blank and the state-machine
# §7d checkpoint ("are these briefs as good as manual research?") was
# unanswerable. These tests pin the fix: the write goes through the gate, only
# on the success path, and touches only the four mapped columns.

def test_successful_summary_is_persisted_to_the_account_row(conn):
    # The regression this ticket closes: a successful summarize_company call
    # must leave the researched profile ON the account row — not just in
    # steps.output_json, where it used to die.
    fake_profile = CompanyProfile(
        one_line_summary="B2B logistics software", industry="Logistics SaaS",
        estimated_size="51-200", geo="US", confidence=0.82,
    )
    with patch("app.tools.summarize_company.call_structured", return_value=fake_profile):
        summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    # The account row the fixture created for tgt_1 must now carry the four
    # mapped fields, each equal to its CompanyProfile counterpart.
    account = conn.execute(
        "SELECT company_summary, industry, estimated_size, geo FROM accounts WHERE account_id='acc_1';"
    ).fetchone()
    assert account["company_summary"] == "B2B logistics software"
    assert account["industry"] == "Logistics SaaS"
    assert account["estimated_size"] == "51-200"
    assert account["geo"] == "US"


def test_failed_summary_persists_nothing_to_the_account_row(conn):
    # Guard for the success-path-only placement: two empty responses mean the
    # target fails — and because the persistence write lives INSIDE the
    # success branch, the account row must remain completely untouched. This
    # is the test that stops someone moving the write outside the branch.
    with patch(
        "app.tools.summarize_company.call_structured",
        side_effect=LLMEmptyResponseError("model returned no tool_use block"),
    ):
        result = summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    assert result is None
    account = conn.execute(
        "SELECT company_summary, industry, estimated_size, geo FROM accounts WHERE account_id='acc_1';"
    ).fetchone()
    assert account["company_summary"] is None
    assert account["industry"] is None
    assert account["estimated_size"] is None
    assert account["geo"] is None
    # No audited profile write may exist either — the gate's write_log is the
    # trail that would expose a write snuck onto the failure path.
    gates = conn.execute(
        "SELECT count(*) AS n FROM write_log WHERE action='update_account_profile';"
    ).fetchone()
    assert gates["n"] == 0
    # And the target still failed, exactly as before this ticket.
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"


def test_confidence_is_not_persisted_and_sql_touches_only_the_mapped_columns(conn):
    # Confidence has no accounts column (per the ticket, schema changes are
    # out of scope) — it lives only in steps.output_json. This test proves
    # the accounts UPDATE touches exactly company_summary / industry /
    # estimated_size / geo plus updated_at: the spy wraps (not replaces) the
    # real write_gate.commit, so the real gate runs and is audited — only the
    # call's arguments are captured for inspection.
    fake_profile = CompanyProfile(
        one_line_summary="B2B logistics software", industry="Logistics SaaS",
        estimated_size="51-200", geo="US", confidence=0.42,
    )
    with patch(
        "app.tools.summarize_company.call_structured", return_value=fake_profile
    ), patch(
        "app.tools.summarize_company.write_gate_commit", wraps=commit
    ) as mock_commit:
        summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    # Exactly one gated write on the success path — nothing more, nothing less.
    assert mock_commit.call_count == 1
    call_kwargs = mock_commit.call_args.kwargs
    assert call_kwargs["action"] == "update_account_profile"
    # The SQL's SET clause touches only the four mapped columns plus
    # updated_at: every expected identifier is present, confidence is not.
    sql = call_kwargs["sql"]
    for column in ("company_summary", "industry", "estimated_size", "geo", "updated_at"):
        assert column in sql
    assert "confidence" not in sql
    # The params tuple maps 1:1 onto the SQL's five ? placeholders: four SET
    # values (in column order) + the WHERE account_id.
    assert call_kwargs["params"] == (
        "B2B logistics software", "Logistics SaaS", "51-200", "US", "acc_1",
    )
    # The audit payload carries exactly the four mapped fields — confidence
    # is excluded there too, so nothing in the write holds it.
    assert set(call_kwargs["payload"].keys()) == {
        "company_summary", "industry", "estimated_size", "geo",
    }
    # And the accounts row itself has no column that could hold confidence.
    account = conn.execute("SELECT * FROM accounts WHERE account_id='acc_1';").fetchone()
    assert "confidence" not in account.keys()


def test_successful_summary_write_goes_through_the_write_gate(conn):
    # The gate's whole point: a raw conn.execute for the accounts update
    # would leave correct-looking values and no trace — invisible to the
    # operator. This test pins the audit row itself: exactly one write_log
    # row for action="update_account_profile", pointing at the account row
    # with the actor and agent recorded.
    fake_profile = CompanyProfile(
        one_line_summary="B2B logistics software", industry="Logistics SaaS",
        estimated_size="51-200", geo="US", confidence=0.82,
    )
    with patch("app.tools.summarize_company.call_structured", return_value=fake_profile):
        summarize_company(
            conn, extracted_text="Acme does logistics software.",
            target_id="tgt_1", run_id="r1", step_id="s1",
        )
    rows = conn.execute(
        "SELECT * FROM write_log WHERE action='update_account_profile';"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["table_name"] == "accounts"
    assert rows[0]["record_id"] == "acc_1"
    assert rows[0]["actor"] == "system"
    assert rows[0]["agent_id"] == "system"
    # The audit payload round-trips the persisted data as JSON.
    payload = json.loads(rows[0]["payload_json"])
    assert payload["company_summary"] == "B2B logistics software"
    assert payload["industry"] == "Logistics SaaS"
    assert payload["estimated_size"] == "51-200"
    assert payload["geo"] == "US"
