import json
from unittest.mock import patch

import pytest

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.ids import new_id  # fresh ids for the raw-source fixture rows the three-tier tests seed
from app.write_gate import commit
from app.llm import (
    LLMEmptyResponseError,
    LLMSchemaValidationError,
    LLMTransportError,
    TRANSPORT_RETRY_SLEEP_SECONDS,
)
from app.schemas import Signal
from app.tools.detect_signals import detect_signals
from app.tools.fetch_sources import (  # the REAL write seam + the findings marker — B2b fixture rows must use the production path, not a hand-rolled INSERT
    FINDINGS_SOURCE_TYPE,
    persist_source_row,
)


class _SignalList:
    """Helper: call_structured for detect_signals must return something
    validating to list[Signal] — mocked here as a simple wrapper."""
    def __init__(self, signals):
        self.signals = signals


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


def test_successful_call_persists_signals_to_db(conn):
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Hiring ops manager",
            signal_strength=0.73,
            # B2a: every Signal now requires an evidence quote — these
            # fixtures only assert on persistence/retry behaviour, so the
            # quote is a placeholder that need not appear in the text.
            evidence_quote="hiring an operations manager for the team",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        result = detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s1")
    assert result == fake_signals
    row = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchone()
    assert row["signal_type"] == "hiring_relevant_role"
    assert row["run_id"] == "r1"


def test_empty_signal_list_is_a_valid_success_not_a_failure(conn):
    with patch("app.tools.detect_signals._call_detect_signals", return_value=[]):
        result = detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s1")
    assert result == []
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "researched"  # untouched — no failure occurred
    step = conn.execute(
        "SELECT status FROM steps WHERE tool_name='detect_signals' AND target_id='tgt_1';"
    ).fetchone()
    assert step["status"] == "success"


def test_malformed_output_retries_once_then_fails_target(conn):
    with patch(
        "app.tools.detect_signals._call_detect_signals",
        side_effect=LLMSchemaValidationError("bad json"),
    ):
        result = detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s1")
    assert result is None
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    # Output-invalid failures keep the original reason string (extended with
    # this assert by the transport-error task: the reason must NOT become
    # llm_transport_error_phase1 for a non-transport failure).
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "llm_output_invalid_phase1"


def test_rerun_on_same_target_does_not_duplicate_signal_rows(conn):
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Hiring ops manager",
            signal_strength=0.73,
            # B2a: every Signal now requires an evidence quote — these
            # fixtures only assert on persistence/retry behaviour, so the
            # quote is a placeholder that need not appear in the text.
            evidence_quote="hiring an operations manager for the team",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s1")
        detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s2")
    rows = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchall()
    assert len(rows) == 1  # same run_id + same signal -> UNIQUE constraint dedupes


# ── Transport error tests (LLMTransportError) ─────────────────────────────────
# Same shape as tests/test_summarize_company.py's transport tests: these patch
# _call_detect_signals (the mock seam for the LLM call) with LLMTransportError
# instances built directly — SDK-level construction is covered in
# tests/test_llm.py.  Every test patches time.sleep so the suite never
# actually pauses for TRANSPORT_RETRY_SLEEP_SECONDS.

def _transport_error(status_code: int | None, retryable: bool) -> LLMTransportError:
    """Build a transport error the way call_structured would raise it."""
    status_part = f"status {status_code}" if status_code is not None else "no HTTP response"
    return LLMTransportError(
        f"anthropic transport error ({status_part}): rate limited",
        provider="anthropic", status_code=status_code, retryable=retryable,
    )


def test_retryable_transport_error_retries_once_then_succeeds(conn):
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Hiring ops manager",
            signal_strength=0.73,
            # B2a: every Signal now requires an evidence quote — these
            # fixtures only assert on persistence/retry behaviour, so the
            # quote is a placeholder that need not appear in the text.
            evidence_quote="hiring an operations manager for the team",
        ),
    ]
    with patch(
        "app.tools.detect_signals._call_detect_signals",
        side_effect=[_transport_error(429, True), fake_signals],
    ) as mock_call, patch("app.tools.detect_signals.time.sleep") as mock_sleep:
        result = detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s1")
    # A retryable transport error must consume the bounded retry and succeed
    # on the second attempt — exactly two calls, no more, no fewer.
    assert result == fake_signals
    assert mock_call.call_count == 2
    # The fixed pause fires once, before the retry, with the shared constant.
    mock_sleep.assert_called_once_with(TRANSPORT_RETRY_SLEEP_SECONDS)
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "researched"  # unchanged — success path never transitions


def test_non_retryable_transport_error_fails_without_second_attempt(conn):
    with patch(
        "app.tools.detect_signals._call_detect_signals",
        side_effect=_transport_error(401, False),
    ) as mock_call, patch("app.tools.detect_signals.time.sleep") as mock_sleep:
        result = detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s1")
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
        "SELECT * FROM steps WHERE tool_name='detect_signals' AND target_id='tgt_1';"
    ).fetchone()
    assert step["status"] == "failed"
    output = json.loads(step["output_json"])
    assert output["error_type"] == "LLMTransportError"
    assert output["retryable"] is False
    assert output["status_code"] == 401


def test_two_retryable_transport_errors_fail_target_with_transport_reason(conn):
    with patch(
        "app.tools.detect_signals._call_detect_signals",
        side_effect=[_transport_error(429, True), _transport_error(503, True)],
    ) as mock_call, patch("app.tools.detect_signals.time.sleep") as mock_sleep:
        result = detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s1")
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
        "app.tools.detect_signals._call_detect_signals",
        side_effect=LLMEmptyResponseError("model returned no tool_use block"),
    ) as mock_call, patch("app.tools.detect_signals.time.sleep") as mock_sleep:
        result = detect_signals(conn, extracted_text="...", target_id="tgt_1", run_id="r1", step_id="s1")
    assert result is None
    assert mock_call.call_count == 2
    mock_sleep.assert_not_called()  # output errors never pause — no transport involved
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "llm_output_invalid_phase1"


# ── Evidence verification tests (plan tasks B2a + B2b) ─────────────────────
# Every signal carries an evidence_quote, and detect_signals records a
# THREE-way verdict (ticket B2b, extending B2a's boolean): 'source' = the
# quote appears in a persisted RAW source text we actually fetched (the
# strongest tier); 'findings' = it appears only in the extracted_text the
# tool was given (the research agent's prose — plausibly from a server-side
# search we cannot capture; NOT a failure); 'unverified' = it appears in
# neither — the fabrication signal.  The check is deliberately deterministic
# — verbatim containment after whitespace normalisation — never an LLM
# re-reading the text, because a model that hallucinated a claim can
# hallucinate agreement with it just as easily.  INVARIANT (B2b):
# evidence_verified = 1 if and only if evidence_tier = 'source' — the two
# columns are written from one computation and can never disagree.  All of
# these tests patch _call_detect_signals, the same mock seam as every test
# above — no live API calls anywhere — and assert against the DATABASE
# (the B1c lesson: a return-value-only check once passed with all
# credentials stripped).

_VERBATIM_TEXT = (
    "Acme Logistics handles bookings on paper and pen. "
    "They are hiring an operations manager to triage WhatsApp and email requests."
)


def _insert_raw_source(conn, text: str, run_id: str = "r1") -> str:
    """Persist one raw fetched page through the REAL write seam
    (fetch_sources.persist_source_row) — the same path fetch_sources uses —
    so these tests exercise the actual sources-table contract, not a
    hand-rolled INSERT that could drift from it."""
    return persist_source_row(
        conn,
        source_id=new_id("src"),
        run_id=run_id,
        target_id="tgt_1",
        step_id=new_id("step"),
        source_type="company_website",
        source_url="https://acme.test",
        extracted_text=text,
        source_confidence=0.8,
        source_priority=1,
        extraction_method="static",
    )


def test_quote_matching_raw_source_gets_source_tier_and_verified(conn):
    # The strongest tier: the quote appears character-for-character in a
    # persisted raw page (the text we actually fetched).  evidence_tier must
    # be 'source' AND evidence_verified must be 1 — the B2b invariant —
    # asserted against the persisted signals row, not the return value.
    _insert_raw_source(conn, _VERBATIM_TEXT)
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Hiring ops manager",
            signal_strength=0.73,
            evidence_quote="hiring an operations manager to triage WhatsApp and email requests",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        result = detect_signals(conn, extracted_text="Acme Logistics is a logistics company.", target_id="tgt_1", run_id="r1", step_id="s1")
    assert result == fake_signals
    row = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchone()
    assert row["evidence_quote"] == fake_signals[0].evidence_quote
    assert row["evidence_tier"] == "source"
    assert row["evidence_verified"] == 1  # the invariant: verified == (tier == 'source')


def test_quote_matching_only_findings_gets_findings_tier(conn):
    # The findings tier: the quote is in the extracted_text the signal agent
    # was GIVEN (the research agent's prose) but in NO persisted raw page.
    # This is NOT a failure and must not read as one: it means "trust the
    # research agent, we cannot independently check" — the legitimate state
    # for quotes derived from the server-side google_search/url_context tools
    # whose text never passes through this process.  evidence_verified must
    # be 0 (verified now means verified against raw text), tier 'findings'.
    _insert_raw_source(conn, "Acme Logistics delivers packages across the region.")
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Hiring ops manager",
            signal_strength=0.73,
            evidence_quote="hiring an operations manager to triage WhatsApp and email requests",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        detect_signals(conn, extracted_text=_VERBATIM_TEXT, target_id="tgt_1", run_id="r1", step_id="s1")
    row = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchone()
    assert row is not None  # persisted — findings is a recorded tier, never a drop
    assert row["evidence_tier"] == "findings"
    assert row["evidence_verified"] == 0


def test_quote_matching_neither_gets_unverified_tier_and_is_not_dropped(conn):
    # The fabrication signal: the quote is in NEITHER the persisted raw pages
    # NOR the findings text — the signal agent produced text that is in no
    # source it was given.  The signal must STILL be persisted, with tier
    # 'unverified' and evidence_verified = 0.  Dropping it would hide the
    # fabrication: deleting turns a detectable lie into an invisible one, and
    # the whole point of B2a/B2b is to make fabrication visible to the
    # operator (and later the ICPJudge) so they can decide what to trust —
    # not to silently erase it.
    _insert_raw_source(conn, _VERBATIM_TEXT)
    fake_signals = [
        Signal(
            signal_type="workflow_complexity_evidence",
            signal_value="Manually triages bookings",
            signal_strength=0.9,
            evidence_quote="the clinic triages all WhatsApp bookings manually every morning",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        result = detect_signals(conn, extracted_text=_VERBATIM_TEXT, target_id="tgt_1", run_id="r1", step_id="s1")
    assert result == fake_signals
    row = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchone()
    assert row is not None  # persisted — NOT dropped, NOT skipped
    assert row["evidence_tier"] == "unverified"
    assert row["evidence_verified"] == 0


def test_signal_in_every_tier_is_written_mark_dont_drop(conn):
    # Mark-don't-drop across the WHOLE tier space: one signal per tier must
    # all land in the signals table — no tier is a reason to discard a
    # signal.  The three quotes: one in the raw page only, one in the
    # findings only, one in neither.  Each pair uses a distinct
    # (signal_type, signal_value) so the UNIQUE dedup constraint doesn't
    # collapse them.
    _insert_raw_source(conn, "The company ships pallets on paper forms.")
    findings = "The company runs bookings on paper and pen. They are expanding their warehouse this quarter."
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Paper shipping forms",
            signal_strength=0.6,
            evidence_quote="ships pallets on paper forms",
        ),
        Signal(
            signal_type="workflow_complexity_evidence",
            signal_value="Paper bookings",
            signal_strength=0.6,
            evidence_quote="runs bookings on paper and pen",
        ),
        Signal(
            signal_type="recent_launch_or_expansion",
            signal_value="Warehouse expansion",
            signal_strength=0.6,
            evidence_quote="opened three new clinics last month",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        detect_signals(conn, extracted_text=findings, target_id="tgt_1", run_id="r1", step_id="s1")
    rows = conn.execute(
        "SELECT evidence_tier FROM signals WHERE target_id='tgt_1' ORDER BY created_at;"
    ).fetchall()
    tiers = sorted(r["evidence_tier"] for r in rows)
    assert tiers == ["findings", "source", "unverified"]  # all three persisted — none dropped


def test_raw_source_wins_over_findings_when_quote_is_in_both(conn):
    # The tier ORDERING is load-bearing (ticket B2b): a quote present in
    # BOTH a raw page and the findings must record as 'source' — the
    # strongest attribution always wins.  If the check were reordered (or
    # collapsed to "check the findings text first"), this quote would be
    # downgraded to 'findings' and the strongest evidence would be recorded
    # as the weakest attributable one — this test pins the order.
    _insert_raw_source(conn, _VERBATIM_TEXT)
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Hiring ops manager",
            signal_strength=0.73,
            evidence_quote="hiring an operations manager to triage WhatsApp and email requests",
        ),
    ]
    # The findings text ALSO contains the quote — the raw page must still win.
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        detect_signals(conn, extracted_text=_VERBATIM_TEXT, target_id="tgt_1", run_id="r1", step_id="s1")
    row = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchone()
    assert row["evidence_tier"] == "source"


def test_whitespace_only_quote_difference_in_raw_source_is_source_tier(conn):
    # False-positive guard: a faithful quote whose line breaks and spacing
    # were reflowed must still verify against the raw page.  "paper  and\npen"
    # normalises to "paper and pen", which appears in the text — without the
    # whitespace normalisation this HONEST quote would be flagged as
    # fabricated, and the check would punish faithful quoting.
    _insert_raw_source(conn, _VERBATIM_TEXT)
    fake_signals = [
        Signal(
            signal_type="workflow_complexity_evidence",
            signal_value="Paper workflows",
            signal_strength=0.6,
            evidence_quote="handles bookings on paper  and\npen",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        detect_signals(conn, extracted_text="unrelated findings text", target_id="tgt_1", run_id="r1", step_id="s1")
    row = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchone()
    assert row["evidence_tier"] == "source"
    assert row["evidence_verified"] == 1


def test_genuinely_different_quote_text_is_unverified(conn):
    # The strictness guard paired with the test above, pinning the other
    # direction: this quote is ALMOST the text but genuinely different
    # ("a digital" inserted, "pen" dropped) — it must land in the
    # 'unverified' tier (not findings, not source).  If the normalisation is
    # ever loosened into fuzzy or substring-of-substring matching, this test
    # fails and says so; whitespace-only is the whole allowance, because
    # that is the only difference faithful quoting produces.
    _insert_raw_source(conn, _VERBATIM_TEXT)
    fake_signals = [
        Signal(
            signal_type="workflow_complexity_evidence",
            signal_value="Paper workflows",
            signal_strength=0.6,
            evidence_quote="handles bookings on paper and a digital pen",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        detect_signals(conn, extracted_text=_VERBATIM_TEXT, target_id="tgt_1", run_id="r1", step_id="s1")
    row = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchone()
    assert row["evidence_tier"] == "unverified"
    assert row["evidence_verified"] == 0


def test_findings_rows_are_never_counted_as_raw_sources(conn):
    # The exclusion filter's guard: a research_findings row must NOT satisfy
    # the raw-text check — if FINDINGS_SOURCE_TYPE ever leaked into the raw
    # filter, a quote matching only agent prose would be mis-recorded as
    # 'source' (attributable to a stored page it never was).  Persist the
    # SAME text as both a raw page and a findings row, then delete the raw
    # row and confirm the tier falls back to 'findings' — proving the
    # findings row alone cannot produce 'source'.
    persist_source_row(
        conn,
        source_id=new_id("src"),
        run_id="r1",
        target_id="tgt_1",
        step_id=new_id("step"),
        source_type=FINDINGS_SOURCE_TYPE,
        source_url=None,
        extracted_text=_VERBATIM_TEXT,
        source_confidence=None,
        source_priority=None,
        extraction_method="agent",
    )
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Hiring ops manager",
            signal_strength=0.73,
            evidence_quote="hiring an operations manager to triage WhatsApp and email requests",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        detect_signals(conn, extracted_text="unrelated findings text", target_id="tgt_1", run_id="r1", step_id="s1")
    row = conn.execute("SELECT * FROM signals WHERE target_id='tgt_1';").fetchone()
    assert row["evidence_tier"] == "unverified"  # the findings row was NOT counted as raw — quote matched neither checked text


def test_step_log_reports_three_tier_counts(conn):
    # The success steps row must carry the three-tier split so an operator
    # scanning the trace sees evidence quality per target without querying
    # the signals table — one source + one findings + one unverified quote
    # below pins the split at 1/1/1.  The three signals use different
    # (signal_type, signal_value) pairs so the UNIQUE dedup constraint
    # doesn't collapse them into one row.
    _insert_raw_source(conn, "The company ships pallets on paper forms.")
    fake_signals = [
        Signal(
            signal_type="hiring_relevant_role",
            signal_value="Paper shipping forms",
            signal_strength=0.6,
            evidence_quote="ships pallets on paper forms",
        ),
        Signal(
            signal_type="workflow_complexity_evidence",
            signal_value="Paper bookings",
            signal_strength=0.6,
            evidence_quote="runs bookings on paper and pen",
        ),
        Signal(
            signal_type="recent_launch_or_expansion",
            signal_value="Warehouse expansion",
            signal_strength=0.6,
            evidence_quote="opened three new clinics last month",
        ),
    ]
    with patch("app.tools.detect_signals._call_detect_signals", return_value=fake_signals):
        detect_signals(conn, extracted_text="The company runs bookings on paper and pen.", target_id="tgt_1", run_id="r1", step_id="s1")
    step = conn.execute(
        "SELECT * FROM steps WHERE tool_name='detect_signals' AND target_id='tgt_1';"
    ).fetchone()
    assert step["status"] == "success"
    output = json.loads(step["output_json"])
    assert output["signal_count"] == 3
    assert output["source_count"] == 1
    assert output["findings_count"] == 1
    assert output["unverified_count"] == 1
