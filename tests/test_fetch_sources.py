from unittest.mock import patch

import pytest

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.tools.fetch_sources import fetch_sources, NormalizedSource


@pytest.fixture
def conn(scratch_db_target):
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # B2b: fetch_sources now persists every successful fetch through the
    # write gate (insert_source), and commit() refuses unregistered agents —
    # so this fixture must seed the registry exactly like the pipeline
    # fixtures in test_detect_signals.py / test_agents_phase1.py do.
    seed_agent_registry(c, run_id="r0", step_id="s0")
    yield c
    c.close()


def test_successful_fetch_returns_normalized_source(conn):
    with patch("app.tools.fetch_sources._fetch_static_page") as mock_fetch:
        mock_fetch.return_value = "<html><body><h1>Acme Inc</h1><p>We do logistics.</p></body></html>"
        sources = fetch_sources(conn, domain="acme.test", target_id="tgt_1", run_id="r1", step_id="s1")
    assert len(sources) == 1
    assert isinstance(sources[0], NormalizedSource)
    assert sources[0].source_type == "company_website"
    assert "Acme Inc" in sources[0].extracted_text
    assert sources[0].extraction_method == "static"


def test_successful_fetch_persists_raw_source_text_to_sources_table(conn):
    # B2b's persistence requirement, asserted against the DATABASE (not the
    # return value — the B1c lesson): the raw text must survive the run in
    # the sources table, because that row is the ground truth detect_signals
    # fact-checks evidence_quote against after the fetch.
    with patch("app.tools.fetch_sources._fetch_static_page") as mock_fetch:
        mock_fetch.return_value = "<html><body><h1>Acme Inc</h1><p>We hire operations staff.</p></body></html>"
        fetch_sources(conn, domain="acme.test", target_id="tgt_1", run_id="r1", step_id="s1")
    rows = conn.execute(
        "SELECT * FROM sources WHERE target_id='tgt_1' AND run_id='r1';"
    ).fetchall()
    assert len(rows) == 1  # exactly one evidence row per successful fetch
    row = rows[0]
    assert row["source_type"] == "company_website"  # raw fetched page, not agent prose
    assert row["source_url"] == "https://acme.test"
    assert "hire operations staff" in row["extracted_text"]  # the raw text itself, persisted verbatim
    assert row["source_confidence"] == 0.8  # mirrors the NormalizedSource dataclass value
    assert row["source_priority"] == 1
    assert row["extraction_method"] == "static"
    # The write must have gone through the gate — a write_log row records it.
    write = conn.execute(
        "SELECT * FROM write_log WHERE action='insert_source' AND record_id=?;",
        (row["source_id"],),
    ).fetchone()
    assert write is not None


def test_failed_fetch_persists_no_source_row(conn):
    # A fetch that never produced text has no evidence to persist — the
    # sources table must stay empty for that attempt, and the failure is
    # recorded only in the steps row (the existing B2a-era contract).
    with patch("app.tools.fetch_sources._fetch_static_page", side_effect=TimeoutError("timed out")):
        sources = fetch_sources(conn, domain="acme.test", target_id="tgt_1", run_id="r1", step_id="s1")
    assert sources == []
    rows = conn.execute("SELECT * FROM sources;").fetchall()
    assert rows == []  # nothing fetched = nothing persisted


def test_persistence_failure_raises_instead_of_silently_losing_evidence(conn):
    # B2b's loud-failure contract (docs/data-flow.md §9i): a fetch/parse
    # failure is rescued, but a PERSISTENCE failure must propagate — if the
    # raw text were silently lost, detect_signals would mark every signal
    # derived from it 'unverified' (the fabrication label), which is worse
    # than failing the target.  Simulate the failure by patching the write
    # gate to raise; the fetch itself succeeds, so only the persistence seam
    # is exercised.
    with patch("app.tools.fetch_sources._fetch_static_page") as mock_fetch, \
         patch("app.tools.fetch_sources.write_gate_commit", side_effect=RuntimeError("db down")):
        mock_fetch.return_value = "<html><body><p>Acme Inc</p></body></html>"
        with pytest.raises(RuntimeError, match="db down"):
            fetch_sources(conn, domain="acme.test", target_id="tgt_1", run_id="r1", step_id="s1")
    # No evidence row was written — the raise happened before any commit.
    rows = conn.execute("SELECT * FROM sources;").fetchall()
    assert rows == []


def test_timeout_is_rescued_and_logged_not_raised(conn):
    with patch("app.tools.fetch_sources._fetch_static_page", side_effect=TimeoutError("timed out")):
        sources = fetch_sources(conn, domain="acme.test", target_id="tgt_1", run_id="r1", step_id="s1")
    assert sources == []
    step = conn.execute(
        "SELECT * FROM steps WHERE tool_name='fetch_company_page' AND target_id='tgt_1';"
    ).fetchone()
    assert step["status"] == "failed"


def test_blocked_page_is_rescued_and_logged(conn):
    with patch("app.tools.fetch_sources._fetch_static_page", side_effect=PermissionError("403 blocked")):
        sources = fetch_sources(conn, domain="acme.test", target_id="tgt_1", run_id="r1", step_id="s1")
    assert sources == []


def test_never_raises_even_on_unexpected_error(conn):
    with patch("app.tools.fetch_sources._fetch_static_page", side_effect=ValueError("unexpected")):
        sources = fetch_sources(conn, domain="acme.test", target_id="tgt_1", run_id="r1", step_id="s1")
    assert sources == []
