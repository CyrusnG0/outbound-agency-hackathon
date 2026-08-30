import json

import pytest

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.write_gate import commit
from app.tools.fetch_sources import NormalizedSource
from app.tools.normalize_sources import normalize_sources


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
        params=("tgt_1", "acc_1", "off_1", "csv", "new"),
    )
    yield c
    c.close()


def test_combines_multiple_sources_into_one_text_blob(conn):
    sources = [
        NormalizedSource("company_website", "https://acme.test", "We do logistics.", "t", 0.8, 1, "static"),
        NormalizedSource("search_result", "https://news.test/1", "Acme raised Series A.", "t", 0.6, 2, "search"),
    ]
    text = normalize_sources(conn, sources=sources, target_id="tgt_1", run_id="r1", step_id="s1")
    assert "We do logistics." in text
    assert "Acme raised Series A." in text


def test_zero_sources_transitions_target_to_failed(conn):
    result = normalize_sources(conn, sources=[], target_id="tgt_1", run_id="r1", step_id="s1")
    assert result is None
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "no_sources_available"


def test_zero_sources_does_not_call_any_llm(conn, monkeypatch):
    def fail_if_called(*a, **k):
        raise AssertionError("LLM should never be called with zero sources")
    # normalize_sources itself makes no LLM calls; this test documents that
    # invariant so a future change can't silently add one.
    result = normalize_sources(conn, sources=[], target_id="tgt_1", run_id="r1", step_id="s1")
    assert result is None


def test_sources_with_no_extractable_text_fail_like_zero_sources(conn):
    # Production failure this test prevents: a source fetches successfully
    # (HTTP 200) but yields no extractable text — a JS-only page whose static
    # HTML is an empty shell.  Pre-A7, the empty combined string was handed
    # to summarize_company, which called the LLM with nothing and got Vertex
    # 400 "Model input cannot be empty" — a wasted call AND a misleading
    # llm_transport_error_phase1 label.  normalize_sources must instead fail
    # fast down the exact same no_sources_available route as zero sources.
    sources = [
        NormalizedSource("company_website", "https://acme.test", "", "t", 0.8, 1, "static"),
    ]
    result = normalize_sources(conn, sources=sources, target_id="tgt_1", run_id="r1", step_id="s1")
    # No text to summarize → None, exactly like the zero-sources case.
    assert result is None
    # Target must be routed to failed with the SAME §7c reason — no new
    # reason string is invented for "fetched but empty".
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "no_sources_available"
    # Never skip logging: the failed step row must exist (golden rule), and
    # it must be distinguishable from the zero-sources row — source_count 1
    # proves a source WAS fetched, unlike case (a)'s source_count 0.
    step_row = conn.execute(
        "SELECT input_json, output_json, status FROM steps "
        "WHERE target_id='tgt_1' AND tool_name='normalize_sources';"
    ).fetchone()
    assert step_row is not None
    assert step_row["status"] == "failed"
    assert json.loads(step_row["input_json"]) == {"source_count": 1}
    assert json.loads(step_row["output_json"]) == {"chars": 0}


def test_whitespace_only_text_fails_like_zero_sources(conn):
    # Production failure this test prevents: a naive `== ""` check misses
    # text that is ONLY whitespace ("\n\n   \t\n").  To the LLM it is exactly
    # as useless as "" and Vertex rejects it with the identical 400
    # INVALID_ARGUMENT.  The guard must strip before deciding nothing usable
    # exists — this is the case the .strip() comment names in the source.
    sources = [
        NormalizedSource("company_website", "https://acme.test", "\n\n   \t\n", "t", 0.8, 1, "static"),
    ]
    result = normalize_sources(conn, sources=sources, target_id="tgt_1", run_id="r1", step_id="s1")
    # Whitespace-only text must behave identically to "" and to zero sources.
    assert result is None
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "failed"
    transition_row = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()
    assert transition_row["reason"] == "no_sources_available"
    # The step row's chars field records the RAW length of the whitespace
    # blob (> 0), so an operator can see text DID arrive — it just stripped
    # to nothing — and tell this apart from both zero sources and pure "".
    step_row = conn.execute(
        "SELECT input_json, output_json, status FROM steps "
        "WHERE target_id='tgt_1' AND tool_name='normalize_sources';"
    ).fetchone()
    assert step_row is not None
    assert step_row["status"] == "failed"
    assert json.loads(step_row["input_json"]) == {"source_count": 1}
    assert json.loads(step_row["output_json"]) == {"chars": len("\n\n   \t\n")}


def test_mixed_sources_keep_the_good_text_when_one_sibling_is_empty(conn):
    # Production failure this test prevents: an over-broad fix that fails the
    # whole batch when ANY source is empty.  The guard must fire only when
    # there is nothing usable AT ALL — one empty sibling must never discard a
    # good source's real text.  This is the test that stops the fix from
    # regressing the common case where one of several fetches comes back
    # empty and the rest carry the content.
    sources = [
        NormalizedSource("company_website", "https://acme.test", "", "t", 0.8, 1, "static"),
        NormalizedSource("search_result", "https://news.test/1", "Acme raised Series A.", "t", 0.6, 2, "search"),
    ]
    result = normalize_sources(conn, sources=sources, target_id="tgt_1", run_id="r1", step_id="s1")
    # The combined blob is "\n\nAcme raised Series A." — not None, and the
    # real text survives the empty sibling's blank contribution.
    assert result is not None
    assert "Acme raised Series A." in result
    # The target must NOT be routed to failed — usable text exists, so the
    # pipeline continues toward summarization.
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "new"
    step_row = conn.execute(
        "SELECT status FROM steps "
        "WHERE target_id='tgt_1' AND tool_name='normalize_sources';"
    ).fetchone()
    assert step_row["status"] == "success"
