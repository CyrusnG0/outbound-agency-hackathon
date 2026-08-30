import json

import pytest

from app.db import connect, apply_schema
from app.tools.log_step import log_step


@pytest.fixture
def conn(scratch_db_target):
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    yield c
    c.close()


def test_log_step_writes_a_steps_row(conn):
    log_step(
        conn,
        run_id="run_1",
        step_id="step_1",
        target_id="tgt_1",
        tool_name="fetch_sources",
        agent_id="system",  # Required since plan A3 — which registered agent ran this step.
        input_data={"domain": "acme.test"},
        output_data={"sources_found": 3},
        status="success",
    )
    row = conn.execute("SELECT * FROM steps WHERE step_id='step_1';").fetchone()
    assert row["tool_name"] == "fetch_sources"
    assert row["agent_id"] == "system"
    assert json.loads(row["input_json"]) == {"domain": "acme.test"}
    assert json.loads(row["output_json"]) == {"sources_found": 3}
    assert row["status"] == "success"
    assert row["model_call_hash"] is None


def test_log_step_records_model_call_hash_for_llm_nodes(conn):
    log_step(
        conn,
        run_id="run_1",
        step_id="step_2",
        target_id="tgt_1",
        tool_name="summarize_company",
        agent_id="system",
        input_data={"text": "..."},
        output_data={"summary": "..."},
        status="success",
        model_call_hash="sha256:abc123",
    )
    row = conn.execute("SELECT * FROM steps WHERE step_id='step_2';").fetchone()
    assert row["model_call_hash"] == "sha256:abc123"
