"""Tests for draft_cli's batch-manifest step (ticket U2-fix2).

The live run view (app/console/app.py::_fetch_run_steps) derives a run's
completeness from a batch-manifest step naming the run's FULL intended
target set — the structural fix for the quiet-period heuristic's latency
race.  draft_cli must log that manifest (tool_name='draft_batch_manifest',
target_id NULL, output_json.target_ids = the eligible batch) BEFORE its
per-target loop, and must log it EVEN when the batch is empty.

Everything but the manifest is stubbed so the test stays offline: the draft
LoopAgent factory and the per-target runner are monkeypatched, so no ADK
agent and no LLM call is ever built (tests/conftest.py's autouse live-client
guard stays untripped).  The CLI's own DB seeding (apply_schema +
seed_agent_registry) still runs for real; the scored target is seeded through
the write gate before the CLI opens the same scratch target.
"""

from app.agents_registry import seed_agent_registry
from app.db import apply_schema, connect
from app.draft_cli import main as draft_cli_main
from app.write_gate import commit


def _seed_scored_target(conn, *, target_id="tgt_d1"):
    """Seed the one row draft_cli's eligible-set SELECT needs: a target in
    state 'scored' (docs/state-machine.md §3 — the first-touch draft path).
    Offer and account rows are FK'd by the target, so all three go through
    the write gate (the repo's seeding convention)."""
    commit(
        conn, action="insert_offer", table_name="offers", record_id="off_d1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_d1", "acme", 1),
    )
    commit(
        conn, action="insert_account", table_name="accounts", record_id="acc_d1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               created_at, updated_at) VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_d1", "Fixture Co", "fixture.test", "fixture.test"),
    )
    commit(
        conn, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, offer_id, source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, "acc_d1", "off_d1", "csv", "scored"),
    )


def _read_manifest_steps(conn, *, run_id, tool_name):
    """Return every manifest step logged under one run as plain dicts, in
    log order — the assertion reads the SAME columns the live view does."""
    rows = conn.execute(
        "SELECT tool_name, target_id, output_json, status, agent_id "
        "FROM steps WHERE run_id=? AND tool_name=? ORDER BY created_at, step_id;",
        (run_id, tool_name),
    ).fetchall()
    return [dict(row) for row in rows]


def test_draft_cli_logs_batch_manifest_before_the_loop(scratch_db_target, monkeypatch):
    """A draft run over a non-empty eligible batch logs exactly one
    draft_batch_manifest step naming that batch, with target_id NULL (a
    batch-level step, matching get_targets's convention), status success, and
    agent_id system."""
    import json

    # Seed one scored target through the write gate, then let the CLI re-open
    # the same scratch target (it applies schema and seeds the registry itself).
    conn = connect(scratch_db_target)
    apply_schema(conn)
    seed_agent_registry(conn, run_id="r0", step_id="s0")
    _seed_scored_target(conn, target_id="tgt_d1")
    conn.close()

    # Stub the two model boundaries: building the draft LoopAgent and running
    # a target through it.  The manifest is logged BEFORE either is needed,
    # but the agent is built unconditionally, so both must be patched for a
    # fully offline run (tests/conftest.py blocks real genai clients).
    monkeypatch.setattr("app.draft_cli.build_draft_agent", lambda conn: object())
    monkeypatch.setattr(
        "app.draft_cli.run_target_through_draft",
        lambda agent, conn, target_id, run_id, offers_dir: "awaiting_review",
    )

    exit_code = draft_cli_main(["--db", scratch_db_target, "--limit", "5"])
    assert exit_code == 0, "a clean draft run must exit 0"

    # Re-open and read the run's manifest step.  The run_id is not known to
    # the test, so grab the run that owns the manifest step.
    conn = connect(scratch_db_target)
    manifest_rows = conn.execute(
        "SELECT run_id FROM steps WHERE tool_name='draft_batch_manifest' ORDER BY created_at DESC LIMIT 1;"
    ).fetchall()
    assert manifest_rows, "draft_cli must log a draft_batch_manifest step"
    run_id = manifest_rows[0]["run_id"]
    steps = _read_manifest_steps(conn, run_id=run_id, tool_name="draft_batch_manifest")
    conn.close()

    assert len(steps) == 1, "exactly one manifest step per draft run"
    step = steps[0]
    assert step["target_id"] is None, "a batch manifest is not scoped to one target"
    assert step["status"] == "success"
    assert step["agent_id"] == "system"
    # output_json carries the FULL batch this run intends to draft.
    assert json.loads(step["output_json"]) == {"target_ids": ["tgt_d1"]}


def test_draft_cli_logs_empty_batch_manifest(scratch_db_target, monkeypatch):
    """A draft run with ZERO eligible targets still logs its manifest — an
    empty manifest (target_ids: []) is a legitimate result, never a skipped
    log.  The live view needs it to know the run had nothing to do (an empty
    manifest means the run can never be mid-batch, so it may complete once
    no target is pending)."""
    import json

    # No targets seeded at all — the eligible set is empty.
    monkeypatch.setattr("app.draft_cli.build_draft_agent", lambda conn: object())
    monkeypatch.setattr(
        "app.draft_cli.run_target_through_draft",
        lambda agent, conn, target_id, run_id, offers_dir: "awaiting_review",
    )

    exit_code = draft_cli_main(["--db", scratch_db_target, "--limit", "5"])
    assert exit_code == 0

    conn = connect(scratch_db_target)
    rows = conn.execute(
        "SELECT run_id, output_json FROM steps "
        "WHERE tool_name='draft_batch_manifest' ORDER BY created_at DESC LIMIT 1;"
    ).fetchall()
    conn.close()
    assert rows, "the empty batch must still log its manifest"
    assert json.loads(rows[0]["output_json"]) == {"target_ids": []}
