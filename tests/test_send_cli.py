"""Tests for send_cli's batch-manifest step (ticket U2-fix2).

The live run view (app/console/app.py::_fetch_run_steps) derives a run's
completeness from a batch-manifest step naming the run's FULL intended
target set — the structural fix for the quiet-period heuristic's latency
race.  send_cli must log that manifest (tool_name='send_batch_manifest',
target_id NULL, output_json.target_ids = the approved batch) BEFORE its
per-target loop, and must log it EVEN when the batch is empty.

The DRY_RUN send itself is stubbed (the fake returns refused=False), so the
test stays offline and focuses on the manifest: no .eml artifact is written
and no state transition fires (both live in the real send_email, which is
not invoked).  The approved target is seeded through the write gate; the CLI
applies schema and seeds the registry itself on the shared scratch target.
"""

from app.agents_registry import seed_agent_registry
from app.db import apply_schema, connect
from app.send_cli import main as send_cli_main
from app.write_gate import commit


class _FakeSendResult:
    """The minimal send_email result the CLI reads: refused flag + outbox
    path.  refused=False makes the loop record a successful dry-run line."""

    refused = False
    outbox_path = "fake.eml"


def _seed_approved_target(conn, *, target_id="tgt_s1"):
    """Seed the one row send_cli's eligible-set SELECT needs: a target in
    state 'approved' — the state machine's only inbound edge to
    dry_run_sent (docs/state-machine.md §7e), so exactly the eligible set."""
    commit(
        conn, action="insert_offer", table_name="offers", record_id="off_s1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_s1", "acme", 1),
    )
    commit(
        conn, action="insert_account", table_name="accounts", record_id="acc_s1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               created_at, updated_at) VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_s1", "Fixture Co", "fixture.test", "fixture.test"),
    )
    commit(
        conn, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, offer_id, source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, "acc_s1", "off_s1", "csv", "approved"),
    )


def test_send_cli_logs_batch_manifest_before_the_loop(scratch_db_target, tmp_path, monkeypatch):
    """A send run over a non-empty approved batch logs exactly one
    send_batch_manifest step naming that batch, with target_id NULL (a
    batch-level step, matching get_targets's convention), status success, and
    agent_id system."""
    import json

    conn = connect(scratch_db_target)
    apply_schema(conn)
    seed_agent_registry(conn, run_id="r0", step_id="s0")
    _seed_approved_target(conn, target_id="tgt_s1")
    conn.close()

    # Stub the DRY_RUN send so the test needs no draft/review chain and no
    # .eml artifact — the manifest is logged before send_email is ever called.
    monkeypatch.setattr("app.send_cli.send_email", lambda conn, target_id, run_id, outbox_dir: _FakeSendResult())

    exit_code = send_cli_main(["--db", scratch_db_target, "--outbox", str(tmp_path / "outbox"), "--limit", "5"])
    assert exit_code == 0, "a clean send run must exit 0"

    conn = connect(scratch_db_target)
    manifest_rows = conn.execute(
        "SELECT run_id FROM steps WHERE tool_name='send_batch_manifest' ORDER BY created_at DESC LIMIT 1;"
    ).fetchall()
    assert manifest_rows, "send_cli must log a send_batch_manifest step"
    run_id = manifest_rows[0]["run_id"]
    steps = [
        dict(r) for r in conn.execute(
            "SELECT tool_name, target_id, output_json, status, agent_id "
            "FROM steps WHERE run_id=? AND tool_name='send_batch_manifest' "
            "ORDER BY created_at, step_id;",
            (run_id,),
        ).fetchall()
    ]
    conn.close()

    assert len(steps) == 1, "exactly one manifest step per send run"
    step = steps[0]
    assert step["target_id"] is None, "a batch manifest is not scoped to one target"
    assert step["status"] == "success"
    assert step["agent_id"] == "system"
    assert json.loads(step["output_json"]) == {"target_ids": ["tgt_s1"]}


def test_send_cli_logs_empty_batch_manifest(scratch_db_target, tmp_path, monkeypatch):
    """A send run with ZERO approved targets still logs its manifest — an
    empty manifest (target_ids: []) is a legitimate result, never a skipped
    log.  The live view needs it to know the run had nothing to do."""
    import json

    monkeypatch.setattr("app.send_cli.send_email", lambda conn, target_id, run_id, outbox_dir: _FakeSendResult())

    exit_code = send_cli_main(["--db", scratch_db_target, "--outbox", str(tmp_path / "outbox"), "--limit", "5"])
    assert exit_code == 0

    conn = connect(scratch_db_target)
    rows = conn.execute(
        "SELECT run_id, output_json FROM steps "
        "WHERE tool_name='send_batch_manifest' ORDER BY created_at DESC LIMIT 1;"
    ).fetchall()
    conn.close()
    assert rows, "the empty batch must still log its manifest"
    assert json.loads(rows[0]["output_json"]) == {"target_ids": []}
