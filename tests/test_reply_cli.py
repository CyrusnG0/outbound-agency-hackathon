"""Tests for reply_cli's batch-manifest step (ticket U2-fix2).

The live run view (app/console/app.py::_fetch_run_steps) derives a run's
completeness from a batch-manifest step naming the run's FULL intended
target set — the structural fix for the quiet-period heuristic's latency
race.  reply_cli must log that manifest (tool_name='reply_batch_manifest',
target_id NULL, output_json.target_ids = the targets the fetched replies
belong to) BEFORE its per-reply classify/route loop, and must log it EVEN
when the batch is empty.

The manifest lists TARGETS, not reply_ids — so reply_cli resolves each
reply_id to its target via the same replies→messages join its crash path
uses.  The resolution is proven against REAL seeded rows here (a target, an
outbound message, and one or two reply rows).  docs/reply-routing.md §5
allows multiple replies on one thread, so two replies resolving to the SAME
target must appear ONCE in the manifest — the dedup decision.

Everything after the manifest is stubbed so the test stays offline: the
fetch is replaced with a canned InboxFetchResult, and the reply-agent
factory + per-reply classifier are monkeypatched (no ADK agent, no LLM
call — tests/conftest.py's autouse live-client guard stays untripped).
"""

from app.agents_registry import seed_agent_registry
from app.db import apply_schema, connect
from app.reply_cli import main as reply_cli_main
from app.tools.fetch_inbox import InboxFetchResult
from app.write_gate import commit


def _seed_reply_chain(conn, *, target_id="tgt_r1", message_id="msg_r1", reply_ids=("rpl_r1",)):
    """Seed the rows the manifest resolution needs: a target (dry_run_sent —
    the state fetch_inbox would have left a linked target in), an outbound
    message pointing at it, and one reply per given reply_id pointing at the
    message.  All rows go through the write gate (the repo's seeding
    convention); replies are written by the fixture with classification NULL
    — exactly what fetch_inbox leaves before the classifier runs."""
    commit(
        conn, action="insert_offer", table_name="offers", record_id="off_r1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_r1", "acme", 1),
    )
    commit(
        conn, action="insert_account", table_name="accounts", record_id="acc_r1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               created_at, updated_at) VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_r1", "Fixture Co", "fixture.test", "fixture.test"),
    )
    commit(
        conn, action="insert_contact", table_name="contacts", record_id="con_r1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
               email_verified, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("con_r1", "acc_r1", "Jane Doe", "jane@fixture.test", 1),
    )
    commit(
        conn, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id, source,
               state, created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, "acc_r1", "con_r1", "off_r1", "csv", "dry_run_sent"),
    )
    commit(
        conn, action="insert_message", table_name="messages", record_id=message_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO messages (message_id, target_id, contact_id, direction,
               subject, body, status, created_at)
               VALUES (?,?,?,?,?,?,?,datetime('now'))""",
        params=(message_id, target_id, "con_r1", "outbound", "A subject", "A body", "dry_run_sent"),
    )
    for reply_id in reply_ids:
        commit(
            conn, action="insert_reply", table_name="replies", record_id=reply_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO replies (reply_id, message_id, from_email, raw_text,
                   redacted_text, created_at)
                   VALUES (?,?,?,?,?,datetime('now'))""",
            params=(reply_id, message_id, "jane@fixture.test", "A reply body", "A reply body"),
        )


def _run_cli(scratch_db_target, monkeypatch, *, reply_ids):
    """Drive reply_cli with the model boundaries stubbed and the fetch canned
    to the given reply_ids; returns the CLI exit code."""
    monkeypatch.setattr(
        "app.reply_cli.fetch_inbox",
        lambda conn, inbox_dir, run_id, limit: InboxFetchResult(
            files_seen=len(reply_ids), replies_created=list(reply_ids), skipped=[], errors=[]
        ),
    )
    monkeypatch.setattr("app.reply_cli.build_reply_agent", lambda conn: object())
    monkeypatch.setattr(
        "app.reply_cli.classify_and_route_reply",
        lambda agent, conn, reply_id, run_id: "routed",
    )
    return reply_cli_main(["--db", scratch_db_target, "--inbox", "/nonexistent-inbox", "--limit", "5"])


def test_reply_cli_logs_batch_manifest_with_resolved_targets(scratch_db_target, monkeypatch):
    """A reply run over a fetched batch logs exactly one reply_batch_manifest
    step naming the TARGETS the replies resolve to (not the reply_ids), with
    target_id NULL (a batch-level step, matching get_targets's convention),
    status success, and agent_id system."""
    import json

    conn = connect(scratch_db_target)
    apply_schema(conn)
    seed_agent_registry(conn, run_id="r0", step_id="s0")
    _seed_reply_chain(conn, target_id="tgt_r1", message_id="msg_r1", reply_ids=("rpl_r1",))
    conn.close()

    exit_code = _run_cli(scratch_db_target, monkeypatch, reply_ids=["rpl_r1"])
    assert exit_code == 0, "a clean reply run must exit 0"

    conn = connect(scratch_db_target)
    manifest_rows = conn.execute(
        "SELECT run_id FROM steps WHERE tool_name='reply_batch_manifest' ORDER BY created_at DESC LIMIT 1;"
    ).fetchall()
    assert manifest_rows, "reply_cli must log a reply_batch_manifest step"
    run_id = manifest_rows[0]["run_id"]
    steps = [
        dict(r) for r in conn.execute(
            "SELECT tool_name, target_id, output_json, status, agent_id "
            "FROM steps WHERE run_id=? AND tool_name='reply_batch_manifest' "
            "ORDER BY created_at, step_id;",
            (run_id,),
        ).fetchall()
    ]
    conn.close()

    assert len(steps) == 1, "exactly one manifest step per reply run"
    step = steps[0]
    assert step["target_id"] is None, "a batch manifest is not scoped to one target"
    assert step["status"] == "success"
    assert step["agent_id"] == "system"
    # The reply_id resolved to its target: the manifest carries the target.
    assert json.loads(step["output_json"]) == {"target_ids": ["tgt_r1"]}


def test_reply_cli_manifest_dedupes_targets_across_replies_on_one_thread(scratch_db_target, monkeypatch):
    """Two replies on the SAME thread (docs/reply-routing.md §5 — both link
    to the same outbound message, hence the same target) must appear ONCE in
    the manifest: it is the set of targets this run will touch, not a
    per-reply list."""
    import json

    conn = connect(scratch_db_target)
    apply_schema(conn)
    seed_agent_registry(conn, run_id="r0", step_id="s0")
    _seed_reply_chain(
        conn, target_id="tgt_r1", message_id="msg_r1",
        reply_ids=("rpl_r1", "rpl_r2"),
    )
    conn.close()

    exit_code = _run_cli(scratch_db_target, monkeypatch, reply_ids=["rpl_r1", "rpl_r2"])
    assert exit_code == 0

    conn = connect(scratch_db_target)
    rows = conn.execute(
        "SELECT run_id, output_json FROM steps "
        "WHERE tool_name='reply_batch_manifest' ORDER BY created_at DESC LIMIT 1;"
    ).fetchall()
    conn.close()
    assert rows, "reply_cli must log a reply_batch_manifest step"
    # Both replies resolve to tgt_r1 — deduped to one entry.
    assert json.loads(rows[0]["output_json"]) == {"target_ids": ["tgt_r1"]}


def test_reply_cli_logs_empty_batch_manifest(scratch_db_target, monkeypatch):
    """A reply run with ZERO fetched replies still logs its manifest — an
    empty manifest (target_ids: []) is a legitimate result, never a skipped
    log.  The live view needs it to know the run had nothing to do."""
    import json

    exit_code = _run_cli(scratch_db_target, monkeypatch, reply_ids=[])
    assert exit_code == 0

    conn = connect(scratch_db_target)
    rows = conn.execute(
        "SELECT run_id, output_json FROM steps "
        "WHERE tool_name='reply_batch_manifest' ORDER BY created_at DESC LIMIT 1;"
    ).fetchall()
    conn.close()
    assert rows, "the empty batch must still log its manifest"
    assert json.loads(rows[0]["output_json"]) == {"target_ids": []}
