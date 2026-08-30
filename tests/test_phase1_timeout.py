# tests/test_phase1_timeout.py — ticket B1g: the per-target wall-clock
# ceiling and the SDK-level request timeouts.
#
# Production failure this prevents: the 2026-08-22 run stalled for 9h48m on
# 1.09s of CPU — the ResearchAgent's next model turn parked in an
# ESTABLISHED-but-idle socket that never raised, so neither A4c's retry
# (sees only exceptions) nor B1f's crash guard (sees only exceptions) could
# fire.  The tests here prove the NEW property: a run that would hang
# forever is cut off by a wall-clock deadline, recorded as a clean
# "failed"/phase1_timeout outcome (not a crash), and the rest of the batch
# still runs.
#
# Hermeticity: every test hangs the pipeline with a STUB agent that sleeps
# (an await that never resolves — the same shape as the real hang) or raises
# the SDK's timeout exception, and the ceiling is set to a deliberately tiny
# value via the PHASE1_TARGET_TIMEOUT_SECONDS env var — never the production
# default — so no test ever waits out a real timeout and the added runtime
# is milliseconds.  The B1c autouse guard is untouched: no test here
# constructs a live model client.
import asyncio  # the hanging stub's never-resolving await
import csv as csv_module  # the batch test's CSV writer (same pattern as test_phase1_cli.py)
import json  # steps.output_json is a JSON string — parse it to assert the timeout detail
import time  # wall-clock bound on the hanging test — the suite must never wait out a real hang
from unittest.mock import patch  # patch the model/network boundaries so every test stays offline (the B1c guard's companion)

import httpx  # the SDK-level timeout exception the second stub raises
import pytest  # test runner + the ValueError assertion
from google.adk.agents import BaseAgent  # base class of the hanging/sdk-timeout stubs
from google.adk.events import Event, EventActions  # the stubs' (unreachable) happy-path yield

from app.agents.phase1 import (  # the module under test
    DEFAULT_PHASE1_TARGET_TIMEOUT_SECONDS,
    build_phase1_agent,
    run_target_through_phase1,
)
from app.agents_registry import seed_agent_registry  # register "system" so transition()'s write gate accepts writes
from app.db import apply_schema, connect  # per-test temp sqlite database
from app.phase1_cli import main  # the batch test drives the REAL CLI so the ceiling composes with B1f
from app.schemas import CompanyProfile  # the fake profile summarize_company's mock returns
from app.tools.fetch_sources import NormalizedSource  # the fake source shape fetch_sources' mock returns
from app.write_gate import commit  # seed offer/account/target rows through the single write path


class _HangingResearchAgent(BaseAgent):
    """Offline stand-in whose run never finishes — the B1g hang shape.

    The real hang was an ESTABLISHED-but-idle socket await inside ADK's model
    loop; asyncio.sleep(3600) is the same shape in stub form: an await that
    only ever resolves if something CANCELS it.  The ceiling's
    asyncio.wait_for must be that something."""

    def __init__(self):
        super().__init__(name="research")  # same stable name as the real agent

    async def _run_async_impl(self, ctx):
        await asyncio.sleep(3600)  # never resolves on its own — only wait_for's cancellation can end this
        # Unreachable unless the sleep is cancelled and the coroutine somehow
        # continued — the yield exists only to make this an async generator
        # (ADK's node contract), and is never expected to run.
        yield Event(
            author=self.name, invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"extracted_text": "never"}),
        )


class _SdkTimeoutResearchAgent(BaseAgent):
    """Offline stand-in that raises the SDK-level timeout exception instead of
    hanging — the shape produced when the per-request timeouts (part 2 of
    B1g) fire BEFORE the per-target ceiling.  It must land in the same
    phase1_timeout bucket, not B1f's CRASHED bucket."""

    def __init__(self):
        super().__init__(name="research")  # same stable name as the real agent

    async def _run_async_impl(self, ctx):
        # The exact exception google-genai lets escape unwrapped on a socket
        # read timeout (measured, app/llm.py §9c) — ADK propagates it out of
        # the agent loop.
        raise httpx.ReadTimeout("simulated SDK-level read timeout")
        yield  # pragma: no cover — makes this an async generator; the raise above always fires first


@pytest.fixture
def conn(scratch_db_target):
    """Temp DB with schema + one new target — the same seeding pattern as
    tests/test_agents_phase1.py (registry seed, offer, account, target)."""
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # Register the system agent (plan A3) — transition()'s write gate refuses
    # unregistered agents, so the timeout path's writes would fail without it.
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


def _assert_phase1_timeout_recorded(conn, target_id: str):
    """The DB-backed assertions every timeout test shares (ticket B1c's
    lesson: a test that only checks a return value passed with all
    credentials stripped — the database is the artifact an operator reviews).

    Three facts must hold: the target is ``failed``; a state_transitions row
    carries the NEW reason ``phase1_timeout`` (not an existing reason, not a
    crash reason); and a failed steps row records the timeout."""
    row = conn.execute("SELECT state FROM targets WHERE target_id=?;", (target_id,)).fetchone()
    assert row["state"] == "failed"
    trn = conn.execute(
        "SELECT previous_state, new_state, reason FROM state_transitions WHERE target_id=?;",
        (target_id,),
    ).fetchone()
    assert trn is not None, "no state_transitions row for the timed-out target"
    assert trn["new_state"] == "failed"
    assert trn["reason"] == "phase1_timeout"  # the NEW reason — distinct from unhandled_error_phase1 and llm_transport_error_phase1
    assert trn["previous_state"] == "new"  # the timeout fired before any node transitioned — the row must say so truthfully
    step = conn.execute(
        "SELECT output_json, status FROM steps WHERE target_id=? AND tool_name='phase1_target_timeout';",
        (target_id,),
    ).fetchone()
    assert step is not None, "no phase1_target_timeout steps row — never skip logging"
    assert step["status"] == "failed"
    out = json.loads(step["output_json"])
    assert out["detail"]  # the row must say WHICH layer fired (ceiling vs SDK timeout)


def test_hanging_target_is_bounded_and_fails_cleanly_with_phase1_timeout(conn, monkeypatch):
    # The ceiling must cut off a run that would hang forever, and the outcome
    # must be a clean "failed" — not a raise, not a crash.  The deadline is
    # injected through the env var at a deliberately tiny 50ms so this test
    # cannot wait out anything real.
    monkeypatch.setenv("PHASE1_TARGET_TIMEOUT_SECONDS", "0.05")
    with patch("app.agents.phase1.build_research_agent", return_value=_HangingResearchAgent()):
        agent = build_phase1_agent(conn)  # built INSIDE the patch so the stub is wired in
        start = time.monotonic()  # the wall-clock bound: the whole point is that this returns quickly
        final_state = run_target_through_phase1(
            agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
        )
        elapsed = time.monotonic() - start
    assert final_state == "failed"  # clean terminal state, returned normally — the CLI records it in results, not crashed
    assert elapsed < 5.0  # bounded: 50ms deadline + cleanup, never a real hang (fails at ~3600s if the ceiling is broken)
    _assert_phase1_timeout_recorded(conn, "tgt_1")


def test_sdk_level_timeout_exception_lands_in_phase1_timeout_bucket(conn, monkeypatch):
    # Part 2 of B1g gives the ADK path a per-request timeout; when it fires
    # FIRST (a single stalled request) the httpx timeout exception must be
    # caught at the seam and routed into the SAME phase1_timeout failure —
    # a timed-out target must never look like B1f's CRASHED bucket.
    monkeypatch.setenv("PHASE1_TARGET_TIMEOUT_SECONDS", "30")  # any normal value — the SDK exception fires long before it
    with patch("app.agents.phase1.build_research_agent", return_value=_SdkTimeoutResearchAgent()):
        agent = build_phase1_agent(conn)  # built INSIDE the patch so the stub is wired in
        final_state = run_target_through_phase1(
            agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
        )
    assert final_state == "failed"
    _assert_phase1_timeout_recorded(conn, "tgt_1")
    # The step row must name the SDK exception so the trace shows WHICH layer
    # fired — ReadTimeout here, "asyncio.wait_for cancelled" in the other test.
    step = conn.execute(
        "SELECT output_json FROM steps WHERE tool_name='phase1_target_timeout';"
    ).fetchone()
    assert "ReadTimeout" in json.loads(step["output_json"])["detail"]


def test_invalid_timeout_env_value_fails_loudly(conn, monkeypatch):
    # A non-numeric override is a wiring mistake and must be an immediate
    # error naming the var — never a silent fallback to the default that
    # would hide the misconfiguration until a run hangs again.
    monkeypatch.setenv("PHASE1_TARGET_TIMEOUT_SECONDS", "ten")
    with patch("app.agents.phase1.build_research_agent", return_value=_HangingResearchAgent()):
        agent = build_phase1_agent(conn)
        with pytest.raises(ValueError, match="PHASE1_TARGET_TIMEOUT_SECONDS"):
            run_target_through_phase1(
                agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
            )


def test_default_ceiling_leaves_headroom_but_is_far_below_forever():
    # Pin the default's envelope so a future edit cannot silently turn it
    # into "fires on healthy targets" (< observed 20-60s healthy time) or
    # "fires never" (the 9h48m hang it exists to prevent).
    assert 60 <= DEFAULT_PHASE1_TARGET_TIMEOUT_SECONDS <= 3600


# ── Batch composition with B1f (ticket requirement: "verify, don't assume") ──

@pytest.fixture
def offers_dir(tmp_path):
    d = tmp_path / "offers"
    d.mkdir()
    (d / "acme-offer.yaml").write_text(
        "pitch: p\npersona_hint: h\ntemplate: t\nfrom_address: a@b.test\n"
    )
    return d


def test_batch_survives_a_timing_out_target(tmp_path, offers_dir, capsys, monkeypatch):
    # One target that hangs must fail cleanly (phase1_timeout, exit code 0 —
    # NOT B1f's CRASHED exit code 1) and the targets after it must still run.
    # The healthy targets use the same offline patch stack as the B1f tests;
    # the hanging target drives the REAL run_target_through_phase1 with the
    # hanging stub, so the REAL wait_for ceiling is what cuts it off — the
    # composition this test proves is ceiling-inside-runner, runner-inside-B1f.
    csv_path = tmp_path / "targets.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=["company_name", "domain", "offer_id"])
        writer.writeheader()
        writer.writerows([
            {"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer"},
            {"company_name": "HangCo", "domain": "hangco.test", "offer_id": "acme-offer"},
            {"company_name": "Beta", "domain": "beta.test", "offer_id": "acme-offer"},
        ])
    db_path = str(tmp_path / "outbound.db")
    # 1.0s: long enough for the fully-offline healthy targets to finish
    # comfortably (they run real sqlite writes through ADK), short enough to
    # keep the test's added runtime negligible.
    monkeypatch.setenv("PHASE1_TARGET_TIMEOUT_SECONDS", "1.0")

    from app.agents.phase1 import FetchAndNormalizeNode  # B1b's retained offline stand-in for healthy targets

    def timing_out_runner(agent, *, conn, target_id, domain, run_id, offers_dir="config/offers"):
        # The hang target runs the REAL runner against a genuinely hanging
        # agent; every other target delegates to the real (offline-patched)
        # pipeline, exactly like the B1f crash wrapper.
        # B2c: offers_dir accepted and forwarded to the delegated runs.
        if domain == "hangco.test":
            return run_target_through_phase1(
                _HangingResearchAgent(), conn=conn, target_id=target_id, domain=domain, run_id=run_id,
                offers_dir=offers_dir,
            )
        return run_target_through_phase1(
            agent, conn=conn, target_id=target_id, domain=domain, run_id=run_id,
            offers_dir=offers_dir,
        )

    with patch("app.tools.fetch_sources.fetch_sources",
               return_value=[NormalizedSource(
                   "company_website", "https://acme.test", "Acme does logistics.", "t", 0.8, 1, "static"
               )]), \
         patch("app.tools.summarize_company.call_structured",
               return_value=CompanyProfile(one_line_summary="Acme does logistics", confidence=0.8)), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=[]), \
         patch("app.agents.phase1.build_research_agent",
               side_effect=lambda conn: FetchAndNormalizeNode(name="research", conn=conn)), \
         patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None), \
         patch("app.phase1_cli.run_target_through_phase1", side_effect=timing_out_runner):
        exit_code = main([
            "--csv", str(csv_path), "--offer", "acme-offer",
            "--db", db_path, "--offers-dir", str(offers_dir),
        ])

    # A timed-out target is a clean failed outcome, NOT a crash: the batch is
    # clean (exit 0) and the summary must not mark it CRASHED — that would be
    # the mislabelling this ticket forbids.
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "CRASHED" not in out

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT t.target_id, t.state, a.normalized_domain FROM targets t "
            "JOIN accounts a ON t.account_id = a.account_id;"
        ).fetchall()
        by_domain = {r["normalized_domain"]: r for r in rows}
        # First and third targets still reached terminal Phase 1 states — the
        # batch survived the second target's hang.
        assert by_domain["acme.test"]["state"] in ("scored", "watchlist", "not_target")
        assert by_domain["beta.test"]["state"] in ("scored", "watchlist", "not_target")
        # The hung target is failed with the NEW reason, with its own failed
        # step row — the same DB-backed assertions as the unit tests.
        hung = by_domain["hangco.test"]
        assert hung["state"] == "failed"
        _assert_phase1_timeout_recorded(conn, hung["target_id"])
        # The CLI summary prints TARGET IDs (not domains), so the clean-failure
        # claim is asserted against the hung target's actual id: it must
        # appear as an ordinary "…: failed" result line, never a CRASHED line.
        assert f"{hung['target_id']}: failed" in out
    finally:
        conn.close()
