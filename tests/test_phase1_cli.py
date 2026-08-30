# tests/test_phase1_cli.py
import csv as csv_module
import re  # extracts the run_id from the CLI's printed summary so the steps assertion can attribute rows to THIS run
from contextlib import ExitStack  # one `with` that enters all four offline patches together
from unittest.mock import patch

import pytest
from aiohttp.client_exceptions import ServerDisconnectedError  # the exact exception class from the 2026-08-21 batch death (raised by ADK's aiohttp transport)

from app.agents.phase1 import (  # B1b's retained offline stand-in for the research LlmAgent (see the e2e test)
    FetchAndNormalizeNode,
    run_target_through_phase1,  # the REAL runner the crashing wrappers delegate non-crash targets to
)
from app.db import connect  # re-open the CLI's DB file after main() returns, to assert on the persisted state
from app.ids import new_id  # test setup simulating mid-pipeline progress needs a step id for the real state machine
from app.phase1_cli import main
from app.schemas import CompanyProfile  # the fake profile shape summarize_company's mock returns
from app.state_machine import transition  # test setup moves a target to "researched" through the real gate, never a raw UPDATE
from app.tools.fetch_sources import NormalizedSource  # the fake source shape fetch_sources' mock returns


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def offers_dir(tmp_path):
    d = tmp_path / "offers"
    d.mkdir()
    (d / "acme-offer.yaml").write_text(
        "pitch: p\npersona_hint: h\ntemplate: t\nfrom_address: a@b.test\n"
    )
    return d


def test_cli_batch_size_cap_rejects_oversized_batch(tmp_path, offers_dir):
    csv_path = tmp_path / "targets.csv"
    rows = [{"company_name": f"Co{i}", "domain": f"co{i}.test", "offer_id": "acme-offer"} for i in range(20)]
    write_csv(csv_path, rows, fieldnames=["company_name", "domain", "offer_id"])

    exit_code = main([
        "--csv", str(csv_path), "--offer", "acme-offer",
        "--db", str(tmp_path / "outbound.db"),
        "--offers-dir", str(offers_dir),
    ])
    assert exit_code != 0  # batch of 20 exceeds the 10-15/run cap


def test_cli_runs_a_small_batch_end_to_end(tmp_path, offers_dir, capsys):
    # B1c hardening (was: assert exit_code == 0 — vacuous).  Before this
    # ticket the test built the FULL pipeline through main(), and
    # build_phase1_agent's first stage is now a REAL ADK research LlmAgent
    # (B1b) that was never patched here — so every pytest run made live
    # billable Vertex calls and real network requests to acme.test (~37s),
    # and the lone exit-code assertion could not even tell the research stage
    # was dead (it passed with GOOGLE_API_KEY etc. unset).  Two fixes below:
    # (1) the research stage is stood in by B1b's retained deterministic
    # FetchAndNormalizeNode — the exact offline pattern used in
    # tests/test_agents_phase1.py and tests/test_phase1_checkpoint.py; (2) the
    # test now asserts on the PERSISTED database, the things "end to end"
    # actually claims: a terminal Phase 1 state, steps rows for the run, and
    # a state_transitions row for the target.
    csv_path = tmp_path / "targets.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer"}],
        fieldnames=["company_name", "domain", "offer_id"],
    )
    db_path = str(tmp_path / "outbound.db")

    # Patch the three pre-existing external boundaries, unchanged from before:
    # the network fetch, the summarize LLM call, and the detect-signals LLM
    # call — no real network or billable API traffic from these either.
    fake_source_patch = patch(
        "app.tools.fetch_sources.fetch_sources",
        return_value=[NormalizedSource(
            "company_website", "https://acme.test", "Acme does logistics.", "t", 0.8, 1, "static"
        )],
    )
    profile_patch = patch(
        "app.tools.summarize_company.call_structured",
        return_value=CompanyProfile(
            one_line_summary="Acme does logistics", confidence=0.8
        ),
    )
    signals_patch = patch("app.tools.detect_signals._call_detect_signals", return_value=[])

    # The B1b-required fourth patch — the one this test was missing.  Patch
    # the exact call site build_phase1_agent uses to build the research stage
    # (app/agents/phase1.py imports build_research_agent and calls it inside
    # build_phase1_agent), returning the retained deterministic
    # FetchAndNormalizeNode instead of the live LlmAgent.  side_effect (not
    # return_value) is the B1b pattern adapted to the CLI: main() opens its
    # OWN connection to db_path, and the stand-in node must write its steps
    # rows through that same connection — the lambda receives it from
    # build_phase1_agent(conn) and binds it, exactly as the checkpoint test
    # binds its fixture conn.  The stand-in then drives the SAME mocked
    # fetch_sources above and the REAL normalize_sources, so the pipeline
    # downstream of research still runs for real.
    research_patch = patch(
        "app.agents.phase1.build_research_agent",
        side_effect=lambda conn: FetchAndNormalizeNode(name="research", conn=conn),
    )

    # The B2c-required fifth patch: the score node now runs the ICP judge, so
    # the judge boundary is stubbed with the documented failure fallback
    # (None → the deterministic label stands) — the e2e assertions below
    # describe the pre-B2c deterministic routing, which the fallback keeps
    # byte-identical.  The judge's happy path is covered in
    # tests/test_judge_icp.py.
    judge_patch = patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None)

    with fake_source_patch, profile_patch, signals_patch, research_patch, judge_patch:
        exit_code = main([
            "--csv", str(csv_path), "--offer", "acme-offer",
            "--db", db_path,
            "--offers-dir", str(offers_dir),
        ])
    assert exit_code == 0  # the CLI contract: a completed run returns 0

    # Grab the run_id from the CLI's own printed summary so the steps
    # assertions below attribute rows to THIS run rather than "any row in the
    # temp DB" — the DB is fresh per test, but the id makes the claim exact.
    out = capsys.readouterr().out
    run_match = re.search(r"Phase 1 run (\S+) complete", out)
    assert run_match is not None, f"CLI did not print its run summary: {out!r}"
    run_id = run_match.group(1)

    # Re-open the DB file main() wrote and assert on PERSISTED rows — the
    # database is the artifact an operator actually reviews, not the stdout
    # summary, so the end-to-end claims must hold against it.
    conn = connect(db_path)
    try:
        # ASSERTION 1 — terminal Phase 1 state, read from targets.state, NOT
        # from the printed output.  This is the assertion that bites: if the
        # research stage is dead (e.g. the stand-in yields no text), the
        # target ends "failed" and this line fails where exit_code==0 alone
        # used to pass.
        target = conn.execute("SELECT target_id, state FROM targets;").fetchone()
        assert target is not None  # the CSV row must have imported
        assert target["state"] in ("scored", "watchlist", "not_target"), (
            f"target ended in {target['state']!r} — expected a terminal Phase 1 "
            f"state (scored/watchlist/not_target); a dead research stage lands "
            f"in 'failed' and must fail this test"
        )

        # ASSERTION 2 — at least one steps row exists for the run (never skip
        # logging).  Strengthened beyond the ticket minimum with the
        # score_lead row: import_csv and normalize_sources also write steps
        # rows, so a bare count would pass even on the failed path — but
        # score_lead only runs when research → summarize → detect all
        # succeeded, so its row proves the pipeline ran to its scoring stage.
        steps = conn.execute(
            "SELECT COUNT(*) AS n FROM steps WHERE run_id=?;", (run_id,)
        ).fetchone()
        assert steps["n"] >= 1, "no steps rows logged for the run"
        score_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM steps WHERE run_id=? AND tool_name='score_lead';",
            (run_id,),
        ).fetchone()
        assert score_rows["n"] >= 1, "score_lead never ran — the pipeline did not reach scoring"

        # ASSERTION 3 — the target's state changes were audited: every
        # transition goes through state_machine.transition(), which writes
        # exactly this table, so a terminal state without a row here would
        # mean the state changed outside the single state-change gate.
        transitions = conn.execute(
            "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id=?;",
            (target["target_id"],),
        ).fetchone()
        assert transitions["n"] >= 1, "no state_transitions row for the target"
    finally:
        conn.close()  # explicit close — the file lives under tmp_path and dies with the test


# ── B1f: one target's crash must not destroy the batch ────────────────────
# Shared offline setup for the B1f tests below.  The four patches are the
# exact set the e2e test above establishes — no real network, no billable
# LLM calls, the research stage stood in by B1b's deterministic
# FetchAndNormalizeNode — so any test here that runs the REAL pipeline for a
# target stays fully offline (the B1c autouse guard would refuse a live
# genai.Client anyway).


def _offline_pipeline_patch_stack() -> ExitStack:
    """Enter the five offline boundary patches and return the ExitStack.

    Caller does `with _offline_pipeline_patch_stack():` — the stack un-enters
    every patch when the with block exits.  The fifth patch (B2c) stubs the
    ICP judge with the failure fallback so the deterministic routing these
    tests assert stays byte-identical to before the judge existed.
    """
    stack = ExitStack()
    stack.enter_context(patch(
        "app.tools.fetch_sources.fetch_sources",
        return_value=[NormalizedSource(
            "company_website", "https://acme.test", "Acme does logistics.", "t", 0.8, 1, "static"
        )],
    ))
    stack.enter_context(patch(
        "app.tools.summarize_company.call_structured",
        return_value=CompanyProfile(
            one_line_summary="Acme does logistics", confidence=0.8
        ),
    ))
    stack.enter_context(patch("app.tools.detect_signals._call_detect_signals", return_value=[]))
    stack.enter_context(patch(
        "app.agents.phase1.build_research_agent",
        side_effect=lambda conn: FetchAndNormalizeNode(name="research", conn=conn),
    ))
    stack.enter_context(patch(
        "app.agents.phase1.judge_icp_module.judge_icp", return_value=None
    ))
    return stack


def _write_batch_csv(tmp_path, rows):
    """Write a company_name/domain/offer_id targets CSV and return its path."""
    csv_path = tmp_path / "targets.csv"
    write_csv(csv_path, rows, fieldnames=["company_name", "domain", "offer_id"])
    return csv_path


def _cli_args(csv_path, db_path, offers_dir):
    """The shared argv shape every B1f test passes to main()."""
    return [
        "--csv", str(csv_path), "--offer", "acme-offer",
        "--db", db_path,
        "--offers-dir", str(offers_dir),
    ]


def test_one_target_crashing_does_not_abort_the_batch(tmp_path, offers_dir, capsys):
    # Production failure this prevents: the 2026-08-21 incident — a real
    # 10-target run died at target 4 with
    # aiohttp.client_exceptions.ServerDisconnectedError: Server disconnected.
    # The loop called run_target_through_phase1 with no exception handling, so
    # the error propagated out of the loop and aborted the whole batch: six
    # targets were left at state "new", and the operator paid for ten targets
    # of Gemini calls and web scraping while keeping four.  Here the SECOND of
    # three targets raises that exact exception inside the call; the first and
    # third must still reach terminal states and the run must complete.
    csv_path = _write_batch_csv(tmp_path, [
        {"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer"},
        {"company_name": "CrashCo", "domain": "crashco.test", "offer_id": "acme-offer"},
        {"company_name": "Beta", "domain": "beta.test", "offer_id": "acme-offer"},
    ])
    db_path = str(tmp_path / "outbound.db")

    def crashing_runner(agent, *, conn, target_id, domain, run_id, offers_dir="config/offers"):
        # The crash target raises the exact exception class from the incident;
        # every other target delegates to the real (offline-patched) pipeline
        # so their terminal states are genuinely reached, not faked.
        # B2c: offers_dir is forwarded so the delegated pipeline judge reads
        # the same offer context the CLI passed in.
        if domain == "crashco.test":
            raise ServerDisconnectedError("Server disconnected")
        return run_target_through_phase1(
            agent, conn=conn, target_id=target_id, domain=domain, run_id=run_id,
            offers_dir=offers_dir,
        )

    with _offline_pipeline_patch_stack(), patch(
        "app.phase1_cli.run_target_through_phase1", side_effect=crashing_runner
    ):
        exit_code = main(_cli_args(csv_path, db_path, offers_dir))

    # A batch that lost a target to an unhandled error is not a clean run.
    assert exit_code != 0
    # The summary must mark the crash distinctly from an ordinary "failed"
    # line, with the exception type visible to the operator.
    out = capsys.readouterr().out
    assert "CRASHED" in out
    assert "ServerDisconnectedError" in out

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT t.target_id, t.state, a.normalized_domain FROM targets t "
            "JOIN accounts a ON t.account_id = a.account_id;"
        ).fetchall()
        by_domain = {r["normalized_domain"]: r for r in rows}
        # First and third targets still reached terminal Phase 1 states — the
        # batch survived the second target's crash.
        assert by_domain["acme.test"]["state"] in ("scored", "watchlist", "not_target")
        assert by_domain["beta.test"]["state"] in ("scored", "watchlist", "not_target")
        # The crashed target is failed in the DB with the new reason string.
        crashed_row = by_domain["crashco.test"]
        assert crashed_row["state"] == "failed"
        trn = conn.execute(
            "SELECT new_state, reason FROM state_transitions WHERE target_id=?;",
            (crashed_row["target_id"],),
        ).fetchone()
        assert trn is not None, "no state_transitions row for the crashed target"
        assert trn["new_state"] == "failed"
        assert trn["reason"] == "unhandled_error_phase1"
    finally:
        conn.close()


def test_crash_transition_reads_current_state_not_hardcoded_new(tmp_path, offers_dir):
    # Production failure this prevents: hardcoding from_state="new" in the
    # crash transition.  A crash can happen at any stage — a target that died
    # mid-pipeline (e.g. after research moved it to "researched") must be
    # recorded as "researched -> failed", not "new -> failed"; otherwise the
    # audit trail lies about where the target actually was when it died, and
    # the operator re-runs the wrong stage.  This test moves the target to
    # "researched" through the REAL state machine before the crash and asserts
    # previous_state == "researched" in the resulting transition row.
    csv_path = _write_batch_csv(tmp_path, [
        {"company_name": "CrashCo", "domain": "crashco.test", "offer_id": "acme-offer"},
    ])
    db_path = str(tmp_path / "outbound.db")

    def crashing_after_research(agent, *, conn, target_id, domain, run_id, offers_dir="config/offers"):
        # B2c: offers_dir accepted (and ignored — this wrapper never
        # delegates) so the CLI's new keyword does not crash the fake.
        # Simulate that research completed before the crash: move the target
        # to "researched" through the real state-machine gate (never a raw
        # UPDATE, even in test setup), then blow up with the incident's
        # exception.
        transition(
            conn, target_id=target_id, from_state="new", to_state="researched",
            reason="test_simulated_research", actor="system",
            run_id=run_id, step_id=new_id("step"),
        )
        raise ServerDisconnectedError("Server disconnected")

    with _offline_pipeline_patch_stack(), patch(
        "app.phase1_cli.run_target_through_phase1", side_effect=crashing_after_research
    ):
        main(_cli_args(csv_path, db_path, offers_dir))

    conn = connect(db_path)
    try:
        target = conn.execute("SELECT target_id, state FROM targets;").fetchone()
        assert target["state"] == "failed"
        trn = conn.execute(
            "SELECT previous_state, new_state, reason FROM state_transitions "
            "WHERE target_id=? AND reason=?;",
            (target["target_id"], "unhandled_error_phase1"),
        ).fetchone()
        assert trn is not None, "no unhandled_error_phase1 transition row"
        # The read-current-state requirement: the row must record where the
        # target ACTUALLY was ("researched"), not a hardcoded "new".
        assert trn["previous_state"] == "researched"
        assert trn["new_state"] == "failed"
    finally:
        conn.close()


def test_crash_writes_a_step_row_with_the_exception_type_name(tmp_path, offers_dir):
    # Production failure this prevents: skipping the log for a crashed target
    # (CLAUDE.md §3 — never skip logs).  Without the failed step row, an
    # operator reading the trace later could not tell a ServerDisconnectedError
    # (a transport failure worth retrying) from a KeyError in our own code (a
    # bug worth fixing) — output_data carries type(exc).__name__ precisely so
    # the two are distinguishable from the steps table alone.
    csv_path = _write_batch_csv(tmp_path, [
        {"company_name": "CrashCo", "domain": "crashco.test", "offer_id": "acme-offer"},
    ])
    db_path = str(tmp_path / "outbound.db")

    with _offline_pipeline_patch_stack(), patch(
        "app.phase1_cli.run_target_through_phase1",
        side_effect=ServerDisconnectedError("Server disconnected"),
    ):
        exit_code = main(_cli_args(csv_path, db_path, offers_dir))
    assert exit_code != 0  # a crashed batch is not a clean run

    conn = connect(db_path)
    try:
        target = conn.execute("SELECT target_id FROM targets;").fetchone()
        step = conn.execute(
            "SELECT output_json, status FROM steps WHERE target_id=? AND status='failed';",
            (target["target_id"],),
        ).fetchone()
        assert step is not None, "the crash left no failed steps row"
        # output_json is the JSON text of output_data — the exception TYPE
        # NAME must be in it (the message alone could not distinguish a
        # transport failure from a coding bug).
        assert "ServerDisconnectedError" in step["output_json"]
        assert "Server disconnected" in step["output_json"]
    finally:
        conn.close()


def test_exit_code_stays_zero_when_a_target_merely_disqualifies(tmp_path, offers_dir):
    # Production failure this prevents: someone "simplifying" the exit-code
    # rule into "any target that didn't score fails the run" — that would turn
    # every not_target/watchlist batch into a red run and drown the crash
    # signal in noise.  Disqualification is a NORMAL Phase 1 result, not batch
    # damage; only an unhandled crash may make the exit code non-zero.
    csv_path = _write_batch_csv(tmp_path, [
        {"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer"},
    ])
    db_path = str(tmp_path / "outbound.db")

    with _offline_pipeline_patch_stack():
        exit_code = main(_cli_args(csv_path, db_path, offers_dir))
    assert exit_code == 0  # no crash -> clean run, even though the target disqualified

    conn = connect(db_path)
    try:
        target = conn.execute("SELECT state FROM targets;").fetchone()
        # The offline pipeline lands here deterministically (no contact data,
        # no signals, confidence 0.8 -> fit_score 18 -> not_target) — exactly
        # the "merely disqualified" shape the exit-code rule must ignore.
        assert target["state"] == "not_target"
    finally:
        conn.close()


def test_keyboard_interrupt_propagates_and_is_not_swallowed(tmp_path, offers_dir):
    # Production failure this prevents: the crash guard catching BaseException
    # instead of Exception.  An operator pressing Ctrl-C mid-batch must stop
    # the run — if the guard swallowed KeyboardInterrupt, the batch would keep
    # spending money on Gemini calls while the operator believes it stopped,
    # which is worse than the one-target crash the guard exists to contain.
    csv_path = _write_batch_csv(tmp_path, [
        {"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer"},
    ])
    db_path = str(tmp_path / "outbound.db")

    with _offline_pipeline_patch_stack(), patch(
        "app.phase1_cli.run_target_through_phase1", side_effect=KeyboardInterrupt
    ):
        with pytest.raises(KeyboardInterrupt):
            main(_cli_args(csv_path, db_path, offers_dir))

    conn = connect(db_path)
    try:
        target = conn.execute("SELECT state FROM targets;").fetchone()
        # BaseException bypasses the crash guard entirely: no "failed"
        # bookkeeping, no unhandled_error_phase1 reason — the target stays
        # "new", exactly as an aborted run should leave it.
        assert target["state"] == "new"
    finally:
        conn.close()


def test_cleanup_failure_does_not_abort_the_batch_or_mask_the_original_error(tmp_path, offers_dir, capsys):
    # Production failure this prevents: the bookkeeping itself killing the
    # batch.  If the DB connection is what broke (the most likely cause of a
    # crash right after a transport error), transition()/log_step() raise too
    # — without the second guard, the CLI would die inside its own error
    # handling, aborting the remaining targets AND hiding the original
    # exception behind the cleanup exception.  Here transition() is patched to
    # raise while the second of three targets crashes; the third target must
    # still complete and the original exception type must still reach the
    # operator's output.
    csv_path = _write_batch_csv(tmp_path, [
        {"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer"},
        {"company_name": "CrashCo", "domain": "crashco.test", "offer_id": "acme-offer"},
        {"company_name": "Beta", "domain": "beta.test", "offer_id": "acme-offer"},
    ])
    db_path = str(tmp_path / "outbound.db")

    def crashing_runner(agent, *, conn, target_id, domain, run_id, offers_dir="config/offers"):
        # B2c: offers_dir accepted and forwarded — same as the other wrappers.
        if domain == "crashco.test":
            raise ServerDisconnectedError("Server disconnected")
        return run_target_through_phase1(
            agent, conn=conn, target_id=target_id, domain=domain, run_id=run_id,
            offers_dir=offers_dir,
        )

    # transition patched to raise simulates the DB being the thing that broke:
    # the CLI's bookkeeping must contain that failure, not die on it.
    with _offline_pipeline_patch_stack(), patch(
        "app.phase1_cli.run_target_through_phase1", side_effect=crashing_runner
    ), patch("app.phase1_cli.transition", side_effect=RuntimeError("database connection lost")):
        exit_code = main(_cli_args(csv_path, db_path, offers_dir))
    assert exit_code != 0  # a crashed batch is still not a clean run

    # The original exception's TYPE must still reach the operator even though
    # the cleanup also failed — never let the cleanup failure mask the crash.
    err = capsys.readouterr().err
    assert "ServerDisconnectedError" in err
    assert "transition also failed" in err  # the cleanup failure is surfaced too, not swallowed

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT t.target_id, t.state, a.normalized_domain FROM targets t "
            "JOIN accounts a ON t.account_id = a.account_id;"
        ).fetchall()
        by_domain = {r["normalized_domain"]: r for r in rows}
        # The third target still completed — the loop continued past the
        # crash AND the failed cleanup.
        assert by_domain["beta.test"]["state"] in ("scored", "watchlist", "not_target")
        # The crashed target was NOT marked failed (transition was broken) —
        # the guard contained the failure honestly instead of faking success.
        assert by_domain["crashco.test"]["state"] == "new"
        # log_step was not patched, so the crash's step row still made it in —
        # never skip logs even when the transition half of the bookkeeping dies.
        step = conn.execute(
            "SELECT output_json FROM steps WHERE target_id=? AND status='failed';",
            (by_domain["crashco.test"]["target_id"],),
        ).fetchone()
        assert step is not None, "no failed steps row even though log_step was healthy"
        assert "ServerDisconnectedError" in step["output_json"]
    finally:
        conn.close()
