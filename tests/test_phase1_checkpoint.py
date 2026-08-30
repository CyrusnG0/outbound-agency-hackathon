# tests/test_phase1_checkpoint.py
# Phase 1 required checkpoint acceptance test (state-machine.md §7d).
# Imports 5 targets across 2 offers, runs the full Phase 1 pipeline
# (mocking external LLM/network calls), and verifies the mechanical
# checkpoint bar: >=4/5 reach `scored`, >=2 offers represented,
# and every scored target has persisted fit_reasons and signals.
#
# B1b adaptation (research LlmAgent): the pipeline's first node is now a
# REAL tool-choosing LlmAgent that would make live billable Vertex calls, so
# both tests stand the RETAINED FetchAndNormalizeNode in for the research
# stage via patch("app.agents.phase1.build_research_agent") — the same
# offline stand-in pattern as tests/test_agents_phase1.py's A6 test.  Every
# assertion below is unchanged: the stand-in drives the same mocked
# fetch_sources and the real normalize_sources, so the no_sources_available
# path (Ironclad Autonomy) and the steps-table logging assertions still
# exercise the real code they always did.  The real research agent's
# behaviour is covered in tests/test_research_agent.py.
from unittest.mock import patch

import pytest

from app.agents_registry import seed_agent_registry
from app.config import sync_offers_table
from app.db import apply_schema, connect
from app.agents.phase1 import (  # the retained node stands in for the research agent (see header)
    FetchAndNormalizeNode,
    build_phase1_agent,
    run_target_through_phase1,
)
from app.ids import new_id
from app.schemas import CompanyProfile, Signal
from app.state_machine import PHASE_1_REACHABLE_STATES
from app.tools.fetch_sources import NormalizedSource
from app.tools.get_targets import import_csv

FIXTURES_DIR = "tests/fixtures"


def _fake_fetch_sources_for(domain: str):
    """Simulate realistic per-target research output — 4 of 5 targets get
    usable content, 1 (Ironclad Autonomy) gets nothing, to exercise the
    no_sources_available path within the same checkpoint run."""
    # Ironclad Autonomy is the deliberately-zero-source target — its domain
    # returns an empty list, which triggers normalize_sources' zero-sources
    # guard and routes it directly to failed with reason=no_sources_available.
    if domain == "ironclad-autonomy.test":
        return []
    # Every other target yields one NormalizedSource — enough for the rest
    # of the pipeline (normalize → summarize → detect_signals → score) to
    # produce a successful scored outcome.
    return [
        NormalizedSource(
            "company_website", f"https://{domain}",
            f"{domain} is a growing company hiring operations staff.",
            "t", 0.8, 1, "static",
        )
    ]


def _fake_profile_for(domain: str) -> CompanyProfile:
    """Return a minimal but valid CompanyProfile — the scoring formula's
    company-fit component checks industry/estimated_size/geo, so we provide
    all three fields to avoid losing points on data completeness."""
    return CompanyProfile(
        one_line_summary=f"Company at {domain}", industry="Services",
        estimated_size="51-200", geo="US", confidence=0.8,
    )


def _fake_signals_for(domain: str) -> list[Signal]:
    """Return one positive signal — score_lead's signal-fit component awards
    points per signal, so this contributes to reaching the scored threshold."""
    return [Signal(
        signal_type="hiring_relevant_role", signal_value="Hiring ops role",
        signal_strength=0.7,
        # B2a: every Signal now requires an evidence quote — a placeholder
        # here; the checkpoint assertions are about scoring and step logging,
        # not quote verification.
        evidence_quote="hiring an operations role for the team",
    )]


@pytest.fixture
def conn(scratch_db_target):
    """Create a fresh in-memory-like SQLite DB (via tmp_path) with the full
    schema applied and offers synced from the fixture directory — each test
    gets a clean database, so the two tests don't interfere."""
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # Register the system agent (plan A3) — every write below flows through
    # the gate, which refuses unregistered agents.
    seed_agent_registry(c, run_id="setup", step_id="setup")
    # Sync both offer fixtures so the CSV import can resolve offer_id by slug.
    sync_offers_table(c, f"{FIXTURES_DIR}/offers", run_id="setup", step_id="setup")
    yield c
    c.close()


def test_five_targets_two_offers_checkpoint_bar_is_measurable(conn, tmp_path):
    """The required Phase 1 acceptance test (state-machine.md §7d,
    ceo-plan.md 'Required Phase 1 acceptance test'): import 5 targets
    across 2 offers, run the full pipeline, and confirm the mechanical
    half of the checkpoint bar (>=4/5 reach a terminal Phase 1 state,
    specifically `scored`) is achievable end to end. The qualitative half
    (3/5 judged as good as manual) is an operator judgment call this test
    cannot automate — it verifies the DATA the operator needs is present
    and correct (persisted fit_reasons, signals, scores)."""
    # Import the 5-target CSV — the entry point for the pipeline.
    run_id = new_id("run")
    target_ids = import_csv(
        conn, csv_path=f"{FIXTURES_DIR}/checkpoint_targets.csv", cli_offer_slug=None,
        run_id=run_id, step_id=new_id("step"),
    )
    assert len(target_ids) == 5  # All 5 rows in the CSV must be imported.

    # Build a domain→target_id map from the database so we can route each
    # target through the pipeline with the domain its mock expects.
    domains_by_target = {}
    for tid in target_ids:
        row = conn.execute(
            "SELECT a.normalized_domain FROM targets t JOIN accounts a ON t.account_id=a.account_id "
            "WHERE t.target_id=?;", (tid,),
        ).fetchone()
        domains_by_target[tid] = row["normalized_domain"]

    # Build the Phase 1 agent once — the same agent runs every target.
    # B1b: the build happens INSIDE the patch so the research stage is the
    # offline stand-in (the retained fetch node), never the live LlmAgent —
    # without this patch the run would make real billable Vertex calls.
    final_states: dict[str, str] = {}
    with patch("app.agents.phase1.build_research_agent",
               return_value=FetchAndNormalizeNode(name="research", conn=conn)):
        agent = build_phase1_agent(conn)
        # Run every target through the pipeline, mocking the three external
        # dependencies: fetch_sources (network), summarize_company (LLM), and
        # detect_signals (LLM). No real network or API calls are made.
        for tid, domain in domains_by_target.items():
            with patch("app.tools.fetch_sources.fetch_sources", return_value=_fake_fetch_sources_for(domain)), \
                 patch("app.tools.summarize_company.call_structured", return_value=_fake_profile_for(domain)), \
                 patch("app.tools.detect_signals._call_detect_signals", return_value=_fake_signals_for(domain)), \
                 patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
                final_states[tid] = run_target_through_phase1(agent, conn=conn, target_id=tid, domain=domain, run_id=run_id)

    # Every target lands somewhere in the Phase 1 reachable set — never a Phase 1b state.
    assert all(state in PHASE_1_REACHABLE_STATES for state in final_states.values())

    # The checkpoint's mechanical bar: >=4/5 reach `scored` specifically
    # (not watchlist/not_target/failed) is what "the pipeline works" means.
    # This fixture is deliberately built so 4/5 succeed and 1/5
    # (Ironclad Autonomy, zero sources) fails — the realistic worst case.
    scored_count = sum(1 for s in final_states.values() if s == "scored")
    assert scored_count >= 4, f"checkpoint bar requires >=4/5 scored, got {scored_count}/5: {final_states}"

    # Both offers are represented among successfully-scored targets — the
    # checkpoint requires >=2 offers, per state-machine.md §7d.
    scored_target_ids = [tid for tid, s in final_states.items() if s == "scored"]
    offers_represented = set()
    for tid in scored_target_ids:
        row = conn.execute(
            "SELECT o.slug FROM targets t JOIN offers o ON t.offer_id = o.offer_id WHERE t.target_id=?;",
            (tid,),
        ).fetchone()
        offers_represented.add(row["slug"])
    assert len(offers_represented) >= 2  # Must include therapy-app AND ai-security.

    # The operator's manual judgment ("3/5 as good as manual") needs
    # fit_reasons and signals to actually be reviewable — verify they're
    # persisted and non-empty for every scored target, not just present
    # as columns.
    for tid in scored_target_ids:
        account = conn.execute(
            "SELECT icp_fit_reasons FROM accounts a JOIN targets t ON a.account_id = t.account_id "
            "WHERE t.target_id=?;", (tid,),
        ).fetchone()
        assert account["icp_fit_reasons"] is not None  # Must be written by score_lead.

        signal_rows = conn.execute("SELECT * FROM signals WHERE target_id=?;", (tid,)).fetchall()
        assert len(signal_rows) >= 1  # Must have at least one signal from detect_signals.

    # The one deliberately-zero-source target failed for the RIGHT reason,
    # not silently.
    failed_target_ids = [tid for tid, s in final_states.items() if s == "failed"]
    assert len(failed_target_ids) == 1  # Exactly Ironclad Autonomy.
    failure_reason = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id=? ORDER BY created_at DESC LIMIT 1;",
        (failed_target_ids[0],),
    ).fetchone()
    assert failure_reason["reason"] == "no_sources_available"


def test_every_tool_call_across_the_batch_is_logged(conn, tmp_path):
    """Per docs/tool-registry.md: a run is not complete until every step
    is logged. Verify the steps table actually has entries for a full
    batch run, not just the happy path."""
    # Import the same 5-target CSV used by the checkpoint test.
    run_id = new_id("run")
    target_ids = import_csv(
        conn, csv_path=f"{FIXTURES_DIR}/checkpoint_targets.csv", cli_offer_slug=None,
        run_id=run_id, step_id=new_id("step"),
    )
    # B1b: build inside the patch so the research stage is the offline
    # stand-in (see the header) — the stand-in runs the real
    # normalize_sources, which is what keeps the steps-table assertions
    # below meaningful.
    with patch("app.agents.phase1.build_research_agent",
               return_value=FetchAndNormalizeNode(name="research", conn=conn)):
        agent = build_phase1_agent(conn)

        # Run every target through the pipeline with mocks — the same setup as
        # the checkpoint test, but this test only cares about step logging, not
        # final state distribution.
        for tid in target_ids:
            row = conn.execute(
                "SELECT a.normalized_domain FROM targets t JOIN accounts a ON t.account_id=a.account_id "
                "WHERE t.target_id=?;", (tid,),
            ).fetchone()
            domain = row["normalized_domain"]
            with patch("app.tools.fetch_sources.fetch_sources", return_value=_fake_fetch_sources_for(domain)), \
                 patch("app.tools.summarize_company.call_structured", return_value=_fake_profile_for(domain)), \
                 patch("app.tools.detect_signals._call_detect_signals", return_value=_fake_signals_for(domain)), \
                 patch("app.agents.phase1.judge_icp_module.judge_icp", return_value=None):
                run_target_through_phase1(agent, conn=conn, target_id=tid, domain=domain, run_id=run_id)

    # The steps table must record both the data-acquisition tool and the
    # scoring tool — if these are missing, logging is broken.
    steps = conn.execute("SELECT DISTINCT tool_name FROM steps WHERE run_id=?;", (run_id,)).fetchall()
    tool_names = {r["tool_name"] for r in steps}
    # At minimum, a source-fetching or source-normalizing step must be logged.
    assert "fetch_company_page" in tool_names or "normalize_sources" in tool_names
    # And the scoring step must be logged for at least the non-failed targets.
    assert "score_lead" in tool_names
