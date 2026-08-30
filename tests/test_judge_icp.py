# tests/test_judge_icp.py — B2c: the ICP judge (deterministic score demoted
# to evidence; the LLM judge issues the final label).
#
# Two kinds of test here:
# 1. Pipeline tests that patch ONLY the judge's LLM seam
#    (app.tools.judge_icp._call_judge_llm) and let the REAL judge_icp run —
#    tier lookup from the signals table, verdict persistence through the
#    write gate, step logging, and the ScoreNode's final routing — so the
#    DoD's "against the database" claims are proven against persisted rows,
#    not return values.
# 2. Pure-model tests on ICPVerdict (the divergence contract) and unit tests
#    on the judge's input construction (the B2b tier payoff must actually
#    reach the prompt).
#
# Every pipeline test here must keep the suite offline: the LLM seam is
# patched, the research stage is stubbed (the B1b pattern), and
# tests/conftest.py's autouse live-client guard stays untouched.

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.agents.phase1 import build_phase1_agent, run_target_through_phase1
from app.agents_registry import seed_agent_registry
from app.db import apply_schema, connect
from app.llm import LLMEmptyResponseError
from app.schemas import CompanyProfile, ICPAssessment, ICPVerdict, Signal
from app.tools.judge_icp import (
    JUDGE_AGENT_ID,
    JUDGE_TOOL_NAME,
    _build_user_content,
    _load_signal_tiers,
    judge_icp,
)
from app.write_gate import commit
from google.adk.agents import BaseAgent  # base class of the offline research stand-in (B1b pattern)
from google.adk.events import Event, EventActions  # how the stand-in publishes extracted_text


class _StubResearchAgent(BaseAgent):
    """Offline stand-in for the B1b research LlmAgent (same shape as the one
    in tests/test_agents_phase1.py) — publishes a fixed extracted_text
    through the state_delta mechanism the real agent's output_key uses, so
    the pipeline tests here never construct a live model client."""

    def __init__(self, findings: str | None):
        super().__init__(name="research")  # same stable name as the real agent
        self._findings = findings  # private attr — pydantic forbids public assignment

    async def _run_async_impl(self, ctx):
        # Mimic output_key: publish the findings under extracted_text, or
        # nothing at all when findings is None.
        if self._findings is not None:
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={"extracted_text": self._findings}),
            )


@pytest.fixture
def conn(scratch_db_target):
    """Fresh SQLite DB with schema, the three seeded principals (including
    icp_judge — the write gate refuses unregistered agents), and one
    offer/account/target at "researched" — the pipeline's expected entry
    state.  Mirrors the fixtures in test_score_lead.py / test_agents_phase1.py."""
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
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


@pytest.fixture
def offers_dir(tmp_path):
    """A tmp offers directory with one offer yaml carrying an icp block —
    the pipeline judge reads its offer context from here."""
    d = tmp_path / "offers"
    d.mkdir()
    (d / "acme.yaml").write_text(
        "pitch: p\nicp:\n  geography: HK\n  disqualifiers:\n    - outside HK\n"
    )
    return d


# ── Shared inputs ─────────────────────────────────────────────────────────────
# A deliberately WEAK profile: no industry/size/geo, confidence 0.5, no
# contact data, one 0.8-strength signal.  Hand-computed deterministic score:
# company 5 (confidence only) + persona 0 + signal round(8*0.8)=6 +
# completeness 5 + evidence 10 = 26 → deterministic label not_target.
# A below-60 score is exactly what the P4 zero-trust test needs: the judge
# may relabel it, but policy must still deny.

def _weak_profile() -> CompanyProfile:
    return CompanyProfile(one_line_summary="Weak fit", confidence=0.5)


def _weak_signals() -> list[Signal]:
    return [Signal(
        signal_type="hiring_relevant_role", signal_value="Hiring ops role",
        signal_strength=0.8,
        # B2a: every Signal requires an evidence quote — a placeholder; the
        # judge's tier lookup reads the signals TABLE, which detect_signals
        # populates for real in these pipeline tests.
        evidence_quote="hiring an operations manager for the team",
    )]


def _rationale(fill: str = "evidence") -> str:
    # A 120+ char rationale (ICPVerdict's minimum) built from a filler word
    # — long enough to pass validation, obviously synthetic for the stub.
    return " ".join([fill] * 25)


def _verdict_for(deterministic_label: str, fit_label: str, justification: str | None = None) -> ICPVerdict:
    """Build a stub verdict that echoes the real deterministic label (so
    _verify_echo passes) and carries a valid rationale + optional
    divergence justification."""
    return ICPVerdict(
        deterministic_fit_label=deterministic_label,
        fit_label=fit_label,
        rationale=_rationale(),
        divergence_justification=justification,
    )


def _patch_judge_llm(verdict_or_side_effect):
    """Patch ONLY the judge's LLM seam — the real judge_icp still runs its
    tier lookup, persistence, logging, and echo verification."""
    return patch(
        "app.tools.judge_icp._call_judge_llm",
        side_effect=verdict_or_side_effect
        if callable(verdict_or_side_effect) else lambda prompt, user_content: verdict_or_side_effect,
    )


def _echoing_verdict(fit_label: str, justification: str | None = None):
    """A judge-LLM side effect that reads the REAL deterministic label out
    of the user_content JSON (so the echo always matches whatever the
    formula actually produced — robust to fixture changes) and returns a
    verdict with the requested final label."""
    def _side_effect(prompt, user_content):
        payload = json.loads(user_content)
        deterministic = payload["deterministic_assessment"]["fit_label"]
        return _verdict_for(deterministic, fit_label, justification)
    return _side_effect


# ── 1. ICPVerdict: the divergence contract is enforced in the model ──────────


def test_divergence_without_justification_fails_validation():
    """A judge that diverges from the deterministic label WITHOUT a
    justification must fail Pydantic validation — the ticket's "must fail
    validation, not slip through" claim, enforced by the model validator."""
    with pytest.raises(ValidationError, match="divergence_justification"):
        ICPVerdict(
            deterministic_fit_label="not_target",
            fit_label="good_fit",  # diverges
            rationale=_rationale(),
            divergence_justification=None,  # missing — must be refused
        )


def test_divergence_with_a_token_justification_fails_validation():
    """A token justification ("n/a") is not a real justification — the
    model's 40-char floor refuses it."""
    with pytest.raises(ValidationError, match="at least 40 characters"):
        ICPVerdict(
            deterministic_fit_label="not_target",
            fit_label="good_fit",
            rationale=_rationale(),
            divergence_justification="n/a",  # token gesture — refused
        )


def test_agreement_with_justification_fails_validation():
    """The inverse direction: a judge that AGREES with the deterministic
    label must NOT carry a justification — a non-empty one would be a fake
    divergence record in the audit trail."""
    with pytest.raises(ValidationError, match="must be empty"):
        ICPVerdict(
            deterministic_fit_label="good_fit",
            fit_label="good_fit",  # agrees
            rationale=_rationale(),
            divergence_justification="the score was right because of evidence",  # contradictory — refused
        )


def test_rationale_minimum_length_is_enforced():
    """The written rationale has a real minimum (120 chars) — the
    boilerplate failure this ticket exists to fix."""
    with pytest.raises(ValidationError, match="rationale"):
        ICPVerdict(
            deterministic_fit_label="good_fit",
            fit_label="good_fit",
            rationale="Strong company-profile match",  # the old boilerplate — too short
        )


def test_judge_output_model_has_no_score_field():
    """The zero-trust claim, tested at the schema level: ICPVerdict has NO
    score field, and extra='forbid' rejects an emitted one.  A judge cannot
    produce, alter, or influence the number policy P4 reads even if it
    tries — the field's absence is the enforcement."""
    assert "fit_score" not in ICPVerdict.model_fields
    assert "score" not in ICPVerdict.model_fields
    # Even a smuggled extra key is refused by extra='forbid'.
    with pytest.raises(ValidationError, match="fit_score"):
        ICPVerdict.model_validate({
            "deterministic_fit_label": "good_fit",
            "fit_label": "good_fit",
            "rationale": _rationale(),
            "fit_score": 99,  # a judge trying to set a score — refused by construction
        })


# ── 2. The judge's label is the target's final state (pipeline + DB) ──────────


def test_diverging_judge_label_is_the_targets_final_state(conn, offers_dir):
    """Run the full pipeline with a judge that diverges upward
    (not_target → watchlist): the target's FINAL state must be the judge's
    label, read from the database — not the deterministic one, and not a
    route-then-reroute double hop."""
    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=_weak_profile()), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=_weak_signals()), \
         _patch_judge_llm(_echoing_verdict(fit_label="watchlist",
                                           justification="The company matches the offer's geography and industry disqualifier checks well enough to re-check later.")):
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(
            agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
            offers_dir=str(offers_dir),
        )

    # The runner's returned terminal state is the judge's label's state.
    assert final_state == "watchlist"
    # The DATABASE agrees — targets.state is what downstream phases read.
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "watchlist"


def test_divergence_persists_both_labels_and_justification(conn, offers_dir):
    """A divergence must persist BOTH labels (deterministic + judge) plus
    the judge's rationale and divergence justification — visible in the
    audit trail without reading code — and the routing transition's reason
    must be greppable for the override."""
    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=_weak_profile()), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=_weak_signals()), \
         _patch_judge_llm(_echoing_verdict(fit_label="watchlist",
                                           justification="The company matches the offer's geography and industry disqualifier checks well enough to re-check later.")):
        agent = build_phase1_agent(conn)
        run_target_through_phase1(
            agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
            offers_dir=str(offers_dir),
        )

    # Both labels on the account row: the deterministic evidence AND the
    # judge's verdict, plus rationale and justification.
    account = conn.execute(
        "SELECT icp_fit_label, judge_fit_label, judge_rationale, "
        "judge_divergence_justification FROM accounts WHERE account_id='acc_1';"
    ).fetchone()
    assert account["icp_fit_label"] == "not_target"  # the deterministic evidence
    assert account["judge_fit_label"] == "watchlist"  # the judge's verdict
    assert account["icp_fit_label"] != account["judge_fit_label"]  # divergence visible in columns
    assert account["judge_rationale"] is not None
    assert account["judge_divergence_justification"] is not None
    # The routing transition's reason names the override — greppable in
    # state_transitions.reason alone.  Filtered by the reason pattern (not
    # ORDER BY created_at, which is second-precision and unstable across
    # same-second transitions).
    trn = conn.execute(
        "SELECT reason FROM state_transitions WHERE target_id='tgt_1' "
        "AND reason LIKE '%judge_overrode%';"
    ).fetchone()
    assert trn is not None, "the judge override must be recorded in state_transitions"
    assert "judge_overrode_deterministic=not_target" in trn["reason"]
    # The judge's step row carries the divergence flag too (greppable in the
    # steps trace without joining accounts).
    judge_step = conn.execute(
        "SELECT output_json, agent_id, status FROM steps WHERE target_id='tgt_1' "
        "AND tool_name=? ORDER BY created_at DESC LIMIT 1;",
        (JUDGE_TOOL_NAME,),
    ).fetchone()
    assert judge_step["status"] == "success"
    assert judge_step["agent_id"] == JUDGE_AGENT_ID  # attributed to the judge principal
    assert json.loads(judge_step["output_json"])["diverged"] is True


def test_judge_verdict_writes_are_attributed_to_icp_judge(conn, offers_dir):
    """Every write the judge's verdict produces (the accounts.judge_*
    columns AND the judge-driven routing transition) must carry
    agent_id=icp_judge in write_log — attributable to the judge principal,
    never to system."""
    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=_weak_profile()), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=_weak_signals()), \
         _patch_judge_llm(_echoing_verdict(fit_label="watchlist",
                                           justification="The company matches the offer's geography and industry disqualifier checks well enough to re-check later.")):
        agent = build_phase1_agent(conn)
        run_target_through_phase1(
            agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
            offers_dir=str(offers_dir),
        )

    # The verdict write (accounts UPDATE) is attributed to the judge.
    verdict_writes = conn.execute(
        "SELECT agent_id FROM write_log WHERE action='update_account_icp_verdict';"
    ).fetchall()
    assert verdict_writes, "the verdict must be persisted through the write gate"
    assert all(r["agent_id"] == JUDGE_AGENT_ID for r in verdict_writes)
    # The judge-driven routing transition's write_log rows are too.
    transition_writes = conn.execute(
        "SELECT agent_id FROM write_log WHERE action='state_transition' "
        "AND payload_json LIKE '%judge_overrode%';"
    ).fetchall()
    assert transition_writes, "the judge-driven transition must be audited"
    assert all(r["agent_id"] == JUDGE_AGENT_ID for r in transition_writes)


# ── 3. The zero-trust claim: a judge cannot move a target past P4 ────────────


def test_judge_cannot_move_target_past_p4_floor(conn, offers_dir):
    """The ticket's zero-trust claim, tested against the database: with a
    deterministic fit_score BELOW 60 (26 here), a judge that labels the
    target good_fit changes the target's state to "scored" — but the policy
    gate still DENIES, because P4 reads the DETERMINISTIC fit_score, which
    the judge cannot touch.  The judge may set the label; it must not be
    able to talk the target past the floor."""
    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=_weak_profile()), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=_weak_signals()), \
         _patch_judge_llm(_echoing_verdict(
             fit_label="good_fit",
             justification="The deterministic score underweights the offer's ICP geography and industry match, which the profile supports.",
         )):
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(
            agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
            offers_dir=str(offers_dir),
        )

    # The judge's label made the target LOOK scored (good_fit stays at
    # "scored") — that is the label power the judge legitimately has.
    assert final_state == "scored"
    # The account row shows the split: deterministic not_target at score 26,
    # judge good_fit.
    account = conn.execute(
        "SELECT icp_fit_label, icp_fit_score, judge_fit_label FROM accounts "
        "WHERE account_id='acc_1';"
    ).fetchone()
    assert account["icp_fit_score"] < 60
    assert account["icp_fit_label"] == "not_target"
    assert account["judge_fit_label"] == "good_fit"
    # But the policy gate DENIED: P4 read the deterministic fit_score (26),
    # not anything the judge produced.  The judge's label travelled nowhere
    # near the gate.
    decision = conn.execute(
        "SELECT decision, matched_rules_json FROM policy_decisions "
        "WHERE target_id='tgt_1';"
    ).fetchone()
    assert decision is not None, "the policy gate must run and record a decision"
    assert decision["decision"] == "deny"
    assert "P4" in json.loads(decision["matched_rules_json"])
    # And targets.score — the routing/query field — still holds the
    # deterministic number, not a judge-influenced one.
    target = conn.execute("SELECT score FROM targets WHERE target_id='tgt_1';").fetchone()
    assert target["score"] == account["icp_fit_score"]


# ── 4. Judge failure degrades to the deterministic label ─────────────────────


def test_judge_llm_failure_degrades_to_deterministic_label(conn, offers_dir):
    """A judge that fails after its bounded retries must degrade to today's
    pre-B2c behaviour: the deterministic label routes the target, the target
    is STILL SCORED (never failed), and the judge's failed attempts are
    logged."""
    def _always_fail(prompt, user_content):
        # Both attempts fail the same way (empty structured output) — the
        # judge's documented failure mode.
        raise LLMEmptyResponseError("judge produced no structured output (stub)")

    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=_weak_profile()), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=_weak_signals()), \
         _patch_judge_llm(_always_fail):
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(
            agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
            offers_dir=str(offers_dir),
        )

    # The deterministic label (not_target, from score 26) routed the target.
    assert final_state == "not_target"
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "not_target"
    # The target was NOT failed: no "failed" transition exists for it.
    failed_hops = conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1' "
        "AND new_state='failed';"
    ).fetchone()
    assert failed_hops["n"] == 0
    # The judge's failed attempts are in the trace, attributed to the judge.
    judge_steps = conn.execute(
        "SELECT status, agent_id FROM steps WHERE target_id='tgt_1' "
        "AND tool_name=? ORDER BY created_at;",
        (JUDGE_TOOL_NAME,),
    ).fetchall()
    assert len(judge_steps) == 2  # the bounded two attempts, both logged
    assert all(s["status"] == "failed" for s in judge_steps)
    assert all(s["agent_id"] == JUDGE_AGENT_ID for s in judge_steps)
    # No judge verdict was persisted — NULL judge columns are the honest
    # "the judge never produced a verdict" value.
    account = conn.execute(
        "SELECT judge_fit_label, judge_rationale FROM accounts WHERE account_id='acc_1';"
    ).fetchone()
    assert account["judge_fit_label"] is None
    assert account["judge_rationale"] is None


def test_judge_echoing_wrong_deterministic_label_is_rejected(conn, offers_dir):
    """A judge that lies about which deterministic label it was given must
    not get its verdict trusted: the echo check refuses it (treated like
    schema-invalid output), and the target degrades to the deterministic
    label instead of routing on a verdict computed against a false premise."""
    def _wrong_echo(prompt, user_content):
        # Echoes "strong_fit" regardless of what the real deterministic
        # label was — the divergence validator would compare the judge's
        # label against this lie.  (A justification is supplied because the
        # stub verdict itself diverges from its own lie and must pass model
        # validation before the echo check can refuse it.)
        return _verdict_for(
            "strong_fit", "good_fit",
            justification="The lied-about deterministic label underweights this target's ICP match.",
        )

    with patch("app.agents.phase1.build_research_agent",
               return_value=_StubResearchAgent(findings="Acme does logistics. Hiring ops manager.")), \
         patch("app.tools.summarize_company.call_structured", return_value=_weak_profile()), \
         patch("app.tools.detect_signals._call_detect_signals", return_value=_weak_signals()), \
         _patch_judge_llm(_wrong_echo):
        agent = build_phase1_agent(conn)
        final_state = run_target_through_phase1(
            agent, conn=conn, target_id="tgt_1", domain="acme.test", run_id="r1",
            offers_dir=str(offers_dir),
        )

    # The deterministic label routed — the lying verdict was never applied.
    assert final_state == "not_target"
    # And no verdict was persisted.
    account = conn.execute(
        "SELECT judge_fit_label FROM accounts WHERE account_id='acc_1';"
    ).fetchone()
    assert account["judge_fit_label"] is None


# ── 5. Missing icp block + the tier payoff ───────────────────────────────────


def test_missing_icp_block_still_works(conn):
    """An offer with NO icp block (and no pitch) must not break scoring:
    the judge still runs with less to go on and its verdict persists.
    (A missing icp block is a documented, supported configuration — the
    fixture offers and every pre-B2c offer yaml are exactly this shape.)"""
    assessment = ICPAssessment(
        fit_label="good_fit", fit_score=65,
        fit_reasons=["Strong company-profile match"], non_fit_reasons=[],
    )
    with _patch_judge_llm(_verdict_for("good_fit", "good_fit")):
        verdict = judge_icp(
            conn,
            company_profile=_weak_profile(),
            signals=_weak_signals(),
            icp_assessment=assessment,
            offer_icp=None,  # no icp block on the offer
            offer_pitch=None,  # no pitch either
            target_id="tgt_1",
            run_id="r1",
            step_id="s1",
        )
    # The judge succeeded and agreed.
    assert verdict is not None
    assert verdict.fit_label == "good_fit"
    account = conn.execute(
        "SELECT judge_fit_label FROM accounts WHERE account_id='acc_1';"
    ).fetchone()
    assert account["judge_fit_label"] == "good_fit"


def test_judge_input_carries_each_signals_persisted_tier(conn):
    """The B2b payoff must actually reach the judge: the user_content JSON
    attaches each signal's persisted evidence tier inline, so a judge cannot
    claim it was not told which evidence is attributable to a fetched page."""
    tiers = {
        ("hiring_relevant_role", "Hiring ops role"): "unverified",
        ("product_or_ops_change", "Expanding ops"): "source",
    }
    signals = [
        Signal(
            signal_type="hiring_relevant_role", signal_value="Hiring ops role",
            signal_strength=0.8, evidence_quote="hiring an operations manager for the team",
        ),
        Signal(
            signal_type="product_or_ops_change", signal_value="Expanding ops",
            signal_strength=0.7, evidence_quote="expanding the operations team this quarter",
        ),
    ]
    assessment = ICPAssessment(
        fit_label="good_fit", fit_score=65, fit_reasons=[], non_fit_reasons=[],
    )
    content = _build_user_content(_weak_profile(), signals, tiers, assessment, None, None)
    payload = json.loads(content)
    # Both signals carry their tier inline — the source-tier one and the
    # unverified one — plus the deterministic assessment as evidence.
    by_value = {s["signal_value"]: s for s in payload["signals_with_evidence_tiers"]}
    assert by_value["Hiring ops role"]["evidence_tier"] == "unverified"
    assert by_value["Expanding ops"]["evidence_tier"] == "source"
    assert payload["deterministic_assessment"]["fit_score"] == 65
    assert payload["offer"]["icp"] is None  # absent icp is represented honestly


def test_signal_tiers_are_loaded_from_the_database(conn):
    """_load_signal_tiers reads the persisted evidence_tier column — the
    same verdict the operator sees in the console — keyed by the signals
    table's UNIQUE (signal_type, signal_value) identity per (target, run)."""
    # Insert two signals with different tiers through the write gate (the
    # only core-table write path), the way detect_signals does.
    for signal_id, signal_type, signal_value, tier in (
        ("sig_1", "hiring_relevant_role", "Hiring ops role", "unverified"),
        ("sig_2", "product_or_ops_change", "Expanding ops", "source"),
    ):
        commit(
            conn,
            action="insert_signal", table_name="signals", record_id=signal_id,
            payload={}, run_id="r1", step_id="s1", actor="system", agent_id="system",
            sql="""INSERT INTO signals
                   (signal_id, run_id, target_id, signal_type, signal_value,
                    signal_strength, evidence_tier, created_at)
                   VALUES (?,?,?,?,?,?,?, datetime('now'))""",
            params=(signal_id, "r1", "tgt_1", signal_type, signal_value, 0.8, tier),
        )
    tiers = _load_signal_tiers(conn, "tgt_1", "r1")
    assert tiers == {
        ("hiring_relevant_role", "Hiring ops role"): "unverified",
        ("product_or_ops_change", "Expanding ops"): "source",
    }
    # A different run sees no tiers — the lookup is run-scoped, so a re-run
    # never inherits a previous run's verdicts.
    assert _load_signal_tiers(conn, "tgt_1", "r_other") == {}
