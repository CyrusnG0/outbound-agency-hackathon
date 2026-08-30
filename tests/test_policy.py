import json

import pytest

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.write_gate import commit
from app.schemas import CompanyProfile, ICPAssessment
from app.policy import policy_check_phase1


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
        params=("tgt_1", "acc_1", "off_1", "csv", "scored"),
    )
    yield c
    c.close()


def test_p3a_does_not_require_verified_email_or_contact_name(conn):
    profile = CompanyProfile(one_line_summary="A real summary", confidence=0.7)
    assessment = ICPAssessment(fit_label="good_fit", fit_score=65, fit_reasons=["x"], non_fit_reasons=[])
    decision = policy_check_phase1(
        conn, company_profile=profile, icp_assessment=assessment, signals=[],
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    assert decision.decision == "allow"
    assert "verified_email" not in decision.required_fields_missing
    assert "contact_candidates[].name" not in decision.required_fields_missing


def test_p3a_denies_missing_company_summary(conn):
    profile = CompanyProfile(one_line_summary="", confidence=0.7)
    assessment = ICPAssessment(fit_label="good_fit", fit_score=65, fit_reasons=[], non_fit_reasons=[])
    decision = policy_check_phase1(
        conn, company_profile=profile, icp_assessment=assessment, signals=[],
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    assert decision.decision in ("deny", "review_required")
    assert "company_profile.one_line_summary" in decision.required_fields_missing


def test_p4_denies_score_below_60(conn):
    profile = CompanyProfile(one_line_summary="x", confidence=0.7)
    assessment = ICPAssessment(fit_label="watchlist", fit_score=45, fit_reasons=[], non_fit_reasons=[])
    decision = policy_check_phase1(
        conn, company_profile=profile, icp_assessment=assessment, signals=[],
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    assert decision.decision == "deny"
    assert "P4" in decision.matched_rules


def test_decision_is_persisted_to_policy_decisions_table(conn):
    profile = CompanyProfile(one_line_summary="x", confidence=0.7)
    assessment = ICPAssessment(fit_label="good_fit", fit_score=65, fit_reasons=[], non_fit_reasons=[])
    policy_check_phase1(
        conn, company_profile=profile, icp_assessment=assessment, signals=[],
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    row = conn.execute("SELECT * FROM policy_decisions WHERE target_id='tgt_1';").fetchone()
    assert row is not None
    # This fixture resolves to decision="allow" (fit_score=65 clears P4, all P3a
    # fields present), so matched_rules is legitimately [] — policy_check_phase1
    # only appends to matched_rules when a rule actually fires. reasons, unlike
    # matched_rules, is guaranteed non-empty on every decision path (the "allow"
    # branch always appends its own explanatory reason), so assert on that instead.
    assert json.loads(row["reasons_json"])
