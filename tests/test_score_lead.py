import pytest

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.write_gate import commit
from app.schemas import CompanyProfile, Signal
from app.tools.score_lead import apply_final_fit_label, score_lead


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
        params=("tgt_1", "acc_1", "off_1", "csv", "researched"),
    )
    yield c
    c.close()


def test_high_signal_strength_with_contact_reaches_scored(conn):
    # industry="Logistics" is required for this fixture to actually clear
    # the >=60 assertion below: hand-computed against the exact formula in
    # Step 3, confidence=0.9 alone (company=9, no industry/size bonus) plus
    # persona=15, signal=15, completeness=10, evidence=10 totals 59 — one
    # point short of "good_fit". Adding industry gives +10 (company=19),
    # landing at 69 with comfortable margin instead of exactly on the
    # boundary. Do not remove this field.
    profile = CompanyProfile(one_line_summary="Great fit company", industry="Logistics", confidence=0.9)
    signals = [
        # B2a: every Signal now requires an evidence quote — a placeholder
        # here; these fixtures exercise the deterministic scoring formula,
        # which reads signal_strength and signal_type, not the quote.
        Signal(
            signal_type="hiring_relevant_role", signal_value="x", signal_strength=1.0,
            evidence_quote="placeholder evidence quote for scoring fixtures",
        ),
        Signal(
            signal_type="product_or_ops_change", signal_value="x", signal_strength=1.0,
            evidence_quote="placeholder evidence quote for scoring fixtures",
        ),
    ]
    assessment = score_lead(
        conn, company_profile=profile, signals=signals, has_contact_data=True,
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    assert assessment.fit_score >= 60
    row = conn.execute("SELECT state, score FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] in ("scored",)
    assert row["score"] == assessment.fit_score

    account = conn.execute("SELECT * FROM accounts WHERE account_id='acc_1';").fetchone()
    assert account["icp_fit_label"] == assessment.fit_label


def test_zero_signals_no_contact_lands_in_watchlist_or_not_target(conn):
    # B2c: score_lead no longer routes scored→watchlist/not_target itself —
    # the final routing happens in apply_final_fit_label, which runs AFTER
    # the ICP judge with the FINAL label.  This test mirrors the pipeline's
    # deterministic fallback: the judge produced nothing, so the
    # deterministic label is applied by the wiring.
    profile = CompanyProfile(one_line_summary="Weak fit", confidence=0.5)
    assessment = score_lead(
        conn, company_profile=profile, signals=[], has_contact_data=False,
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    assert assessment.fit_score < 60
    # After score_lead alone the target sits at "scored" — scoring completed,
    # but no final label has been applied yet (that is the judge's slot).
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "scored"
    # Apply the deterministic label the way ScoreNode does on judge failure.
    apply_final_fit_label(
        conn, target_id="tgt_1", run_id="r1", step_id="s1",
        final_label=assessment.fit_label, deterministic_label=assessment.fit_label,
    )
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] in ("watchlist", "not_target")


def test_max_reachable_score_without_contact_data_is_70(conn):
    """Per policy-matrix.md P4's Phase 1 reachability note: Persona Fit (20)
    and 'clear contact identified' (5) are unreachable without contact
    data, capping the max at 70/100 — not 100/100."""
    profile = CompanyProfile(
        one_line_summary="Perfect fit", industry="x", estimated_size="x", geo="x", confidence=1.0,
    )
    signals = [
        # B2a: every Signal now requires an evidence quote — a placeholder
        # here; these fixtures exercise the deterministic scoring formula,
        # which reads signal_strength and signal_type, not the quote.
        Signal(
            signal_type="hiring_relevant_role", signal_value="x", signal_strength=1.0,
            evidence_quote="placeholder evidence quote for scoring fixtures",
        ),
        Signal(
            signal_type="product_or_ops_change", signal_value="x", signal_strength=1.0,
            evidence_quote="placeholder evidence quote for scoring fixtures",
        ),
        Signal(
            signal_type="recent_launch_or_expansion", signal_value="x", signal_strength=1.0,
            evidence_quote="placeholder evidence quote for scoring fixtures",
        ),
        Signal(
            signal_type="workflow_complexity_evidence", signal_value="x", signal_strength=1.0,
            evidence_quote="placeholder evidence quote for scoring fixtures",
        ),
    ]
    assessment = score_lead(
        conn, company_profile=profile, signals=signals, has_contact_data=False,
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    assert assessment.fit_score <= 70


def test_fit_reasons_and_non_fit_reasons_persisted_to_accounts(conn):
    profile = CompanyProfile(one_line_summary="x", confidence=0.5)
    score_lead(
        conn, company_profile=profile, signals=[], has_contact_data=True,
        target_id="tgt_1", run_id="r1", step_id="s1",
    )
    account = conn.execute("SELECT icp_fit_reasons, icp_non_fit_reasons FROM accounts WHERE account_id='acc_1';").fetchone()
    assert account["icp_fit_reasons"] is not None
    assert account["icp_non_fit_reasons"] is not None


# ── apply_final_fit_label (ticket B2c): the single post-judge routing hop ──


def test_apply_final_fit_label_agreeing_keeps_pre_b2c_reason(conn):
    """When the final label equals the deterministic one, the transition
    reason keeps the pre-B2c vocabulary (fit_label=<label>) — the judge
    merely agreed, so the audit trail's language is unchanged."""
    # Move the target to "scored" first — apply_final_fit_label's contract
    # is scored→{label}, and the fixture target starts at "researched".
    from app.state_machine import transition

    transition(
        conn, target_id="tgt_1", from_state="researched", to_state="scored",
        reason="scoring_complete", actor="system", run_id="r1", step_id="s1",
    )
    apply_final_fit_label(
        conn, target_id="tgt_1", run_id="r1", step_id="s1",
        final_label="not_target", deterministic_label="not_target",
    )
    # Filter by the routing hop's new_state (not ORDER BY created_at, whose
    # second precision cannot order same-second transitions reliably).
    row = conn.execute(
        "SELECT new_state, reason FROM state_transitions WHERE target_id='tgt_1' "
        "AND new_state='not_target';"
    ).fetchone()
    assert row["new_state"] == "not_target"
    assert row["reason"] == "fit_label=not_target"


def test_apply_final_fit_label_divergence_is_greppable_and_attributed(conn):
    """When the judge overrode the deterministic label, the transition reason
    names BOTH labels (greppable divergence) and the write_log rows are
    attributed to the judge principal, not to system."""
    from app.state_machine import transition

    transition(
        conn, target_id="tgt_1", from_state="researched", to_state="scored",
        reason="scoring_complete", actor="system", run_id="r1", step_id="s1",
    )
    apply_final_fit_label(
        conn, target_id="tgt_1", run_id="r1", step_id="s1",
        final_label="watchlist", deterministic_label="not_target",
        agent_id="icp_judge",  # the judge's verdict drove this hop
    )
    # Filter by the routing hop's new_state (see the agreeing test for why
    # ORDER BY created_at is avoided).
    row = conn.execute(
        "SELECT new_state, reason FROM state_transitions WHERE target_id='tgt_1' "
        "AND new_state='watchlist';"
    ).fetchone()
    assert row["new_state"] == "watchlist"
    # The divergence is greppable in state_transitions.reason alone.
    assert "judge_overrode_deterministic=not_target" in row["reason"]
    # Both write_log rows of the transition name the judge principal — the
    # attribution the ticket requires for judge-driven transitions.
    log_rows = conn.execute(
        "SELECT agent_id FROM write_log WHERE table_name IN ('targets','state_transitions') "
        "AND payload_json LIKE '%judge_overrode%' ORDER BY created_at;"
    ).fetchall()
    assert log_rows, "the transition must be audited in write_log"
    assert all(r["agent_id"] == "icp_judge" for r in log_rows)


def test_apply_final_fit_label_strong_fit_needs_no_hop(conn):
    """strong_fit/good_fit final labels stay at "scored" — no second
    transition is written (the scored→watchlist/not_target hop only fires
    for the two routing labels)."""
    from app.state_machine import transition

    transition(
        conn, target_id="tgt_1", from_state="researched", to_state="scored",
        reason="scoring_complete", actor="system", run_id="r1", step_id="s1",
    )
    before = conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()["n"]
    apply_final_fit_label(
        conn, target_id="tgt_1", run_id="r1", step_id="s1",
        final_label="strong_fit", deterministic_label="good_fit",
        agent_id="icp_judge",
    )
    after = conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()["n"]
    assert after == before  # no hop written
    row = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert row["state"] == "scored"
