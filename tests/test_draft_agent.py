# tests/test_draft_agent.py — B3: the draft writer⇄critic LoopAgent.
#
# Every test keeps the suite offline by patching ONLY the two LLM agent
# factories (app.agents.draft._build_writer_agent / _build_critic_agent)
# with offline stand-ins that publish fixed dicts through the same
# state_delta mechanism the real agents' output_key uses — the same pattern
# tests/test_agents_phase1.py applies to build_research_agent, so
# tests/conftest.py's autouse live-client guard is never tripped and the
# real DraftPersistAndDecideNode (validation, transition, gated write,
# logging, loop-exit decision) runs for real against the database.
#
# The four B3 zero-trust boundaries each get a test that fails if someone
# "simplifies" them away: no footer field on EmailDraft (B3-Z1), the
# critic's passed flag never approving/sending (B3-Z2), the gate columns
# always NULL from B3 (B3-Z3), and the audit trail proving the gated write
# path (which is also the B3-Z4-adjacent check that an LLM never writes).

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.agents.draft import (
    DRAFT_CRITIC_AGENT_ID,
    DRAFT_WRITER_AGENT_ID,
    _CRITIC_INSTRUCTION,
    _STYLE_HYPOTHESES,
    _WRITER_INSTRUCTION,
    _build_draft_context,
    _select_style_hypothesis,
    build_draft_agent,
    run_target_through_draft,
)
from app.agents_registry import seed_agent_registry
from app.db import apply_schema, connect
from app.ids import new_id
from app.schemas import DraftCritique, EmailDraft
from app.write_gate import commit
from google.adk.agents import BaseAgent  # base class of the offline writer/critic stand-ins (B1b pattern)
from google.adk.events import Event, EventActions  # how the stand-ins publish their output dicts


# ── Offline stand-ins for the two LLM agents ─────────────────────────────────
# The real writer/critic are ADK LlmAgents that would make live billable
# calls; these stubs publish predetermined dicts under the same state keys
# ("draft" / "critique") the real agents' output_schema + output_key write.
# Both read the session's draft_revision counter (published by the persist
# node at the end of each iteration) to behave differently per iteration —
# exactly the way the real writer consumes the critique feedback.

class _StubWriterAgent(BaseAgent):
    """Offline stand-in for the writer LlmAgent: publishes the Nth draft
    dict under state key "draft" (mimicking output_key), where N is the
    current loop iteration.  Counts invocations so tests can assert the
    loop stopped early."""

    def __init__(self, drafts: list[dict]):
        super().__init__(name="draft_writer")  # same stable name as the real agent
        self._drafts = drafts  # private attr — pydantic forbids public assignment
        self._calls = 0  # invocation counter, read by tests to prove early loop exit

    async def _run_async_impl(self, ctx):
        self._calls += 1  # record this invocation BEFORE publishing — tests assert on the final count
        revision = ctx.session.state.get("draft_revision", 0)  # 0 on iteration 1; the persist node publishes 1, 2, ... after each iteration
        # Clamp to the last provided draft: a never-passing critic drives 3
        # iterations but the stub list may be shorter — the last draft
        # repeats, which is exactly what a writer told "fix this" would do
        # when its stub has no further variations.
        idx = min(revision, len(self._drafts) - 1)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"draft": self._drafts[idx]}),
        )


class _StubCriticAgent(BaseAgent):
    """Offline stand-in for the critic LlmAgent: publishes a
    pass/fail DraftCritique dict under state key "critique", chosen per
    loop iteration from a list of booleans."""

    def __init__(self, verdicts: list[bool]):
        super().__init__(name="draft_critic")  # same stable name as the real agent
        self._verdicts = verdicts  # private attr — pydantic forbids public assignment

    async def _run_async_impl(self, ctx):
        revision = ctx.session.state.get("draft_revision", 0)  # same per-iteration read as the writer stub
        passed = self._verdicts[min(revision, len(self._verdicts) - 1)]
        critique = _pass_critique() if passed else _fail_critique()
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"critique": critique}),
        )


class _SilentWriterAgent(BaseAgent):
    """Offline stand-in that publishes NOTHING — reproducing the real
    writer's failure shape ("the agent produced no output", so state key
    "draft" is absent, exactly like _StubResearchAgent(findings=None))."""

    def __init__(self):
        super().__init__(name="draft_writer")

    async def _run_async_impl(self, ctx):
        # Yield nothing: the "draft" key stays absent from session state,
        # which is the real LlmAgent's behaviour on an empty final turn.
        # The unreachable yield keeps this an ASYNC GENERATOR — ADK's
        # _run_async_impl protocol iterates it, and a yield-less coroutine
        # would be "never awaited" instead of "yielded nothing" (same
        # shape as _StubResearchAgent's findings=None branch, which is a
        # generator because the other branch yields).
        if False:  # pragma: no cover — exists only to make this function a generator
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={}),
            )


# ── Shared draft/critique builders ───────────────────────────────────────────

def _draft(subject: str = "A question about your intake admin") -> dict:
    # A valid EmailDraft serialized to a dict — the exact shape the real
    # writer's output_key stores (model_dump, exclude_none=True).
    return EmailDraft(
        subject=subject,
        body=(
            "Hello, I help mental health practices cut intake admin time in "
            "half, and I noticed your team manages a high volume of booking "
            "coordination. Would a short conversation about automating the "
            "repetitive parts be useful this month?"
        ),
        rationale=(
            "The practice matches the offer's ICP geography and size, and "
            "the intake bottleneck is the operational pain the pitch speaks "
            "to, so the angle leads with that pain rather than the product."
        ),
        confidence=0.8,
    ).model_dump()


def _pass_critique() -> dict:
    # The clean-pass shape — must satisfy DraftCritique's
    # passed-couples-to-evidence validator.
    return DraftCritique(
        passed=True, issues=[], required_changes="", severity="none",
    ).model_dump()


def _fail_critique() -> dict:
    # The fail shape — a concrete required_changes (>= 30 chars) so the
    # validator accepts it and the next writer iteration has feedback.
    return DraftCritique(
        passed=False,
        issues=["The body asserts an unverified hiring signal as established fact."],
        required_changes=(
            "Hedge the hiring claim as a possibility or remove it entirely, "
            "and tighten the body to one clear ask."
        ),
        severity="major",
    ).model_dump()


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conn(scratch_db_target):
    """Fresh SQLite DB with schema, the five seeded principals (including
    draft_writer/draft_critic — the write gate refuses unregistered agents),
    one offer/account/target at "scored" with a policy "allow" decision, and
    a second target at "scored" with NO policy row (the fail-closed case)."""
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
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               industry, estimated_size, geo, company_summary, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_1", "Acme", "acme.test", "acme.test", "Logistics", "11-50", "HK",
                "Acme coordinates logistics bookings."),
    )
    for target_id, state in (("tgt_1", "scored"), ("tgt_2", "scored")):
        commit(
            c, action="insert_target", table_name="targets", record_id=target_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO targets (target_id, account_id, offer_id, source, state, created_at, updated_at)
                   VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(target_id, "acc_1", "off_1", "csv", state),
        )
    # tgt_1 gets a policy "allow" decision (the Phase 1 gate's row — the
    # realistic precondition); tgt_2 deliberately gets NONE so the
    # fail-closed case can run against it.
    _insert_policy_decision(c, "tgt_1", "allow")
    yield c
    c.close()


def _insert_policy_decision(c, target_id: str, decision: str) -> None:
    """Insert one policy_decisions row through the write gate — the same
    path policy_check_phase1 uses (core table, gated write)."""
    commit(
        c, action="insert_policy_decision", table_name="policy_decisions",
        record_id=new_id("pol"), payload={"decision": decision},
        run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO policy_decisions
               (policy_decision_id, run_id, step_id, target_id, action, decision,
                risk_level, reasons_json, matched_rules_json, missing_fields_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=(new_id("pol"), "r0", "s0", target_id, "policy_check_phase1", decision,
                "low", "[]", "[]", "[]"),
    )


@pytest.fixture
def offers_dir(tmp_path):
    """A tmp offers directory with one offer yaml carrying pitch,
    persona_hint, from_address, and an icp block — the draft brief and the
    deterministic footer read their offer context from here."""
    d = tmp_path / "offers"
    d.mkdir()
    (d / "acme.yaml").write_text(
        "pitch: We cut intake admin time in half.\n"
        "persona_hint: Operations lead at a mid-size practice.\n"
        "from_address: outreach@acme.test\n"
        "icp:\n  geography: HK\n  disqualifiers:\n    - outside HK\n"
    )
    return d


def _run_draft(conn, offers_dir, writer, critic, target_id="tgt_1"):
    """Build the draft agent with the two stand-ins patched in, run the
    target, and return the outcome string."""
    with patch("app.agents.draft._build_writer_agent", return_value=writer), \
         patch("app.agents.draft._build_critic_agent", return_value=critic):
        agent = build_draft_agent(conn)
        return run_target_through_draft(
            agent, conn=conn, target_id=target_id, run_id="r1",
            offers_dir=str(offers_dir),
        )


def _draft_rows(conn, target_id="tgt_1"):
    return conn.execute(
        "SELECT * FROM message_draft_versions WHERE target_id=? "
        "ORDER BY revision_number;",
        (target_id,),
    ).fetchall()


# ── 1. Happy path: critic passes on iteration 1 ──────────────────────────────

def test_critic_passes_on_iteration_one_stops_loop_early(conn, offers_dir):
    """A passing critique on the first iteration must persist exactly ONE
    revision, land the target in awaiting_review, and stop the loop early —
    the writer runs once, not three times (the escalate exit, fact §2.1)."""
    writer = _StubWriterAgent([_draft()])
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([True]))

    assert outcome == "awaiting_review"
    rows = _draft_rows(conn)
    assert len(rows) == 1, "a passing first critique must stop the loop after one revision"
    assert rows[0]["revision_number"] == 1
    assert rows[0]["critique_passed"] == 1
    # The early exit is proven by the writer's invocation count — the
    # max-iteration exit would have run it three times.
    assert writer._calls == 1
    # The DATABASE agrees: the target's state is what downstream phases read.
    target = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert target["state"] == "awaiting_review"


# ── 2. Critic fails twice, then passes ───────────────────────────────────────

def test_critic_fails_twice_then_passes_persists_three_revisions(conn, offers_dir):
    """Two failing critiques then a pass must persist three revisions
    (1, 2, 3) with the third carrying critique_passed=1 — the console's
    "watch the agent improve" evidence — and land in awaiting_review."""
    writer = _StubWriterAgent([_draft("First angle"), _draft("Second angle"), _draft("Third angle")])
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([False, False, True]))

    assert outcome == "awaiting_review"
    rows = _draft_rows(conn)
    assert [r["revision_number"] for r in rows] == [1, 2, 3]
    assert [r["critique_passed"] for r in rows] == [0, 0, 1]
    assert writer._calls == 3, "the third iteration's pass must be the one that exits"
    # Each revision carries the critique that produced the NEXT rewrite —
    # the first two rows store failing verdicts as JSON.
    assert json.loads(rows[0]["critique_json"])["passed"] is False
    assert json.loads(rows[2]["critique_json"])["passed"] is True


# ── 3. The critic never passes — B3-Z2 ───────────────────────────────────────

def test_critic_never_passes_still_reaches_awaiting_review(conn, offers_dir):
    """B3-Z2: a never-passing critique is a bounded retry, NOT a failure
    and NOT a block.  Exactly DRAFT_MAX_ITERATIONS rows are persisted, the
    target STILL reaches awaiting_review (the human decides), and it is
    NEVER approved or sent — the critic's passed flag cannot approve
    anything and its absence cannot withhold the human review gate."""
    writer = _StubWriterAgent([_draft()])
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([False, False, False]))

    assert outcome == "awaiting_review"
    rows = _draft_rows(conn)
    assert len(rows) == 3, "the loop is bounded by DRAFT_MAX_ITERATIONS, exactly"
    assert all(r["critique_passed"] == 0 for r in rows)
    assert writer._calls == 3, "the max-iteration exit ran the full bounded budget"
    # The target's state — and nothing beyond: NO transition to approved,
    # sent, or dry_run_sent may exist for this target.
    target = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert target["state"] == "awaiting_review"
    forward_hops = conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1' "
        "AND new_state IN ('approved', 'sent', 'dry_run_sent');"
    ).fetchone()
    assert forward_hops["n"] == 0


# ── 4. B3-Z3: the gate columns stay NULL ─────────────────────────────────────

def test_gate_columns_are_always_null(conn, offers_dir):
    """B3-Z3, plus the G2 runner's boundary: the draft AGENT cannot set its
    own gates, so every version it persists carries NULL — the SEPARATE
    deterministic runner then evaluates only the LATEST revision and writes its
    two columns, while send_gate_passed stays NULL everywhere (the send gate's
    own column)."""
    _run_draft(conn, offers_dir, _StubWriterAgent([_draft()]),
               _StubCriticAgent([False, False, False]))
    rows = _draft_rows(conn)
    assert len(rows) == 3
    # B3 persisted the first two revisions untouched by the runner: their gate
    # columns are still exactly what the drafting agent wrote — NULL.
    for row in rows[:2]:
        assert row["policy_check_passed"] is None, "B3 must never write policy_check_passed"
        assert row["injection_scan_passed"] is None, "B3 must never write injection_scan_passed"
        assert row["send_gate_passed"] is None, "B3 must never write send_gate_passed"
    # The runner fired on the LATEST revision (a clean draft -> both pass), and
    # still never touched send_gate_passed.
    latest = rows[2]
    assert latest["policy_check_passed"] == 1, "the G2 runner must evaluate the latest fresh revision"
    assert latest["injection_scan_passed"] == 1
    assert latest["send_gate_passed"] is None, "send_gate_passed stays the send gate's own"


# ── 5. B3-Z1: no footer field on the model; deterministic footer on rows ─────

def test_draft_output_model_has_no_footer_field():
    """B3-Z1 at the schema level: EmailDraft has NO footer field — a model
    cannot author the compliance footer even if its prompt is subverted,
    because there is no field to emit it into (mirror of
    test_judge_output_model_has_no_score_field)."""
    assert "footer" not in EmailDraft.model_fields
    # Even a smuggled extra key is refused by extra='forbid' — a writer
    # emitting a footer JSON key fails validation instead of slipping
    # through.
    with pytest.raises(ValidationError, match="footer"):
        EmailDraft.model_validate({**_draft(), "footer": "malicious unsubscribe text"})


def test_every_persisted_version_carries_the_deterministic_footer(conn, offers_dir):
    """B3-Z1 at the database level: every persisted version's footer column
    is non-empty and contains the deterministic unsubscribe token — the
    footer is composed by code (from the offer config), never by the model."""
    _run_draft(conn, offers_dir, _StubWriterAgent([_draft()]),
               _StubCriticAgent([False, False, False]))
    for row in _draft_rows(conn):
        assert row["footer"], "footer is NOT NULL — a non-empty deterministic footer is mandatory"
        assert "[unsubscribe:" in row["footer"], "the footer must carry the unsubscribe token"
        assert "{UNSUBSCRIBE_URL}" in row["footer"], (
            "the footer must carry the placeholder token B5 substitutes at "
            "send time — B3 must not invent a URL scheme"
        )


def test_first_touch_footer_never_schedules_even_with_scheduling_enabled(conn, tmp_path):
    """Demo, 2026-08-30 — the follow_up=False guard: scheduling (and the
    earlier booking_url link it replaced) may ONLY ever appear on a
    follow-up draft. This offer has scheduling_enabled: true, but this
    fixture's target is on the FIRST-TOUCH path ("scored", via
    conftest's offers_dir/_run_draft setup) — its footer must carry no
    scheduling line and no meetings row, proving the gate is on
    follow_up, not merely on the config flag. The follow-up-enabled case
    (a real reservation appearing in the footer) is tested end to end in
    tests/test_follow_up_draft.py, which is where a "routed" target
    actually exists to test it against."""
    d = tmp_path / "offers_with_scheduling"
    d.mkdir()
    (d / "acme.yaml").write_text(
        "pitch: We cut intake admin time in half.\n"
        "persona_hint: Operations lead at a mid-size practice.\n"
        "from_address: outreach@acme.test\n"
        "scheduling_enabled: true\n"
        "icp:\n  geography: HK\n  disqualifiers:\n    - outside HK\n"
    )
    _run_draft(conn, d, _StubWriterAgent([_draft()]), _StubCriticAgent([False, False, False]))
    for row in _draft_rows(conn):
        # The unsubscribe token is still there — additive, never load-bearing.
        assert "[unsubscribe: {UNSUBSCRIBE_URL}]" in row["footer"]
        assert "We've held" not in row["footer"], (
            f"a first-touch draft must never carry a scheduled-meeting line; got: {row['footer']!r}"
        )
    assert conn.execute("SELECT COUNT(*) AS n FROM meetings;").fetchone()["n"] == 0


def test_footer_has_no_scheduling_line_when_offer_lacks_it(conn, offers_dir):
    """The absent case, explicit: the SAME offers_dir fixture every other
    footer test in this file uses has no scheduling_enabled key — its
    footer must contain no scheduling line at all, proving the addition
    is truly optional and does not leak a stray phrase when the key is
    missing."""
    _run_draft(conn, offers_dir, _StubWriterAgent([_draft()]), _StubCriticAgent([False, False, False]))
    for row in _draft_rows(conn):
        assert "We've held" not in row["footer"], (
            f"no scheduling line should appear when the offer lacks scheduling_enabled; got: {row['footer']!r}"
        )


# ── 6. Policy refusal (deny, and fail-closed on no row) ──────────────────────

def test_policy_deny_refuses_to_draft(conn, offers_dir):
    """A scored target whose LATEST policy decision is deny must not be
    drafted: zero rows, state unchanged, and a step row records the
    refusal — the §3 trigger for scored→drafted is "policy allows draft"."""
    # tgt_2 is used (not tgt_1) so the deny row is the target's ONLY policy
    # row.  (Ticket B5's insert_seq column makes "latest" deterministic for
    # production rows — the read orders by insert_seq DESC, created_at DESC —
    # but this fixture's raw insert leaves insert_seq NULL, so a same-second
    # pair would still fall back to the ambiguous created_at comparison;
    # using the row-less target keeps the test's intent explicit.)
    _insert_policy_decision(conn, "tgt_2", "deny")
    writer = _StubWriterAgent([_draft()])
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([True]),
                         target_id="tgt_2")

    assert outcome == "policy_denied"
    assert _draft_rows(conn, target_id="tgt_2") == [], "no draft version may exist for a denied target"
    assert writer._calls == 0, "the loop must never run for a denied target"
    target = conn.execute("SELECT state FROM targets WHERE target_id='tgt_2';").fetchone()
    assert target["state"] == "scored"
    # The refusal is in the trace, greppable without joining.
    refusal = conn.execute(
        "SELECT output_json FROM steps WHERE target_id='tgt_2' "
        "AND tool_name='draft_target_run';"
    ).fetchone()
    assert refusal is not None, "a refusal must be a logged step, never a silent skip"
    assert json.loads(refusal["output_json"])["outcome"] == "policy_denied"


def test_missing_policy_decision_fails_closed(conn, offers_dir):
    """A scored target with NO policy_decisions row at all must be refused
    (fail closed — an unmapped action always resolves to deny): zero rows,
    state unchanged, logged refusal."""
    writer = _StubWriterAgent([_draft()])
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([True]),
                         target_id="tgt_2")  # tgt_2 deliberately has no policy row

    assert outcome == "policy_denied"
    assert _draft_rows(conn, target_id="tgt_2") == []
    assert writer._calls == 0
    target = conn.execute("SELECT state FROM targets WHERE target_id='tgt_2';").fetchone()
    assert target["state"] == "scored"


# ── 7. Wrong-state refusal ───────────────────────────────────────────────────

@pytest.mark.parametrize("state", ["researched", "awaiting_review", "new", "failed"])
def test_wrong_state_is_refused(conn, offers_dir, state):
    """scored→drafted is the ONLY inbound edge to drafted: a target in any
    other state is refused with not_draftable, zero rows, no state change."""
    # Move tgt_1 into the tested state.  Direct UPDATE here is TEST SETUP,
    # not a pipeline write path (the tested states are unreachable from
    # "scored" through the real state machine — that is the point) — same
    # direct-setup precedent as test_agent_registry.py's kill-switch
    # fixture.
    conn.execute(
        "UPDATE targets SET state=?, updated_at=datetime('now') WHERE target_id=?",
        (state, "tgt_1"),
    )
    conn.commit()
    writer = _StubWriterAgent([_draft()])
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([True]))

    assert outcome == "not_draftable"
    assert _draft_rows(conn) == []
    assert writer._calls == 0, "the loop must never run for a non-scored target"
    target = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert target["state"] == state, "a refusal must not change the target's state"
    # The refusal is in the trace, greppable without joining.
    refusal = conn.execute(
        "SELECT output_json FROM steps WHERE target_id='tgt_1' "
        "AND tool_name='draft_target_run';"
    ).fetchone()
    assert refusal is not None
    assert json.loads(refusal["output_json"])["outcome"] == "not_draftable"


# ── 8. Writer output invalid: leave the target in scored ─────────────────────

def test_writer_producing_no_output_leaves_target_scored(conn, offers_dir):
    """The B3 failure path, deliberately asymmetric: when the writer
    produces nothing, the persist node logs a failed step and escalates,
    the target STAYS in scored (NOT failed — a drafting outage is not a
    research failure; the next run retries it), and zero draft versions are
    persisted."""
    writer = _SilentWriterAgent()  # publishes nothing — the "draft" key stays absent
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([True]))

    assert outcome == "scored", "zero persisted versions => the target's state is still scored"
    assert _draft_rows(conn) == [], "nothing persistable may reach message_draft_versions"
    target = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert target["state"] == "scored", "the target must NOT be failed — research and score are intact"
    failed_hops = conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1' "
        "AND new_state='failed';"
    ).fetchone()
    assert failed_hops["n"] == 0
    # The failure is in the trace: a failed draft_persist step row exists.
    failed_step = conn.execute(
        "SELECT status FROM steps WHERE target_id='tgt_1' "
        "AND tool_name='draft_persist' AND status='failed';"
    ).fetchall()
    assert len(failed_step) == 1, "the unusable output must be logged as a failed step"


def test_writer_producing_schema_invalid_output_leaves_target_scored(conn, offers_dir):
    """The same failure path for a WRONG-SHAPE draft dict (the re-validation
    half of the zero-trust line): a dict that fails EmailDraft validation
    must never reach the database, and the target stays scored."""
    writer = _StubWriterAgent([{"subject": "too short"}])  # body/rationale/confidence missing — fails re-validation
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([True]))

    assert outcome == "scored"
    assert _draft_rows(conn) == []
    target = conn.execute("SELECT state FROM targets WHERE target_id='tgt_1';").fetchone()
    assert target["state"] == "scored"
    failed_step = conn.execute(
        "SELECT status FROM steps WHERE target_id='tgt_1' "
        "AND tool_name='draft_persist' AND status='failed';"
    ).fetchall()
    assert len(failed_step) == 1


# ── 9. DraftCritique validator: passed couples to the evidence ───────────────

def test_critique_passed_true_with_issues_is_rejected():
    """passed=True with a non-empty issues list is self-contradictory — the
    model validator refuses it (same enforcement shape as ICPVerdict's
    divergence contract)."""
    with pytest.raises(ValidationError, match="passed=True"):
        DraftCritique(passed=True, issues=["still too long"], required_changes="", severity="none")


def test_critique_passed_true_with_required_changes_is_rejected():
    """passed=True with non-empty required_changes is refused — a clean pass
    cannot carry revision instructions."""
    with pytest.raises(ValidationError, match="passed=True"):
        DraftCritique(
            passed=True, issues=[], required_changes="tighten the body",
            severity="none",
        )


def test_critique_passed_false_with_empty_required_changes_is_rejected():
    """passed=False with an empty required_changes is refused — a failing
    verdict must hand the next writer iteration concrete instructions."""
    with pytest.raises(ValidationError, match="required_changes"):
        DraftCritique(passed=False, issues=["too long"], required_changes="", severity="minor")


def test_critique_passed_false_with_token_required_changes_is_rejected():
    """A token required_changes ("fix it") is not actionable — the 30-char
    floor refuses it."""
    with pytest.raises(ValidationError, match="at least 30"):
        DraftCritique(passed=False, issues=["too long"], required_changes="fix it", severity="minor")


def test_critique_passed_false_with_no_issues_is_rejected():
    """passed=False with an empty issues list is refused — a failing verdict
    must name what is wrong."""
    with pytest.raises(ValidationError, match="at least one issue"):
        DraftCritique(passed=False, issues=[], required_changes="shorten the body to one ask", severity="minor")


def test_critique_valid_shapes_are_accepted():
    """Both valid shapes construct: the clean pass and the actionable fail."""
    assert _pass_critique()["passed"] is True
    assert _fail_critique()["passed"] is False


# ── 10. Audit trail: every revision row has a gated, attributed write ────────

def test_every_revision_has_a_write_log_row_attributed_to_draft_writer(conn, offers_dir):
    """The gated-write proof (catches someone replacing the gated write
    with a raw conn.execute): every message_draft_versions row has a
    corresponding write_log row with action=insert_message_draft_version
    and agent_id=draft_writer — the draft agent's writes are attributable
    to the writer principal, never to system and never ungated."""
    _run_draft(conn, offers_dir, _StubWriterAgent([_draft()]),
               _StubCriticAgent([False, False, False]))
    rows = _draft_rows(conn)
    assert len(rows) == 3
    for row in rows:
        log_row = conn.execute(
            "SELECT agent_id, action FROM write_log WHERE record_id=? "
            "AND action='insert_message_draft_version';",
            (row["draft_version_id"],),
        ).fetchone()
        assert log_row is not None, (
            f"draft version {row['draft_version_id']} has no gated write_log row"
        )
        assert log_row["action"] == "insert_message_draft_version"
        assert log_row["agent_id"] == DRAFT_WRITER_AGENT_ID


# ── 11. Registry: the two draft principals are seeded ────────────────────────

def test_draft_principals_are_registered_and_enabled(conn):
    """B3's two new principals must be registered and enabled after
    seed_agent_registry, each carrying the draft_model role alias — an LLM
    principal's registry row names its model role (NULL is the marker for
    deterministic principals only)."""
    rows = conn.execute(
        "SELECT agent_id, model_alias, enabled FROM agent_registry "
        "WHERE agent_id IN (?, ?);",
        (DRAFT_WRITER_AGENT_ID, DRAFT_CRITIC_AGENT_ID),
    ).fetchall()
    by_id = {r["agent_id"]: r for r in rows}
    assert set(by_id) == {DRAFT_WRITER_AGENT_ID, DRAFT_CRITIC_AGENT_ID}, (
        "both draft principals must be seeded"
    )
    for agent_id in (DRAFT_WRITER_AGENT_ID, DRAFT_CRITIC_AGENT_ID):
        assert by_id[agent_id]["enabled"] == 1, "the draft agents start enabled; the kill switch is opt-out"
        assert by_id[agent_id]["model_alias"] == "draft_model", (
            "the draft agents are LLM principals — their model_alias must "
            "name the config/models.yaml draft_model role"
        )


# ── 12. H9: the brief carries the contact's name / salutation rule ───────────

def _link_contact(conn, *, contact_id: str, full_name, title, target_id: str = "tgt_1"):
    """Insert a contacts row and point the target at it — TEST SETUP, not a
    pipeline write path: the direct-UPDATE precedent of
    test_wrong_state_is_refused (the shared conn fixture creates the targets
    without a contact_id, and this seeding only makes the H9 brief reads
    meaningful)."""
    commit(
        conn, action="insert_contact", table_name="contacts", record_id=contact_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts (contact_id, account_id, full_name, title, created_at, updated_at)
               VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=(contact_id, "acc_1", full_name, title),
    )
    conn.execute(
        "UPDATE targets SET contact_id=? WHERE target_id=?",
        (contact_id, target_id),
    )
    conn.commit()


def test_draft_context_includes_contact_name_when_present(conn, offers_dir):
    """H9: when the target has a contact with a name, the brief carries the
    name and title, and the salutation rule tells the writer to use it
    verbatim — the writer is no longer left to reach for a placeholder."""
    _link_contact(conn, contact_id="con_1", full_name="Dr Quraulain Zaidi", title="Clinical Director")
    brief = _build_draft_context(conn, "tgt_1", str(offers_dir))
    assert "Dr Quraulain Zaidi" in brief, "the brief must name the recipient"
    assert "Clinical Director" in brief, "the brief must carry the recipient's title"
    assert "EXACTLY as written" in brief, "the named-case rule must demand verbatim use"
    assert "no name" not in brief.lower(), "a named contact must not read as name-free"


def test_draft_context_handles_null_contact_name(conn, offers_dir):
    """H9: a contact whose name is NULL (Central Minds in the real run) must
    not raise; the brief says no name is recorded and instructs a name-free
    greeting — and explicitly forbids inventing a name or emitting a
    placeholder."""
    _link_contact(conn, contact_id="con_1", full_name=None, title=None)
    brief = _build_draft_context(conn, "tgt_1", str(offers_dir))
    assert "(no name recorded)" in brief
    assert "name-free greeting" in brief
    assert "Do NOT invent or guess a name" in brief
    assert "placeholder" in brief


def test_draft_context_handles_null_contact_id(conn, offers_dir):
    """H9: a target with no contact at all (contact_id NULL — the shared
    fixture's targets have none) must not raise; the brief degrades to the
    same name-free case instead of crashing the draft stage."""
    brief = _build_draft_context(conn, "tgt_1", str(offers_dir))
    assert "(no name recorded)" in brief
    assert "name-free greeting" in brief


def test_writer_instruction_carries_the_salutation_rule():
    """H9 prompt: the writer's static WRITING RULES tell it to follow the
    brief's CONTACT section, never invent/guess a name, and never emit a
    placeholder — the rule must live in the instruction too, so it holds even
    if a future brief degrades."""
    assert "7. Salutation (ticket H9):" in _WRITER_INSTRUCTION
    assert "NEVER emit a placeholder token" in _WRITER_INSTRUCTION


def test_critic_instruction_rejects_placeholder_salutations():
    """H9 prompt: the critic's checklist rejects a draft whose greeting
    invents a name or carries any placeholder, so the writer⇄critic loop
    self-corrects before the content gate ever sees the draft."""
    assert "7. Does the draft's greeting" in _CRITIC_INSTRUCTION
    assert "placeholder token" in _CRITIC_INSTRUCTION
    assert "invented or guessed name" in _CRITIC_INSTRUCTION
    assert "placeholder salutations" in _CRITIC_INSTRUCTION, "the severity mapping must name this class as major"


def test_critic_instruction_sees_the_reply_and_checks_it_was_addressed():
    """2026-08-30 fix: the critic's checklist used to have no way to catch a
    follow-up draft that ignores the prospect's reply and re-pitches cold —
    {follow_up_context?} was never templated into _CRITIC_INSTRUCTION, only
    the writer's, so nothing downstream of the writer's own good intentions
    verified it actually wrote a reply. Caught on a real production draft
    (Therapy Partners) that passed 3/3 critic rounds while reading as a
    fresh cold open. This asserts both halves of the fix: the critic's
    instruction template now carries the placeholder ADK will substitute
    the same follow_up_context state key into (the writer already reads
    this key — no new state wiring), and checklist item 8 states the rule
    a reviewer would expect: unaddressed reply == hard failure."""
    assert "{follow_up_context?}" in _CRITIC_INSTRUCTION
    assert "THE PROSPECT'S REPLY" in _CRITIC_INSTRUCTION
    assert "8. ONLY when THE PROSPECT'S REPLY block above is non-empty" in _CRITIC_INSTRUCTION
    assert "hard failure on this item" in _CRITIC_INSTRUCTION


def test_a_real_first_touch_draft_run_produces_a_trace_row_the_scoreboard_can_read(conn, offers_dir):
    """Integration check, added on review (2026-08-31): the two style-
    hypothesis tickets were each tested in isolation — the writer-wiring
    ticket's own tests never inspect the steps table, and the scoreboard
    ticket's own tests hand-construct a steps row rather than ever calling
    run_target_through_draft for real. Nothing had proven the two halves
    actually agree on the row's shape. This runs the REAL drafting path
    (mocked LLM agents only, same as every other test in this file) against
    tgt_1 — a genuine first-touch target — and feeds the SAME connection
    straight into scripts.hypothesis_scoreboard.compute_scoreboard, the
    exact function the CLI report calls. If the two tickets had disagreed on
    a key name or a tool_name string, this is the test that would catch it;
    the two tickets' own unit tests could not have."""
    # Import here, not at module scope: this is the one test in the file
    # that depends on scripts/, which is otherwise unrelated to draft.py's
    # own test surface.
    from scripts.hypothesis_scoreboard import compute_scoreboard

    # tgt_1 is the shared fixture target, state "scored" with a policy
    # "allow" row (see the conn fixture above) — the ordinary first-touch
    # precondition, no follow_up path involved, so hypothesis selection
    # fires for real.
    writer = _StubWriterAgent([_draft()])
    outcome = _run_draft(conn, offers_dir, writer, _StubCriticAgent([True]))
    assert outcome == "awaiting_review", "the draft must actually persist for there to be a trace row to check"

    # The REAL trace row the REAL persist node wrote — read it exactly as
    # compute_scoreboard does (SELECT + json.loads), not through any test
    # helper, so this assertion is checking the real database state.
    step_row = conn.execute(
        "SELECT input_json FROM steps WHERE tool_name='draft_persist' AND target_id='tgt_1';"
    ).fetchone()
    assert step_row is not None, "the persist node must log a draft_persist step for every persisted revision"
    input_data = json.loads(step_row["input_json"])
    hypothesis_id = input_data.get("hypothesis_id", "")
    # tgt_1 is a first-touch draft, so a real (non-empty) hypothesis tag
    # must have been selected and recorded — the same tag
    # _select_style_hypothesis would deterministically pick for "tgt_1".
    expected_id, _ = _select_style_hypothesis("tgt_1")
    assert hypothesis_id == expected_id, (
        "the real persist node's logged hypothesis_id must match what "
        "_select_style_hypothesis deterministically picks for this target_id"
    )

    # Now hand the SAME connection to the scoreboard's real aggregation
    # function — no reply was seeded, so this target is "tested" but has no
    # trustworthy verdict yet (neither a win nor a loss), which is the
    # correct, honest state for a draft nobody has replied to.
    board = compute_scoreboard(conn)
    assert board[hypothesis_id]["tested"] == 1, (
        "the scoreboard must count this real drafted target toward its "
        "hypothesis's tested total — a mismatch here means the two tickets "
        "disagree on the tool_name or the input_json key name"
    )
    assert board[hypothesis_id]["wins"] == 0 and board[hypothesis_id]["losses"] == 0


# ── 13. Style hypotheses (2026-08-31, demo feature) ──────────────────────────

def test_style_hypothesis_selection_is_deterministic_and_covers_the_range():
    """The style-hypothesis selector must be deterministic (the same
    target_id picks the same hypothesis every time — a re-run or replay of
    a target reproduces the identical selection) and must spread across the
    ten hand-written claims rather than collapsing onto one.  Uses
    new_id()-shaped ids (12 lowercase hex chars, the shape app/ids.py's
    new_id() produces) — the only shape the hex path is documented for."""
    # Same id twice -> the identical (hypothesis_id, hypothesis_text) tuple.
    fixed = "tgt_000000000000"
    assert _select_style_hypothesis(fixed) == _select_style_hypothesis(fixed), (
        "the selector must be deterministic — the same target_id, the same hypothesis"
    )
    # Every selection's text must be one of the ten hand-written claims —
    # never a synthesized or out-of-set string.
    seen_ids = set()
    for i in range(20):  # last 4 hex chars run 0000..0013 -> indices 0..19 mod 10
        hypothesis_id, hypothesis_text = _select_style_hypothesis(f"tgt_{i:012x}")
        assert hypothesis_text in _STYLE_HYPOTHESES, (
            f"selected text must be a member of _STYLE_HYPOTHESES; got {hypothesis_text!r}"
        )
        seen_ids.add(hypothesis_id)
    # range(20) covers every index 0..19 exactly once, so modulo 10 yields
    # all ten ids — assert only the lower bound so a future re-shuffle of
    # the tuple (or a different spread) cannot break the weak claim.
    assert len(seen_ids) >= 5, (
        f"20 new_id-shaped ids must spread across at least 5 distinct hypotheses; got {sorted(seen_ids)}"
    )


def test_writer_instruction_carries_the_hypothesis_placeholder():
    """The writer's instruction templates the style hypothesis as an
    OPTIONAL block: {hypothesis_directive?} is substituted from session
    state ("" on a follow-up or a pre-feature run — seeded by
    run_target_through_draft), and the heading states the block only
    appears on a first-touch draft, so the prompt neither renders an empty
    heading nor leaves the writer guessing when a claim applies."""
    assert "{hypothesis_directive?}" in _WRITER_INSTRUCTION
    assert "STYLE HYPOTHESIS TO APPLY" in _WRITER_INSTRUCTION
