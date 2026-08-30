"""Tests for the reply router (ticket C1): app/agents/reply.py — the
deterministic half of the reply stage, where the classifier's verdict
becomes the right state.

WHAT IS PROVEN HERE (ticket §6, docs/reply-routing.md §2/§5,
docs/policy-matrix.md P4/P5):
1.  Each of the nine classes routes to its specified action and state.
2.  P4 — an unsubscribe at confidence 0.5 does not suppress and does not
    auto-act; it goes to review (the CLAUDE.md §9 test).
3.  P5 — risky never auto-acts regardless of confidence.
4.  A high-confidence unsubscribe writes a suppressions row
    (reason=unsubscribe, added_by=system) and reaches suppressed — from
    ANY non-terminal state (ticket E3), recording the true
    previous_state.
5.  dry_run_sent → replied → routed runs end to end on a seeded target.
6.  A second reply on the same thread gets its own row and a terminal
    state is never overridden (§5).
7.  No reply, of any class, triggers a send — no outbound messages row is
    created.
8.  Classifier failure leaves the reply row unclassified and the target's
    state unchanged.
9.  The audit trail — every replies and suppressions row has a write_log
    row.

Every test keeps the suite offline by patching ONLY the classifier
factory (app.agents.reply._build_classifier_agent) with an offline
stand-in that publishes a fixed dict under the "reply_classification"
state key — the same pattern tests/test_draft_agent.py applies to the
writer/critic factories, so tests/conftest.py's autouse live-client guard
is never tripped and the REAL ReplyRouterNode (validation, P4/P5,
transitions, gated writes, suppression) runs for real against the
database.  A kill-switch fixture keeps the B4a guardrail disengaged (it
is fail-closed: without the fixture every invocation would halt).
"""

import json  # parsing write_log payloads in the audit-trail assertions
from pathlib import Path  # tmp inbox for the end-to-end test
from unittest.mock import patch  # the classifier factory seam — the only model boundary

import pytest  # fixtures, tmp_path, parametrize

from app.agents.reply import (  # the module under test
    P4_CONFIDENCE_FLOOR,
    REPLY_CLASSIFIER_AGENT_ID,
    build_reply_agent,
    classify_and_route_reply,
    decide_route,
)
from app.agents_registry import seed_agent_registry  # the principals — the write gate refuses unregistered writers
from app.db import apply_schema, connect  # fresh per-test SQLite database
from app.kill_switch import write_kill_switch  # the switch writer — tests flip the tmp switch file the env var points at
from app.schemas import ReplyClassification  # the verdict shape the stub publishes and the router re-validates
from app.tools.fetch_inbox import fetch_inbox, redact_text  # the end-to-end test's fetch half; redact_text for safe fixture text
from app.write_gate import commit  # every seeded core-table row goes through the gate, never a raw INSERT
from google.adk.agents import BaseAgent  # base class of the offline classifier stand-in (B1b pattern)
from google.adk.events import Event, EventActions  # how the stand-in publishes its verdict dict


# ── Offline stand-ins for the classifier LlmAgent ────────────────────────────
# The real classifier is an ADK LlmAgent that would make live billable
# calls; these stubs publish predetermined dicts under the state key the
# real agent's output_schema + output_key writes ("reply_classification").
# The REAL router node consumes them — the trust boundary under test is
# the router, so the classifier's internals are the one thing stubbed.


class _StubClassifierAgent(BaseAgent):
    """Offline stand-in for the classifier: publishes the given verdict
    dict under state key "reply_classification" (mimicking output_key)."""

    def __init__(self, verdict: dict):
        super().__init__(name=REPLY_CLASSIFIER_AGENT_ID)  # same stable name as the real agent
        self._verdict = verdict  # private attr — pydantic forbids public assignment

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"reply_classification": self._verdict}),
        )


class _SilentClassifierAgent(BaseAgent):
    """Offline stand-in that publishes NOTHING — reproducing the real
    classifier's failure shape ("the agent produced no output", so the
    state key is absent, exactly like draft.py's _SilentWriterAgent)."""

    def __init__(self):
        super().__init__(name=REPLY_CLASSIFIER_AGENT_ID)

    async def _run_async_impl(self, ctx):
        # Yield nothing: the "reply_classification" key stays absent from
        # session state, which is the real LlmAgent's behaviour on an
        # empty final turn.  The unreachable yield keeps this an ASYNC
        # GENERATOR (ADK iterates _run_async_impl).
        if False:  # pragma: no cover — exists only to make this function a generator
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(state_delta={}),
            )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def switch_path(tmp_path, monkeypatch):
    """A tmp kill-switch file, written DISENGAGED, and the env var
    pointing the guardrail's reader at it — the B4a convention (the
    reader is fail-closed, so without this every reply invocation would
    halt at agent entry)."""
    path = tmp_path / "kill_switch.json"
    write_kill_switch(engaged=False, updated_by="fixture", path=str(path))
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(path))
    return path


@pytest.fixture
def conn(scratch_db_target, switch_path):
    """Fresh SQLite DB with schema + the seeded principals + one shared
    offer.  Targets/replies are added per test via _seed_reply_chain."""
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
    yield c
    c.close()


def _seed_reply_chain(c, *, target_id: str, email: str, message_id: str,
                      reply_id: str | None, state: str = "replied",
                      reply_body: str = "Please send more details.") -> None:
    """Seed the full chain the router needs: account + contact + target
    (in ``state``) + an outbound messages row + one replies row (with the
    redacted copy computed the way the real fetch computes it).

    Idempotent for the chain rows (existence-checked before insert), so
    a second reply on the same thread can be seeded without duplicating
    the target's FK chain.  ``reply_id=None`` skips the replies insert
    entirely — the end-to-end test lets the real fetch create the row.
    All writes go through the write gate — fixtures are normal pipeline
    writes, so the audit-trail test sees them too."""
    account_id = f"acc_{target_id}"
    contact_id = f"con_{target_id}"
    domain = email.split("@", 1)[-1]
    # The chain rows exist once per target; a second _seed_reply_chain
    # call (second reply on the thread) must not re-insert them.  The
    # normalized_domain carries the target_id suffix so two targets that
    # share an email domain cannot trip the UNIQUE constraint.
    if c.execute("SELECT 1 FROM accounts WHERE account_id=?;", (account_id,)).fetchone() is None:
        commit(
            c, action="insert_account", table_name="accounts", record_id=account_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
                   created_at, updated_at) VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
            params=(account_id, "Seed Clinic", domain, f"{target_id}.{domain}"),
        )
    if c.execute("SELECT 1 FROM contacts WHERE contact_id=?;", (contact_id,)).fetchone() is None:
        commit(
            c, action="insert_contact", table_name="contacts", record_id=contact_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
                   email_verified, created_at, updated_at)
                   VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(contact_id, account_id, "Seed Person", email, 1),
        )
    if c.execute("SELECT 1 FROM targets WHERE target_id=?;", (target_id,)).fetchone() is None:
        commit(
            c, action="insert_target", table_name="targets", record_id=target_id,
            payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
            sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id,
                   source, state, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
            params=(target_id, account_id, contact_id, "off_1", "csv", state),
        )
    if c.execute("SELECT 1 FROM messages WHERE message_id=?;", (message_id,)).fetchone() is None:
        commit(
            c, action="insert_message", table_name="messages", record_id=message_id,
            payload={"status": "dry_run_sent"}, run_id="r0", step_id="s0",
            actor="system", agent_id="system",
            sql="""INSERT INTO messages (message_id, target_id, contact_id, direction,
                   provider_message_id, thread_id, subject, body, body_redacted,
                   status, sent_at, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
            params=(message_id, target_id, contact_id, "outbound", None, None,
                    "Cold subject", "Cold body text.", None, "dry_run_sent", None),
        )
    if reply_id is not None:
        commit(
            c, action="insert_reply", table_name="replies", record_id=reply_id,
            payload={"match_method": "test_seed"}, run_id="r0", step_id="s0",
            actor="system", agent_id="system",
            sql="""INSERT INTO replies (reply_id, message_id, thread_id, from_email,
                   raw_text, redacted_text, classification, confidence,
                   routed_action, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))""",
            params=(reply_id, message_id, message_id, email,
                    reply_body, redact_text(reply_body), None, None, None),
        )


def _verdict(reply_class: str, confidence: float) -> dict:
    """A valid ReplyClassification serialized to a dict — the exact shape
    the real classifier's output_key stores (model_dump), so the stub
    publishes what the real agent would."""
    return ReplyClassification(
        reply_class=reply_class,  # type: ignore[arg-type] — the Literal is checked by the model; tests pass the nine legal strings
        confidence=confidence,
        rationale=(
            f"The reply is class {reply_class} because its wording asks for "
            f"that handling, and no other class matches its content better."
        ),
        evidence_quote="Please send more details about this.",
    ).model_dump()


def _classify(c, *, reply_id: str, verdict: dict, run_id: str = "r1") -> str:
    """Run ONE classification with the stub classifier patched in — the
    single seam every test here uses to stay offline."""
    with patch("app.agents.reply._build_classifier_agent",
               return_value=_StubClassifierAgent(verdict)):
        agent = build_reply_agent(c)
        return classify_and_route_reply(agent, conn=c, reply_id=reply_id, run_id=run_id)


def _state(c, target_id: str) -> str:
    """The target's current state — the one-line read every state
    assertion uses."""
    return c.execute(
        "SELECT state FROM targets WHERE target_id=?;", (target_id,)
    ).fetchone()["state"]


# ── The nine classes route to their specified action and state ──────────────


@pytest.mark.parametrize("reply_class,confidence,expected_action,expected_state", [
    ("positive", 0.9, "queue_follow_up_draft", "routed"),
    ("not_now", 0.9, "schedule_reminder", "routed"),
    ("negative", 0.9, "close_not_target", "routed"),
    ("unsubscribe", 0.9, "auto_suppress", "suppressed"),
    ("wrong_person", 0.9, "re_enrich", "routed"),
    ("objection", 0.9, "draft_hold", "routed"),
    ("meeting_request", 0.9, "notify_operator", "routed"),
    ("risky", 0.9, "review_required", "routed"),  # P5 overrides the §2 freeze action with review_required
    ("unclear", 0.9, "human_review", "routed"),
])
def test_each_class_routes_to_its_action_and_state(
    conn, reply_class, confidence, expected_action, expected_state
):
    """docs/reply-routing.md §2, one row per class: the verdict is
    persisted to the replies row (classification/confidence/routed_action)
    and the target lands in the specified state — routed for every class
    but unsubscribe, which suppresses.  risky records review_required
    (P5's vocabulary) — see the parametrization comment."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000001", reply_id="rpl_000000000001")
    outcome = _classify(conn, reply_id="rpl_000000000001",
                        verdict=_verdict(reply_class, confidence))
    row = conn.execute(
        "SELECT classification, confidence, routed_action FROM replies WHERE reply_id='rpl_000000000001';"
    ).fetchone()
    # The verdict is persisted — all three columns filled.
    assert row["classification"] == reply_class
    assert row["confidence"] == confidence
    assert row["routed_action"] == expected_action
    # The state the routing table specifies.
    assert _state(conn, "tgt_1") == expected_state
    # The outcome string agrees with the state.
    if expected_state == "suppressed":
        assert outcome == "suppressed"
    elif expected_action == "review_required":
        assert outcome == "review_required"
    else:
        assert outcome == "routed"


# ── P4 and P5 — the two policy rules that bind ───────────────────────────────


def test_low_confidence_unsubscribe_does_not_suppress_and_goes_to_review(conn):
    """P4 — the CLAUDE.md §9 test: an unsubscribe at confidence 0.5 must
    NOT silently suppress.  No suppressions row, no transition to
    suppressed, the verdict is recorded as review_required, and the
    target sits in routed (the review-pending state) — never suppressed.
    (The ticket's "does not transition" is enforced as "no auto-act
    transition": replied → routed is the classification hop that even a
    review-bound reply completes, and no suppression hop may follow.)"""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000002", reply_id="rpl_000000000002")
    outcome = _classify(conn, reply_id="rpl_000000000002",
                        verdict=_verdict("unsubscribe", confidence=0.5))
    # The verdict went to review, not to the suppression machinery.
    assert outcome == "review_required"
    row = conn.execute(
        "SELECT routed_action FROM replies WHERE reply_id='rpl_000000000002';"
    ).fetchone()
    assert row["routed_action"] == "review_required"
    # NO suppressions row — the §9 guarantee.
    assert conn.execute("SELECT COUNT(*) AS n FROM suppressions;").fetchone()["n"] == 0
    # NO transition to suppressed — and the target is not suppressed.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1' AND new_state='suppressed';"
    ).fetchone()["n"] == 0
    assert _state(conn, "tgt_1") == "routed"


def test_risky_never_auto_acts_regardless_of_confidence(conn):
    """P5: a risky reply at confidence 0.99 still never auto-acts — no
    suppression, no suppression transition, routed_action
    review_required, and the target sits in routed, not suppressed."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000003", reply_id="rpl_000000000003")
    outcome = _classify(conn, reply_id="rpl_000000000003",
                        verdict=_verdict("risky", confidence=0.99))
    assert outcome == "review_required"
    assert conn.execute("SELECT COUNT(*) AS n FROM suppressions;").fetchone()["n"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1' AND new_state='suppressed';"
    ).fetchone()["n"] == 0
    assert _state(conn, "tgt_1") == "routed"


def test_high_confidence_unsubscribe_writes_suppression_and_reaches_suppressed(conn):
    """The one auto side effect: a HIGH-confidence unsubscribe writes a
    suppressions row (reason=unsubscribe, added_by=system — the CHECK
    vocabulary) and the target reaches suppressed through BOTH hops:
    replied → routed → suppressed."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000004", reply_id="rpl_000000000004")
    outcome = _classify(conn, reply_id="rpl_000000000004",
                        verdict=_verdict("unsubscribe", confidence=0.95))
    assert outcome == "suppressed"
    row = conn.execute(
        "SELECT email, reason, added_by FROM suppressions WHERE email='jane@clinic.test';"
    ).fetchone()
    assert row is not None, "the unsubscribe must append a suppressions row"
    assert row["reason"] == "unsubscribe"
    assert row["added_by"] == "system"
    assert _state(conn, "tgt_1") == "suppressed"
    # Both hops are on the record, in order.  The order key is insert_seq
    # (ticket C1's extension of B5's fix to this table): created_at is
    # second-precision TEXT and both hops land in the same second, so
    # ordering by it asserted whichever row SQLite happened to return —
    # passing alone, failing in the full suite.  Every row here is written
    # by transition(), which populates insert_seq via the MAX+1 subquery,
    # so no NULLs arise and plain ASC is deterministic (production history
    # reads use the null-safe prefix — see app/console/app.py).
    hops = conn.execute(
        "SELECT previous_state, new_state FROM state_transitions WHERE target_id='tgt_1' "
        "ORDER BY insert_seq, created_at;"
    ).fetchall()
    assert [(h["previous_state"], h["new_state"]) for h in hops] == [
        ("replied", "routed"), ("routed", "suppressed"),
    ]


# ── Ticket E3: the suppression hop fires from ANY non-terminal state ──────────


@pytest.mark.parametrize("state", ["routed", "awaiting_review", "approved"])
def test_high_confidence_unsubscribe_suppresses_from_any_non_terminal_state(conn, state):
    """E3's fix, per state: a high-confidence unsubscribe must suppress
    from WHATEVER live state the target is actually in — E1 made the
    pipeline cyclical, so a second-reply unsubscribe can land on a target
    that is already routed (an objection came first), awaiting_review, or
    approved (mid follow-up cycle).  Each case reaches suppressed, the
    outcome names it, and the state_transitions row records the TRUE
    previous_state (read from the DB, never hardcoded "routed") — the
    draft_cli B1f rule applied to the reply stage."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000010", reply_id="rpl_000000000010",
                      state=state)
    outcome = _classify(conn, reply_id="rpl_000000000010",
                        verdict=_verdict("unsubscribe", confidence=0.95))
    # The suppression fired — the router reports it and the target closed.
    assert outcome == "suppressed"
    assert _state(conn, "tgt_1") == "suppressed"
    # ONE hop, and it records where the target actually was.
    hops = conn.execute(
        "SELECT previous_state, new_state, reason FROM state_transitions "
        "WHERE target_id='tgt_1' ORDER BY insert_seq, created_at;"
    ).fetchall()
    assert [(h["previous_state"], h["new_state"], h["reason"]) for h in hops] == [
        (state, "suppressed", "unsubscribe_reply"),
    ]


@pytest.mark.parametrize("state", ["suppressed", "not_target", "failed"])
def test_unsubscribe_on_terminal_target_records_verdict_without_transition(conn, state):
    """E3's terminal half, unchanged by construction: a high-confidence
    unsubscribe on a terminal target still records the verdict (the row
    and the step) but fires NO transition and writes NO suppression row —
    step 5's terminal guard returns before the auto side effect, so the
    widened hop can never fire suppressed -> suppressed."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000011", reply_id="rpl_000000000011",
                      state=state)
    outcome = _classify(conn, reply_id="rpl_000000000011",
                        verdict=_verdict("unsubscribe", confidence=0.9))
    # The guard named itself; the verdict IS recorded either way.
    assert outcome == "terminal_no_transition"
    row = conn.execute(
        "SELECT classification, routed_action FROM replies WHERE reply_id='rpl_000000000011';"
    ).fetchone()
    assert row["classification"] == "unsubscribe"
    assert row["routed_action"] == "auto_suppress"
    # Nothing moved, and no suppression machinery ran (the guard returns
    # before the auto side effect).
    assert _state(conn, "tgt_1") == state
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM suppressions;").fetchone()["n"] == 0


@pytest.mark.parametrize("state", ["routed", "awaiting_review"])
def test_low_confidence_unsubscribe_never_suppresses_at_routed_or_awaiting_review(conn, state):
    """E3's P4 regression at more than one state: a LOW-confidence
    unsubscribe must never suppress, whether the target is in routed (a
    second reply) or awaiting_review (mid follow-up cycle) — CLAUDE.md §9
    holds at every turn of the conversation, not only on the first
    reply.  The widened hop is gated on decision.auto_suppress, which P4
    already set to False, so it is never reached."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000012", reply_id="rpl_000000000012",
                      state=state)
    outcome = _classify(conn, reply_id="rpl_000000000012",
                        verdict=_verdict("unsubscribe", confidence=0.5))
    # P4 overrides the class action: review, never suppress.
    assert outcome == "review_required"
    assert _state(conn, "tgt_1") == state  # no hop of any kind fired
    assert conn.execute("SELECT COUNT(*) AS n FROM suppressions;").fetchone()["n"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1' AND new_state='suppressed';"
    ).fetchone()["n"] == 0


# ── End to end: dry_run_sent → replied → routed ──────────────────────────────


def test_dry_run_sent_to_replied_to_routed_end_to_end(conn, tmp_path):
    """The full C1 loop on a seeded target: an .eml arrives in the
    simulated inbox, the fetch links it (dry_run_sent → replied) and
    writes the row, the classifier (positive) judges it, and the router
    moves it replied → routed with the follow-up action recorded."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000005", reply_id=None,  # no pre-seeded reply — the fetch must create it
                      state="dry_run_sent")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "reply.eml").write_text(
        "From: Test Sender <jane@clinic.test>\n"
        "To: outreach@outbound-agency.invalid\n"
        "Subject: Re: Cold subject\n"
        "Date: Fri, 22 Aug 2026 09:14:00 +0800\n"
        "Message-ID: <demo@example.test>\n"
        "In-Reply-To: <1.2.3.msg_000000000005@outbound-agency.invalid>\n"
        "\n"
        "This is interesting, please send more details."
    )
    fetched = fetch_inbox(conn, inbox_dir=str(inbox), run_id="r1")
    assert len(fetched.replies_created) == 1
    reply_id = fetched.replies_created[0]
    # The fetch performed the first hop.
    assert _state(conn, "tgt_1") == "replied"
    # The classification performs the second.
    outcome = _classify(conn, reply_id=reply_id, verdict=_verdict("positive", 0.9))
    assert outcome == "routed"
    assert _state(conn, "tgt_1") == "routed"
    row = conn.execute(
        "SELECT classification, routed_action FROM replies WHERE reply_id=?;", (reply_id,)
    ).fetchone()
    assert row["classification"] == "positive"
    assert row["routed_action"] == "queue_follow_up_draft"


# ── §5: terminal states, second replies, and the no-send guarantee ──────────


def test_second_reply_gets_its_own_row_and_cannot_override_a_terminal_state(conn):
    """§5: a second reply on the same thread is classified independently
    (its own row, its own verdict) and a terminal state is never
    overridden — the target stays suppressed even when the second reply
    is a high-confidence unsubscribe (which would suppress a live
    target)."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000006", reply_id="rpl_000000000006",
                      state="suppressed")
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000006", reply_id="rpl_000000000007",
                      state="suppressed")
    outcome = _classify(conn, reply_id="rpl_000000000007",
                        verdict=_verdict("unsubscribe", confidence=0.9))
    # The terminal guard fired — classified and recorded, never transitioned.
    assert outcome == "terminal_no_transition"
    row = conn.execute(
        "SELECT classification, routed_action FROM replies WHERE reply_id='rpl_000000000007';"
    ).fetchone()
    assert row["classification"] == "unsubscribe"  # the verdict IS recorded
    assert row["routed_action"] == "auto_suppress"
    # The terminal state held — and no suppression machinery ran either
    # (the guard returns before the auto side effect).
    assert _state(conn, "tgt_1") == "suppressed"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()["n"] == 0


def test_no_reply_triggers_a_send(conn):
    """The v1 no-auto-send guarantee: after routing a positive reply (and
    an unsubscribe), NO outbound messages row is created — the reply
    stage has no outbound code path, and the seeded outbound row is the
    only one that exists."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000007", reply_id="rpl_000000000008")
    _classify(conn, reply_id="rpl_000000000008", verdict=_verdict("positive", 0.9))
    _classify(conn, reply_id="rpl_000000000008", verdict=_verdict("unsubscribe", 0.9))
    # The only outbound row is the seeded one — nothing new was "sent".
    outbound = conn.execute(
        "SELECT message_id FROM messages WHERE direction='outbound';"
    ).fetchall()
    assert [r["message_id"] for r in outbound] == ["msg_000000000007"]


# ── Failure paths ────────────────────────────────────────────────────────────


def test_classifier_failure_leaves_row_unclassified_and_state_unchanged(conn):
    """The classifier produces nothing usable: the reply row persists
    UNCLASSIFIED (all three verdict columns NULL), the target's state is
    unchanged (still replied), a failed step is logged, and — the
    deliberate asymmetry — NO transition to failed fires (a classifier
    outage is not the target's fault; the B2c judge precedent)."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000008", reply_id="rpl_000000000009")
    with patch("app.agents.reply._build_classifier_agent",
               return_value=_SilentClassifierAgent()):
        agent = build_reply_agent(conn)
        outcome = classify_and_route_reply(agent, conn=conn,
                                           reply_id="rpl_000000000009", run_id="r1")
    assert outcome == "classification_failed"
    row = conn.execute(
        "SELECT classification, confidence, routed_action FROM replies WHERE reply_id='rpl_000000000009';"
    ).fetchone()
    assert row["classification"] is None
    assert row["confidence"] is None
    assert row["routed_action"] is None
    # The state is whatever the fetch left it in — unchanged by the failure.
    assert _state(conn, "tgt_1") == "replied"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM state_transitions WHERE target_id='tgt_1';"
    ).fetchone()["n"] == 0
    # The failure IS in the trace (never skip logs).
    steps = conn.execute(
        "SELECT status, output_json FROM steps WHERE tool_name='reply_router';"
    ).fetchall()
    assert any(s["status"] == "failed" for s in steps)
    assert any("error_type" in json.loads(s["output_json"]) for s in steps if s["output_json"])


# ── The audit trail and the pure decision function ───────────────────────────


def test_every_reply_and_suppression_row_is_gated(conn):
    """The audit-trail guarantee: every replies row (the insert from the
    fetch and the verdict UPDATE) and every suppressions row has a
    matching write_log row — a raw conn.execute replacing the gated
    writes would leave rows with no audit row and this test fails."""
    _seed_reply_chain(conn, target_id="tgt_1", email="jane@clinic.test",
                      message_id="msg_000000000009", reply_id="rpl_000000000010")
    _classify(conn, reply_id="rpl_000000000010",
              verdict=_verdict("unsubscribe", confidence=0.9))
    for row in conn.execute("SELECT reply_id FROM replies;").fetchall():
        audit = conn.execute(
            "SELECT 1 FROM write_log WHERE record_id=? AND action='insert_reply';",
            (row["reply_id"],),
        ).fetchone()
        assert audit is not None, f"reply row {row['reply_id']} has no insert_reply write_log row"
        audit = conn.execute(
            "SELECT 1 FROM write_log WHERE record_id=? AND action='update_reply_classification';",
            (row["reply_id"],),
        ).fetchone()
        assert audit is not None, f"reply row {row['reply_id']} has no update_reply_classification write_log row"
    for row in conn.execute("SELECT email FROM suppressions;").fetchall():
        audit = conn.execute(
            "SELECT 1 FROM write_log WHERE record_id=? AND action='insert_suppression';",
            (row["email"],),
        ).fetchone()
        assert audit is not None, f"suppression row {row['email']} has no write_log row"


def test_decide_route_is_pure_and_enforces_p4_and_p5():
    """The decision function itself, unit-tested without a database: P5
    dominates (risky → review_required at any confidence), P4 gates the
    floor (below 0.7 → review_required for every class, including
    unsubscribe), and a floor-exact confidence is NOT below the floor."""
    # P5 dominates confidence.
    risky = decide_route(ReplyClassification(
        reply_class="risky", confidence=0.99,
        rationale="The reply threatens legal action and demands data deletion.",
        evidence_quote="We reserve all rights.",
    ))
    assert risky.routed_action == "review_required"
    assert risky.auto_suppress is False
    # P4: a low-confidence unsubscribe is exactly the §9 case.
    low = decide_route(ReplyClassification(
        reply_class="unsubscribe", confidence=P4_CONFIDENCE_FLOOR - 0.01,
        rationale="The wording asks to stop contact but is very short and ambiguous.",
        evidence_quote="Please stop contacting me.",
    ))
    assert low.routed_action == "review_required"
    assert low.auto_suppress is False
    # The floor itself passes (the check is strictly below).
    exact = decide_route(ReplyClassification(
        reply_class="unsubscribe", confidence=P4_CONFIDENCE_FLOOR,
        rationale="The reply explicitly demands removal from the mailing list.",
        evidence_quote="Remove this address from your mailing list.",
    ))
    assert exact.routed_action == "auto_suppress"
    assert exact.auto_suppress is True
