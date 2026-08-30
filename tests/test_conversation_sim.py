# tests/test_conversation_sim.py — ticket E2: the counterparty simulator.
#
# E1 made "routed -> drafted" possible; this file proves the system can
# hold a MULTI-TURN conversation.  Seven scripted personas (D3 added
# negative/not_now/wrong_person to the original four)
# (app/conversation_sim.py) are walked through the REAL pipeline —
# real fetch_inbox threading, real decide_route (P4/P5), real
# state_machine.transition(), real write_gate writes, real E1 follow-up
# drafting, real review-gate approvals, real DRY_RUN sends — with ONLY
# the three LLM agent factories stubbed (classifier, writer, critic), the
# same offline-stand-in pattern tests/test_follow_up_draft.py applies, so
# tests/conftest.py's autouse live-client guard is never tripped.
#
# THE INVARIANTS (ticket §2.3), asserted ACROSS turns, not once:
#  1. No reply of any class ever triggers a send — every outbound (every
#     follow-up included) requires a recorded review_decisions approval.
#  2. Terminal states are never overridden — a suppressed target's later
#     replies are recorded but change nothing.
#  3. P4 (confidence < 0.7) and P5 (risky) never auto-act — at ANY turn.
#  4. The E1 follow-up cap holds across a long positive thread: the third
#     follow-up attempt is refused with the logged follow_up_cap_reached.
#  5. Suppression is permanent — after an unsubscribe, no later turn may
#     produce a draft or a send.  Held in BOTH cases: the first-reply
#     unsubscribe (test_suppressed_target_later_replies_change_nothing)
#     and the second-reply unsubscribe on an already-routed thread
#     (test_unsubscribe_on_routed_target_blocks_later_drafts) — the case
#     that failed under E2 and that ticket E3 fixed in
#     app/agents/reply.py by firing the unsubscribe hop from ANY
#     non-terminal state, not only from a same-run replied -> routed.
#  6. Every reply gets its own row and is classified independently.

import ast  # the structural no-raw-writes test parses app/conversation_sim.py
import hashlib  # proving the guard test never modified data/outbound.db
import json  # parsing steps/write_log payloads in the audit assertions
import re  # the write-keyword pattern for the structural test
from email import policy as email_policy  # parsing generated inbox .eml files
from email.parser import BytesParser  # RFC-5322 parsing — reading only, never transport
from pathlib import Path  # resolving the real-database path and the module path
from unittest.mock import patch  # the classifier/writer/critic factory seams — the only model boundaries

import pytest  # fixtures, tmp_path, and the xfail marker

from app.agents.draft import (  # the E1 follow-up machinery under test
    FOLLOW_UP_ROUTED_ACTION,
    build_draft_agent,
    run_target_through_draft,
    select_draft_eligible_targets,
)
from app.agents.reply import build_reply_agent, classify_and_route_reply  # the real classifier+router runner
from app.agents_registry import seed_agent_registry  # the principals — the write gate refuses unregistered writers
from app.conversation_sim import (  # the module under test
    CONVO_FILE_PREFIX,
    PERSONAS,
    ScriptedTurn,
    generate_next_turn,
    main,
)
from app.db import apply_schema, connect  # fresh per-test SQLite database
from app.demo_seed import DEMO_SOURCE, seed_demo_data  # the REAL demo seed — the walk's preconditions come from it
from app.ids import new_id  # ids for run/step attribution
from app.kill_switch import write_kill_switch  # the switch writer — tests flip the tmp switch file the env var points at
from app.review import ReviewDecisionRequest, record_review_decision  # the REAL review gate — the walk's approvals
from app.schemas import DraftCritique, EmailDraft, MeetingProposal  # valid offline stand-in payloads for the draft loop; MeetingProposal stubs the real-scheduling seam (demo, 2026-08-30)
from app.send_gate import evaluate_send_gate  # the suppression-permanence check (gate-level, direct)
from app.tools.fetch_inbox import fetch_inbox  # the REAL simulated fetch — the walk's inbound half
from app.tools.send_email import send_email  # the REAL DRY_RUN send — deterministic, no model call
from google.adk.agents import BaseAgent  # base class of the offline stand-ins (B1b pattern)
from google.adk.events import Event, EventActions  # how the stand-ins publish their output dicts


# ── Offline stand-ins for the three LLM agents ───────────────────────────────
# The real classifier/writer/critic are ADK LlmAgents that would make
# live billable calls; these stubs publish predetermined dicts under the
# same state keys the real agents' output_schema + output_key write
# ("reply_classification" / "draft" / "critique").  The REAL deterministic
# halves — the reply router, the draft persist node, the state machine,
# the gates — run unmodified; the LLM internals are the one thing stubbed
# (the exact trust boundary test_follow_up_draft.py draws).


class _StubClassifierAgent(BaseAgent):
    """Offline stand-in for the reply classifier: publishes one
    predetermined verdict dict per run, in the order given, and records
    the REDACTED reply text it was handed (the P8 boundary — raw_text
    must never reach the model)."""

    def __init__(self, verdicts: list[dict]):
        super().__init__(name="reply_classifier")  # the registered principal's name
        self._verdicts = list(verdicts)  # private attr — pydantic forbids public assignment
        self._seen_texts: list[str] = []  # what the classifier was handed, per run

    async def _run_async_impl(self, ctx):
        # Snapshot the reply text the REAL runner seeded into state —
        # the P8 assertion reads this later.
        self._seen_texts.append(ctx.session.state["reply_text"])
        # Publish the next predetermined verdict under the exact key the
        # real classifier's output_key writes; the REAL router node
        # re-validates it before any write.
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={"reply_classification": self._verdicts.pop(0)}
            ),
        )


class _StubWriterAgent(BaseAgent):
    """Offline stand-in for the draft writer: publishes one fixed
    EmailDraft dict under state key "draft" (mimicking output_key)."""

    def __init__(self, draft: dict):
        super().__init__(name="draft_writer")  # same stable name as the real agent
        self._draft = draft  # private attr — pydantic forbids public assignment

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"draft": self._draft}),
        )


class _StubCriticAgent(BaseAgent):
    """Offline stand-in for the critic: publishes a passing critique, so
    the loop exits after exactly one iteration (the conversation tests
    care about the transitions, not the loop mechanics)."""

    def __init__(self):
        super().__init__(name="draft_critic")  # same stable name as the real agent

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"critique": _pass_critique()}),
        )


def _verdict(reply_class: str, confidence: float) -> dict:
    """A classifier verdict dict matching ReplyClassification's shape
    (rationale >= 40 chars, evidence_quote >= 10 chars — the schema's
    floors, met with honest stub text)."""
    return {
        "reply_class": reply_class,
        "confidence": confidence,
        "rationale": (
            f"The scripted counterparty turn is worded to elicit the "
            f"{reply_class!r} class, and the classifier agrees with the script."
        ),
        "evidence_quote": "the scripted counterparty reply body",
    }


def _draft() -> dict:
    # A valid EmailDraft serialized to a dict — the exact shape the real
    # writer's output_key stores (model_dump).
    return EmailDraft(
        subject="Re: your interest — next step",
        body=(
            "Thanks for your reply — happy to send the details you asked "
            "for. Would a short call this week work to walk through how it "
            "fits your intake flow?"
        ),
        rationale=(
            "The prospect asked for more information, so the draft answers "
            "their question directly and proposes one concrete next step."
        ),
        confidence=0.8,
    ).model_dump()


def _pass_critique() -> dict:
    # The clean-pass shape — must satisfy DraftCritique's
    # passed-couples-to-evidence validator (the loop then exits early).
    return DraftCritique(
        passed=True, issues=[], required_changes="", severity="none",
    ).model_dump()


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def switch_path(tmp_path, monkeypatch):
    """A tmp kill-switch file, written DISENGAGED, and the env var
    pointing the guardrail's reader at it — the B4a convention (the
    reader is fail-closed, so without this every draft/reply/review
    invocation would halt or refuse)."""
    path = tmp_path / "kill_switch.json"
    write_kill_switch(engaged=False, updated_by="fixture", path=str(path))
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(path))
    return path


def _build_seeded_db(tmp_path, switch_path) -> tuple:
    """Build a fresh SQLite DB the way the demo builds it: schema,
    registered principals, then the REAL demo seed (three targets walked
    to approved with reserved-domain contacts, real policy allow rows,
    real recorded operator approvals).  Returns (conn, db_path); the
    walk's preconditions are the seed's, exactly as the demo's are."""
    db_path = str(tmp_path / "demo.db")
    conn = connect(db_path)
    apply_schema(conn)
    seed_agent_registry(conn, run_id="r0", step_id="s0")
    seed_demo_data(conn, run_id="rseed")
    return conn, db_path


@pytest.fixture
def seeded_db(tmp_path, switch_path):
    """The connection half of _build_seeded_db — every walk uses it."""
    conn, _ = _build_seeded_db(tmp_path, switch_path)
    yield conn
    conn.close()


def _demo_target_id(conn) -> str:
    """The Serenity Clinic target — the demo's interested-reply target —
    identified by its seeded contact address (never by id guessing)."""
    row = conn.execute(
        "SELECT t.target_id FROM targets t "
        "JOIN contacts c ON t.contact_id = c.contact_id "
        "WHERE t.source=? AND c.email='dr.chan@serenity-clinic.test';",
        (DEMO_SOURCE,),
    ).fetchone()
    assert row is not None, "the demo seed must create the Serenity Clinic target"
    return row["target_id"]


# ── Walk helpers (the real pipeline, models stubbed) ─────────────────────────


def _state(conn, target_id: str) -> str:
    # The target's current state — the assertion anchor at every turn.
    return conn.execute(
        "SELECT state FROM targets WHERE target_id=?;", (target_id,)
    ).fetchone()["state"]


def _hops(conn, target_id: str) -> list[tuple[str, str, str]]:
    """The target's state_transitions history, in insertion order — the
    audit answer to 'what happened, in what order' (the walk's final
    assertion compares this against the full expected sequence)."""
    return [
        (r["previous_state"], r["new_state"], r["reason"])
        for r in conn.execute(
            "SELECT previous_state, new_state, reason FROM state_transitions "
            "WHERE target_id=? ORDER BY insert_seq, created_at;",
            (target_id,),
        ).fetchall()
    ]


def _generate(conn, outbox, inbox, script, target_id):
    """Advance the thread one turn via the simulator (the same function
    the CLI drives) and assert it actually wrote a file."""
    result = generate_next_turn(
        conn,
        outbox_dir=str(outbox),
        inbox_dir=str(inbox),
        persona_script=script,
        target_id=target_id,
    )
    assert result.written_path is not None, (
        f"turn generation refused: {result.refusal_reason}"
    )
    return result


def _fetch_and_classify(conn, inbox, run_id, verdicts):
    """Run the REAL fetch over the inbox, then the REAL classifier+router
    runner per created reply with the stub classifier publishing the
    given verdicts in order.  Returns (InboxFetchResult, outcome dict)."""
    fetched = fetch_inbox(conn, inbox_dir=str(inbox), run_id=run_id, limit=100)
    with patch(
        "app.agents.reply._build_classifier_agent",
        return_value=_StubClassifierAgent(verdicts),
    ):
        agent = build_reply_agent(conn)
        outcomes = {
            reply_id: classify_and_route_reply(
                agent, conn=conn, reply_id=reply_id, run_id=run_id
            )
            for reply_id in fetched.replies_created
        }
    return fetched, outcomes


def _fake_scheduler_verdict(system_prompt, user_content):
    """The scheduler LLM stub (demo, 2026-08-30): real config/offers/
    therapy-app.yaml now carries scheduling_enabled: true, so a follow-up
    draft run against the real committed offer (as this harness
    deliberately does) invokes schedule_meeting for real — this stub keeps
    ONLY the model call offline, the same offline-stand-in discipline
    _StubWriterAgent/_StubCriticAgent/_StubClassifierAgent apply. It picks
    the FIRST slot the real calendar computation actually offered (parsed
    straight out of the real prompt payload), never an invented one."""
    offered = json.loads(user_content)
    return MeetingProposal(
        chosen_slot_label=offered["available_slots"][0],
        company_name=offered["company_name"],
        reasoning="earliest available slot",
    )


def _run_follow_up_draft(conn, target_id, run_id, *, writer_stub=None):
    """Run the REAL E1 follow-up draft runner with the writer/critic
    stubbed — the real preconditions, selection semantics, cap check,
    transitions, and gated revision write all execute.  ``writer_stub``
    overrides the clean follow-up draft so a test can plant a draft whose
    gate verdict is fail."""
    draft_dict = _draft() if writer_stub is None else writer_stub
    with patch("app.agents.draft._build_writer_agent", return_value=_StubWriterAgent(draft_dict)), \
         patch("app.agents.draft._build_critic_agent", return_value=_StubCriticAgent()), \
         patch("app.tools.schedule_meeting._call_scheduler_llm", side_effect=_fake_scheduler_verdict):
        agent = build_draft_agent(conn)
        return run_target_through_draft(
            agent, conn=conn, target_id=target_id, run_id=run_id,
            offers_dir="config/offers",  # the real committed offer — the draft brief reads it read-only
        )


def _approve(conn, target_id, run_id):
    """The REAL review gate: record an operator approval (the same door
    the console's Approve button uses)."""
    outcome = record_review_decision(
        conn,
        request=ReviewDecisionRequest(
            target_id=target_id,
            decision="approve",
            reason="E2 conversation walk: the operator approves the follow-up draft",
        ),
        run_id=run_id,
    )
    assert not outcome.refused, f"review gate refused: {outcome.refusal_reason}"
    return outcome


def _send(conn, target_id, run_id, outbox):
    """The REAL DRY_RUN send (19-check preflight, .eml artifact, messages
    row, approved -> dry_run_sent)."""
    return send_email(
        conn, target_id=target_id, run_id=run_id, outbox_dir=str(outbox),
    )


def _assert_follow_up_gate_columns_written(conn, target_id):
    """G2's close of the old E2 finding: the real draft loop's follow-up
    revision now carries NON-NULL policy_check_passed / injection_scan_passed
    because the deterministic draft gate runner fired inside
    run_target_through_draft.  A clean follow-up draft is 1/1, so the real
    send gate no longer needs the old fixture seeding."""
    revision = conn.execute(
        "SELECT draft_version_id, policy_check_passed, injection_scan_passed "
        "FROM message_draft_versions WHERE target_id=? "
        "ORDER BY revision_number DESC, insert_seq DESC, created_at DESC LIMIT 1;",
        (target_id,),
    ).fetchone()
    assert revision["policy_check_passed"] == 1, "the G2 runner must evaluate the fresh follow-up revision"
    assert revision["injection_scan_passed"] == 1


def _assert_reply_turn(conn, run_id, reply_id, *, reply_class, confidence,
                       routed_action, review_required, target_state):
    """The per-turn assertion block (ticket §2.2: at EVERY turn, assert
    the classification, the routed action, the target's state, the rows
    written — and that each went through the gate — and the steps
    trace).  One call per turn, so a regression names the turn it broke.
    """
    # The replies row: classified independently, exactly as the stub
    # verdict said — and the class/action pair matches the routing table.
    row = conn.execute(
        "SELECT classification, confidence, routed_action FROM replies "
        "WHERE reply_id=?;",
        (reply_id,),
    ).fetchone()
    assert row is not None, f"reply {reply_id} has no row"
    assert row["classification"] == reply_class
    assert row["confidence"] == confidence
    assert row["routed_action"] == routed_action
    # The row's writes went through the gate — both the fetch's insert
    # and the router's verdict update, attributed to the right agents.
    actions = {
        r["action"]
        for r in conn.execute(
            "SELECT action FROM write_log WHERE run_id=? AND record_id=?;",
            (run_id, reply_id),
        ).fetchall()
    }
    assert actions == {"insert_reply", "update_reply_classification"}, (
        f"reply {reply_id} was not written through the gate: {actions}"
    )
    # The steps trace: the fetch row names the reply in its output, the
    # router row names it in its input and carries the verdict + the
    # resulting state in its output.
    fetch_step = conn.execute(
        "SELECT output_json FROM steps WHERE tool_name='fetch_inbox' "
        "AND output_json LIKE ?;",
        (f'%"{reply_id}"%',),
    ).fetchone()
    assert fetch_step is not None, "every inbound message gets a fetch_inbox step row"
    router_step = conn.execute(
        "SELECT output_json FROM steps WHERE tool_name='reply_router' "
        "AND input_json LIKE ?;",
        (f'%"{reply_id}"%',),
    ).fetchone()
    assert router_step is not None, "every classified reply gets a reply_router step row"
    router_out = json.loads(router_step["output_json"])
    assert router_out["reply_class"] == reply_class
    assert router_out["routed_action"] == routed_action
    assert router_out["review_required"] == review_required
    assert router_out["target_state"] == target_state
    # The target's actual state after the turn — the state machine's
    # answer, read fresh (replies has no target column; join through the
    # matched message).
    target_row = conn.execute(
        "SELECT t.state FROM targets t JOIN messages m ON m.target_id = t.target_id "
        "JOIN replies r ON r.message_id = m.message_id WHERE r.reply_id=?;",
        (reply_id,),
    ).fetchone()
    assert target_row["state"] == target_state


# ── 1. The persona scripts: complete, pinned, and address-free ───────────────


def test_persona_scripts_are_pinned_and_address_free():
    """The seven required personas exist with the EXACT class sequences
    the ticket specifies — the data block is pinned so a silent edit to
    a persona's shape fails here — and no scripted text contains an
    email address (the simulator never invents a real person)."""
    assert set(PERSONAS) == {
        "warms_up", "pushes_back_then_leaves", "goes_legal", "stays_vague",
        "negative", "not_now", "wrong_person",
    }
    assert [t.reply_class for t in PERSONAS["warms_up"]] == [
        "positive", "meeting_request", "positive", "positive",
    ]
    assert [t.reply_class for t in PERSONAS["pushes_back_then_leaves"]] == [
        "objection", "unsubscribe",
    ]
    assert [t.reply_class for t in PERSONAS["goes_legal"]] == ["risky", "risky"]
    assert [t.reply_class for t in PERSONAS["stays_vague"]] == ["unclear", "unclear"]
    assert [t.reply_class for t in PERSONAS["negative"]] == ["negative"]
    assert [t.reply_class for t in PERSONAS["not_now"]] == ["not_now"]
    assert [t.reply_class for t in PERSONAS["wrong_person"]] == ["wrong_person"]
    for name, script in PERSONAS.items():
        for turn in script:
            assert "@" not in turn.body, (
                f"{name} turn text contains an email address — the script "
                f"must never invent a real-looking sender or recipient"
            )


# ── 2. The simulator's own contract ──────────────────────────────────────────


def test_generated_turn_is_deterministic(seeded_db, tmp_path):
    """The same thread state always produces the same next message:
    generating a turn, deleting the file, and generating again yields
    byte-identical artifacts (the turn index comes from the recorded
    reply count, and the Date header from the outbound artifact — never
    the wall clock)."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    assert not _send(conn, target_id, new_id("run"), outbox).refused

    first = _generate(conn, outbox, inbox, PERSONAS["warms_up"], target_id)
    first_bytes = Path(first.written_path).read_bytes()
    Path(first.written_path).unlink()  # remove the artifact — same state, fresh write
    second = _generate(conn, outbox, inbox, PERSONAS["warms_up"], target_id)
    second_bytes = Path(second.written_path).read_bytes()
    assert first_bytes == second_bytes, (
        "the same thread state must produce byte-identical turns"
    )


def test_generated_senders_are_all_reserved_domains(seeded_db, tmp_path):
    """Ticket §2.1 hard requirement: every simulated sender address is on
    an RFC 2606 reserved domain — asserted by parsing each generated
    .eml's From header, one turn per persona (the reported count is 7)."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    assert not _send(conn, target_id, new_id("run"), outbox).refused

    parser = BytesParser(policy=email_policy.default)
    senders = []
    for name, script in PERSONAS.items():
        result = _generate(conn, outbox, inbox, script, target_id)
        msg = parser.parsebytes(Path(result.written_path).read_bytes())
        sender = msg["From"]
        assert "@" in sender, f"{name} turn has no From address"
        # The address half, brackets stripped, TLD parsed — must be one
        # of the reserved three (the same rule demo_seed enforces).
        address = sender.rsplit("<", 1)[-1].rstrip(">").strip()
        domain = address.split("@", 1)[-1].lower()
        assert domain.rsplit(".", 1)[-1] in ("test", "invalid", "example"), (
            f"{name} sender {sender!r} is not on a reserved domain"
        )
        senders.append(address)
        Path(result.written_path).unlink()  # leave no unprocessed file for the next persona
    assert len(senders) == 7  # the count the ticket asks to report


def test_conversation_sim_has_no_raw_core_table_writes():
    """Structural guarantee, in the spirit of the console and demo-seed
    tests: every conn.execute() call in app/conversation_sim.py must
    carry SELECT-only SQL (the simulator's only output is .eml files),
    and the module must import demo_seed's threading helpers — the
    ticket's 'reuse, not duplicate' contract, made structural."""
    path = Path(__file__).resolve().parent.parent / "app" / "conversation_sim.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    write_sql = re.compile(r"\b(INSERT|UPDATE|DELETE|REPLACE)\b", re.IGNORECASE)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            assert not write_sql.search(node.args[0].value), (
                f"app/conversation_sim.py issues a raw write via conn.execute(): "
                f"{node.args[0].value!r}"
            )
    # The positive half: the reuse contract — the outbox→inbox threading
    # is demo_seed's (lead-verified), imported, never duplicated.
    assert "from app.demo_seed import" in source
    assert "_compose_reply_bytes" in source
    assert "_reserved_domain_of" in source
    assert "_guard_violation" in source


# ── 3. Persona 1: warms_up — the long positive thread, walked end to end ─────


def test_follow_up_send_passes_without_seeded_gate_columns(seeded_db, tmp_path):
    """G2's close of the old E2 finding: a follow-up draft produced by the
    real E1 loop now carries its own NON-NULL gate columns because the
    deterministic draft gate runner fires inside run_target_through_draft.
    A clean follow-up therefore passes the real send gate WITHOUT any
    hand-seeded columns — the old 'refused until seeded' premise is gone."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    # Exchange 1: the seeded send, a positive reply, a real follow-up
    # draft, and a real operator approval.
    assert not _send(conn, target_id, new_id("run"), outbox).refused
    _generate(conn, outbox, inbox, PERSONAS["warms_up"], target_id)
    _fetch_and_classify(conn, inbox, new_id("run"), [_verdict("positive", 0.9)])
    assert _run_follow_up_draft(conn, target_id, new_id("run")) == "awaiting_review"
    # The runner wrote both gate columns on the fresh follow-up revision.
    _assert_follow_up_gate_columns_written(conn, target_id)
    _approve(conn, target_id, new_id("run"))
    # The follow-up send passes the SAME live gate with no seeding.
    assert not _send(conn, target_id, new_id("run"), outbox).refused
    assert _state(conn, target_id) == "dry_run_sent"


def test_follow_up_failing_draft_is_refused_at_the_gate(seeded_db, tmp_path):
    """The fail-closed half that moved with the rewrite above: a follow-up
    draft whose runner verdict is FAIL still gets refused at the real send
    gate with the reason named — no artifact, and the target stays in
    approved for a corrected retry."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    assert not _send(conn, target_id, new_id("run"), outbox).refused
    _generate(conn, outbox, inbox, PERSONAS["warms_up"], target_id)
    _fetch_and_classify(conn, inbox, new_id("run"), [_verdict("positive", 0.9)])
    # A schema-valid draft that the CONTENT POLICY refuses: a banned
    # pressure phrase ("limited time") the writer's own rule 5 forbids.
    failing_draft = EmailDraft(
        subject="Re: your interest — next step",
        body=(
            "This is a limited time offer you should grab right now before "
            "it goes away. Would a short call this week work to walk "
            "through how it fits your intake flow?"
        ),
        rationale=(
            "The prospect asked for more information, so the draft answers "
            "their question directly and proposes one concrete next step."
        ),
        confidence=0.8,
    ).model_dump()
    assert _run_follow_up_draft(conn, target_id, new_id("run"), writer_stub=failing_draft) == "awaiting_review"
    _approve(conn, target_id, new_id("run"))
    # The runner wrote policy_check_passed=0 on the failing draft, so the
    # live send gate refuses and names the check.
    refused = _send(conn, target_id, new_id("run"), outbox)
    assert refused.refused, "a draft the runner failed must be refused"
    assert "policy_check_passed" in refused.refusal_reason
    assert _state(conn, target_id) == "approved", "a refused send must not move the target"
    assert len(list(outbox.glob("*.eml"))) == 1, "no artifact for the refused follow-up"


def test_warms_up_full_thread(seeded_db, tmp_path):
    """Persona 1, the whole thread: positive -> follow-up -> meeting
    request -> positive -> follow-up -> positive -> CAP.  Walked through
    the real pipeline with the classifier/writer/critic stubbed; every
    invariant in ticket §2.3 is asserted ACROSS the turns, and the walk
    ends with the full state_transitions sequence."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    script = PERSONAS["warms_up"]
    before_hops = _hops(conn, target_id)  # the seed's own five hops — asserted unchanged at the end

    # ── Exchange 1 ─────────────────────────────────────────────────────
    assert not _send(conn, target_id, new_id("run"), outbox).refused
    assert _state(conn, target_id) == "dry_run_sent"
    outbox_count = 1  # the running invariant: no reply may ever add a send

    # Turn 1: positive -> routed with queue_follow_up_draft.
    _generate(conn, outbox, inbox, script, target_id)
    run1 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run1, [_verdict("positive", 0.9)])
    reply1 = fetched.replies_created[0]
    assert outcomes[reply1] == "routed"
    _assert_reply_turn(conn, run1, reply1, reply_class="positive", confidence=0.9,
                       routed_action=FOLLOW_UP_ROUTED_ACTION, review_required=True,
                       target_state="routed")
    assert len(list(outbox.glob("*.eml"))) == outbox_count, "a positive reply must not send"

    # Follow-up 1: the real E1 loop -> awaiting_review -> approval ->
    # send (gate columns seeded per the labelled fixture practice).
    assert _run_follow_up_draft(conn, target_id, new_id("run")) == "awaiting_review"
    assert _state(conn, target_id) == "awaiting_review"
    _approve(conn, target_id, new_id("run"))
    _assert_follow_up_gate_columns_written(conn, target_id)
    assert not _send(conn, target_id, new_id("run"), outbox).refused
    outbox_count += 1
    # The approval invariant: every send has its own recorded decision
    # (the seed's own approval is decision #1).
    approvals = conn.execute(
        "SELECT decision FROM review_decisions WHERE target_id=? "
        "ORDER BY insert_seq, created_at;", (target_id,)
    ).fetchall()
    assert [r["decision"] for r in approvals] == ["approve", "approve"]

    # Turn 2: meeting_request -> routed with notify_operator, review
    # required — and NOTHING is drafted (the §2 table queues no draft).
    _generate(conn, outbox, inbox, script, target_id)
    run2 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run2, [_verdict("meeting_request", 0.9)])
    reply2 = fetched.replies_created[0]
    assert outcomes[reply2] == "routed"
    _assert_reply_turn(conn, run2, reply2, reply_class="meeting_request", confidence=0.9,
                       routed_action="notify_operator", review_required=True,
                       target_state="routed")
    assert select_draft_eligible_targets(conn, limit=10) == [], (
        "a meeting_request must not queue a follow-up draft"
    )
    assert len(list(outbox.glob("*.eml"))) == outbox_count, "a meeting request must not send"

    # Turn 3: positive again -> follow-up 2 (the cap is at 2, not 1).
    _generate(conn, outbox, inbox, script, target_id)
    run3 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run3, [_verdict("positive", 0.9)])
    reply3 = fetched.replies_created[0]
    assert outcomes[reply3] == "routed"
    _assert_reply_turn(conn, run3, reply3, reply_class="positive", confidence=0.9,
                       routed_action=FOLLOW_UP_ROUTED_ACTION, review_required=True,
                       target_state="routed")
    assert _run_follow_up_draft(conn, target_id, new_id("run")) == "awaiting_review"
    _approve(conn, target_id, new_id("run"))
    _assert_follow_up_gate_columns_written(conn, target_id)
    assert not _send(conn, target_id, new_id("run"), outbox).refused
    outbox_count += 1

    # Turn 4: positive forever -> the E1 cap refuses the third follow-up
    # attempt, with the refusal LOGGED (greppable outcome).
    _generate(conn, outbox, inbox, script, target_id)
    run4 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run4, [_verdict("positive", 0.9)])
    reply4 = fetched.replies_created[0]
    assert outcomes[reply4] == "routed"
    _assert_reply_turn(conn, run4, reply4, reply_class="positive", confidence=0.9,
                       routed_action=FOLLOW_UP_ROUTED_ACTION, review_required=True,
                       target_state="routed")
    assert _run_follow_up_draft(conn, target_id, new_id("run")) == "follow_up_cap_reached"
    refusal = conn.execute(
        "SELECT output_json FROM steps WHERE target_id=? "
        "AND tool_name='draft_target_run';", (target_id,)
    ).fetchone()
    assert refusal is not None, "the cap refusal must be a logged step"
    assert json.loads(refusal["output_json"])["outcome"] == "follow_up_cap_reached"
    assert _state(conn, target_id) == "routed", "the cap refusal never moves the target"
    assert len(list(outbox.glob("*.eml"))) == outbox_count, "the cap must not send"

    # ── The across-turn invariants, asserted once at the end ────────────
    # Invariant 6: every reply got its own row, classified independently,
    # in insertion order.
    reply_rows = conn.execute(
        "SELECT r.classification, r.confidence FROM replies r "
        "JOIN messages m ON r.message_id = m.message_id "
        "WHERE m.target_id=? ORDER BY r.insert_seq, r.created_at;",
        (target_id,),
    ).fetchall()
    assert [(r["classification"], r["confidence"]) for r in reply_rows] == [
        ("positive", 0.9), ("meeting_request", 0.9), ("positive", 0.9), ("positive", 0.9),
    ]
    # Invariant 1: three sends, three recorded approvals beyond the
    # seed's (the seeded approval is decision #1), three outbox artifacts.
    assert len(list(outbox.glob("*.eml"))) == 3
    outbound = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE target_id=? AND direction='outbound';",
        (target_id,),
    ).fetchone()["n"]
    assert outbound == 3, "one messages row per allowed send, no more"
    # Invariants 2/3: the full hop sequence — every state change went
    # through the state machine, in exactly the expected order (turn 3's
    # fetch and classification add no hops: the target is already
    # routed, and nothing may re-route it).
    assert _hops(conn, target_id) == before_hops + [
        ("approved", "dry_run_sent", "send_gate_success_dry_run"),
        ("dry_run_sent", "replied", "inbound_message_linked"),
        ("replied", "routed", "classified_and_routed"),
        ("routed", "drafted", FOLLOW_UP_ROUTED_ACTION),
        ("drafted", "awaiting_review", "draft_complete"),
        ("awaiting_review", "approved", "operator_approval"),
        ("approved", "dry_run_sent", "send_gate_success_dry_run"),
        ("dry_run_sent", "replied", "inbound_message_linked"),
        ("replied", "routed", "classified_and_routed"),
        ("routed", "drafted", FOLLOW_UP_ROUTED_ACTION),
        ("drafted", "awaiting_review", "draft_complete"),
        ("awaiting_review", "approved", "operator_approval"),
        ("approved", "dry_run_sent", "send_gate_success_dry_run"),
        ("dry_run_sent", "replied", "inbound_message_linked"),
        ("replied", "routed", "classified_and_routed"),
    ]


# ── 4. Persona 2: pushes_back_then_leaves — objection, then unsubscribe ──────


def test_pushes_back_then_leaves(seeded_db, tmp_path):
    """Persona 2, turns 1-2: an objection (draft_hold, review required,
    nothing drafted), then an unsubscribe on the same thread.  The
    suppression row lands, the target reaches suppressed from its REAL
    current state (routed — ticket E3's hop, fired on a second reply
    rather than a same-run replied -> routed), and the SEND half of
    suppression permanence is proven at the gate: the suppressed address
    can never be sent to.  (The DRAFT half — what a later positive turn
    may still produce — is
    test_unsubscribe_on_routed_target_blocks_later_drafts' subject.)"""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    script = PERSONAS["pushes_back_then_leaves"]

    assert not _send(conn, target_id, new_id("run"), outbox).refused
    outbox_count = 1

    # Turn 1: objection -> routed, draft_hold, review required, and the
    # draft stage's eligible set stays empty (draft_hold queues nothing).
    _generate(conn, outbox, inbox, script, target_id)
    run1 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run1, [_verdict("objection", 0.9)])
    reply1 = fetched.replies_created[0]
    assert outcomes[reply1] == "routed"
    _assert_reply_turn(conn, run1, reply1, reply_class="objection", confidence=0.9,
                       routed_action="draft_hold", review_required=True,
                       target_state="routed")
    assert select_draft_eligible_targets(conn, limit=10) == []
    assert len(list(outbox.glob("*.eml"))) == outbox_count

    # Turn 2: a HIGH-confidence unsubscribe on the same thread.  The
    # suppression row must land (the one auto side effect that exists),
    # the router reports "suppressed", and — ticket E3 — the target
    # closes from its REAL current state (routed, set by turn 1), so the
    # audit row records previous_state=routed, not a hardcoded guess.
    _generate(conn, outbox, inbox, script, target_id)
    run2 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run2, [_verdict("unsubscribe", 0.95)])
    reply2 = fetched.replies_created[0]
    assert outcomes[reply2] == "suppressed"
    _assert_reply_turn(conn, run2, reply2, reply_class="unsubscribe", confidence=0.95,
                       routed_action="auto_suppress", review_required=False,
                       target_state="suppressed")
    assert _hops(conn, target_id)[-1] == ("routed", "suppressed", "unsubscribe_reply"), (
        "the unsubscribe hop must record where the target actually was"
    )
    suppression = conn.execute(
        "SELECT 1 FROM suppressions WHERE email='dr.chan@serenity-clinic.test';"
    ).fetchone()
    assert suppression is not None, "an unsubscribe must add a suppression row"
    # Immediately after the unsubscribe, no draft is eligible — the
    # latest reply's action is auto_suppress, not queue_follow_up_draft.
    assert select_draft_eligible_targets(conn, limit=10) == []
    assert len(list(outbox.glob("*.eml"))) == outbox_count, "an unsubscribe must not send"

    # The SEND half of suppression permanence, proven at the gate itself:
    # the suppressed address is a hard refusal, whatever else is true.
    decision = evaluate_send_gate(conn, target_id=target_id, run_id=new_id("run"), step_id=new_id("step"))
    assert decision.suppression_hit, "the gate must flag the suppressed address"
    assert any("suppression list" in reason for reason in decision.reasons)


def test_unsubscribe_on_routed_target_blocks_later_drafts(seeded_db, tmp_path):
    """Ticket §2.3 invariant: 'Suppression is permanent — after an
    unsubscribe, no later turn can produce a draft or a send.'  Walked
    on a thread where the unsubscribe is the SECOND reply (objection
    first): under E2 this FAILED — the router recorded the suppression
    but never closed the target, so the positive turn re-entered the E1
    draft path.  E3 fixed the hop to fire from any non-terminal state;
    this test now proves the fix on the exact thread that broke it."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    script = PERSONAS["pushes_back_then_leaves"] + (
        ScriptedTurn(reply_class="positive", body="Actually, do send more details."),
    )

    assert not _send(conn, target_id, new_id("run"), outbox).refused
    _generate(conn, outbox, inbox, script, target_id)
    _fetch_and_classify(conn, inbox, new_id("run"), [_verdict("objection", 0.9)])
    assert _state(conn, target_id) == "routed"

    # Turn 2: the SECOND-reply unsubscribe (the target is already routed
    # from turn 1) — E3's hop must close it, recording the true
    # previous_state rather than a hardcoded "routed"-or-nothing.
    _generate(conn, outbox, inbox, script, target_id)
    _fetch_and_classify(conn, inbox, new_id("run"), [_verdict("unsubscribe", 0.95)])
    assert _state(conn, target_id) == "suppressed", (
        "a second-reply unsubscribe must suppress an already-routed target"
    )
    assert _hops(conn, target_id)[-1] == ("routed", "suppressed", "unsubscribe_reply")

    # A later positive turn — the persona has "left", but the invariant
    # says later turns change nothing: the verdict is recorded, the
    # terminal guard fires, and no draft may be produced.
    _generate(conn, outbox, inbox, script, target_id)
    fetched, outcomes = _fetch_and_classify(conn, inbox, new_id("run"), [_verdict("positive", 0.9)])
    assert outcomes[fetched.replies_created[0]] == "terminal_no_transition"

    # THE INVARIANT ASSERTIONS:
    # (a) the eligible set must stay empty, and
    # (b) the draft runner must refuse, producing no new revision.
    assert select_draft_eligible_targets(conn, limit=10) == [], (
        "a suppressed contact must never re-enter the draft eligible set"
    )
    revisions_before = conn.execute(
        "SELECT COUNT(*) AS n FROM message_draft_versions WHERE target_id=?;",
        (target_id,),
    ).fetchone()["n"]
    outcome = _run_follow_up_draft(conn, target_id, new_id("run"))
    assert outcome != "awaiting_review", "no follow-up draft may be produced after an unsubscribe"
    revisions_after = conn.execute(
        "SELECT COUNT(*) AS n FROM message_draft_versions WHERE target_id=?;",
        (target_id,),
    ).fetchone()["n"]
    assert revisions_after == revisions_before, "a suppressed thread must gain no draft"


# ── 5. Persona 3: goes_legal — risky replies at every turn ───────────────────


def test_goes_legal(seeded_db, tmp_path):
    """Persona 3, two turns: P5 must hold at EVERY turn — a risky reply
    routes to review_required with NO auto side effect, however high the
    confidence, and the second risky reply is recorded independently
    (its own row, its own verdict)."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    script = PERSONAS["goes_legal"]

    assert not _send(conn, target_id, new_id("run"), outbox).refused
    outbox_count = 1

    # Turn 1: risky at confidence 0.99 — still review_required (P5), no
    # suppression, no draft, no send.
    _generate(conn, outbox, inbox, script, target_id)
    run1 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run1, [_verdict("risky", 0.99)])
    reply1 = fetched.replies_created[0]
    assert outcomes[reply1] == "review_required"
    _assert_reply_turn(conn, run1, reply1, reply_class="risky", confidence=0.99,
                       routed_action="review_required", review_required=True,
                       target_state="routed")
    assert conn.execute("SELECT COUNT(*) AS n FROM suppressions;").fetchone()["n"] == 0
    assert select_draft_eligible_targets(conn, limit=10) == []
    assert len(list(outbox.glob("*.eml"))) == outbox_count

    # Turn 2: risky again, on the same thread — recorded independently,
    # still review_required, still no auto side effect, no state change
    # (the target is already routed, and nothing may re-route it).
    hops_before_turn2 = _hops(conn, target_id)
    _generate(conn, outbox, inbox, script, target_id)
    run2 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run2, [_verdict("risky", 0.99)])
    reply2 = fetched.replies_created[0]
    assert reply2 != reply1, "every reply gets its own row"
    assert outcomes[reply2] == "review_required"
    _assert_reply_turn(conn, run2, reply2, reply_class="risky", confidence=0.99,
                       routed_action="review_required", review_required=True,
                       target_state="routed")
    assert conn.execute("SELECT COUNT(*) AS n FROM suppressions;").fetchone()["n"] == 0
    assert _hops(conn, target_id) == hops_before_turn2, "a second risky reply changes no state"
    assert len(list(outbox.glob("*.eml"))) == outbox_count


# ── 6. Persona 4: stays_vague — low confidence at every turn ─────────────────


def test_stays_vague(seeded_db, tmp_path):
    """Persona 4: P4 must hold at EVERY turn — a below-floor confidence
    routes to review_required whatever the class, and a low-confidence
    UNSUBSCRIBE (turn 3, on the same thread) must NOT suppress: the
    CLAUDE.md §9 case, exercised at turn 3 rather than turn 1."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    # The persona's two vague turns plus the low-confidence unsubscribe
    # as the third — a full 3-turn script (the turn index maps into the
    # whole script, so custom walks pass the complete conversation).
    script = PERSONAS["stays_vague"] + (
        ScriptedTurn(reply_class="unsubscribe", body="Please stop contacting me."),
    )

    assert not _send(conn, target_id, new_id("run"), outbox).refused
    outbox_count = 1

    # Turns 1-2: unclear at 0.5 then 0.4 — both review_required, each on
    # its own row, each classified independently.
    for confidence in (0.5, 0.4):
        _generate(conn, outbox, inbox, script, target_id)
        run_id = new_id("run")
        fetched, outcomes = _fetch_and_classify(
            conn, inbox, run_id, [_verdict("unclear", confidence)]
        )
        reply_id = fetched.replies_created[0]
        assert outcomes[reply_id] == "review_required"
        _assert_reply_turn(conn, run_id, reply_id, reply_class="unclear",
                           confidence=confidence,
                           routed_action="review_required", review_required=True,
                           target_state="routed")
        assert select_draft_eligible_targets(conn, limit=10) == []
        assert len(list(outbox.glob("*.eml"))) == outbox_count

    # Turn 3: a low-confidence UNSUBSCRIBE — P4 overrides the class's
    # auto-action: review_required, and NO suppression row may appear.
    _generate(conn, outbox, inbox, script, target_id)
    run_id = new_id("run")
    fetched, outcomes = _fetch_and_classify(
        conn, inbox, run_id, [_verdict("unsubscribe", 0.5)]
    )
    reply_id = fetched.replies_created[0]
    assert outcomes[reply_id] == "review_required"
    _assert_reply_turn(conn, run_id, reply_id, reply_class="unsubscribe",
                       confidence=0.5,
                       routed_action="review_required", review_required=True,
                       target_state="routed")
    assert conn.execute("SELECT COUNT(*) AS n FROM suppressions;").fetchone()["n"] == 0, (
        "a low-confidence unsubscribe must NOT suppress (CLAUDE.md §9)"
    )
    assert len(list(outbox.glob("*.eml"))) == outbox_count


# ── 7. Personas 5-7: negative, not_now, wrong_person — reachability ────────────


def test_negative_persona(seeded_db, tmp_path):
    """Persona 5, one turn: a plain, polite decline proves the negative
    class is reachable through the REAL pipeline — classified, routed to
    the §2 close_not_target action at high confidence with review NOT
    required, and the target moves replied -> routed like any classified
    reply.  (The router RECORDS close_not_target; it does not yet move
    the target to the not_target state — reply-routing.md §2 vs
    app/agents/reply.py.)"""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    script = PERSONAS["negative"]

    assert not _send(conn, target_id, new_id("run"), outbox).refused
    outbox_count = 1

    _generate(conn, outbox, inbox, script, target_id)
    run1 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run1, [_verdict("negative", 0.9)])
    reply1 = fetched.replies_created[0]
    assert outcomes[reply1] == "routed"
    _assert_reply_turn(conn, run1, reply1, reply_class="negative", confidence=0.9,
                       routed_action="close_not_target", review_required=False,
                       target_state="routed")
    assert select_draft_eligible_targets(conn, limit=10) == [], (
        "a negative reply must not queue a follow-up draft"
    )
    assert len(list(outbox.glob("*.eml"))) == outbox_count, "a negative reply must not send"


def test_not_now_persona(seeded_db, tmp_path):
    """Persona 6, one turn: a clear 'not at this time, check back later'
    proves the not_now class is reachable — classified, routed to the §2
    schedule_reminder action at high confidence with review NOT required,
    and no draft is queued.  (No reminder scheduler exists yet — the
    router RECORDS schedule_reminder; it does not create a reminder —
    reply-routing.md §2 vs app/agents/reply.py.)"""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    script = PERSONAS["not_now"]

    assert not _send(conn, target_id, new_id("run"), outbox).refused
    outbox_count = 1

    _generate(conn, outbox, inbox, script, target_id)
    run1 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run1, [_verdict("not_now", 0.9)])
    reply1 = fetched.replies_created[0]
    assert outcomes[reply1] == "routed"
    _assert_reply_turn(conn, run1, reply1, reply_class="not_now", confidence=0.9,
                       routed_action="schedule_reminder", review_required=False,
                       target_state="routed")
    assert select_draft_eligible_targets(conn, limit=10) == [], (
        "a not_now reply must not queue a follow-up draft"
    )
    assert len(list(outbox.glob("*.eml"))) == outbox_count, "a not_now reply must not send"


def test_wrong_person_persona(seeded_db, tmp_path):
    """Persona 7, one turn: a clear 'not the right contact, reach out to
    someone else' proves the wrong_person class is reachable — classified,
    routed to the §2 re_enrich action at high confidence with review NOT
    required, and no draft is queued.  (No re-enrichment path exists yet —
    the router RECORDS re_enrich; it does not re-run enrichment —
    reply-routing.md §2 vs app/agents/reply.py.)"""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    script = PERSONAS["wrong_person"]

    assert not _send(conn, target_id, new_id("run"), outbox).refused
    outbox_count = 1

    _generate(conn, outbox, inbox, script, target_id)
    run1 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run1, [_verdict("wrong_person", 0.9)])
    reply1 = fetched.replies_created[0]
    assert outcomes[reply1] == "routed"
    _assert_reply_turn(conn, run1, reply1, reply_class="wrong_person", confidence=0.9,
                       routed_action="re_enrich", review_required=False,
                       target_state="routed")
    assert select_draft_eligible_targets(conn, limit=10) == [], (
        "a wrong_person reply must not queue a follow-up draft"
    )
    assert len(list(outbox.glob("*.eml"))) == outbox_count, "a wrong_person reply must not send"


# ── 8. Terminal states are never overridden (first-reply unsubscribe) ────────


def test_suppressed_target_later_replies_change_nothing(seeded_db, tmp_path):
    """Ticket §2.3 invariant 2, on the FIRST-reply unsubscribe path: a
    high-confidence unsubscribe suppresses the target (replied -> routed
    -> suppressed); a LATER positive reply on the thread is recorded and
    classified, but changes NOTHING — no transition, no draft, no send,
    and the verdict cannot resurrect the thread."""
    conn = seeded_db
    target_id = _demo_target_id(conn)
    outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
    script = (
        ScriptedTurn(reply_class="unsubscribe", body="Please stop contacting me."),
        ScriptedTurn(reply_class="positive", body="Actually, do send more details."),
    )

    assert not _send(conn, target_id, new_id("run"), outbox).refused

    # Turn 1: unsubscribe at 0.95 — the full suppression path fires.
    _generate(conn, outbox, inbox, script, target_id)
    run1 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run1, [_verdict("unsubscribe", 0.95)])
    assert outcomes[fetched.replies_created[0]] == "suppressed"
    assert _state(conn, target_id) == "suppressed"

    # Turn 2: a positive reply lands on the suppressed thread.
    _generate(conn, outbox, inbox, script, target_id)
    hops_before = _hops(conn, target_id)
    run2 = new_id("run")
    fetched, outcomes = _fetch_and_classify(conn, inbox, run2, [_verdict("positive", 0.9)])
    reply2 = fetched.replies_created[0]
    # The router's terminal guard: the verdict is RECORDED (the row and
    # the step exist), but the outcome names the guard that fired.
    assert outcomes[reply2] == "terminal_no_transition"
    _assert_reply_turn(conn, run2, reply2, reply_class="positive", confidence=0.9,
                       routed_action=FOLLOW_UP_ROUTED_ACTION, review_required=True,
                       target_state="suppressed")
    assert _hops(conn, target_id) == hops_before, "a terminal state is never overridden"
    assert _state(conn, target_id) == "suppressed"
    # The verdict cannot resurrect the thread: the eligible set reads the
    # STATE, not the reply row — a suppressed target is never drafted.
    assert select_draft_eligible_targets(conn, limit=10) == []
    assert len(list(outbox.glob("*.eml"))) == 1, "no send may follow a suppressed target"


# ── 9. The CLI ───────────────────────────────────────────────────────────────


def test_converse_refuses_real_outbound_db(real_outbound_db, tmp_path):
    """The imported demo_seed guard holds for this CLI too: --db pointing
    at the real database is refused (exit 1) before any connection, and
    the real file is byte-identical afterwards.  The real_outbound_db
    fixture creates a stand-in on a fresh clone, so this runs instead of
    dying with FileNotFoundError (ticket H7)."""
    real_db = real_outbound_db  # the file the fixture manages: a created stand-in OR the operator's real DB
    before = hashlib.md5(real_db.read_bytes()).hexdigest()  # the file's fingerprint before the attempt
    code = main([
        "converse", "--db", str(real_db),  # converse shares demo_seed's _guard_violation
        "--persona", "warms_up", "--target", "tgt_whatever",
        "--outbox", str(tmp_path / "outbox"), "--inbox", str(tmp_path / "inbox"),
    ])
    assert code == 1  # the simulator CLI refuses the real database
    assert hashlib.md5(real_db.read_bytes()).hexdigest() == before  # byte-identical: the guard ran before any connect


def test_converse_refuses_unknown_persona_and_lists_them(tmp_path):
    """An unknown persona name is refused with the roster printed by
    --list-personas; no database is needed for either path."""
    assert main(["converse", "--db", str(tmp_path / "nope.db"),
                 "--persona", "not_a_persona", "--target", "tgt_1"]) == 1
    assert main(["converse", "--list-personas"]) == 0


def test_converse_advances_one_turn_and_refuses_unprocessed(tmp_path, switch_path):
    """The demo loop, one step: converse writes turn 1 threaded against
    the real outbound Message-ID, refuses to advance again while that
    file is unprocessed (exit 1), and — once the reply is recorded —
    the persona's exhaustion is a clean refusal too (a 1-turn script)."""
    conn, db_path = _build_seeded_db(tmp_path, switch_path)
    try:
        target_id = _demo_target_id(conn)
        outbox, inbox = tmp_path / "outbox", tmp_path / "inbox"
        assert not _send(conn, target_id, new_id("run"), outbox).refused

        # One-turn script so the exhaustion path is reachable in this test.
        one_turn = (ScriptedTurn(reply_class="positive", body="Tell me more."),)
        code = main([
            "converse", "--db", db_path,
            "--persona", "warms_up", "--target", target_id,
            "--outbox", str(outbox), "--inbox", str(inbox),
        ])
        assert code == 0
        files = sorted(inbox.glob(f"{CONVO_FILE_PREFIX}{target_id}_*.eml"))
        assert len(files) == 1
        # The artifact genuinely threads: its In-Reply-To carries the
        # real outbox Message-ID token.
        msg = BytesParser(policy=email_policy.default).parsebytes(files[0].read_bytes())
        outbox_msg = BytesParser(policy=email_policy.default).parsebytes(
            sorted(outbox.glob("*.eml"))[0].read_bytes()
        )
        assert msg["In-Reply-To"] == outbox_msg["Message-ID"]

        # Advancing again while the turn is unprocessed: refused.
        assert main([
            "converse", "--db", db_path,
            "--persona", "warms_up", "--target", target_id,
            "--outbox", str(outbox), "--inbox", str(inbox),
        ]) == 1

        # Record the reply (real fetch; classification is not needed for
        # the turn counter), then the persona (now exhausted after one
        # turn) refuses cleanly instead of inventing a second message.
        fetched = fetch_inbox(conn, inbox_dir=str(inbox), run_id=new_id("run"), limit=100)
        assert len(fetched.replies_created) == 1
        result = generate_next_turn(
            conn, outbox_dir=str(outbox), inbox_dir=str(inbox),
            persona_script=one_turn, target_id=target_id,
        )
        assert result.refusal_reason == "persona_exhausted"
    finally:
        conn.close()
