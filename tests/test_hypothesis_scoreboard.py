# tests/test_hypothesis_scoreboard.py — the read-only style-hypothesis
# scoreboard report (scripts/hypothesis_scoreboard.py).
#
# WHAT IS PROVEN HERE: compute_scoreboard() reads the audit trail (the steps
# rows the draft persist node logs, plus the replies rows the router
# classifies) and aggregates a per-hypothesis win/loss record.  These tests
# seed REAL rows through the same paths the pipeline uses — write_gate.commit
# for the core-table chain and log_step for the trace — and assert the
# aggregation:
#
#   1. a target whose LATEST reply the router trusted positive
#      (routed_action queue_follow_up_draft) is a WIN for its hypothesis;
#   2. a target whose latest reply the router trusted negative
#      (routed_action close_not_target) is a LOSS;
#   3. a draft logged with hypothesis_id "" (a follow-up draft, or a
#      pre-feature row) is NEVER counted at all, and a drafted target with
#      NO trustworthy verdict (no reply, or a non-trusted routed_action) is
#      counted toward "tested" but is neither a win nor a loss;
#   4. the writer⇄critic loop can log several draft_persist rows for ONE
#      target, and the scoreboard dedupes by target_id so the target
#      contributes exactly one tested count and one verdict.
#
# Every test runs against the shared scratch_db_target fixture, so a plain
# pytest is unchanged (SQLite) and a run with the shared scratch-target env
# set exercises the same assertions against Postgres — the report must run
# identically on both dialects, which is exactly why input_json is parsed
# with json.loads and never with dialect-specific JSON SQL.

import pytest  # fixtures

from app.agents.draft import DRAFT_PERSIST_TOOL_NAME  # the exact tool_name the persist node logs — never re-typed
from app.agents_registry import seed_agent_registry  # the principals — the write gate refuses unregistered writers
from app.db import apply_schema, connect  # fresh per-test database via the shared scratch target
from app.ids import new_id  # unique prefixed ids for every seeded row
from app.tools.log_step import log_step  # the trace writer the persist node itself uses — real steps rows, not raw INSERTs
from app.write_gate import commit  # every seeded core-table row goes through the gate, never a raw INSERT
from scripts.hypothesis_scoreboard import compute_scoreboard  # the aggregation under test — imported, not re-implemented


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conn(scratch_db_target):
    """Fresh SQLite/scratch DB with schema, the seeded principals, and the
    shared offer/account/contact that every target's FK chain needs.

    Targets, messages, replies and steps are added per test via the seed
    helpers below, so each test seeds exactly the shape it asserts on."""
    # scratch_db_target honours the shared scratch-target convention (SQLite
    # by default, Postgres when the env var is set) — the dialect guard in
    # tests/test_dialect_coverage.py fails if a conn fixture hand-rolls its
    # own sqlite path instead of taking this argument.
    c = connect(scratch_db_target)
    apply_schema(c)
    seed_agent_registry(c, run_id="r0", step_id="s0")
    # The shared offer row — the target's offer_id FK must point somewhere.
    commit(
        c, action="insert_offer", table_name="offers", record_id="off_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=("off_1", "acme", 1),
    )
    # The shared account row — the target's account_id FK, and the contacts
    # row's account_id FK.  normalized_domain is UNIQUE, so one account only.
    commit(
        c, action="insert_account", table_name="accounts", record_id="acc_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO accounts (account_id, company_name, domain, normalized_domain,
               created_at, updated_at) VALUES (?,?,?,?,datetime('now'),datetime('now'))""",
        params=("acc_1", "Acme", "acme.test", "acme.test"),
    )
    # The shared contact row — messages.contact_id is NOT NULL, so every
    # target that gets a reply also needs this row.
    commit(
        c, action="insert_contact", table_name="contacts", record_id="con_1",
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO contacts (contact_id, account_id, full_name, email,
               email_verified, created_at, updated_at)
               VALUES (?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=("con_1", "acc_1", "Jane", "jane@acme.test", 1),
    )
    yield c
    c.close()


# ── Seed helpers (TEST SETUP — the same gated paths the pipeline uses) ───────

def _seed_target(c, target_id: str) -> None:
    """Insert one target row linked to the shared account/contact/offer."""
    commit(
        c, action="insert_target", table_name="targets", record_id=target_id,
        payload={}, run_id="r0", step_id="s0", actor="system", agent_id="system",
        sql="""INSERT INTO targets (target_id, account_id, contact_id, offer_id,
               source, state, created_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))""",
        params=(target_id, "acc_1", "con_1", "off_1", "csv", "awaiting_review"),
    )


def _seed_message(c, *, target_id: str, message_id: str) -> None:
    """Insert one outbound messages row linking the target to a reply."""
    commit(
        c, action="insert_message", table_name="messages", record_id=message_id,
        payload={"status": "dry_run_sent"}, run_id="r0", step_id="s0",
        actor="system", agent_id="system",
        sql="""INSERT INTO messages (message_id, target_id, contact_id, direction,
               status, created_at) VALUES (?,?,?,?,?,datetime('now'))""",
        params=(message_id, target_id, "con_1", "outbound", "dry_run_sent"),
    )


def _seed_reply(c, *, message_id: str, reply_id: str, routed_action: str) -> None:
    """Insert one replies row whose routed_action is the verdict under test.

    Only routed_action matters to the scoreboard; classification/confidence
    stay NULL exactly as a reply that has not been judged yet would carry."""
    commit(
        c, action="insert_reply", table_name="replies", record_id=reply_id,
        payload={"routed_action": routed_action}, run_id="r0", step_id="s0",
        actor="system", agent_id="system",
        sql="""INSERT INTO replies (reply_id, message_id, thread_id, from_email,
               raw_text, redacted_text, classification, confidence, routed_action,
               created_at) VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))""",
        params=(reply_id, message_id, message_id, "jane@acme.test", "body", "body",
                None, None, routed_action),
    )


def _seed_draft_persist(c, *, target_id: str, hypothesis_id: str) -> None:
    """Insert one draft_persist steps row via the REAL trace writer — the
    exact shape the persist node logs, with hypothesis_id in input_json."""
    log_step(
        c, run_id="r1", step_id=new_id("step"), target_id=target_id,
        tool_name=DRAFT_PERSIST_TOOL_NAME, agent_id="draft_writer",
        input_data={"stage": "draft_persist", "revision_number": 1,
                    "hypothesis_id": hypothesis_id},
        output_data={}, status="success",
    )


# ── 1. A trusted positive reply is a WIN ─────────────────────────────────────

def test_scoreboard_counts_a_win(conn):
    """A target whose latest reply the router trusted POSITIVE
    (routed_action queue_follow_up_draft) is a WIN for the hypothesis that
    drafted its first-touch email: tested=1, wins=1, losses=0, score=1."""
    _seed_target(conn, "tgt_win")
    _seed_message(conn, target_id="tgt_win", message_id="msg_win")
    _seed_reply(conn, message_id="msg_win", reply_id="rpl_win",
                routed_action="queue_follow_up_draft")
    _seed_draft_persist(conn, target_id="tgt_win", hypothesis_id="H3")

    board = compute_scoreboard(conn)

    assert board["H3"] == {"tested": 1, "wins": 1, "losses": 0, "score": 1}


# ── 2. A trusted negative reply is a LOSS ────────────────────────────────────

def test_scoreboard_counts_a_loss(conn):
    """A target whose latest reply the router trusted NEGATIVE
    (routed_action close_not_target) is a LOSS for the drafting hypothesis:
    tested=1, wins=0, losses=1, score=-1."""
    _seed_target(conn, "tgt_loss")
    _seed_message(conn, target_id="tgt_loss", message_id="msg_loss")
    _seed_reply(conn, message_id="msg_loss", reply_id="rpl_loss",
                routed_action="close_not_target")
    _seed_draft_persist(conn, target_id="tgt_loss", hypothesis_id="H3")

    board = compute_scoreboard(conn)

    assert board["H3"] == {"tested": 1, "wins": 0, "losses": 1, "score": -1}


# ── 3. Empty hypothesis is never counted; no-verdict targets are "tested" ────

def test_scoreboard_ignores_empty_hypothesis_and_untrusted_verdicts(conn):
    """Two things must never move the win/loss numbers:

    - a draft logged with hypothesis_id "" (a follow-up draft, or a database
      seeded before the feature existed) contributes to NO hypothesis's
      counts at all — not even "tested";
    - a drafted target with NO trustworthy verdict — here, a target with no
      reply row at all, and a separate target whose latest reply carried a
      non-trusted routed_action (schedule_reminder, not a win or loss) — is
      counted toward "tested" but is NEITHER a win NOR a loss.

    This is the P4 confidence-floor consequence: only a verdict the router
    acted on may move the number, and a missing/untrusted verdict is honest
    data that the summary line reports as "no trustworthy verdict yet"."""
    # A follow-up / pre-feature draft: hypothesis_id "" must never count.
    _seed_target(conn, "tgt_nohyp")
    _seed_draft_persist(conn, target_id="tgt_nohyp", hypothesis_id="")
    # A drafted target with NO reply row at all: tested but no verdict.
    _seed_target(conn, "tgt_noreply")
    _seed_draft_persist(conn, target_id="tgt_noreply", hypothesis_id="H3")
    # A drafted target whose latest reply is a non-trusted action: also
    # tested but no verdict (H5 stays out of H3's row).
    _seed_target(conn, "tgt_untrusted")
    _seed_message(conn, target_id="tgt_untrusted", message_id="msg_untrusted")
    _seed_reply(conn, message_id="msg_untrusted", reply_id="rpl_untrusted",
                routed_action="schedule_reminder")
    _seed_draft_persist(conn, target_id="tgt_untrusted", hypothesis_id="H5")

    board = compute_scoreboard(conn)

    assert board["H3"] == {"tested": 1, "wins": 0, "losses": 0, "score": 0}
    assert board["H5"] == {"tested": 1, "wins": 0, "losses": 0, "score": 0}
    # The empty-hypothesis target contributes to NO hypothesis's tested count
    # anywhere — only the two real-hypothesis targets above are tested.
    assert sum(stats["tested"] for stats in board.values()) == 2


# ── 4. Multiple revisions of one target are deduped ──────────────────────────

def test_scoreboard_dedupes_multiple_revisions_per_target(conn):
    """The writer⇄critic loop can log several draft_persist rows for ONE
    target (revision attempts before the draft passes).  The scoreboard must
    dedupe by target_id so the target contributes exactly ONE tested count
    and exactly ONE verdict — the hypothesis_id is the same across every
    revision of one drafting run, so last-write-wins changes nothing here."""
    _seed_target(conn, "tgt_dup")
    _seed_message(conn, target_id="tgt_dup", message_id="msg_dup")
    _seed_reply(conn, message_id="msg_dup", reply_id="rpl_dup",
                routed_action="queue_follow_up_draft")
    _seed_draft_persist(conn, target_id="tgt_dup", hypothesis_id="H4")  # revision 1
    _seed_draft_persist(conn, target_id="tgt_dup", hypothesis_id="H4")  # revision 2 (same run)

    board = compute_scoreboard(conn)

    # Two steps rows, but ONE target: tested=1 (not 2) and wins=1 (not 2).
    assert board["H4"] == {"tested": 1, "wins": 1, "losses": 0, "score": 1}
