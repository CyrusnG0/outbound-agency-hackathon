"""Tests for the demo seed (ticket D3a): the placeholder-data module that
makes the full loop reachable so the demo can show the REAL gates running.

Covered here, one test per ticket requirement:

- the reserved-domain rule: every seeded email address is on an RFC 2606
  reserved TLD (.test/.invalid/.example) — parsed from the DB, not trusted
  from the data block;
- the real-database guard: seed and replies refuse data/outbound.db (both
  via --db and via OUTBOUND_DB_TARGET) and never open the file;
- idempotency: re-running seed leaves every row count stable;
- the state walk: the seeded target's state_transitions rows form the
  real hop sequence new → researched → scored → drafted → awaiting_review
  → approved;
- the structural guarantee (in the spirit of the console tests): no raw
  core-table INSERT/UPDATE exists in app/demo_seed.py — every
  conn.execute() call in that file carries SELECT-only SQL, and the module
  imports the write gate and the state machine;
- the full-loop proof: seed → real send_email (the 19-check preflight) →
  replies subcommand → the generated In-Reply-To values match the real
  outbox Message-IDs, string-compared;
- the console's DEMO DATA banner appears on a seeded database and stays
  absent on an unseeded one.
"""

import ast  # the structural test parses app/demo_seed.py the way the console tests parse app/console/
import hashlib  # proving the guard test never modified data/outbound.db
import re  # the write-keyword pattern for the structural test
from email import policy as email_policy  # parsing the generated inbox .eml files
from email.parser import BytesParser  # RFC-5322 parsing — reading only, never transport
from pathlib import Path  # resolving the real-database path and the module path

import pytest  # fixtures and the monkeypatch fixture for the env-var guard test
from fastapi.testclient import TestClient  # the console banner tests

from app.console.app import app  # the module-level console instance (the same one the deploy uses)
from app.db import apply_schema, connect  # building scratch databases the same way the CLIs do
from app.demo_seed import DEMO_REPLY_PREFIX, DEMO_SOURCE, RESERVED_TLDS, main  # the module under test
from app.ids import new_id  # the send run id in the full-loop test
from app.tools.send_email import send_email  # the REAL DRY_RUN send — deterministic, no model call

# The write keywords a raw conn.execute() must never carry in demo_seed —
# the same vocabulary the console's SELECT-only test refuses.
_WRITE_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|REPLACE)\b", re.IGNORECASE)


@pytest.fixture
def seeded_db(tmp_path):
    """Run the seed subcommand against a scratch database and return the
    path — the same invocation the operator runs (main() directly, so the
    guard and CLI plumbing are exercised, not just the inner function)."""
    db = str(tmp_path / "demo.db")
    code = main(["seed", "--db", db])
    assert code == 0, "the seed subcommand must exit 0 on a fresh scratch DB"
    return db


def _row_counts(db: str) -> dict[str, int]:
    """Count every row family the seed writes — the idempotency test
    compares two runs on these numbers."""
    conn = connect(db)
    counts = {
        table: conn.execute(f"SELECT COUNT(*) AS n FROM {table};").fetchone()["n"]
        for table in (
            "offers", "accounts", "contacts", "targets", "signals",
            "message_draft_versions", "policy_decisions", "review_decisions",
            "state_transitions",
        )
    }
    conn.close()
    return counts


# ── The real-database guard ───────────────────────────────────────────────────


def test_guard_refuses_real_outbound_db(real_outbound_db, tmp_path, monkeypatch):
    """The guard must refuse data/outbound.db — via --db AND via the
    OUTBOUND_DB_TARGET env convention — and must never open (let alone
    modify) the real file.  The md5 before/after is the proof.  On a fresh
    clone (no data/outbound.db) the real_outbound_db fixture creates a
    stand-in at that exact path, so this runs instead of dying with
    FileNotFoundError (ticket H7)."""
    real_db = real_outbound_db  # the file the fixture manages: a created stand-in OR the operator's real DB
    before = hashlib.md5(real_db.read_bytes()).hexdigest()  # the file's fingerprint before any attempt
    # --db pointing straight at the real database: refused with exit 1.
    assert main(["seed", "--db", str(real_db)]) == 1
    # The replies subcommand shares the guard: refused identically.
    assert main(["replies", "--db", str(real_db)]) == 1
    # The env convention: even with a scratch --db, an environment that
    # points at the real database must refuse (the demo tool has zero
    # ways to reach production data).
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(real_db))
    assert main(["seed", "--db", str(tmp_path / "scratch.db")]) == 1
    # The real file is byte-identical — the guard ran before any connect.
    after = hashlib.md5(real_db.read_bytes()).hexdigest()
    assert before == after


def test_seed_accepts_scratch_database(tmp_path):
    """The guard's flip side: a scratch path passes and the seed runs."""
    assert main(["seed", "--db", str(tmp_path / "ok.db")]) == 0


# ── The seeded data ───────────────────────────────────────────────────────────


def test_seeded_emails_all_reserved_domains(seeded_db):
    """Every seeded contact email must be on an RFC 2606 reserved TLD —
    parsed from the database, asserting the TLD set directly (the hard
    requirement the ticket pins to a constant plus a test)."""
    conn = connect(seeded_db)
    emails = [
        row["email"] for row in conn.execute("SELECT email FROM contacts;").fetchall()
    ]
    conn.close()
    assert emails, "the seed must create contacts"
    # Parse each address's TLD (the final label of the domain half) and
    # assert the whole set sits inside the reserved vocabulary.
    tlds = set()
    for email in emails:
        domain = email.split("@", 1)[-1].lower()
        tlds.add(domain.rsplit(".", 1)[-1])
    assert tlds and tlds <= set(RESERVED_TLDS), (
        f"seeded emails use non-reserved TLDs: {tlds} (allowed: {RESERVED_TLDS})"
    )


def test_seed_is_idempotent(seeded_db):
    """Running seed twice must not duplicate rows or crash — the sentinel
    (targets.source='demo_seed') makes the second run a no-op."""
    first = _row_counts(seeded_db)
    code = main(["seed", "--db", seeded_db])  # the second run
    assert code == 0
    second = _row_counts(seeded_db)
    assert second == first, f"row counts changed on re-seed: {first} -> {second}"


def test_state_transition_walk_matches_real_hop_sequence(seeded_db):
    """Each seeded target's state_transitions rows must form the REAL hop
    sequence — new → researched → scored → drafted → awaiting_review →
    approved — in insertion order (insert_seq, the C1 ordering fix), with
    the final hop attributable to the operator's recorded approval."""
    conn = connect(seeded_db)
    target_ids = [
        row["target_id"]
        for row in conn.execute(
            "SELECT target_id FROM targets WHERE source=?;", (DEMO_SOURCE,)
        ).fetchall()
    ]
    assert target_ids, "the seed must create demo targets"
    expected = [
        ("new", "researched"),
        ("researched", "scored"),
        ("scored", "drafted"),
        ("drafted", "awaiting_review"),
        ("awaiting_review", "approved"),
    ]
    for target_id in target_ids:
        # insert_seq ordering (with the NULL-last prefix the console uses)
        # gives the chronological hop list — the audit trail's own order.
        hops = [
            (row["previous_state"], row["new_state"])
            for row in conn.execute(
                "SELECT previous_state, new_state FROM state_transitions "
                "WHERE target_id=? "
                "ORDER BY (insert_seq IS NULL) DESC, insert_seq, created_at;",
                (target_id,),
            ).fetchall()
        ]
        assert hops == expected, f"target {target_id} walked {hops}"
        # The target must actually BE approved — the walk's destination.
        state = conn.execute(
            "SELECT state FROM targets WHERE target_id=?;", (target_id,)
        ).fetchone()["state"]
        assert state == "approved"
    conn.close()


# ── The structural guarantee: no raw core-table writes ───────────────────────


def test_demo_seed_has_no_raw_core_table_writes():
    """In the spirit of the console's structural tests: every
    conn.execute() call in app/demo_seed.py must carry SELECT-only SQL
    (all writes flow through write_gate.commit / state_machine.transition
    / log_step), and the module must import those gates.  This is the test
    that proves the gate was not bypassed — a behavioural test could pass
    a seed that simply has not written raw SQL yet."""
    path = Path(__file__).resolve().parent.parent / "app" / "demo_seed.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    # Walk every call: any .execute(...) whose first argument is a string
    # literal containing a write keyword is a raw write path.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            assert not _WRITE_SQL.search(node.args[0].value), (
                f"app/demo_seed.py issues a raw write via conn.execute(): "
                f"{node.args[0].value!r}"
            )
    # The positive half: the module actually USES the gates (a seed that
    # simply wrote nothing would pass the negative check vacuously).
    assert "from app.write_gate import commit" in source
    assert "from app.state_machine import transition" in source


# ── The full loop: seed → real send → real threading ─────────────────────────


def test_full_loop_send_and_replies_thread(seeded_db, tmp_path):
    """Ticket §4.3's proof, as a test: the seeded preconditions let the
    REAL send gate allow every approved target (no refusals), the DRY_RUN
    send writes one .eml per target, and the replies subcommand generates
    inbox files whose In-Reply-To values match the real outbox Message-IDs
    (string comparison).  No model call anywhere in this test."""
    conn = connect(seeded_db)
    target_ids = [
        row["target_id"]
        for row in conn.execute("SELECT target_id FROM targets WHERE state='approved';").fetchall()
    ]
    assert len(target_ids) == 3, "the seed must leave exactly 3 approved targets"
    outbox = tmp_path / "outbox"
    results = [
        send_email(conn, target_id=tid, run_id=new_id("run"), outbox_dir=str(outbox))
        for tid in target_ids
    ]
    # Every send must be ALLOWED — the seeded preconditions satisfy the
    # real 19-check preflight; a refusal here is a seeded precondition
    # that the real gate actually wants (never loosen the check).
    for result in results:
        assert not result.refused, (
            f"target {result.target_id} refused by the real gate: "
            f"{result.refusal_reason}"
        )
    conn.close()
    # One artifact per allowed send, named by message id.
    outbox_files = sorted(outbox.glob("*.eml"))
    assert len(outbox_files) == 3

    # The replies subcommand, run exactly as the demo runs it.
    inbox = tmp_path / "inbox"
    code = main([
        "replies", "--db", seeded_db,
        "--outbox", str(outbox), "--inbox", str(inbox),
    ])
    assert code == 0
    generated = sorted(inbox.glob(f"{DEMO_REPLY_PREFIX}*.eml"))
    assert len(generated) == 3, "one threaded reply per outbound message"

    # Parse both sides and string-compare: every generated In-Reply-To
    # token must equal a real outbox Message-ID token, and each outbox
    # message must be answered exactly once.
    parser = BytesParser(policy=email_policy.default)
    outbox_tokens = {
        parser.parsebytes(path.read_bytes())["Message-ID"] for path in outbox_files
    }
    reply_tokens = {
        parser.parsebytes(path.read_bytes())["In-Reply-To"] for path in generated
    }
    assert len(outbox_tokens) == 3, "every outbox .eml carries a Message-ID"
    assert reply_tokens == outbox_tokens, (
        f"generated In-Reply-To values do not match the real outbox "
        f"Message-IDs: {reply_tokens} vs {outbox_tokens}"
    )
    # Senders on reserved domains only — the generated replies' From
    # addresses must parse to a reserved TLD.
    for path in generated:
        msg = parser.parsebytes(path.read_bytes())
        sender = msg["From"]
        assert "@" in sender, f"generated reply {path.name} has no From address"
        tld = sender.split("@", 1)[-1].split(">", 1)[0].rsplit(".", 1)[-1].lower()
        assert tld in RESERVED_TLDS, (
            f"generated reply {path.name} sender {sender!r} is not on a "
            f"reserved domain"
        )


# ── The console's honesty surface ────────────────────────────────────────────


def test_console_shows_demo_banner_on_seeded_db(seeded_db, monkeypatch):
    """The console must show the unmissable DEMO DATA indicator when the
    database is a seeded one — detected by a single SELECT on the seed's
    steps marker, with no new console import."""
    monkeypatch.setenv("OUTBOUND_DB_TARGET", seeded_db)
    # H11: the console now requires auth — supply the secret AND a credential
    # on every request so these tests actually reach the handler.  The auth
    # layer itself is tested in tests/test_console_auth.py; here the
    # credential is just the key that opens the door.
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", "test-console-secret")
    client = TestClient(app)
    _auth = {"X-Internal-API-Key": "test-console-secret"}
    # The banner div on the index, the detail page, and the review
    # surface — the div is what only renders when demo_data is true (the
    # words also appear in CSS comments, so assert on the element).
    assert 'class="demo-banner"' in client.get("/", headers=_auth).text
    target_id = connect(seeded_db).execute(
        "SELECT target_id FROM targets WHERE source=? LIMIT 1;", (DEMO_SOURCE,)
    ).fetchone()["target_id"]
    assert 'class="demo-banner"' in client.get(f"/targets/{target_id}", headers=_auth).text
    assert 'class="demo-banner"' in client.get("/review/queue", headers=_auth).text


def test_console_shows_no_demo_banner_on_unseeded_db(tmp_path, monkeypatch):
    """The flip side: an ordinary database (schema applied, no seed
    marker) renders no banner — the indicator must not cry wolf on real
    data."""
    db = str(tmp_path / "plain.db")
    conn = connect(db)
    apply_schema(conn)
    conn.close()
    monkeypatch.setenv("OUTBOUND_DB_TARGET", db)
    # H11: supply the secret AND a credential (see the seeded-banner test).
    # Without the credential this would 503, and the assertion below would
    # pass vacuously on the error body — this test must reach the handler to
    # prove the banner is truly absent on real data.
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", "test-console-secret")
    client = TestClient(app)
    # The banner element must not render on a non-seeded database.
    resp = client.get("/", headers={"X-Internal-API-Key": "test-console-secret"})
    assert resp.status_code == 200, "expected a rendered index page, not an auth/error body"
    assert 'class="demo-banner"' not in resp.text
