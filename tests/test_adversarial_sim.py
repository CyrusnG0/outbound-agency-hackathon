# tests/test_adversarial_sim.py — ticket F1: the adversarial harness.
#
# E2 proved the system behaves when the counterparty is cooperative; this
# file proves it holds when the counterparty is hostile.  A fixed corpus of
# attacks (app/adversarial_sim.py) is driven through the REAL pipeline —
# real fetch_inbox threading, real decide_route (P4/P5), real
# state_machine.transition(), real write_gate writes, real E1 follow-up
# drafting, real send_gate, real app/review.py — with ONLY the three LLM
# agent factories stubbed, the same offline-stand-in pattern
# tests/test_conversation_sim.py applies.
#
# THE TEST ITSELF IS PARAMETRIZED OVER THE CORPUS, so adding an attack to
# app/adversarial_sim.ATTACKS automatically adds coverage.  A breach is
# preserved as @pytest.mark.xfail(strict=True) — it stays green while the
# finding remains recorded; the suite must never paper over it.

import ast  # the structural no-raw-writes test parses app/adversarial_sim.py
import hashlib  # proving the guard test never modified data/outbound.db
import re  # the write-keyword and address-literal patterns for the structural tests
import sqlite3  # the scratch-file test writes a real row before reset
from pathlib import Path  # resolving the module path and the real-database path

import pytest  # fixtures, tmp_path, and the xfail marker

from app.adversarial_sim import (  # the module under test
    ATTACKS,
    RESERVED_TLDS,
    _open_scratch,
    main,
    run_attack,
)
from app.db import reset_scratch_database, scratch_target_violation  # the dialect-aware reset under test

# The write keywords a raw conn.execute() must never carry in adversarial_sim
# — the same vocabulary the demo-seed and conversation-sim structural tests
# refuse.  All writes flow through write_gate.commit / state_machine.transition.
_WRITE_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|REPLACE)\b", re.IGNORECASE)

# The minimum attack ids the ticket requires — the corpus must contain all of
# them, so a silent deletion of one attack fails here.
_REQUIRED_ATTACK_IDS = {
    "A1", "A2", "A3",
    "B1", "B2", "B3",
    "C1", "C2", "C3",
    "D1", "D2", "D3",
}


def _attack_params() -> list:
    """Build the parametrize list: one pytest.param per corpus attack, with
    a strict xfail mark ONLY on attacks the corpus declares to be a known
    breach.  None are marked today: C2 — the one attack that was — was
    closed by F1b (the email_normalized matching key + shared
    normalize_email() fold) and is retained in the corpus as a regression
    guard.  The id is the attack id so a failure names it."""
    params = []
    for attack in ATTACKS:
        marks = []
        if attack.breach:
            marks.append(pytest.mark.xfail(strict=True, reason=attack.breach_reason))
        params.append(pytest.param(attack, marks=marks, id=attack.id))
    return params


# ── 1. Every attack in the corpus, run through the real pipeline ────────────


@pytest.mark.parametrize("attack", _attack_params())
def test_attack_holds(attack, tmp_path):
    """Run one attack and assert the SAFE expectation held: verdict PASS,
    no generic-invariant violations.  C2 (suppression evasion) was the one
    attack formerly marked xfail(strict=True); F1b closed it, so it now
    asserts PASS like every other attack and is retained as a regression
    guard.  The xfail mechanism stays for any FUTURE attack whose corpus
    entry sets breach=True."""
    result = run_attack(
        attack,
        db_path=str(tmp_path / "adv.db"),
        switch_path=str(tmp_path / "kill_switch.json"),
        outbox_dir=str(tmp_path / "outbox"),
        inbox_dir=str(tmp_path / "inbox"),
    )
    assert result.verdict == "PASS", (
        f"{result.attack_id} did not hold: {result.observed}\n"
        f"invariant violations: {result.invariant_violations}\n"
        f"audit: {result.audit}"
    )


# ── 2. The corpus is complete and pinned ─────────────────────────────────────


def test_corpus_contains_every_required_attack():
    """The ticket's minimum attack set is present, with no duplicate ids —
    adding an attack means appending to ATTACKS, deleting one means editing
    this list deliberately."""
    ids = [attack.id for attack in ATTACKS]
    assert len(ids) == len(set(ids)), "attack ids must be unique"
    assert set(ids) >= _REQUIRED_ATTACK_IDS, (
        f"missing required attacks: {_REQUIRED_ATTACK_IDS - set(ids)}"
    )


def test_every_simulated_address_is_on_a_reserved_domain():
    """Ticket §0.5 / verification item 4: every email-like string literal in
    app/adversarial_sim.py must be on an RFC 2606 reserved TLD, and the
    count is asserted to be non-zero (the module actually declares
    addresses, so a vacuous pass is impossible)."""
    path = Path(__file__).resolve().parent.parent / "app" / "adversarial_sim.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    addresses = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            addresses.update(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", node.value))
    # The corpus must actually declare addresses — this is what makes the
    # reserved-domain check non-vacuous.
    assert addresses, "app/adversarial_sim.py declares no email-like strings"
    for address in addresses:
        domain = address.split("@", 1)[-1].lower()
        tld = domain.rsplit(".", 1)[-1]
        assert tld in RESERVED_TLDS, (
            f"simulated address {address!r} is not on a reserved domain "
            f"(allowed: {RESERVED_TLDS})"
        )
    # The count the ticket asks the report to give.
    assert len(addresses) >= 1


# ── 3. Structural guarantee: no raw core-table writes ────────────────────────


def test_adversarial_sim_has_no_raw_core_table_writes():
    """In the spirit of the demo-seed and conversation-sim structural tests:
    every conn.execute() call in app/adversarial_sim.py must carry
    SELECT-only SQL (all writes flow through write_gate.commit /
    state_machine.transition), and the module must import those gates."""
    path = Path(__file__).resolve().parent.parent / "app" / "adversarial_sim.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
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
                f"app/adversarial_sim.py issues a raw write via conn.execute(): "
                f"{node.args[0].value!r}"
            )
    # The positive half: the module actually USES the gates (a module that
    # wrote nothing would pass the negative check vacuously).
    assert "from app.write_gate import" in source
    assert "write_gate_commit" in source
    assert "from app.state_machine import transition" in source


# ── 4. The report CLI refuses the real database ──────────────────────────────


def test_report_refuses_real_outbound_db(real_outbound_db):
    """The shared demo_seed guard holds for the report CLI too: --db pointing
    at data/outbound.db is refused (exit 1) before any connection, and the
    real file is byte-identical afterwards.  The real_outbound_db fixture
    creates a stand-in on a fresh clone, so this runs instead of dying with
    FileNotFoundError (ticket H7)."""
    real_db = real_outbound_db  # the file the fixture manages: a created stand-in OR the operator's real DB
    before = hashlib.md5(real_db.read_bytes()).hexdigest()  # the file's fingerprint before the attempt
    code = main(["report", "--db", str(real_db)])  # report shares demo_seed's _guard_violation
    assert code == 1  # the report CLI refuses the real database
    assert hashlib.md5(real_db.read_bytes()).hexdigest() == before  # byte-identical: the guard ran before any connect


# ── 5. Dialect-aware scratch reset (ticket H2) ───────────────────────────────


def test_reset_scratch_database_deletes_sqlite_file_with_rows(tmp_path):
    """The file-shaped reset preserves today's behaviour: a real SQLite file
    that has committed rows is gone after the call, so the next connect()
    recreates it empty."""
    db = tmp_path / "scratch.db"
    # Write a real row via the stdlib driver so the file is non-trivial —
    # the point is that reset deletes the file, not that it empties tables.
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);")
    conn.execute("INSERT INTO t (v) VALUES ('row');")
    conn.commit()
    conn.close()
    reset_scratch_database(str(db))
    assert not db.exists()


@pytest.mark.parametrize(
    "target",
    [
        "postgresql://u@h/outbound",
        "postgres://u@h:5432/outbound_prod",
        "cloudsql://proj:region:inst/outbound",
        "postgresql://u@h",  # URL with no database name — names nothing to reset
    ],
)
def test_scratch_target_violation_refuses_unmarked_urls(target):
    """A URL without the scratch/test marker (or without any database name)
    must be refused — never assumed to be scratch."""
    msg = scratch_target_violation(target)
    assert msg is not None
    assert "scratch" in msg.lower()
    assert "test" in msg.lower()


@pytest.mark.parametrize(
    "target",
    [
        "postgresql://u@h/outbound_scratch",
        "postgresql://u@h/h1_test",
        "cloudsql://proj:region:inst/outbound_test",
        "postgresql://u@h/OUTBOUND_SCRATCH",  # case-insensitive marker
        "data/demo.db",  # file-shaped: safe, guarded upstream by _guard_violation
    ],
)
def test_scratch_target_violation_allows_scratch_and_file(target):
    """Marked scratch/test URLs and file-shaped targets are safe to reset."""
    assert scratch_target_violation(target) is None


def test_reset_scratch_database_refused_url_raises_without_connecting(monkeypatch):
    """The refusal must happen before any connection: monkeypatch connect() to
    raise if it is ever called, and assert the ValueError is what surfaces."""
    def boom(target):
        raise AssertionError("connected")

    monkeypatch.setattr("app.db.connect", boom)
    with pytest.raises(ValueError, match="scratch"):
        reset_scratch_database("postgresql://u@h/outbound")


def test_report_refuses_unmarked_url_before_any_attack(monkeypatch, capsys):
    """The report CLI refuses a non-scratch URL up front (exit 1, stderr),
    and connect() is never reached — proving no attack ran."""
    def boom(target):
        raise AssertionError("connected")

    monkeypatch.setattr("app.db.connect", boom)
    code = main(["report", "--db", "postgresql://u@h/outbound"])
    assert code == 1
    err = capsys.readouterr().err
    assert "ERROR:" in err
    assert "scratch" in err.lower()
    assert "test" in err.lower()


def test_open_scratch_routes_its_reset_through_the_dialect_aware_helper(tmp_path, monkeypatch):
    """LEAD-ADDED (H2 review). Pin the FIX, not just the guard.

    The worker's own sabotage broke `scratch_target_violation` and six tests
    failed — but reverting `_open_scratch` to the original
    `Path(db_path).unlink(missing_ok=True)` left the ENTIRE suite green. Every
    other test runs on SQLite, where unlink is correct, and the one Postgres
    test skips without a server, so nothing exercised the URL branch at all.
    That is the exact shape of hole found in E1 (raising the follow-up cap
    2 -> 99 stayed green) and F1b (an over-broad normaliser caught by one
    incidental test): the mechanism was covered, the decision was not.

    This test pins the decision offline, with no Postgres required — it spies
    on the helper and asserts `_open_scratch` actually routes through it, so a
    silent regression to the file-only reset fails here.
    """
    calls: list[str] = []

    def spy(target: str) -> None:
        # Record the target, then do the real file-shaped reset so the rest of
        # _open_scratch (connect, apply_schema, seed) still works normally.
        calls.append(target)
        Path(target).unlink(missing_ok=True)

    # Patch the name as _open_scratch resolves it: adversarial_sim imports the
    # helper into its OWN namespace (`from app.db import reset_scratch_database`),
    # so patching app.db would miss the bound name entirely.
    monkeypatch.setattr("app.adversarial_sim.reset_scratch_database", spy)

    db = tmp_path / "scratch.db"
    switch = tmp_path / "kill_switch.json"
    conn = _open_scratch(str(db), str(switch))
    try:
        # The assertion that matters: the reset went through the dialect-aware
        # helper with the target it was given. A revert to Path(...).unlink()
        # never calls the helper and leaves this list empty.
        assert calls == [str(db)]
    finally:
        conn.close()
