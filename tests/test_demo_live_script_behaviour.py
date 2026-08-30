"""
tests/test_demo_live_script_behaviour.py — structural guard for scripts/demo_live.sh (ticket D2).

This is the D2 companion to tests/test_deploy_script_behaviour.py and
tests/test_demo_replay_script.py.  Almost everything it asserts is TEXT-LEVEL
— the deploy and replay scripts shell out to free tools (gcloud/docker/curl,
cp/cmp), so running them under stubs is cheap; the live demo script shells out
to `python -m app.phase1_cli` / `draft_cli` / `reply_cli`, which make REAL,
billable Gemini calls.  The D2 ticket's explicit requirement is that these
tests must never invoke those CLIs and must never spend money in CI.

There is ONE execution-level exception (ticket D2-fix Finding 2): a semantic
subprocess test that runs `finish` against a scratch db holding an
`awaiting_review` target to prove the human-gate refusal actually fires BEFORE
send_cli.  That test is safe by construction — the gate's refusal exits before
the send stage starts, and PYTHON_BIN is stubbed with a fake interpreter that
executes only the stdin-heredoc helper the gates use and refuses any
`-m app.*` invocation with exit 99, so a real CLI can never be reached even
under a regression that broke the gate.

What the file pins (each is a documented D2 property, not an implementation
detail):
  - the default scratch db is data/live_demo.db, NEVER the operator's real
    data/outbound.db or the seeded D1/D3a data/demo.db;
  - the real-database guard (_guard_db) is the FIRST DB-touching action in
    BOTH setup() and finish() — it runs before any command that reads or
    mutates the database;
  - in finish(), the human-gate guard (_require_approved) runs BEFORE
    send_cli — asserted by string OFFSET, not mere presence, so a future edit
    that moved the send above the gate fails;
  - the script never names/imports a mail transport (the same banned-module
    list tests/test_send_gate.py enforces for app/), and every python
    invocation targets `python -m app.*` or a stdin heredoc helper — never a
    third-party tool;
  - data/demo_live_target.csv is a real single-row import: exactly one data
    row, whose company_name/domain are verbatim from data/hk_therapy_targets.csv
    (parsed with the stdlib csv module, never hand-parsed strings), and whose
    contact_email sits on an RFC 2606 reserved TLD (the same RESERVED_TLDS
    constant demo_seed/conversation_sim enforce);
  - the script sets `set -euo pipefail` near the top so a failing stage aborts
    the whole run instead of silently continuing;
  - the script never invokes a raw transport/network tool directly — no MTA
    client, no ssh/telnet, no bash /dev/tcp, no bare interpreter with an
    optional absolute path prefix (ticket D2-fix2 Issue A);
  - finish() pins send_cli to `--limit 1` so a batch dry-run send can never
    fire, and setup() refuses a multi-row --csv so a live demo db is
    single-target by construction (ticket D2-fix2 Issue B).
"""

import csv  # parsing both CSVs structurally — never hand-splitting strings
import os  # subprocess env: build a hermetic env for the semantic gate test
import re  # the text-offset / banned-module regex checks
import subprocess  # the one semantic test: run finish against a scratch db
import sys  # the fake interpreter's shebang must be the interpreter running pytest
from pathlib import Path  # resolving the repo root and the default db path shape

import pytest  # fixtures for the script text and function-body segments

# The reserved, non-routable TLDs every demo email address must use.  Imported
# from app.demo_seed — the single constant the seed AND conversation_sim tests
# already rely on — rather than re-hand-rolled, so the rule cannot drift
# between the two demo surfaces (ticket: reuse the existing list/helper).
from app.db import apply_schema, connect  # the semantic gate test seeds a real scratch db
from app.demo_seed import RESERVED_TLDS

# The repo root and the REAL files — the tests assert on what is actually
# committed, never on a copy.
ROOT = Path(__file__).resolve().parent.parent
DEMO_LIVE_SCRIPT = ROOT / "scripts" / "demo_live.sh"
DEMO_LIVE_CSV = ROOT / "data" / "demo_live_target.csv"
REAL_TARGETS_CSV = ROOT / "data" / "hk_therapy_targets.csv"

# The two database files the live demo must NEVER default to: the operator's
# real run data (data/outbound.db) and the seeded D1/D3a demo database
# (data/demo.db).  A live run defaulting to either would silently mutate real
# state or build on synthetic state — the two things the demo must not do.
_FORBIDDEN_DB_BASENAMES = ("outbound.db", "demo.db")

# The mail-transport module roots no part of the repo may import or name.
# Mirror of tests/test_send_gate.py's _FORBIDDEN_TRANSPORT_MODULES: stdlib
# SMTP/IMAP/POP transports, the Gmail API client stack, and the common
# third-party mail SDKs.  This script shells out to `python -m app.*` only; if
# a future edit ever names one of these, it is the first step toward a real
# send, which the repo forbids outright.
_MAIL_TRANSPORT_MODULES = (
    "smtplib", "aiosmtplib", "poplib", "imaplib", "smtpd",
    "googleapiclient", "google_auth_oauthlib", "google.oauth2",
    "yagmail", "redmail", "sendgrid", "mailgun", "mailjet",
    "exchangelib", "imapclient", "imbox", "emails",
)

# Word-shaped raw transport/network tools the script must never invoke
# directly (round-2 review Issue A, ticket D2-fix2).  A ``\b`` word boundary on
# both sides keeps `mail` from matching inside `email`/`sendmail`/`outbox`; the
# `mail` entry itself is what catches the BSD `mail -s hi a@b.com` client.
_RAW_TRANSPORT_WORD_TOKENS = (
    "sendmail", "mailx", "mail", "ssmtp", "msmtp",
    "postfix", "ssh", "telnet",
)

# Non-word-shaped forms matched as literal substrings (no ``\b``): `/dev/tcp`
# is bash's built-in TCP redirection (`exec 3<>/dev/tcp/...` — preceded by the
# `<>` redirection operator, never a word char, so a word boundary would MISS
# it), and `openssl s_client` is the exact TLS-client phrase — a bare `openssl`
# is LEGAL (this script itself mints a console key with `openssl rand -hex 32`),
# so only the `s_client` mode is banned.
_RAW_TRANSPORT_LITERAL_TOKENS = ("/dev/tcp", "openssl s_client")


# ── Comment stripping for order/presence assertions (ticket D2-fix F1) ────────


def _strip_comments(text: str) -> str:
    """Return ``text`` with every bash comment (``#`` to end-of-line) removed.

    The order/presence assertions must run against the code that would actually
    EXECUTE, not the raw text: a sabotage that comments out a guard call
    (`# _require_approved ...`) leaves the STRING in the file, so an
    offset/`.index()`/`.count()` check on raw text would still pass while the
    script would actually skip the guard.  Stripping ``#`` to end-of-line makes
    a commented-out call disappear from the text the assertions read, so the
    sabotage fails loudly here instead of silently unguarding the script.

    A per-line split is safe for THIS file because no ``#`` sits inside a bash
    double-quoted or single-quoted string anywhere in scripts/demo_live.sh
    (verified by inspection).  The only other ``#`` characters live inside the
    ``<<'PY'`` python heredocs, where they are python comments — stripping them
    is harmless because none of the patterns the offset checks search for
    (`"${PYTHON_BIN}"`, ``_guard_db "${db}"``, ``app.send_cli``, ...) can
    appear inside heredoc content (they are bash expansions / bash call
    syntax, never python).
    """
    return "\n".join(line.split("#", 1)[0] for line in text.split("\n"))


# ── Fixtures: the real script text and its two subcommand bodies ──────────────
# Reading the committed file once at module scope keeps every test cheap and
# means a test failure points at the real script, not a fixture copy.


@pytest.fixture(scope="module")
def script_text() -> str:
    """The committed demo_live.sh text — the ground truth every text assertion reads."""
    return DEMO_LIVE_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def segments(script_text: str) -> dict:
    """The setup()/finish()/_require_approved() bodies, sliced then comment-stripped.

    Each segment starts at its function's `name() {` line and ends right before
    the next top-level section, so offset comparisons inside a segment are
    within the one function and cannot accidentally span into the other.

    Slicing happens on RAW text first (the marker offsets come from the
    committed file exactly as written), THEN each slice is passed through
    _strip_comments, so every order/presence assertion reads only the code that
    would actually execute.  A sabotage that comments out a guard call leaves
    the string in the raw text — and therefore in the file — but the stripped
    text the assertions search no longer contains it (ticket D2-fix Finding 1).
    """
    setup_start = script_text.index("setup() {")      # the setup() definition
    finish_start = script_text.index("finish() {")    # the finish() definition
    gate_start = script_text.index("_require_approved() {")  # the human-gate function
    target_id_state_start = script_text.index("_target_id_state() {")  # next function def
    dispatch_start = script_text.index("# ── Dispatch")  # the bottom dispatch block
    return {
        "setup": _strip_comments(script_text[setup_start:finish_start]),
        "finish": _strip_comments(script_text[finish_start:dispatch_start]),
        "gate": _strip_comments(script_text[gate_start:target_id_state_start]),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _first_db_touching_index(segment: str) -> int:
    """Earliest offset in a function body of anything that reads or mutates the db.

    Three shapes count, all of which the script uses to touch the database:
      1. the human-gate guard `_require_approved "${db}" "${target}"` — a
         SELECT against the target row;
      2. an `rm -f "${db}"` — the setup --fresh deletion of the db file;
      3. any `"${PYTHON_BIN}"` line carrying `--db` (every stage CLI:
         phase1/draft/send/reply/conversation_sim-with-db) or the stdin-heredoc
         db arg (`"${PYTHON_BIN}" - "${db}"` — the python helper readouts).
    The `--list-personas` passthrough carries NEITHER `--db` nor the heredoc
    arg, so it is correctly NOT counted as DB-touching (it only prints the
    persona roster and exits).
    """
    candidates: list[int] = []
    gate = segment.find('_require_approved "${db}" "${target}"')
    if gate != -1:
        candidates.append(gate)  # the human gate reads the target's state
    rm = segment.find('rm -f "${db}"')
    if rm != -1:
        candidates.append(rm)  # the --fresh deletion mutates the db file
    start = 0
    while True:
        i = segment.find('"${PYTHON_BIN}"', start)
        if i == -1:
            break  # no more python invocations in this body
        line_end = segment.find("\n", i)
        line = segment[i:line_end]
        if "--db" in line or '- "${db}"' in line:
            candidates.append(i)  # a stage CLI / heredoc helper touching the db
        start = i + 1
    assert candidates, "function body has no DB-touching command to guard against"
    return min(candidates)  # the FIRST thing that could touch the db


def _read_csv(path: Path) -> list[list[str]]:
    """Parse a committed CSV with the stdlib csv module — structural, never
    hand-split strings (a quoted comma would silently break a naive split)."""
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def _write_fake_python(tmp_path) -> Path:
    """Write the hermetic fake interpreter the semantic subprocess tests share.

    `python - <db> [<target>]` (the stdin-heredoc form _guard_db /
    _require_approved / setup's CSV row-count check use) executes the heredoc
    from stdin with the trailing argv; every `python -m app.*` invocation — a
    real CLI like send_cli / reply_cli / phase1_cli — prints the marker and
    exits 99 instead of running.  Shared so every semantic test builds the
    identical stub: no real CLI can ever be reached even under a regression
    that broke the guard being tested.
    """
    fake_python = tmp_path / "fake_python.py"
    fake_python.write_text(
        "#!{}\n"
        "import sys\n"
        "if len(sys.argv) >= 2 and sys.argv[1] == '-':\n"
        "    sys.argv = sys.argv[1:]\n"
        "    exec(compile(sys.stdin.read(), '<demo_live_heredoc>', 'exec'))\n"
        "    raise SystemExit(0)\n"
        "print('FAKE_PYTHON_BLOCKED_MODULE=' + ' '.join(sys.argv[1:]), file=sys.stderr)\n"
        "raise SystemExit(99)\n".format(sys.executable),
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


# ── 1. The default database is a scratch file, never a real/seeded one ────────


def test_db_default_is_scratch_never_real_or_seeded(script_text: str) -> None:
    """DB_DEFAULT must be the dedicated scratch file data/live_demo.db.

    The documented env-override shape is
    `DB_DEFAULT="${DB_DEMO_LIVE:-${REPO_ROOT}/data/live_demo.db}"`; extracting
    the literal fallback path and comparing its BASENAME (not a substring,
    which would false-positive: `live_demo.db` CONTAINS `demo.db`) pins that
    the default is neither the operator's real run data nor the seeded demo db.
    Even though _guard_db would refuse those two at runtime, the default itself
    must never point at them — fail-loud at the source, not at the gate.
    """
    m = re.search(
        r'^DB_DEFAULT="\$\{DB_DEMO_LIVE:-\$\{REPO_ROOT\}/(?P<path>[^"}]+)\}"',
        script_text,
        re.MULTILINE,
    )
    assert m, (
        "DB_DEFAULT must follow the documented env-override shape "
        r'"${DB_DEMO_LIVE:-${REPO_ROOT}/data/live_demo.db}"'
    )
    default = Path(m.group("path"))
    assert default.parts[0] == "data", (
        f"DB_DEFAULT literal {default!s} is not under data/ — a live demo must "
        "write to a repo scratch path"
    )
    assert default.name not in _FORBIDDEN_DB_BASENAMES, (
        f"DB_DEFAULT resolves to {default.name!r}, but the live demo must never "
        f"default to the operator's real {_FORBIDDEN_DB_BASENAMES[0]!r} or the "
        f"seeded {_FORBIDDEN_DB_BASENAMES[1]!r}"
    )


# ── 2. _guard_db is the FIRST DB-touching action in BOTH subcommands ──────────


def test_guard_db_precedes_all_db_io_in_setup(segments: dict) -> None:
    """setup() runs the real-database guard before ANY db read or mutation.

    The D2 safety contract: an operator who passes --db data/outbound.db (or
    data/demo.db) by mistake is refused before a single connect or CLI touches
    the file.  The guard is also asserted to precede the --fresh `rm -f`, which
    deletes the db file — a mutation that must never happen on a guarded path.
    """
    seg = segments["setup"]
    assert seg.count('_guard_db "${db}"') == 1, (
        "setup() must call the real-database guard exactly once"
    )
    # Belt-and-braces on top of comment-stripping: the guard call must sit at
    # the start of a line (only whitespace before it), so a commented-out
    # `# _guard_db ...` cannot match even if the strip ever regressed.
    assert re.search(r"^\s*_guard_db\b", seg, re.MULTILINE), (
        "setup() must CALL _guard_db at the start of a line — a comment "
        "containing the name is not a call"
    )
    guard = seg.index('_guard_db "${db}"')
    first_db = _first_db_touching_index(seg)
    assert guard < first_db, (
        f"setup() runs a DB-touching command at offset {first_db} BEFORE the "
        f"real-database guard at {guard} — the guard must be the first thing "
        "that could touch the database."
    )


def test_guard_db_precedes_all_db_io_in_finish(segments: dict) -> None:
    """finish() runs the real-database guard before ANY db read or mutation.

    Same contract as setup, applied to finish: the guard must come before the
    human-gate SELECT (_require_approved) and before the send/reply python
    invocations, so a --db mistake is refused even on the send path.
    """
    seg = segments["finish"]
    assert seg.count('_guard_db "${db}"') == 1, (
        "finish() must call the real-database guard exactly once"
    )
    # Same belt-and-braces as the setup test: the guard call must start a line,
    # so `# _guard_db ...` cannot match even if the strip ever regressed.
    assert re.search(r"^\s*_guard_db\b", seg, re.MULTILINE), (
        "finish() must CALL _guard_db at the start of a line — a comment "
        "containing the name is not a call"
    )
    guard = seg.index('_guard_db "${db}"')
    first_db = _first_db_touching_index(seg)
    assert guard < first_db, (
        f"finish() runs a DB-touching command at offset {first_db} BEFORE the "
        f"real-database guard at {guard} — the guard must be the first thing "
        "that could touch the database."
    )


# ── 3. The product beat: the human gate runs BEFORE the send ──────────────────


def test_finish_require_approved_before_send_cli(segments: dict) -> None:
    """finish() must not send until a human approved — asserted by OFFSET.

    The D2 ticket's single most important demo beat is that the pipeline STOPS
    at awaiting_review and will not proceed.  A mere "both strings exist"
    assertion would stay green through a regression that moved send_cli ABOVE
    the gate; comparing the two offsets makes that regression fail.  The gate
    also runs after _guard_db (covered by the finish test above), so the order
    is guard -> human gate -> send.
    """
    seg = segments["finish"]
    # Belt-and-braces on top of comment-stripping: the gate call must sit at
    # the start of a line, so a commented-out `# _require_approved ...` cannot
    # match even if the strip ever regressed (ticket D2-fix Finding 1).
    assert re.search(r"^\s*_require_approved\b", seg, re.MULTILINE), (
        "finish() must CALL _require_approved at the start of a line — a "
        "comment containing the name is not a call"
    )
    gate = seg.index('_require_approved "${db}" "${target}"')
    send = seg.index("app.send_cli")
    assert gate < send, (
        f"finish() runs send_cli at offset {send} BEFORE the human-gate guard "
        f"at {gate} — the D2 product beat is that the pipeline stops at "
        "awaiting_review and never auto-proceeds."
    )


def test_finish_gate_predicate_pins_approved_only(segments: dict) -> None:
    """The human gate must refuse anything that is not the exact state 'approved'.

    The order test above pins WHERE the gate runs but never WHAT it checks: a
    sabotage that flips the predicate (e.g. `!= "approved"` -> `== "failed"`)
    would keep every order assertion green while letting an awaiting_review
    target straight through to send_cli (ticket D2-fix Finding 2).  Pinning the
    exact literal comparison inside the _require_approved function body — the
    `gate` segment, not merely anywhere in the file — makes that flip fail here.
    The behavioural test below is the real proof; this is the cheap static half.
    """
    gate = segments["gate"]
    assert re.search(r'row\["state"\]\s*!=\s*"approved"', gate), (
        "the human gate must refuse anything that is not the exact state "
        "'approved' — the predicate `row[\"state\"] != \"approved\"` must "
        "appear inside _require_approved.  A flipped comparison would let "
        "awaiting_review targets through to send_cli."
    )


def test_finish_send_cli_is_pinned_to_a_single_send(segments: dict) -> None:
    """finish() must pass --limit 1 to send_cli — never the default batch.

    send_cli is a BATCH command: its own default limit is 10 and it dry-runs
    EVERY `approved` target up to that limit, ordered by created_at (see
    app/send_cli.py).  finish's human gate verified exactly ONE target, so
    without `--limit 1` a db holding several approved targets (a hand-made db,
    or one built before setup refused multi-row CSVs) would dry-run ALL of them
    in a single finish run — the original finding-4 batch-send bug.  Pinning
    the exact `--limit 1` on the send_cli line makes deleting it (or changing
    the value) fail here.  (ticket D2-fix2 Issue B point 1.)
    """
    seg = segments["finish"]
    send = seg.index("app.send_cli")  # the send_cli invocation, comment-stripped
    line_start = seg.rfind("\n", 0, send) + 1  # start of the line holding it
    line_end = seg.find("\n", send)  # end of that line
    send_line = seg[line_start:line_end]
    assert re.search(r"--limit\s+1(\s|$)", send_line), (
        "finish() must pass --limit 1 to send_cli — without it send_cli's "
        f"default batch limit would dry-run every approved target.\nsend_cli "
        f"line: {send_line!r}"
    )


def test_finish_refuses_unapproved_target_before_any_send(tmp_path) -> None:
    """A finish run over an awaiting_review target must fail at the gate.

    The SEMANTIC proof behind every text assertion: seed a scratch SQLite db
    with the REAL schema and one target in `awaiting_review` — the state a
    human has not approved — then run the actual script's `finish` subcommand
    as a subprocess.  _require_approved must refuse with a non-zero exit BEFORE
    send_cli is ever reached.

    No model call or network can happen, by construction:
      - the gate's refusal exits the subprocess before the send stage starts;
      - as belt-and-braces against a regression that WOULD reach send_cli,
        PYTHON_BIN is pointed at a FAKE interpreter that executes only the
        stdin-heredoc helper form the gates use (`python - <db> [<target>]`)
        and refuses any `-m app.*` invocation with a distinct marker + exit 99.
        So if the gate ever stopped firing, the subprocess would emit
        FAKE_PYTHON_BLOCKED_MODULE (failing this test) instead of calling a
        real CLI and spending a billable Gemini call.
    """
    # Seed the real schema + one awaiting_review target.  The inserts are plain
    # SQL because the gate only SELECTs the target's state — the write gate is
    # not part of this script-behaviour guard's scope.
    db = tmp_path / "live_demo.db"
    conn = connect(str(db))
    apply_schema(conn)
    conn.execute(
        "INSERT INTO offers (offer_id, slug, active, created_at) "
        "VALUES (?,?,?,datetime('now'))",
        ("off_d2", "therapy-app", 1),
    )
    conn.execute(
        "INSERT INTO accounts (account_id, company_name, domain, "
        "normalized_domain, created_at, updated_at) "
        "VALUES (?,?,?,?,datetime('now'),datetime('now'))",
        ("acc_d2", "Demo Co", "demo.test", "demo.test"),
    )
    conn.execute(
        "INSERT INTO targets (target_id, account_id, offer_id, source, state, "
        "created_at, updated_at) VALUES (?,?,?,?,?,datetime('now'),datetime('now'))",
        ("tgt_d2", "acc_d2", "off_d2", "csv", "awaiting_review"),
    )
    conn.close()

    # The fake interpreter (see _write_fake_python): the stdin-heredoc helper
    # form the gates use executes from stdin; every real `-m app.*` CLI prints
    # the marker and exits 99.
    fake_python = _write_fake_python(tmp_path)

    # Hermetic subprocess env: never inherit the operator's real OUTBOUND_DB_TARGET
    # (the _guard_db heredoc would refuse a real db path) and force the fake
    # interpreter so no real CLI can run.  PYTHONPATH is set to the repo root so
    # the fake's exec of the heredocs can `import app` — a script-file python
    # invocation puts the SCRIPT's dir on sys.path, not cwd (the real `python -`
    # gets cwd), so without PYTHONPATH the heredoc imports would fail.
    env = os.environ.copy()
    env.pop("OUTBOUND_DB_TARGET", None)
    env["PYTHON_BIN"] = str(fake_python)
    env["PYTHONPATH"] = str(ROOT)

    proc = subprocess.run(
        ["scripts/demo_live.sh", "finish", "--db", str(db), "--target", "tgt_d2",
         "--persona", "warms_up"],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    # The gate must refuse: non-zero exit, the gate's own "not 'approved'"
    # message on stderr, and NO fake-python marker (which would prove a real
    # CLI was reached past the gate).
    assert proc.returncode != 0, (
        "finish must refuse an unapproved (awaiting_review) target — it exited "
        "0, which means the send stage may have run.\nstdout:\n"
        f"{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "not 'approved'" in proc.stderr, (
        "finish must refuse with the human gate's 'not approved' message.\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "FAKE_PYTHON_BLOCKED_MODULE" not in proc.stderr, (
        "a real CLI (send_cli/reply_cli) was reached — the human gate did not "
        "stop the run before the send stage.\nstderr:\n{proc.stderr}"
    )


def test_setup_refuses_multi_row_csv_before_any_cli(tmp_path) -> None:
    """setup() must refuse a --csv with more than one data row BEFORE phase1_cli.

    The demo is deliberately single-target end to end (finish pins --limit 1,
    _target_id_state reads ONE target, the console review is per-target), but
    setup accepts an arbitrary --csv with no row-count check in the round-2
    code — pointing it at data/hk_therapy_targets.csv (10 real rows, already
    in the repo) silently produces a multi-target live_demo.db that the
    single-target readouts cannot drive correctly.  This semantic test runs the
    real setup against the real 10-row CSV under the fake interpreter and
    asserts it refuses with the multi-row message BEFORE any CLI (phase1_cli)
    is reached.  (ticket D2-fix2 Issue B point 3.)
    """
    fake_python = _write_fake_python(tmp_path)
    db = tmp_path / "live_demo.db"  # scratch path — must pass _guard_db first
    env = os.environ.copy()
    env.pop("OUTBOUND_DB_TARGET", None)
    env["PYTHON_BIN"] = str(fake_python)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        ["scripts/demo_live.sh", "setup", "--csv", "data/hk_therapy_targets.csv",
         "--db", str(db)],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    # The multi-row CSV must be refused with a non-zero exit, the multi-row
    # message on stderr, and NO fake-python marker (phase1_cli never reached).
    assert proc.returncode != 0, (
        "setup must refuse a multi-row --csv — it exited 0, which means "
        "phase1_cli may have imported 10 targets.\nstdout:\n"
        f"{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "data rows" in proc.stderr and "single-target" in proc.stderr, (
        "setup must refuse a multi-row CSV with the explicit message naming "
        "the row count and the single-target design.\nstderr:\n{proc.stderr}"
    )
    assert "FAKE_PYTHON_BLOCKED_MODULE" not in proc.stderr, (
        "a real CLI (phase1_cli) was reached — setup did not refuse the "
        "multi-row CSV before importing it.\nstderr:\n{proc.stderr}"
    )


def test_guard_db_refuses_real_database_before_any_send(tmp_path) -> None:
    """finish() over the operator's real data/outbound.db must refuse at the guard.

    The text tests pin that _guard_db is CALLED first in both subcommands, but
    never that it actually REFUSES anything in THIS script's wiring — a
    `return 0` injected at the top of _guard_db's body keeps every order
    assertion green.  _guard_db delegates to app.demo_seed._guard_violation
    (which has its own coverage), so this pins the WIRING: running finish with
    --db data/outbound.db must exit non-zero with the real-database guard's
    message BEFORE any CLI is reached.  (ticket D2-fix2 Issue C.)
    """
    fake_python = _write_fake_python(tmp_path)
    env = os.environ.copy()
    env.pop("OUTBOUND_DB_TARGET", None)
    env["PYTHON_BIN"] = str(fake_python)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        ["scripts/demo_live.sh", "finish", "--db", "data/outbound.db",
         "--target", "tgt_x", "--persona", "warms_up"],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    # The guard must refuse before any db I/O: non-zero exit, the guard's own
    # real-database message on stderr, and NO fake-python marker (a real CLI
    # would prove the guard did not stop the run).
    assert proc.returncode != 0, (
        "finish --db data/outbound.db must be refused by the real-database "
        "guard — it exited 0.\nstdout:\n"
        f"{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "refusing to run the demo seed against" in proc.stderr, (
        "finish --db data/outbound.db must refuse with the guard's own "
        "real-database message.\nstderr:\n{proc.stderr}"
    )
    assert "FAKE_PYTHON_BLOCKED_MODULE" not in proc.stderr, (
        "a real CLI was reached — the real-database guard did not stop the "
        "run before the send stage.\nstderr:\n{proc.stderr}"
    )


# ── 4. No mail transport anywhere in the script ───────────────────────────────


def test_script_never_imports_or_names_a_mail_transport(script_text: str) -> None:
    """The script must never name/import a mail-transport module.

    Greps the whole text for the same banned module list tests/test_send_gate.py
    enforces for app/ (stdlib SMTP/IMAP/POP, the Gmail client stack, third-party
    SDKs).  The script's job is to ORCHESTRATE the repo's stage CLIs, which are
    themselves structurally incapable of transmitting; a mail import here would
    be the first step toward a real send and is forbidden by the standing rule.
    """
    for mod in _MAIL_TRANSPORT_MODULES:
        assert not re.search(rf"\b{re.escape(mod)}\b", script_text), (
            f"demo_live.sh names {mod!r} — a mail-transport module. The script "
            "only shells out to `python -m app.*` and must never import or "
            "reference one."
        )


def test_script_only_shells_out_to_app_python(script_text: str) -> None:
    """Every python invocation must target an app module or a stdin helper.

    The D2 ticket: "this script only shells out to `python -m app.*`".  Read
    against the actual script, the full legal set of `"${PYTHON_BIN}"` lines is:
    the stage CLIs (`-m app.<module>`), the console-server launch instruction
    (`-m uvicorn app.console.app:app` — printed inside setup's cat <<EOF so the
    operator can launch the console; the script itself never runs it), and the
    stdin-heredoc helpers (`- "${db}" <<'PY'` — python reading the script from
    stdin with the db path as argv; `- "${csv}" <<'PY'` — the setup CSV
    row-count guard reads the target CSV the same way).  A future edit adding a
    `python -c "import smtplib ..."` or a third-party tool invocation fails
    here.
    """
    allowed = re.compile(r'(-m app\.|-m uvicorn app\.|- "\$\{db\}"|- "\$\{csv\}")')
    offending = []
    start = 0
    while True:
        i = script_text.find('"${PYTHON_BIN}"', start)
        if i == -1:
            break  # no more python invocations in the script
        line_end = script_text.find("\n", i)
        line = script_text[i:line_end]
        if not allowed.search(line):
            offending.append(line.strip())
        start = i + 1
    assert offending == [], (
        "these python invocations are neither `-m app.*` nor the stdin-heredoc "
        f"helper form:\n" + "\n".join(offending)
    )


def test_script_has_no_raw_transport_tools(script_text: str) -> None:
    """The script must never call a raw network/transport tool directly.

    The check above only scans ``${PYTHON_BIN}`` lines.  A raw ``curl`` /
    ``wget`` / ``nc`` / bare interpreter (not ``${PYTHON_BIN}``) / MTA client /
    ``ssh`` / ``telnet`` / bash ``/dev/tcp`` calling out to a URL would
    transmit mail without ever appearing on a ``${PYTHON_BIN}`` line, so this
    greps the WHOLE script for those tokens plus any literal http(s) URL and
    asserts none appear anywhere (ticket D2-fix Finding 3, widened for the
    round-2 blind spots in D2-fix2 Issue A).  The banned-module check above
    stays too — it catches smtplib & co.

    The scan runs on the COMMENT-STRIPPED code text (the same _strip_comments
    the order tests use): a comment that merely documents the script ("the
    python interpreter", "the curl-based fetcher") is not a transport, and only
    code that would actually execute can transmit.  The ONE legitimate
    ``python3.14`` is the documented PYTHON_BIN default PATH value
    (`PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.14}"`): it is a path,
    not a command.  The command-position regex allows an optional absolute path
    PREFIX (`/usr/bin/python3 -c ...` is a real risk — the script's own default
    uses exactly that shape), but restricts the prefix to path-safe characters
    so the greedy prefix cannot reach backwards across the `${PYTHON_BIN:-...}`
    assignment and flag the legit default value (verified: the reviewer's
    proposed `\\S*/` variant false-positives on it).

    OUT OF SCOPE, deliberately (assessed in the round-2 D2-fix2 review as a
    genuine ceiling for this regex-over-bash-source approach — do NOT re-litigate
    it against this test): variable-indirection obfuscation
    (`PY=python3; $PY -c ...`) and command-substitution obfuscation
    (`$(echo cu)$(echo rl)`).  Closing those would need a different technique.
    """
    # Scan the code-only text: every bash comment stripped, so a comment that
    # mentions "python"/"curl" can never be mistaken for an invocation.
    code_only = _strip_comments(script_text)
    # A python/perl/ruby/node in COMMAND position — not the PYTHON_BIN default
    # path (the optional path prefix is restricted to path-safe chars so it
    # cannot cross the `${PYTHON_BIN:-...` of the default assignment) and not a
    # `${PYTHON_BIN}` expansion (uppercase).  `\s` (not just a literal space)
    # covers a tab-indented invocation.
    interpreter_command = re.compile(
        r"(^|[;&|(\s])(?:[A-Za-z0-9_./-]*/)?(?:python[0-9.]*|perl|ruby|node)\b"
    )
    for lineno, line in enumerate(code_only.splitlines(), start=1):
        if not line.strip():
            continue  # a pure-comment line strips to nothing — nothing to check
        for token in ("curl", "wget"):
            if re.search(rf"\b{token}\b", line):
                pytest.fail(
                    f"line {lineno} references {token!r} — a raw network tool. "
                    "The script must only shell out through ${PYTHON_BIN} to "
                    "`python -m app.*`."
                )
        if re.search(r"\bnc\b", line):
            pytest.fail(
                f"line {lineno} references 'nc' — a raw network tool. The "
                "script must only shell out through ${PYTHON_BIN} to "
                "`python -m app.*`."
            )
        if interpreter_command.search(line):
            pytest.fail(
                f"line {lineno} invokes a bare interpreter in command position "
                "(python/perl/ruby/node, with or without an absolute path) — "
                "the script must shell out only through ${PYTHON_BIN}."
            )
        for token in _RAW_TRANSPORT_WORD_TOKENS:
            if re.search(rf"\b{re.escape(token)}\b", line):
                pytest.fail(
                    f"line {lineno} references {token!r} — a raw MTA/transport "
                    "tool. The script must only shell out through ${PYTHON_BIN} "
                    "to `python -m app.*`."
                )
        for token in _RAW_TRANSPORT_LITERAL_TOKENS:
            if token in line:
                pytest.fail(
                    f"line {lineno} references {token!r} — a raw network "
                    "transport. The script must only shell out through "
                    "${PYTHON_BIN} to `python -m app.*`."
                )
        if re.search(r"https?://", line):
            pytest.fail(
                f"line {lineno} contains a literal http(s):// URL — the script "
                "must never reach out to the network directly."
            )


# ── 5. The demo CSV: exactly one REAL row on a reserved domain ────────────────


def test_demo_live_csv_has_exactly_one_data_row() -> None:
    """data/demo_live_target.csv must be header + EXACTLY one non-empty data row.

    A live demo is deliberately single-target so the judge can follow the whole
    pipeline on one company; a multi-row import would blur the demo and a
    zero-row import would import nothing.  Asserting the row count structurally
    pins both ends.
    """
    rows = _read_csv(DEMO_LIVE_CSV)
    assert len(rows) == 2, (
        f"expected header + exactly 1 data row, got {len(rows) - 1} data rows"
    )
    assert rows[0][0].strip().lower() == "company_name", (
        f"expected a company_name header, got {rows[0]!r}"
    )
    assert len(rows[1]) == 7, f"the data row must have 7 columns, got {rows[1]!r}"
    assert any(cell.strip() for cell in rows[1]), (
        "the single data row must be non-empty"
    )


def test_demo_live_csv_company_and_domain_match_a_real_row() -> None:
    """The demo target's company_name/domain must be verbatim from the real import.

    Parses BOTH CSVs with the stdlib csv module and compares fields exactly
    (never hand-parsed strings).  The whole D2 premise is "research a company
    nobody pre-selected" — but that company must be a REAL row the operator
    already imports, not an invented one, so the demo proves the real pipeline
    on a real target.  A future edit that invents a company fails here.
    """
    live = _read_csv(DEMO_LIVE_CSV)
    real = _read_csv(REAL_TARGETS_CSV)
    live_header, live_row = live[0], live[1]
    real_header = real[0]
    live_name = live_row[live_header.index("company_name")]
    live_domain = live_row[live_header.index("domain")]
    real_name_col = real_header.index("company_name")
    real_domain_col = real_header.index("domain")
    match = any(
        row[real_name_col] == live_name and row[real_domain_col] == live_domain
        for row in real[1:]
    )
    assert match, (
        f"demo target {live_name!r} ({live_domain!r}) is not any row in "
        f"{REAL_TARGETS_CSV.name} — a live demo must import a real company, "
        "never invent one."
    )


def test_demo_live_csv_contact_email_is_reserved_domain() -> None:
    """The demo CSV's contact_email must sit on an RFC 2606 reserved TLD.

    Uses the SAME RESERVED_TLDS constant the demo_seed and conversation_sim
    tests enforce, so the three demo surfaces cannot drift apart.  A `.test`
    address can never resolve in DNS, so the DRY_RUN send's recipient can never
    collide with a real inbox — the standing no-real-email rule, applied to the
    demo's one address.
    """
    rows = _read_csv(DEMO_LIVE_CSV)
    header, row = rows[0], rows[1]
    email = row[header.index("contact_email")]
    domain = email.split("@", 1)[-1].lower()  # the domain half after the @
    tld = domain.rsplit(".", 1)[-1]  # the final label — the routability test
    assert tld in RESERVED_TLDS, (
        f"demo contact_email {email!r} is not on a reserved TLD "
        f"(allowed: {RESERVED_TLDS}) — the DRY_RUN send could target a real "
        "inbox."
    )


# ── 6. Strict bash mode, so a failing stage aborts the run ────────────────────


def test_script_sets_strict_bash_mode_near_the_top(script_text: str) -> None:
    """`set -euo pipefail` must be set near the top, before any function body.

    -e aborts on the first failing command, -u refuses unset variables, and
    -o pipefail fails a pipeline when any stage fails — the three flags together
    mean a failed research/draft/send stage can never silently continue and look
    like success (CLAUDE.md §3: failures surface clearly).  "Near the top"
    (before setup()) is asserted because a late set would leave the arg-parsing
    and guard setup unguarded.
    """
    m = re.search(r"^set -euo pipefail\s*$", script_text, re.MULTILINE)
    assert m is not None, "the script must set `set -euo pipefail`"
    assert m.start() < script_text.index("setup() {"), (
        "`set -euo pipefail` must appear before the first function body, not "
        "partway through the run."
    )
