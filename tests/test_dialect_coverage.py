"""Dialect-coverage guardrail (ticket H3).

The shared ``scratch_db_target`` fixture makes every ``conn`` /
``conn_with_offers`` fixture honour ``OUTBOUND_TEST_DB_TARGET``, so one
``pytest`` run with that env var set exercises the Postgres path across the
whole suite instead of only ``test_db_postgres.py``.  The coverage number is
not a sentence in a report — it is a tested property here:

- a new module with a hardcoded SQLite ``conn`` fixture must fail
  ``test_every_conn_fixture_uses_scratch_db_target_or_is_allowlisted``;
- the allowlist cannot rot into a rubber stamp (every name must be a real
  file, and its length is pinned), so coverage can never silently regress;
- a test BODY that hand-builds ``tmp_path/"<name>.db"`` while also holding a
  dialect-aware fixture (conn / conn_with_offers / scratch_db_target) must fail
  ``test_no_converted_test_body_hand_builds_a_sqlite_path`` — the fixture-only
  walk could not see H4a's divergence (a body pointing a CLI at a SQLite file
  while its conn fixture was Postgres);
- a module with a ``conn``/``conn_with_offers`` fixture must not reference the
  production database target by any AST-visible spelling: the env-var name
  ``OUTBOUND_DB_TARGET`` (as a substring of any string constant), the
  operator's real database's default path ``data/outbound.db``, or the
  ``app/console/app.py::_db_target()`` resolver name —
  ``test_no_db_fixture_module_references_the_production_target``.  The
  fixture-only walk inspects ARG NAMES, never fixture or module bodies, so an
  allowlisted module whose fixture read the production variable (S2's
  ``test_db_postgres``) slipped straight through; this closes that blind spot.
"""

import ast
import fnmatch  # S2e: matching pytest's python_files basename patterns (test_*.py / *_test.py)
import os  # S2e round 5: os.walk(followlinks=True) — rglob misses symlinked directories, see below
import re  # H7: the gitignored data/ path patterns the regression guard refuses
import tomllib  # S2e: reading pyproject.toml's [tool.pytest.ini_options] for a python_files override
from pathlib import Path


# Every test module deliberately left OUTSIDE the shared dialect-aware
# scratch fixture, each with the concrete reason.  All eight are SQLite-only
# by design (they exercise engine internals or pass a filesystem path to a
# CLI).  test_db_postgres was the ninth and was removed at S2: its conn()
# fixture was converted onto the shared scratch_db_target fixture, after it
# had been allowlisted pre-S2 with the reason "cannot be routed through the
# scratch fixture" — which the S2 conversion proved false (and which let its
# fixture read the PRODUCTION database variable until the incident; see the
# S2 section below).  Together with the 28 converted modules these are the
# honest coverage denominator.
#
# S2b accuracy check (verified 2026-08-28): NONE of these eight currently
# defines a conn/conn_with_offers fixture at all, so the allowlist branch of
# the fixture-coverage walk below is today never taken — every non-allowlisted
# module already routes its fixture through scratch_db_target.  What the
# pinned count actually protects now is the documented SET itself: it pins
# that no ninth module has silently joined the SQLite-only list, and it keeps
# these modules exempt from the H4a body-walk (an exemption that is likewise
# currently moot, since none of their test bodies takes a dialect-aware
# fixture).  The count is a regression guard on the set and its rationale,
# not on fixture routing.
SQLITE_ONLY_MODULES = frozenset({
    "test_db",  # asserts SQLite WAL journal mode / PRAGMA / sqlite_master / BEGIN IMMEDIATE — engine internals with no Postgres equivalent.
    "test_db_dialect",  # pins the SQLite side of the dialect contract (PRAGMA journal_mode, BEGIN IMMEDIATE on a temp file).
    "test_console",  # passes a SQLite filesystem path to the console via OUTBOUND_DB_TARGET (path-resolved, not a URL).
    "test_demo_seed",  # passes --db as a filesystem path to the seed CLI and asserts the path-resolved real-db guard.
    "test_phase1_cli",  # passes --db as a filesystem path to the phase1 CLI.
    "test_config",  # opens a filesystem db_path directly in its sync_offers idempotency test (no conn fixture to convert).
    "test_adversarial_sim",  # passes a filesystem db_path (+ outbox/inbox dirs) to run_attack/main.
    "test_conversation_sim",  # passes a filesystem --db path (+ outbox/inbox dirs) to the simulator CLI.
})

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent  # S2e: pyproject.toml — the pytest config the shared discovery helper reads — lives here.


# ── S2e: ONE shared candidate-discovery mechanism, not four ad-hoc globs ─────
#
# Every guard below used to discover its files with its own
# `sorted(TESTS_DIR.glob("test_*.py"))` (the S2 guard appended one literal
# conftest.py path on top).  Four review rounds on the same guard found four
# new spelling variants; round 4 then found the real structural reason they
# kept reopening — the discovery mechanism itself does not match pytest's
# actual collection rules, so every patch that appended another literal path
# merely moved the gap to the next shape.  Centralising discovery in
# _pytest_collectable_python_files() fixes the mechanism once for every guard
# instead of patching the next instance.


def _pytest_python_files_patterns():
    """Return pytest's ``python_files`` basename patterns for this repo.

    pytest's default is ``["test_*.py", "*_test.py"]``.  pyproject.toml's
    [tool.pytest.ini_options] today declares only testpaths and pythonpath —
    no python_files override (re-verified at S2e) — but this reads the file at
    runtime so a FUTURE override is honoured rather than hardcoding pytest's
    default blindly.
    """
    try:
        # tomllib is stdlib on the >=3.11 runtime this repo already requires.
        ini = tomllib.loads(PROJECT_ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8"))
        patterns = ini["tool"]["pytest"]["ini_options"].get("python_files")
    except (KeyError, OSError, tomllib.TOMLDecodeError):
        # No [tool.pytest.ini_options] section, or the file cannot be read —
        # fall back to pytest's built-in default (the current state of this repo).
        patterns = None
    if not patterns:
        return ["test_*.py", "*_test.py"]
    # pytest accepts python_files as a single string or a list of patterns.
    # (S2e round 5) Anything else — a TOML table, an int, a bool — is a
    # malformed config this function has no business interpreting.  The
    # unsafe failure mode here is silent: list({"a": 1}) yields ["a"], which
    # would make the guard scan only files literally named "a" with every
    # check reporting green — a coverage collapse with no error anywhere,
    # exactly the "masks a real config change" shape this guard exists to
    # avoid elsewhere.  Falling back to pytest's real default on any
    # unexpected shape is over-inclusive (matches at least what pytest
    # collects today), which is the same safe direction the rest of this file
    # already prefers over silent narrowing.
    if isinstance(patterns, str):
        return [patterns]
    if isinstance(patterns, list) and all(isinstance(p, str) for p in patterns):
        return list(patterns)
    return ["test_*.py", "*_test.py"]


def _pytest_collectable_python_files():
    """Return every ``*.py`` under tests/ that pytest would actually collect,
    sorted — the one discovery call every guard in this file shares.

    pytest's collection is RECURSIVE: it walks every subdirectory under
    testpaths and collects any ``*.py`` whose BASENAME matches one of the
    ``python_files`` patterns, plus every ``conftest.py`` it loads on the way
    down (conftest.py is loaded regardless of ``python_files``).  The old
    per-guard globs matched only top-level ``test_*.py``, so a ``*_test.py``
    module, a nested ``test_*.py``, or a nested ``conftest.py`` was collected
    by a plain `pytest` run yet invisible to every guard in this file — the
    round-4 finding that motivated centralising discovery here.
    """
    patterns = _pytest_python_files_patterns()
    files = set()
    # (S2e round 5) os.walk(followlinks=True), not Path.rglob: rglob does NOT
    # descend into a directory SYMLINK by default, and the recurse_symlinks
    # flag that would fix that is Python 3.13+ — this repo's floor is 3.11
    # (pyproject.toml). pytest itself DOES collect through a symlinked
    # subdirectory, so rglob alone reproduced the exact "pytest sees it, the
    # guard doesn't" class S2e exists to close — proven live: a
    # tests/linkdir -> /tmp/outside_dir symlink holding a leaking test file
    # was collected by pytest and invisible to every guard here until this
    # walk was switched. _seen_real_dirs breaks a symlink CYCLE (a directory
    # symlinked into its own descendant), which followlinks=True does not
    # protect against on its own.
    _seen_real_dirs = set()
    for root, dirnames, filenames in os.walk(TESTS_DIR, followlinks=True):
        real_root = os.path.realpath(root)
        if real_root in _seen_real_dirs:
            dirnames[:] = []  # Already visited via another path — stop descending, break the cycle.
            continue
        _seen_real_dirs.add(real_root)
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(root) / name
            if name == "conftest.py":
                # conftest.py is loaded at every directory level regardless of
                # python_files, so a nested tests/<sub>/conftest.py is a real
                # suite-wide fixture source and is always included.
                files.add(path)
                continue
            # The basename filter keeps this to exactly what pytest would
            # collect, so a helper .py under tests/ (e.g. a future
            # tests/helpers.py) is not mistaken for a test module.
            rel = path.relative_to(TESTS_DIR).as_posix()  # for a slash pattern like "tests/test_*.py"
            if any(fnmatch.fnmatch(rel if "/" in p else name, p) for p in patterns):
                files.add(path)
    return sorted(files)


def _iter_fixture_defs(tree, names):
    """Yield ``(display_name, func_node)`` for every module- OR class-level
    FunctionDef/AsyncFunctionDef named in ``names`` in one module.

    Classes are walked because pytest lets a fixture live on a class
    (``class TestX: @pytest.fixture def conn(self, ...)``) — the round-4
    finding that the S2/H3 walks missed.  This mirrors the over-inclusive-is-
    safe rule _iter_class_test_methods documents below for the H7 guard: any
    class is walked regardless of its name, because a false positive costs a
    review while a false negative reintroduces the S2/H3 blind spot silently.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            yield from _iter_class_fixture_defs(node, node.name, names)


def _iter_class_fixture_defs(cls, cls_dotted_name, names):
    """Yield ``(display_name, func_node)`` for defs named in ``names`` inside
    one class, recursing into nested classes — the same recursion shape as
    _iter_class_test_methods, adapted to fixture names instead of ``test_*``."""
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            yield f"{cls_dotted_name}.{node.name}", node
        elif isinstance(node, ast.ClassDef):
            yield from _iter_class_fixture_defs(node, f"{cls_dotted_name}.{node.name}", names)


def _conn_fixtures_in(path: Path):
    """Yield ``(fixture_name, arg_names)`` for every fixture named ``conn`` or
    ``conn_with_offers`` in one test file — at module scope OR inside a class.

    Walks defs only (never bodies, via _iter_fixture_defs), so a helper
    function named ``conn`` nested inside a test body cannot be mistaken for a
    fixture.  Both FunctionDef AND AsyncFunctionDef are matched — `async def
    conn(...)` is a legal async fixture (pytest-asyncio), and a walk that only
    saw sync defs would let a hardcoded-SQLite async fixture slip straight
    through, the same latent class this whole module exists to catch.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for _display_name, node in _iter_fixture_defs(tree, ("conn", "conn_with_offers")):
        args = [a.arg for a in node.args.args]
        yield node.name, args


def test_every_conn_fixture_uses_scratch_db_target_or_is_allowlisted():
    """A new module with a hardcoded SQLite conn fixture must fail this.

    The rule: every ``conn`` / ``conn_with_offers`` fixture either takes
    ``scratch_db_target`` (so it honours OUTBOUND_TEST_DB_TARGET) or its module
    is deliberately listed in SQLITE_ONLY_MODULES.
    """
    offenders = []
    for path in _pytest_collectable_python_files():
        for name, args in _conn_fixtures_in(path):
            # A fixture that takes the shared target is already dialect-aware.
            if "scratch_db_target" in args:
                continue
            # A deliberately-exempt module is allowed to keep its own target.
            if path.stem in SQLITE_ONLY_MODULES:
                continue
            offenders.append(f"{path.stem}.{name}({', '.join(args)})")
    assert not offenders, (
        "conn/conn_with_offers fixtures must take scratch_db_target or be "
        f"allowlisted in SQLITE_ONLY_MODULES; offenders: {offenders}"
    )


def test_sqlite_only_modules_all_exist():
    """The allowlist must not rot into a rubber stamp: every name must be a
    real test file, so a renamed or deleted module fails here."""
    for stem in SQLITE_ONLY_MODULES:
        assert (TESTS_DIR / f"{stem}.py").is_file(), f"{stem}.py is missing"


def test_sqlite_only_module_count_is_pinned():
    """Pin the allowlist length — a regression guard on the documented SET.

    28 modules route their conn/conn_with_offers fixture through the shared
    scratch_db_target.  These 8 are the SQLite-only set (all by design — the
    ninth, test_db_postgres, was converted onto the shared fixture at S2).
    None of the 8 defines a conn/conn_with_offers fixture today, so this
    count no longer gates fixture routing (every routed module already uses
    scratch_db_target); it pins the documented set so a ninth SQLite-only
    module cannot join silently.  Raising this number is a deliberate
    decision (a module became SQLite-only), not a fix — the same pattern as
    the follow-up-cap pin that a sabotage once raised 2 -> 99.
    """
    assert len(SQLITE_ONLY_MODULES) == 8


# ── H4a: a hardcoded SQLite path in a TEST BODY must fail the walk too ────────
#
# The fixture-only walk above cannot see a path built inside a test body.
# H4a's divergence was exactly that: the draft-gate e2e test took the
# dialect-aware conn fixture (Postgres on a PG run) but pointed send_cli at
# tmp_path/"test.db" — a SQLite file — so on Postgres the send opened an empty
# database and wrote 0 .eml artifacts instead of 1.  The predicate below flags
# the same shape: a tmp_path/"<name>.db" BinOp in a test that ALSO holds a
# dialect-aware database target by ANY of the three fixture names below.  A
# test with none of those fixtures cannot mix dialects — it is self-contained
# on its own SQLite file (the same class as the allowlisted CLI modules) or
# has no DB at all — so it is not a divergence and is deliberately not flagged.

# The three fixture names that make a test body dialect-aware: conn and
# conn_with_offers OPEN the target; scratch_db_target IS the target string.
# On a Postgres run all three hand the test a Postgres DSN, so hand-building a
# SQLite file beside any of them is exactly the divergence class this guard
# exists to catch.  scratch_db_target was added at H4a review: a test taking
# it directly (no conn) is just as dialect-aware, and the conn-only set let a
# tmp_path/"x.db" beside it slip straight through.
_DIALECT_AWARE_FIXTURES = frozenset({"conn", "conn_with_offers", "scratch_db_target"})


def _is_tmp_path_db_path(node):
    """True when ``node`` is a ``tmp_path / "<something>.db"`` BinOp.

    Matches the plain shape, a path chained through a subdirectory
    (``tmp_path / "dir" / "x.db"``), and the shape nested anywhere inside a
    function body (e.g. inside ``str(...)``) because the caller walks the
    whole body.  ``tmp_path.joinpath(...)`` is a Call, not a BinOp, and is not
    matched — no module in the repo uses it for a DB path today.
    """
    if not isinstance(node, ast.BinOp):
        return False  # Only a "/" join expression can be a tmp_path path build.
    if not isinstance(node.op, ast.Div):
        return False  # "/" (ast.Div) is the operator pathlib overloads to join a path.
    right = node.right
    if not (
        isinstance(right, ast.Constant)
        and isinstance(right.value, str)
        and right.value.endswith(".db")
    ):
        return False  # The leaf must be a literal "<name>.db" — that is the SQLite-file tell.
    # The left operand must reference tmp_path, so a coincidental "/" with a
    # ".db" string on some unrelated value is not flagged.
    return any(isinstance(n, ast.Name) and n.id == "tmp_path" for n in ast.walk(node.left))


def _hand_built_sqlite_paths_in(path):
    """Yield ``(func_name, lineno)`` for test bodies in one file that build a
    SQLite path by hand while also holding a dialect-aware database target
    (any of conn / conn_with_offers / scratch_db_target).

    Walks only top-level defs (FunctionDef OR AsyncFunctionDef) named
    ``test_*`` — a fixture or helper with any other name is not a test body, so
    its paths are not this guard's concern.  ``ast.walk`` over the whole
    function finds the BinOp anywhere: an assignment, a ``str(...)`` argument,
    or a call argument.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        # Match BOTH sync and async test defs: an `async def test_...` holds a
        # dialect-aware target exactly as a sync one does, so the guard must not
        # depend on the `def` keyword — pytest-asyncio is installed and Phase
        # 1b/2 anticipates async I/O, so an async test is a latent case, which
        # is precisely the class this guard exists to catch.
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        arg_names = {a.arg for a in node.args.args}
        if not (_DIALECT_AWARE_FIXTURES & arg_names):
            # No dialect-aware fixture in this test -> it cannot mix dialects,
            # so a hand-built SQLite file here is self-contained, not a divergence.
            continue
        for child in ast.walk(node):
            if _is_tmp_path_db_path(child):
                yield node.name, child.lineno


def test_no_converted_test_body_hand_builds_a_sqlite_path():
    """A test body that builds ``tmp_path/"<name>.db"`` while also holding a
    dialect-aware target (conn / conn_with_offers / scratch_db_target) is the
    divergence the fixture-only walk cannot see (H4a #2).  Reuses
    SQLITE_ONLY_MODULES so the SQLite-only-by-design modules keep their
    deliberate exemption — no second list to keep in sync, and the list cannot
    be weakened without failing the pinned-count test.
    """
    offenders = []
    for path in _pytest_collectable_python_files():
        if path.stem in SQLITE_ONLY_MODULES:
            continue  # The same allowlist as the fixture walk — a module is either exempt or it is scanned.
        for func_name, lineno in _hand_built_sqlite_paths_in(path):
            offenders.append(f"{path.stem}.{func_name}:{lineno}")
    assert not offenders, (
        "test bodies must use a dialect-aware fixture (conn / conn_with_offers / "
        f"scratch_db_target) for their DB target, not build tmp_path/'*.db' by hand; "
        f"offenders: {offenders}"
    )


# ── LEAD-ADDED (H3 review): pin the fixture's BEHAVIOUR, not just the wiring ──
#
# The worker reported this gap itself, and it was right: the AST test above
# pins that each conn fixture *declares* scratch_db_target, but nothing pinned
# that scratch_db_target *honours* OUTBOUND_TEST_DB_TARGET.  The worker's own
# sabotage — making the fixture ignore the variable and always return the tmp
# path — left test_dialect_coverage.py fully green.  That sabotage reproduces
# the EXACT failure this ticket exists to kill: a suite that looks
# dialect-aware and silently runs every test on SQLite anyway.  Same shape as
# E1's follow-up cap, F1b's normaliser, and H2's reset — the mechanism was
# covered, the decision was not.
#
# The fixture is exercised through its undecorated function (pytest stores it
# on __wrapped__), because a test body runs AFTER fixture setup and so cannot
# monkeypatch the environment the fixture already read.

import pytest

from conftest import (
    TEST_DB_TARGET_ENV,
    real_outbound_db as _real_outbound_db_fixture,
    scratch_db_target as _scratch_db_target_fixture,
)


def _run_fixture(tmp_path):
    """Drive the fixture's generator to its yield and return the target."""
    # __wrapped__ is the plain generator function underneath @pytest.fixture.
    gen = _scratch_db_target_fixture.__wrapped__(tmp_path)
    return next(gen), gen


def test_scratch_db_target_defaults_to_sqlite_when_env_unset(tmp_path, monkeypatch):
    """No env var → today's exact SQLite tmp path, so a plain `pytest` is
    unchanged. This is the default the whole suite still runs on."""
    monkeypatch.delenv(TEST_DB_TARGET_ENV, raising=False)
    target, _ = _run_fixture(tmp_path)
    assert target == str(tmp_path / "test.db")


def test_scratch_db_target_honours_the_env_var(tmp_path, monkeypatch):
    """THE regression guard: with the variable set to a safe scratch target the
    fixture must yield THAT target, not the tmp path. A fixture that ignores
    the variable fails here — which is exactly what the H3 sabotage did while
    every other test stayed green."""
    # A file-shaped scratch target keeps this test offline: no Postgres server
    # is needed to prove the fixture reads and returns the variable.
    configured = tmp_path / "configured_scratch.db"
    monkeypatch.setenv(TEST_DB_TARGET_ENV, str(configured))
    target, _ = _run_fixture(tmp_path)
    assert target == str(configured)
    assert target != str(tmp_path / "test.db")


def test_scratch_db_target_fails_loudly_on_an_unsafe_target(tmp_path, monkeypatch):
    """A configured-but-unsafe target must FAIL, never skip. A silently
    skipping suite is how the 1-of-10 coverage gap arose in the first place."""
    # An unmarked Postgres URL: scratch_target_violation refuses it, so the
    # fixture must raise pytest's Failed rather than quietly falling back.
    monkeypatch.setenv(TEST_DB_TARGET_ENV, "postgresql://u@h/outbound")
    # pytest.fail raises Failed, which derives from BaseException, NOT
    # Exception — pytest.raises(Exception) silently misses it.  Catch the
    # documented alias so this test cannot pass by not raising at all.
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _run_fixture(tmp_path)
    message = str(excinfo.value)
    assert TEST_DB_TARGET_ENV in message  # names the variable the operator must fix
    assert "scratch" in message.lower()   # names the marker rule


def test_scratch_db_target_ignores_an_empty_env_var(tmp_path, monkeypatch):
    """An exported-but-empty variable is a misconfiguration that must fall back
    to the SQLite default — never reach reset_scratch_database("")."""
    monkeypatch.setenv(TEST_DB_TARGET_ENV, "")
    target, _ = _run_fixture(tmp_path)
    assert target == str(tmp_path / "test.db")


# ── S2: a conn/conn_with_offers fixture module must never reach production ────
#
# WHY THIS IS H3'S BLIND SPOT, CLOSED: the fixture walk above inspects only
# ARG NAMES — it checks that a conn/conn_with_offers fixture declares
# scratch_db_target (or that its module is allowlisted).  It never looks
# INSIDE the fixture or the module.  test_db_postgres's pre-S2 conn() was
# exactly that case: the module set its target from the PRODUCTION database
# variable at MODULE scope (so even a fixture-BODY-only walk would have
# missed it), the conn fixture used that module constant, and the module sat
# in the allowlist with a reason asserting it "cannot be routed through the
# scratch fixture" — which S2 proved false.  The allowlist trusted a comment
# without checking what the fixture actually connected to, so a destructive
# test bound to the production database variable ran as documented and wrote
# 13 live rows into Cloud SQL (2026-08-27/28), cleared by hand before a demo
# run could be restored.
#
# The rule: any module that defines a conn/conn_with_offers fixture must not
# reference the production database target by any of the three AST-visible
# spellings below: (1) the env-var spelling OUTBOUND_DB_TARGET as a SUBSTRING
# of any string constant — a docstring that merely names the variable trips
# the guard, because the S2 defect was documented-and-dismissed exactly that
# way; comments are NOT in the AST, so a comment naming the variable can
# never be checked (a boundary stated in the test docstring); (2) the
# operator's real database's default path "data/outbound.db" — the exact file
# tests/conftest.py's real_outbound_db fixture (H7) exists to protect; and
# (3) the app/console/app.py::_db_target() resolver name, whose whole job is
# resolving the production target.  A DB-touching fixture must go through
# scratch_db_target / OUTBOUND_TEST_DB_TARGET; the production target is the
# operator's real database, so no test fixture may ever reach it.  Checked
# REGARDLESS of the allowlist: the allowlist exists for SQLite-only modules,
# and a SQLite-only module has no reason to read the production target either.
#
# S2b note on the old bare-Name branch: the pre-S2b guard also flagged any
# bare Name spelled OUTBOUND_DB_TARGET.  That was dead code overclaiming —
# the real S2 defect's alias pattern (TARGET = os.environ.get(
# "OUTBOUND_DB_TARGET"), aliased to a DIFFERENT name) carries the env-var
# string literal, which the substring check (1) already catches; and a bare
# Name spelled OUTBOUND_DB_TARGET with no such literal in the module is
# either a harmless local constant (the old false positive) or an import that
# cannot exist (no module exports that name).  The branch was removed at S2b
# rather than left claiming a protection it does not give.
_PRODUCTION_DB_TARGET_ENV = "OUTBOUND_DB_TARGET"
_PRODUCTION_DB_DEFAULT_PATH = "data/outbound.db"
_DB_TARGET_RESOLVER_NAME = "_db_target"
# app/demo_seed.py exports REAL_DB_PATH = "data/outbound.db" under its own
# self-documenting name (S2b review, item 1) — a conn fixture doing
# `from app.demo_seed import REAL_DB_PATH; connect(REAL_DB_PATH)` reaches the
# operator's real database with no literal path, no env var, and no
# piece-building anywhere in the fixture module: the "author who builds the
# target from pieces has already thought about it" rationale that excuses the
# path-construction escapes does not hold here, because there is no
# construction — it is a one-line import of a constant this repo itself
# exports for exactly this purpose. Given the SAME three-branch (Name /
# Attribute / import-alias) treatment as _DB_TARGET_RESOLVER_NAME below, so
# `REAL_DB_PATH`, `demo_seed.REAL_DB_PATH`, and
# `from app.demo_seed import REAL_DB_PATH as X` are all caught alike.
_REAL_DB_PATH_CONSTANT_NAME = "REAL_DB_PATH"


def _module_references_production_target(path):
    """True when ``path`` defines a conn/conn_with_offers fixture AND
    references the production database target by any of the three AST-visible
    spellings the S2 comment block above lists."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # First: does the module define a conn/conn_with_offers fixture at all —
    # at module scope OR inside a class?  Only DB-touching modules are this
    # guard's concern — a module without a DB fixture (e.g. test_demo_seed,
    # which SETS the production variable to a tmp SQLite file for a CLI test)
    # is self-contained on its own path, not a divergence, and is deliberately
    # not flagged.  Classes are walked because pytest collects class-level
    # fixtures too (S2e round-4 finding 2); reusing _iter_fixture_defs keeps
    # this walk identical to the one _conn_fixtures_in uses.
    has_db_fixture = next(_iter_fixture_defs(tree, ("conn", "conn_with_offers")), None) is not None
    if not has_db_fixture:
        return False
    # The whole-module walk is deliberate: the original defect stored the
    # variable at MODULE scope and used the alias inside the fixture, so a
    # fixture-body-only walk would have missed it.
    for node in ast.walk(tree):
        # (1) The env-var spelling as a SUBSTRING of any string constant: a
        # module docstring (an ast.Constant) that merely names
        # OUTBOUND_DB_TARGET trips the guard, matching the S2 "documented-and-
        # dismissed" pattern.  Substring (not exact equality) is what makes
        # this claim true; it also catches the real S2 alias pattern,
        # TARGET = os.environ.get("OUTBOUND_DB_TARGET") — the literal rides
        # inside the os.environ.get call.  Comments are not ast.Constants, so
        # they are invisible here by construction (COVERAGE BOUNDARY below).
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _PRODUCTION_DB_TARGET_ENV in node.value
        ):
            return True
        # (2) The operator's real database's default path as a literal:
        # connect("data/outbound.db") reaches the same file real_outbound_db
        # (H7) protects, with no env var involved at all.  Reusing that
        # fixture's judgement (the exact default path) rather than deriving a
        # new one keeps the two guards in agreement.
        if isinstance(node, ast.Constant) and node.value == _PRODUCTION_DB_DEFAULT_PATH:
            return True
        # (3) The _db_target() resolver, in every AST spelling: a bare Name
        # (_db_target() — the import-and-call bypass), an Attribute
        # (app.console.app._db_target() after a dotted import), or an import
        # alias (from app.console.app import _db_target).  The function's
        # whole job is resolving OUTBOUND_DB_TARGET with a data/outbound.db
        # fallback, so a DB-fixture module reaching for it is production
        # access by construction.
        if isinstance(node, ast.Name) and node.id == _DB_TARGET_RESOLVER_NAME:
            return True
        if isinstance(node, ast.Attribute) and node.attr == _DB_TARGET_RESOLVER_NAME:
            return True
        if (
            isinstance(node, ast.alias)
            and (
                node.name == _DB_TARGET_RESOLVER_NAME
                or node.asname == _DB_TARGET_RESOLVER_NAME
            )
        ):
            return True
        # (3b) _db_target reached through getattr(obj, "_db_target") — the
        # resolver's name still appears as a plain string constant. This
        # CANNOT reuse spelling (1)'s bare substring check: "_db_target" is
        # also a substring of "scratch_db_target", the SAFE fixture name that
        # nearly every converted module's docstring names on purpose (proven
        # by running this exact guard against the repo before this line was
        # narrowed — it flagged test_db_postgres.py's own incident-explaining
        # docstring, the file S2 fixed). Exact-match against the resolver
        # name only, never a substring test, avoids that collision.
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == _DB_TARGET_RESOLVER_NAME
        ):
            return True
        # (4) REAL_DB_PATH, in the same three AST spellings as (3) above — see
        # the constant's own comment for why this is not a "pieces" escape.
        if isinstance(node, ast.Name) and node.id == _REAL_DB_PATH_CONSTANT_NAME:
            return True
        if isinstance(node, ast.Attribute) and node.attr == _REAL_DB_PATH_CONSTANT_NAME:
            return True
        if (
            isinstance(node, ast.alias)
            and (
                node.name == _REAL_DB_PATH_CONSTANT_NAME
                or node.asname == _REAL_DB_PATH_CONSTANT_NAME
            )
        ):
            return True
        # (4b) REAL_DB_PATH reached through getattr(obj, "REAL_DB_PATH") /
        # obj.__dict__["REAL_DB_PATH"] / a variable holding the string —
        # (3b)'s reasoning applies identically: exact-match against the
        # constant's name, never substring, and nothing in this repo
        # contains "REAL_DB_PATH" as a proper substring of another
        # identifier, so this carries no equivalent collision risk.
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == _REAL_DB_PATH_CONSTANT_NAME
        ):
            return True
    return False


def test_no_db_fixture_module_references_the_production_target():
    """A module with a conn/conn_with_offers fixture must not reference the
    production database target by any AST-visible spelling.

    This is the S2 blind spot closed: the arg-name walk above cannot see what
    a fixture BODY (or the module constant it reads) points at, and the
    allowlist is comment-trusted, so an allowlisted module's fixture reading
    the production variable slipped straight through (test_db_postgres, until
    S2 converted it onto scratch_db_target).

    S2b review round 2 (independent, adversarial) found and closed two more
    real escapes: `from app.demo_seed import REAL_DB_PATH; connect(REAL_DB_PATH)`
    reached the operator's real database with no literal path and no env var
    at all — the "author who builds the target from pieces has already
    thought about it" excuse below does not apply, since there is no
    construction, just a one-line import of a constant this repo exports
    under its own self-documenting name. And `getattr(console_app,
    "_db_target")()` reached the resolver via a plain string rather than a
    Name/Attribute node. That string check is EXACT-match, not substring:
    "_db_target" is a substring of "scratch_db_target" — the SAFE fixture
    name nearly every converted module's docstring names on purpose — so a
    substring check here false-positived on test_db_postgres.py itself
    (caught by running the guard, not by inspection, before this fix shipped).

    A THIRD review round (independently re-checking the lead's own fix for
    the second round, since nobody else had looked at it yet) found the same
    two spellings — Name/Attribute exact-match, and the getattr string form —
    were not yet given to REAL_DB_PATH; both are now added, closing it
    symmetrically with _db_target. It also found tests/conftest.py was never
    scanned at all: the glob below only matched test_*.py, so a
    conn/conn_with_offers fixture placed in the SHARED conftest — reachable
    by every module in the suite, a wider blast radius than the
    single-module S2 defect — was invisible. Now scanned explicitly.
    conftest.py defines no such fixture today; this closes a scope gap
    before it is ever exercised, not a live defect.
    A FOURTH review round (S2e) found all three previous patches had fixed the
    wrong SHAPE — appending literal paths to a test_*.py glob — and reproduced
    live scope gaps that shape could not see: pytest's real python_files
    default is BOTH test_*.py AND *_test.py; a conn fixture inside a class is
    invisible to a module-body walk; and a nested tests/<sub>/conftest.py is
    never opened.  Discovery is now centralized in one
    _pytest_collectable_python_files() helper that matches pytest's real
    collection rules, and the fixture walk recurses into classes (see below).

    COVERAGE BOUNDARY — the shapes this walk deliberately does NOT catch, so
    the next reader knows the guard's actual limit instead of over-trusting
    it: this is a static AST check, so it cannot follow data flow across
    files or around function boundaries.  A base64-encoded or otherwise
    obfuscated spelling, a target read from a config file, a path OR
    attribute-name built from pieces (os.path.join("data", "outbound.db"),
    Path("data") / "outbound.db", f"data/{name}.db",
    getattr(o, "_db" + "_target")), a path variant ("./data/outbound.db", an
    absolute path), and any indirection that hides the literal
    (TARGET = _load_db_target(); connect(TARGET)) all escape.  Comments are
    not part of the AST, so a comment naming the variable can NEVER trip this
    guard — the docstring substring match is the counter to the S2
    "documented-and-dismissed" pattern, and a comment is invisible here by
    construction.  File-discovery escapes are the OTHER boundary axis (found
    at S2e round 4) and are disclosed here rather than silently claimed: a
    conn/conn_with_offers fixture re-exported from another module
    (``from app.something import conn``) is an ast.alias, not a def, so this
    walk never opens the module where the fixture body actually lives; and
    ``pytest_plugins = [...]`` declared in tests/conftest.py makes a plugin
    module's fixtures suite-wide available without this walk ever opening that
    file.  Neither is caught — both are genuine cross-file indirection that a
    same-file static AST walk cannot follow, materially harder than the
    same-file escapes above.  This guard is against accident, not malice — an
    author who builds the target from pieces or obfuscates it has already
    thought about it, which is the friction the guard exists to create."""
    # File discovery is centralized in _pytest_collectable_python_files(), not
    # a literal glob here: the S2b review found conftest.py was never scanned
    # (patched by appending one literal path), and the S2e round-4 review found
    # that patch shape still missed pytest's real rules — *_test.py modules,
    # class-level fixtures, and nested conftest.py files.  A conn/conn_with_offers
    # fixture placed in the SHARED conftest is reachable by every module in the
    # suite — a wider blast radius than the single-module S2 defect — and the
    # recursive rglob below is what makes a nested tests/<sub>/conftest.py
    # visible too. conftest.py today defines no such fixture, so this remains
    # a scope widening with no behaviour change yet.
    candidates = _pytest_collectable_python_files()
    offenders = []
    for path in candidates:
        if _module_references_production_target(path):
            # tests/-relative path, not just the stem: a nested
            # tests/<sub>/conftest.py and the root tests/conftest.py share the
            # stem "conftest", so the stem alone could not say which file leaked.
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert not offenders, (
        "a module with a conn/conn_with_offers fixture must not reference "
        "the production database variable (OUTBOUND_DB_TARGET) anywhere in "
        "the file — route the fixture through scratch_db_target / "
        f"OUTBOUND_TEST_DB_TARGET instead. offenders: {offenders}"
    )


# ── H7: a test must not depend on untracked gitignored data/ files ───────────
#
# The three real-database guard tests used to md5 data/outbound.db directly.
# data/*.db is gitignored as runtime state, so on a fresh clone the file did
# not exist and all three died with FileNotFoundError (ticket H7). H7 routed
# them through the real_outbound_db fixture (tests/conftest.py) so a fresh
# clone still exercises the guard. The walk below makes that a tested property:
# a test body that builds a Path (or open) to a gitignored data/ file MUST take
# the real_outbound_db fixture, so a future test cannot silently reintroduce a
# dependency on untracked runtime state. Same shape as H4a's
# test_no_converted_test_body_hand_builds_a_sqlite_path — an AST walk over test
# bodies — because the fixture-only walk cannot see a path built inside a body.

# The gitignored runtime-state patterns under data/ (.gitignore lines 13-17):
# data/*.db, data/*.db-journal, data/*.db-wal, data/*.db-shm, data/outbox/.
# gitignore's "*" does not cross "/", so the db files live directly under data/
# (one path segment); data/outbox/ ignores the whole directory. A TRACKED
# data/ file (data/hk_therapy_targets.csv, data/inbox/*.eml) matches none of
# these and is deliberately not flagged.
_GITIGNORED_DATA_RE = re.compile(
    r"^data/[^/]+\.db(?:-journal|-wal|-shm)?$|^data/outbox/"
)

# The fixture (ticket H7 step 1) that makes the real-database path legitimate:
# it creates the stand-in on a fresh clone and never touches a real database.
_REAL_OUTBOUND_DB_FIXTURE = "real_outbound_db"


def _gitignored_data_access(node):
    """Return the gitignored data/ path when ``node`` is a Path(...) or
    open(...) call whose first argument is a constant gitignored data/ path,
    else None.

    Only the pathlib.Path constructor and the builtin open() turn a data/
    string into a filesystem access. A bare string passed to a pure predicate
    (e.g. scratch_target_violation("data/demo.db")) reads nothing and is not
    flagged — that is a legit value-under-test, not a dependency on the file.
    """
    if not isinstance(node, ast.Call):
        return None  # Not a call at all.
    if not isinstance(node.func, ast.Name):
        return None  # A method/attribute call (e.g. Path.resolve) — the path is built elsewhere.
    if node.func.id not in ("Path", "open"):
        return None  # Only these two constructors open the filesystem.
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None  # No positional constant -> not a literal data/ path.
    arg = node.args[0].value
    if not isinstance(arg, str) or not _GITIGNORED_DATA_RE.match(arg):
        return None  # Not a gitignored data/ path (tracked data/ files pass through).
    return arg


def _iter_pytest_test_defs(tree):
    """Yield ``(display_name, func_node)`` for every pytest-collectable test
    def in one module: top-level ``test_*`` functions AND ``test_*`` methods
    inside classes.

    Classes are walked for two reasons: pytest collects class-based tests
    natively (``class TestFoo: def test_x(self): ...``), and this guard exists
    to catch a FUTURE reintroduction, not to document the past. Any class is
    walked regardless of its name — pytest's default ``python_classes`` is a
    ``*Test``-style glob, but being over-inclusive here is the safe direction:
    a false positive costs a deliberate review, a false negative reintroduces
    the fresh-clone failure silently.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            yield node.name, node  # A top-level test function.
        elif isinstance(node, ast.ClassDef):
            yield from _iter_class_test_methods(node, node.name)


def _iter_class_test_methods(cls, cls_dotted_name):
    """Yield ``(display_name, func_node)`` for ``test_*`` methods inside one
    class, recursing into nested classes (a class inside a class is collectable
    by pytest too)."""
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            yield f"{cls_dotted_name}.{node.name}", node  # Name the class so the failure is greppable.
        elif isinstance(node, ast.ClassDef):
            yield from _iter_class_test_methods(node, f"{cls_dotted_name}.{node.name}")


def _gitignored_data_accesses_without_fixture(path):
    """Yield ``(display_name, lineno, data_path)`` for test bodies in one file
    that build a Path/open to a gitignored data/ file WITHOUT the
    real_outbound_db fixture.

    Walks every pytest-collectable def — top-level ``test_*`` functions and
    ``test_*`` methods inside classes, FunctionDef OR AsyncFunctionDef (the same
    both-sync-and-async shape as the H4a guard) — and ast.walk()s each body so
    the offending call is found anywhere: an assignment, a call argument, or a
    nested expression. For a class-based test the fixture check reads the
    METHOD's args (e.g. ``def test_x(self, real_outbound_db)``), which is where
    pytest injects fixtures.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for display_name, func in _iter_pytest_test_defs(tree):
        arg_names = {a.arg for a in func.args.args}
        has_fixture = _REAL_OUTBOUND_DB_FIXTURE in arg_names
        for child in ast.walk(func):
            target = _gitignored_data_access(child)
            if target is not None and not has_fixture:
                yield display_name, child.lineno, target


def test_no_test_reads_a_gitignored_data_file_without_the_fixture():
    """A test body that builds Path("data/<gitignored>") / open(...) without
    the real_outbound_db fixture is the H7 bug class: it either dies on a
    fresh clone (file absent) or silently depends on untracked runtime state.
    A test WITH the fixture is fine — that is the sanctioned way to touch
    data/outbound.db (the fixture owns its stand-in and never touches a real
    database).

    COVERAGE BOUNDARY — the shapes this walk deliberately does NOT catch, so
    the next reader knows the guard's actual limit instead of over-trusting it:
    a path assembled from pieces (os.path.join("data", "x.db"),
    Path("data") / "x.db"), an f-string value (f"data/{name}.db"), and
    variable indirection (db = "data/x.db"; Path(db)) all escape this walk,
    which only sees a literal string passed directly to Path(...) or open(...).
    This guard is against accident, not malice — an author who builds the path
    from pieces has already thought about it, which is the friction the guard
    exists to create."""
    offenders = []
    for path in _pytest_collectable_python_files():
        for func_name, lineno, data_path in _gitignored_data_accesses_without_fixture(path):
            offenders.append(f"{path.stem}.{func_name}:{lineno} ({data_path})")
    assert not offenders, (
        "test bodies must not build Path/open to a gitignored data/ file "
        "(data/*.db, data/*.db-*, data/outbox/) without the real_outbound_db "
        f"fixture — on a fresh clone the file does not exist. offenders: {offenders}"
    )


# ── H7 review: a symlink at data/outbound.db must be left alone ──────────────
#
# The real_outbound_db fixture decides ownership from the filesystem. A BROKEN
# symlink (link target missing) reports exists() == False, so — before the
# is_symlink() branch was added — the fixture took the "absent -> create
# stand-in" path: write_bytes() wrote THROUGH the link (creating a brand-new
# file at the link's target) and teardown unlinked the operator's symlink
# itself, leaking a junk file where the database belongs. data/outbound.db is
# real runtime state that grows with every run; an operator symlinking it to
# external storage is an ordinary thing to do, and that volume can be unmounted.
# The fixture must treat ANY symlink as not-ours: never write through it, never
# remove it. These tests pin that rule.

def _run_real_outbound_db_fixture():
    """Drive real_outbound_db's generator to its yield and return (path, gen).

    __wrapped__ is the plain generator function underneath @pytest.fixture.
    The caller resumes the generator past the yield (next(gen) -> StopIteration)
    to run the fixture's teardown, because a test body runs AFTER fixture setup
    and so cannot let pytest run the teardown itself.
    """
    gen = _real_outbound_db_fixture.__wrapped__()
    return next(gen), gen


def test_real_outbound_db_leaves_a_broken_symlink_alone(tmp_path, monkeypatch):
    """A broken symlink at data/outbound.db must be treated as not-ours: no file
    may be created at the link target, and the symlink must survive both the
    fixture's setup and its teardown.  Guards the H7-review BLOCKER."""
    # chdir to a temp dir so the fixture's RELATIVE Path("data/outbound.db")
    # resolves under tmp_path and can never touch the repo's real database.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    target = tmp_path / "external_target.db"  # the link's destination — deliberately absent
    link = tmp_path / "data" / "outbound.db"
    link.symlink_to(target)  # a BROKEN symlink: the target does not exist

    _path, gen = _run_real_outbound_db_fixture()  # fixture setup, up to the yield

    # (a) No file was created THROUGH the broken link at its target...
    assert not target.exists(), "fixture wrote THROUGH the broken symlink"
    # ...and the link is still the operator's pointer.
    assert link.is_symlink(), "fixture removed the operator's symlink during setup"

    # Resume the generator past the yield to run the fixture's teardown.
    with pytest.raises(StopIteration):
        next(gen)
    # (b) Teardown must not have unlinked the symlink or created the target.
    assert link.is_symlink(), "teardown removed the operator's symlink"
    assert not target.exists(), "teardown left a file at the link target"


def test_real_outbound_db_leaves_a_resolving_symlink_alone(tmp_path, monkeypatch):
    """A symlink that points at an EXISTING file must also be left alone.  This
    case already behaved before the is_symlink() branch (exists() follows the
    link, and the SQLite-magic check says not-ours), but it pins the ONE rule —
    any symlink is not-ours — so a future fix cannot treat broken and resolving
    links differently."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    target = tmp_path / "external_target.db"
    target.write_bytes(b"real operator database bytes")  # a resolving link target
    link = tmp_path / "data" / "outbound.db"
    link.symlink_to(target)

    _path, gen = _run_real_outbound_db_fixture()  # fixture setup, up to the yield

    assert target.read_bytes() == b"real operator database bytes"  # untouched through the link
    assert link.is_symlink(), "fixture removed the operator's symlink during setup"

    with pytest.raises(StopIteration):
        next(gen)  # teardown: still not ours -> must not unlink
    assert link.is_symlink(), "teardown removed the operator's symlink"
    assert target.read_bytes() == b"real operator database bytes"
