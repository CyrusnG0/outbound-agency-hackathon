# tests/conftest.py — suite-wide hermeticity guard (ticket B1c).
#
# WHAT THIS PREVENTS
# ------------------
# This is a cost-and-correctness control, not ceremony.  Ticket B1b put a real
# ADK LlmAgent into build_phase1_agent, and two test files build the full
# pipeline.  tests/test_phase1_cli.py::test_cli_runs_a_small_batch_end_to_end
# patched fetch_sources / call_structured / _call_detect_signals but NOT the
# research LlmAgent, so with the operator's real .env present it made live
# billable Vertex calls and real network requests to acme.test on every pytest
# run — 37 seconds of wall time for one test (the suite went from ~11s to
# ~49s).  Worse, its only assertion was `exit_code == 0`, so it could not tell
# a working pipeline from a completely dead research stage: with
# GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_CLOUD_PROJECT / GOOGLE_API_KEY all unset
# (a state in which the research agent cannot work at all) it still passed.
# This was the third unmasked model boundary to leak into the suite (B1b caught
# the same class of bug in tests/test_phase1_checkpoint.py).  Fixing one test
# is not enough — this autouse fixture makes the NEXT unmocked boundary fail
# loudly and instantly instead of silently costing money and 30 seconds.
#
# WHY ONE PATCH TARGET IS ENOUGH (verified against the pinned SDKs)
# -----------------------------------------------------------------
# Both consumers resolve the SAME module attribute at call time:
#   - app/llm.py does `from google import genai` at module top and then
#     `genai.Client(...)` inside _build_client — an attribute lookup on the
#     google.genai module object at call time.
#   - ADK's GoogleLlm (google/adk/models/google_llm.py, google-adk==2.7.1)
#     does `from google.genai import Client` INSIDE the lazy `api_client`
#     cached_property — `from X import Y` is also a getattr on the
#     google.genai module at call time.  (Constructing an LlmAgent alone never
#     builds the client — it is only built when an LLM request is first made.)
# Patching the single attribute `google.genai.Client` therefore intercepts
# BOTH paths.  google.genai.types does NOT re-export Client (checked), and
# google.genai's __all__ is ['Client', 'interactions', 'types'].
#
# WHY THE STUB IS A CLASS, NOT A FUNCTION
# --------------------------------------
# A plain function would make any `isinstance(x, genai.Client)` check in code
# under test raise TypeError ("arg 2 must be a type").  A class keeps
# isinstance checks working (they simply return False) and only the
# construction itself — the live boundary — raises.

import os
from pathlib import Path  # H7: resolving the real-database stand-in path the three guard tests share

import pytest

import google.genai as genai_module

from app.db import reset_scratch_database, scratch_target_violation


def pytest_configure(config):
    # Register the opt-out marker so `pytest --strict-markers` never complains
    # about it.  No test in the repo uses it today; it exists so a future live
    # smoke test can be written DELIBERATELY (and reviewed) instead of a test
    # accidentally reaching Vertex and billing the operator.
    config.addinivalue_line(
        "markers",
        "live_model: opt out of the autouse guard that blocks real "
        "google.genai.Client construction — use ONLY for a deliberate live "
        "smoke test, never for routine pipeline tests",
    )


@pytest.fixture(autouse=True)
def _block_live_genai_clients(request, monkeypatch):
    """Refuse (not stub) any test that tries to construct a real genai client.

    Autouse on every test in the suite.  The guard REPLACES google.genai.Client
    with a class whose __init__ raises — it never silently fakes model
    responses, because a test passing against a fake is exactly the vacuous
    exit_code==0 failure this ticket exists to kill.  A test that needs a
    client must patch the model boundary ITSELF (that patch replaces this
    guard for the duration of the test — see tests/test_llm.py, which patches
    app.llm.genai.Client with its own mock), or opt out with
    @pytest.mark.live_model for a genuinely live smoke test.
    """
    if request.node.get_closest_marker("live_model"):
        # Explicit opt-out: a test that genuinely intends a live call.  The
        # marker is registered in pytest_configure above; nothing in the repo
        # uses it today.
        yield
        return

    nodeid = request.node.nodeid  # captured now: the error below must name the test

    class _GuardedClient:
        """Stand-in whose ONLY job is to refuse construction loudly."""

        def __init__(self, *args, **kwargs):
            # This raise IS the guard: constructing a real client is the
            # billable boundary, so the failure names the test and the fix
            # instead of letting the test spend money and 30 seconds silently.
            raise RuntimeError(
                f"{nodeid} attempted to construct a real google.genai.Client. "
                f"The autouse hermeticity guard in tests/conftest.py blocks live model "
                f"clients so the suite can never make billable Vertex/API calls "
                f"(ticket B1c: an unmasked research LlmAgent once cost 37s per pytest "
                f"run while its only assertion stayed green). Patch the model boundary "
                f"in this test — e.g. patch 'app.llm.genai.Client' with a mock, or "
                f"patch 'app.agents.phase1.build_research_agent' with the offline "
                f"FetchAndNormalizeNode stand-in — or, ONLY for a deliberate live smoke "
                f"test, mark it with @pytest.mark.live_model."
            )

    # Replace the single resolution point both app/llm.py and ADK look up at
    # call time (see the module docstring for why one target covers both).
    monkeypatch.setattr(genai_module, "Client", _GuardedClient)
    yield
    # monkeypatch restores the real Client when the test ends, so a
    # deliberately live test later in the session is unaffected.


# A NEW, SEPARATE environment variable — deliberately NOT OUTBOUND_DB_TARGET.
# OUTBOUND_DB_TARGET is the repo-wide convention for where the operator's real
# database is (docs/gcp-setup.md §6); the console and every CLI read it.  The
# scratch_db_target fixture below is destructive: it empties its target before
# every test.  Binding it to OUTBOUND_DB_TARGET would mean that an operator who
# has their production Cloud SQL instance exported in their shell and then
# types `pytest` destroys it.
TEST_DB_TARGET_ENV = "OUTBOUND_TEST_DB_TARGET"


@pytest.fixture
def scratch_db_target(tmp_path):
    """Return the per-test database target, honouring OUTBOUND_TEST_DB_TARGET.

    Deliberately NOT autouse: only the modules whose conn/conn_with_offers
    fixture asks for it get dialect-aware behaviour.  Every other module keeps
    its own explicit target.
    """
    # `not target` (not `is None`): an exported-but-EMPTY variable is a
    # misconfiguration that must fall back to the SQLite default, not reach
    # reset_scratch_database("") — which would be a Path("").unlink().
    target = os.environ.get(TEST_DB_TARGET_ENV)
    if not target:
        # Default path — the exact SQLite tmp file every conn fixture used to
        # build by hand, so a plain `pytest` (no env var) is unchanged.
        yield str(tmp_path / "test.db")
        return
    # Configured path — the target may be a URL (Postgres) or a file.  A
    # misconfiguration must fail loudly, never skip: a suite that silently
    # skips when misconfigured is how the "1 of 10" coverage gap arose.
    violation = scratch_target_violation(target)
    if violation is not None:
        # Fail, do not skip: name the variable and the marker rule so the
        # operator can correct the DSN instead of losing coverage silently.
        pytest.fail(
            f"{TEST_DB_TARGET_ENV}={target!r} is not a safe scratch target: "
            f"{violation}"
        )
    # Reset is per-test, so tests stay isolated exactly as tmp_path isolated
    # them before: every test starts from an empty database.
    reset_scratch_database(target)
    yield target


# ── H7: the real-database guard tests need data/outbound.db to exist ─────────
#
# Three tests (test_demo_seed::test_guard_refuses_real_outbound_db,
# test_adversarial_sim::test_report_refuses_real_outbound_db,
# test_conversation_sim::test_converse_refuses_real_outbound_db) prove the
# CLI's real-database guard by md5-fingerprinting data/outbound.db before and
# after the refusal. On a fresh `git clone` that file does not exist —
# .gitignore excludes data/*.db as runtime state — so all three died with
# FileNotFoundError before reaching their assertion (ticket H7). This fixture
# creates a stand-in at that exact path when none exists, so the guard is
# genuinely exercised on a fresh clone, and removes ONLY a file it created. If
# data/outbound.db ALREADY exists, it is the operator's real database: the
# fixture uses it exactly as the old tests did and never touches it (in
# particular, never deletes it in teardown) — the exact disaster
# scratch_db_target was designed around in H3.

# The sentinel prefix distinguishing a fixture-created stand-in from the
# operator's real database. A real SQLite file begins with the 16-byte magic
# "SQLite format 3\x00" (app/db.py opens data/outbound.db as SQLite), so a file
# whose first bytes are this ASCII sentinel can ONLY be a stand-in this fixture
# wrote — never the operator's data. This is how a leftover stand-in from a
# CRASHED prior run stays recognisably OURS: the next run detects the sentinel,
# replaces it with a fresh stand-in, and removes it at teardown, instead of
# mistaking it for the operator's database (or, worse, treating an unrecognised
# file as disposable).
_H7_STANDIN_MARKER = b"OUTBOUND_H7_TEST_STANDIN_MARKER\n"

# The stand-in's payload after the marker. Arbitrary, but FIXED for the run, so
# the md5-before/after proof runs against known bytes: the test knows exactly
# what the file contained before the CLI attempt, which is strictly stronger
# than fingerprinting an unknown real file. 1 KiB so the file is non-trivial —
# a zero-byte file would make the "before" state trivially reproducible by a
# stray create, weakening the proof.
_H7_STANDIN_BODY = b"x" * 1024


def _is_h7_stand_in(path: Path) -> bool:
    """True when ``path`` is a stand-in this fixture wrote (survives a crashed
    run).  Reads only the marker length, never the whole file, so checking the
    operator's real multi-MB database costs bytes, not megabytes.

    Raises PermissionError if the file exists but is unreadable — the exception
    propagates uncaught, which is fail-safe: an unreadable file is never
    silently treated as disposable (the loud failure touches nothing)."""
    if not path.is_file():
        return False  # No file -> nothing to recognise -> not ours.
    with path.open("rb") as f:
        # First N bytes only — enough to tell the sentinel from SQLite magic.
        head = f.read(len(_H7_STANDIN_MARKER))
    return head == _H7_STANDIN_MARKER  # Sentinel match => fixture-created stand-in.


def _create_h7_stand_in(path: Path) -> None:
    """Write a fresh stand-in file at ``path``: the marker prefix + fixed body."""
    # data/ is tracked in the repo (hk_therapy_targets.csv, inbox/), but be
    # defensive — a sparse checkout may lack it, and mkdir is idempotent.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic per run, so the md5 proof has known "before" bytes.
    path.write_bytes(_H7_STANDIN_MARKER + _H7_STANDIN_BODY)


@pytest.fixture
def real_outbound_db():
    """Yield the path to data/outbound.db, creating a stand-in only when none
    exists (or the only file there is a leftover stand-in from a crashed run).

    The fixture owns a file ONLY when it created it: teardown removes a file
    the fixture created and never touches a file that was already present and
    is not recognisably ours. That is the H7 safety rule — pytest must never
    delete the operator's real database.
    """
    # The exact path the three guard CLIs refuse (app/demo_seed.py REAL_DB_PATH).
    path = Path("data/outbound.db")
    created = False  # Ownership flag: True only for a file THIS fixture created.
    if path.is_symlink():
        # A symlink — broken or not — is an existing thing the fixture did NOT
        # create, so it is never ours. is_symlink() MUST be tested before
        # exists(): exists() follows the link and returns False for a BROKEN
        # link, which would otherwise fall into the "absent" branch and
        # write_bytes() would write THROUGH the link (creating a new file at
        # the link's target) while teardown unlinked the operator's symlink
        # itself. A resolving link would already be caught by the SQLite-magic
        # check below, but routing both through this single branch keeps the
        # rule one rule, not two accidents. Either way: touch nothing, and
        # `created` stays False so teardown never removes it.
        created = False  # explicit, not reliant on the initial default: a symlink is never ours.
    elif not path.exists():
        # Fresh tree (a fresh clone): no file at all — create our stand-in so
        # the guard is exercised rather than skipped.
        _create_h7_stand_in(path)
        created = True
    elif _is_h7_stand_in(path):
        # A leftover stand-in from a crashed prior run: the sentinel proves it
        # is ours, so it is safe to overwrite with a fresh stand-in (and it
        # WILL be removed at teardown). A real database can never match the
        # sentinel (SQLite magic), so this branch can never touch operator data.
        _create_h7_stand_in(path)
        created = True
    # else: an existing file that is not recognisably ours — the operator's real
    # database (or any other real file at that path). Use it exactly as the old
    # tests did and leave it untouched: `created` stays False, so teardown is a
    # no-op for it. This is the H3 disaster-avoidance rule, verbatim.
    yield path
    if created:
        # Only a file this fixture created may be removed. missing_ok=True: if
        # the test itself already removed it, teardown must not fail.
        path.unlink(missing_ok=True)
