"""
tests/test_demo_replay_script.py — RESULT-level guard for scripts/demo_replay.sh (ticket D1).

Same philosophy as tests/test_deploy_script_behaviour.py: run the REAL script
and assert on what it DOES, not on what its text says.  The D1 ticket's single
most important property is a SAFETY one — the replay console must serve a
DISPOSABLE COPY of the pristine source (data/e2e_run2.db), never the source
itself, so a judge clicking the always-rendered kill-switch toggle (the one
console write door not gated on target state) can never mutate the ONE pristine
copy of the restored run.  A text assertion on the script would stay green
through a regression that deleted the copy step; running it proves the copy is
made, that OUTBOUND_DB_TARGET points at the copy and NOT the source, and that
the source is never written.

Mechanics: every test runs the REAL script under subprocess from the repo root
with:
  - a fake python interpreter (PYTHON_BIN) that logs its invocation and the
    console-relevant env, then exits 0 — so no real server can ever start;
  - a stub `openssl` FIRST on PATH that logs and emits a fixed hex key — so no
    real key is generated;
  - a controlled tmp source (DEMO_REPLAY_SOURCE) and tmp scratch path
    (REPLAY_SCRATCH_PATH) — so the tests never touch the repo's real
    data/e2e_run2.db (gitignored, and absent on a fresh clone — H7).

COVERAGE BOUNDARY — what the stubs do and do not simulate:
  DO simulate: the launch (fake python), the key generation (stub openssl).
  DO NOT simulate: a real uvicorn server, real cp/cmp (both run for real on tmp
  files — they are harmless filesystem ops and are the point of the test), or
  real git (the repo's real git resolves the root, exactly as the script does).
"""

import json  # the controlled kill-switch source content (never the real config file)
import os
import subprocess
from pathlib import Path

import pytest

# The repo root and the REAL script — the tests run this file, never a copy.
ROOT = Path(__file__).resolve().parent.parent
REPLAY_SCRIPT = ROOT / "scripts" / "demo_replay.sh"

# The fixed key the openssl stub emits: 64 hex chars (openssl rand -hex 32).
FIXED_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

# The fake python interpreter: logs its full invocation + the four console
# env vars the script exports, then exits 0 — the script's uvicorn launch
# becomes a logged non-event instead of a real server.  OUTBOUND_KILL_SWITCH_PATH
# is the D1-fix addition: it is how the test proves the console's kill-switch
# toggle is pointed at the DISPOSABLE COPY, never the real config file.
_FAKE_PYTHON = r'''#!/bin/bash
echo "python $*" >> "$STUB_LOG_DIR/calls"
echo "OUTBOUND_DB_TARGET=$OUTBOUND_DB_TARGET" >> "$STUB_LOG_DIR/env"
echo "OUTBOUND_REPLAY_MODE=$OUTBOUND_REPLAY_MODE" >> "$STUB_LOG_DIR/env"
echo "OUTBOUND_CONSOLE_API_KEY=$OUTBOUND_CONSOLE_API_KEY" >> "$STUB_LOG_DIR/env"
echo "OUTBOUND_KILL_SWITCH_PATH=$OUTBOUND_KILL_SWITCH_PATH" >> "$STUB_LOG_DIR/env"
exit 0
'''

# The stub openssl: logs the invocation and emits FIXED_KEY (no trailing
# newline — the exact shape the real `openssl rand -hex 32` prints).
_OPENSSL_STUB = r'''#!/bin/bash
echo "openssl $*" >> "$STUB_LOG_DIR/calls"
printf '%s' "${STUB_OPENSSL_KEY:-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef}"
'''


class _Harness:
    """Bundle the per-test stub dir, log dir and ready-to-use env."""

    def __init__(self, *, env: dict, tmp_path: Path):
        self.env = env
        self.tmp_path = tmp_path


def _write_stubs(stub_dir: Path, log_dir: Path) -> Path:
    """Write the openssl stub (on PATH) and the fake python (via PYTHON_BIN)."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    openssl = stub_dir / "openssl"
    openssl.write_text(_OPENSSL_STUB, encoding="utf-8")
    openssl.chmod(0o755)  # a stub that cannot execute must fail loudly
    fake_py = stub_dir / "fake_python"
    fake_py.write_text(_FAKE_PYTHON, encoding="utf-8")
    fake_py.chmod(0o755)
    return fake_py


@pytest.fixture()
def harness(tmp_path):
    """Hermetic harness: stub dir FIRST on PATH, fake python via PYTHON_BIN."""
    stub_dir = tmp_path / "bin"
    log_dir = tmp_path / "log"
    fake_py = _write_stubs(stub_dir, log_dir)
    env = os.environ.copy()
    env["PATH"] = str(stub_dir) + os.pathsep + env.get("PATH", "")
    env["STUB_LOG_DIR"] = str(log_dir)
    env["PYTHON_BIN"] = str(fake_py)
    # D1-fix: every test also points the KILL-SWITCH source and scratch at tmp
    # paths (the controlled source is created here so the script's refuse-loud
    # missing-switch check passes for every happy path).  This mirrors how
    # DEMO_REPLAY_SOURCE / REPLAY_SCRATCH_PATH protect the real
    # data/e2e_run2.db: a test can never read or write the real, git-TRACKED
    # config/kill_switch.json, even if it fails partway.
    env["DEMO_REPLAY_KILL_SWITCH_SOURCE"] = str(_make_kill_switch_source(tmp_path))
    env["REPLAY_KILL_SWITCH_SCRATCH_PATH"] = str(tmp_path / "kill_switch_scratch.json")
    # The script reads git to resolve the repo root; make sure it finds the
    # real one (the stub dir's fake python is only the uvicorn interpreter).
    return _Harness(env=env, tmp_path=tmp_path)


def _run(env: dict) -> subprocess.CompletedProcess:
    """Run the real script under bash, bounded, from the repo root."""
    return subprocess.run(
        ["bash", str(REPLAY_SCRIPT)],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=90,
    )


def _log_lines(log_dir: Path, name: str) -> list[str]:
    """Read a stub log file as a list of stripped lines ([] when never written)."""
    p = log_dir / name
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _calls(log_dir: Path) -> list[str]:
    """Every fake-python/openssl invocation, in order — the script's trace."""
    return _log_lines(log_dir, "calls")


def _env(log_dir: Path) -> list[str]:
    """The env vars the fake python observed at launch time."""
    return _log_lines(log_dir, "env")


def _make_source(tmp_path: Path, name: str = "source.db") -> Path:
    """A controlled tmp 'pipeline run' for the script to replay."""
    source = tmp_path / name
    # Not a real sqlite file — the script never opens it, it only copies it.
    source.write_bytes(b"fake-sqlite-bytes")
    return source


def _make_kill_switch_source(tmp_path: Path, name: str = "kill_switch_source.json") -> Path:
    """A controlled tmp 'kill switch file' for the script to copy — NEVER the
    real config/kill_switch.json.  Written in the runbook.md §1 shape, so the
    copy test can prove byte-identical content lands at the scratch path.  The
    D1-fix tests must control BOTH ends (source + scratch) via env exactly like
    the database test, so a test that fails partway can never leave the real
    tracked switch mutated."""
    source = tmp_path / name
    source.write_text(
        json.dumps(
            {"enabled": False, "updated_at": "2026-07-30T00:00:00Z", "updated_by": "test"}
        ),
        encoding="utf-8",
    )
    return source


# ── The scenarios ────────────────────────────────────────────────────────────


def test_missing_source_refuses_loudly_and_never_launches(harness):
    """A missing/unreadable source must exit 1 with a clear message — and
    must NOT create a scratch copy or launch anything (the D1 contract: refuse,
    never regenerate, never silently degrade)."""
    env = dict(harness.env)
    missing = str(harness.tmp_path / "no_such_dir" / "missing.db")
    scratch = str(harness.tmp_path / "scratch.db")
    env["DEMO_REPLAY_SOURCE"] = missing
    env["REPLAY_SCRATCH_PATH"] = scratch

    proc = _run(env)

    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}"
    assert "does not exist" in proc.stderr
    assert "cannot be regenerated" in proc.stderr  # the why, not just the what
    assert not Path(scratch).exists(), "no scratch copy may be made without a source"
    assert _calls(harness.tmp_path / "log") == [], (
        "nothing may be launched when the source is missing"
    )


def test_serves_scratch_copy_never_the_source(harness):
    """THE safety property: the console is launched against a DISPOSABLE COPY
    that is byte-identical to the source, never the source itself — and the
    source is never written. Pinned at the result level because a regression
    that deleted the copy step would keep every text-level assertion green."""
    env = dict(harness.env)
    source = _make_source(harness.tmp_path)
    scratch = harness.tmp_path / "scratch" / "replay.db"
    env["DEMO_REPLAY_SOURCE"] = str(source)
    env["REPLAY_SCRATCH_PATH"] = str(scratch)
    before = source.read_bytes()

    proc = _run(env)

    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}"
    # A separate file exists, byte-identical to the source…
    assert scratch.exists()
    assert scratch.read_bytes() == before
    assert scratch.resolve() != source.resolve()
    # …and the source was never written.
    assert source.read_bytes() == before

    # The launch env points the console at the COPY, never the source, in
    # replay mode.
    env_lines = _env(harness.tmp_path / "log")
    assert f"OUTBOUND_DB_TARGET={scratch}" in env_lines, env_lines
    assert f"OUTBOUND_DB_TARGET={source}" not in env_lines, env_lines
    assert "OUTBOUND_REPLAY_MODE=1" in env_lines, env_lines

    # The exact README §6 invocation was used.
    calls = _calls(harness.tmp_path / "log")
    assert any(
        ln.startswith("python -m uvicorn app.console.app:app --port 8080")
        for ln in calls
    ), calls


def test_kill_switch_served_from_scratch_copy_never_the_real_switch(harness):
    """THE second half of the safety property (ticket D1-fix): the console's
    kill-switch toggle (write_kill_switch resolves OUTBOUND_KILL_SWITCH_PATH,
    app/kill_switch.py) must be pointed at a DISPOSABLE COPY of the switch
    file, never the real git-TRACKED config/kill_switch.json — whose mutation
    during a demo would flip the operator's REAL, live switch for every OTHER
    run on this machine.  The test controls BOTH ends
    (DEMO_REPLAY_KILL_SWITCH_SOURCE and REPLAY_KILL_SWITCH_SCRATCH_PATH)
    exactly like the database test, so the real repo file is never read or
    written — even if this test fails partway."""
    env = dict(harness.env)
    # Same hermetic DB control as the database test — this test is about the
    # switch, but it must not touch the real run either.
    env["DEMO_REPLAY_SOURCE"] = str(_make_source(harness.tmp_path))
    env["REPLAY_SCRATCH_PATH"] = str(harness.tmp_path / "scratch.db")
    source = Path(env["DEMO_REPLAY_KILL_SWITCH_SOURCE"])
    # A fresh nested scratch path proves the script creates the parent dir,
    # exactly like the DB scratch test.
    scratch = harness.tmp_path / "ks_scratch" / "kill_switch.json"
    env["REPLAY_KILL_SWITCH_SCRATCH_PATH"] = str(scratch)
    before = source.read_bytes()

    proc = _run(env)

    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}"
    # A separate file exists, byte-identical to the controlled source…
    assert scratch.exists()
    assert scratch.read_bytes() == before
    assert scratch.resolve() != source.resolve()
    # …and the controlled source was never written.
    assert source.read_bytes() == before

    # The launch env points the toggle at the COPY, never the source — and
    # the copy is NOT the real tracked config/kill_switch.json.
    env_lines = _env(harness.tmp_path / "log")
    assert f"OUTBOUND_KILL_SWITCH_PATH={scratch}" in env_lines, env_lines
    assert f"OUTBOUND_KILL_SWITCH_PATH={source}" not in env_lines, env_lines
    assert (
        f"OUTBOUND_KILL_SWITCH_PATH={ROOT / 'config' / 'kill_switch.json'}"
        not in env_lines
    ), env_lines
    # The DB side still points at its own scratch copy (the two protections
    # coexist in one launch env).
    assert f"OUTBOUND_DB_TARGET={env['REPLAY_SCRATCH_PATH']}" in env_lines, env_lines


def test_reuses_existing_api_key_without_calling_openssl(harness):
    """An operator's pre-set OUTBOUND_CONSOLE_API_KEY is reused verbatim (never
    overwritten) and openssl is never invoked."""
    env = dict(harness.env)
    env["DEMO_REPLAY_SOURCE"] = str(_make_source(harness.tmp_path))
    env["REPLAY_SCRATCH_PATH"] = str(harness.tmp_path / "scratch.db")
    env["OUTBOUND_CONSOLE_API_KEY"] = "preset-operator-key"

    proc = _run(env)

    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}"
    assert "Reusing existing OUTBOUND_CONSOLE_API_KEY" in proc.stdout
    openssl_calls = [
        ln for ln in _calls(harness.tmp_path / "log") if ln.startswith("openssl")
    ]
    assert openssl_calls == [], "openssl must not run when a key is already set"
    assert "OUTBOUND_CONSOLE_API_KEY=preset-operator-key" in _env(
        harness.tmp_path / "log"
    )


def test_generates_and_prints_api_key_when_unset(harness):
    """With no key in the environment, the script generates one via openssl,
    prints it unmissably (with the Basic-auth username contract), and passes it
    to the console."""
    env = dict(harness.env)
    env["DEMO_REPLAY_SOURCE"] = str(_make_source(harness.tmp_path))
    env["REPLAY_SCRATCH_PATH"] = str(harness.tmp_path / "scratch.db")
    env.pop("OUTBOUND_CONSOLE_API_KEY", None)

    proc = _run(env)

    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}"
    calls = _calls(harness.tmp_path / "log")
    assert any(ln.startswith("openssl rand -hex 32") for ln in calls), calls
    assert "username: operator" in proc.stdout
    assert FIXED_KEY in proc.stdout  # printed clearly and unmissably
    assert f"OUTBOUND_CONSOLE_API_KEY={FIXED_KEY}" in _env(
        harness.tmp_path / "log"
    )
