"""
tests/test_deploy_script_behaviour.py — RESULT-level guard for the deploy script (ticket H14).

This is the repo's first test that RUNS the real scripts/deploy_console.sh and
asserts on what it DOES, not on what it says. Text assertions
(tests/test_deploy_artifacts.py) stayed green through three execution-level
defects, all found only by running the script:

  1. IMAGE defaulted to a hardcoded stale tag, so every deploy shipped pre-H11
     code and the console sat on a public URL with no auth.
  2. The "was IMAGE pinned?" flag was captured AFTER the default assignment, so
     it was always true and the build-and-push branch was dead code that never
     once executed.
  3. `docker build ... .` used the operator's cwd, so it failed outright when
     run from scripts/.

None of these is visible in the file text, which is exactly why they shipped.
The lead verified the current script by running it end-to-end with stubbed
gcloud/docker/curl on PATH; those runs are now permanent tests.

Mechanics: every test executes the REAL script under subprocess with a stub
directory FIRST on PATH, so no cloud call, no docker build and no network
request can ever happen. The stubs log every invocation and emit scripted
HTTP codes / IAM results; the tests assert on the resulting exit code, the IAM
binding log and the exact gcloud invocation line — execution properties a
file-text test cannot reach.

COVERAGE BOUNDARY — what the stubs do and do not simulate:
  DO simulate: gcloud run deploy success/failure, `services describe` (the
  deployed URL), add/remove-iam-policy-binding (logged), secrets describe,
  artifacts describe, docker build/push/imagetools-inspect, and curl HTTP
  status codes fed through the smoke loops.
  DO NOT simulate: real gcloud IAM semantics (add-iam-policy-binding is LOGGED,
  never applied), real IAM propagation timing (the settle window is collapsed
  to 6s and every scenario resolves on the FIRST probe), real Cloud Run
  revision creation, or real docker builds. A regression that only appears
  against real GCP — a flag gcloud rejects, a quota error, a credential
  failure — is outside these tests. They prove the script's CONTROL FLOW and
  MUTATION ORDER, not that the cloud calls are valid.
"""

import os
import subprocess
from pathlib import Path

import pytest

# The repo root and the REAL script — the tests run this file, never a copy.
ROOT = Path(__file__).resolve().parent.parent
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy_console.sh"

# Collapse the smoke-check settle window so passing scenarios resolve on the
# first probe and failing ones break immediately. 6s/2s still exercises the
# bounded-retry loop shape without the 180s production default (CLAUDE.md §7:
# retries must be bounded).
SMOKE_WINDOW_SECONDS = "6"
SMOKE_INTERVAL_SECONDS = "2"


# ── Stub sources ───────────────────────────────────────────────────────────────
# Written verbatim into a tmp_path/bin directory by _write_stubs, so every run
# is hermetic and leaves nothing behind. Each stub appends its full invocation
# to $STUB_LOG_DIR/calls so the tests can assert on call PRESENCE and ORDER —
# the whole point of running the real script instead of grepping it.

_GCLOUD_STUB = r'''#!/bin/bash
# Log every invocation first: the tests assert on what gcloud was asked to do.
echo "gcloud $*" >> "$STUB_LOG_DIR/calls"
case "$*" in
  *"run deploy"*)
      # Simulate a failed revision when STUB_DEPLOY_FAILS=true (scenario 2:
      # today's incident — gcloud applies the IAM flag BEFORE the revision).
      if [[ "${STUB_DEPLOY_FAILS:-false}" == "true" ]]; then
        echo "ERROR: (gcloud.run.deploy) simulated revision failure" >&2; exit 1
      fi
      echo "Service [outbound-console] deployed."; exit 0 ;;
  *"services describe"*) echo "https://stub-console.example.test"; exit 0 ;;
  *"add-iam-policy-binding"*)  echo "added" >> "$STUB_LOG_DIR/iam"; exit 0 ;;
  *"remove-iam-policy-binding"*) echo "removed" >> "$STUB_LOG_DIR/iam"; exit 0 ;;
  *"secrets describe"*) exit 0 ;;
  *"artifacts docker images describe"*) exit 0 ;;
  *) exit 0 ;;
esac
'''

_DOCKER_STUB = r'''#!/bin/bash
echo "docker $*" >> "$STUB_LOG_DIR/calls"
case "$*" in
  *imagetools*)
      # The script inspects with --format '{{json .Image}}', and the two real
      # manifest shapes print DIFFERENTLY -- which is the whole point of this
      # stub. A single-platform manifest (what --provenance=false produces)
      # yields one object with "architecture"/"os"; a multi-platform index
      # yields a MAP keyed by platform. The first version of the script's
      # check only understood the index shape and therefore REJECTED every
      # correctly built amd64 image. Both shapes are simulated so that
      # regression can never come back silently.
      plat="${STUB_PLATFORM:-linux/amd64}"
      arch="${plat#*/}"
      if [[ "${STUB_MANIFEST_SHAPE:-single}" == "index" ]]; then
        printf '{\n  "%s": {\n    "architecture": "%s",\n    "os": "linux"\n  }\n}\n' "${plat}" "${arch}"
      else
        printf '{\n  "architecture": "%s",\n  "os": "linux"\n}\n' "${arch}"
      fi
      exit 0 ;;
  *) exit 0 ;;
esac
'''

_CURL_STUB = r'''#!/bin/bash
# Serve successive HTTP codes from $STUB_CODES (space-separated), repeating the
# last one once exhausted — mirrors the smoke-check retry loop's assumption that
# a settled service returns a STABLE code on every probe. $STUB_LOG_DIR/n is the
# per-test cursor so codes advance exactly once per curl call.
i=$(cat "$STUB_LOG_DIR/n" 2>/dev/null || echo 0)
arr=(${STUB_CODES})
last=$(( ${#arr[@]} - 1 ))
idx=$(( i < last ? i : last ))
echo -n "${arr[$idx]:-000}"
echo $((i+1)) > "$STUB_LOG_DIR/n"
'''


class _Harness:
    """Bundle the per-test stub dir, log dir and ready-to-use env for the subprocess."""

    def __init__(self, *, stub_dir: Path, log_dir: Path, env: dict):
        self.stub_dir = stub_dir
        self.log_dir = log_dir
        self.env = env


def _write_stubs(stub_dir: Path, log_dir: Path) -> None:
    """Write the gcloud/docker/curl stubs into stub_dir (executable) and ensure log_dir exists."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, src in (
        ("gcloud", _GCLOUD_STUB),
        ("docker", _DOCKER_STUB),
        ("curl", _CURL_STUB),
    ):
        p = stub_dir / name
        p.write_text(src, encoding="utf-8")
        p.chmod(0o755)  # a stub that cannot execute must fail loudly, never fall through


def _base_env(stub_dir: Path, log_dir: Path) -> dict:
    """Env for the script subprocess: stubs FIRST on PATH + the test's run switches.

    STUB_LOG_DIR tells the stubs where to write their logs. ALLOW_DIRTY=true
    bypasses the script's dirty-tree guard (the tree is dirty while the H14 fix
    is uncommitted, and the guard would otherwise refuse to run at all).
    SMOKE_WINDOW/INTERVAL collapse the settle loops to first-probe speed.
    STUB_CODES has a safe default for scenarios that never reach a probe.
    """
    env = os.environ.copy()
    env["PATH"] = str(stub_dir) + os.pathsep + env.get("PATH", "")
    env["STUB_LOG_DIR"] = str(log_dir)
    env["ALLOW_DIRTY"] = "true"
    env["SMOKE_WINDOW_SECONDS"] = SMOKE_WINDOW_SECONDS
    env["SMOKE_INTERVAL_SECONDS"] = SMOKE_INTERVAL_SECONDS
    env.setdefault("STUB_CODES", "403")
    return env


def _assert_stubs_resolve(stub_dir: Path, env: dict) -> None:
    """Fail the test if any of gcloud/docker/curl would resolve to a REAL binary.

    The whole hermeticity contract rests on the stub directory being first on
    PATH. If a stub is missing, `command -v` resolves to the operator's real
    gcloud/docker/curl and the script would silently reach the cloud — the exact
    thing these tests exist to prevent. Resolve explicitly so a missing stub
    FAILS the test instead of silently falling through (CLAUDE.md §7).
    """
    for tool in ("gcloud", "docker", "curl"):
        proc = subprocess.run(
            ["bash", "-c", f"command -v {tool}"],
            env=env, capture_output=True, text=True, timeout=15,
        )
        assert proc.returncode == 0, (
            f"command -v {tool} failed inside the harness PATH — PATH={env['PATH']!r}"
        )
        resolved = Path(proc.stdout.strip())
        assert resolved == stub_dir / tool, (
            f"{tool} resolved to {resolved}, not the stub at {stub_dir / tool}. "
            "A real binary would be reached on every run — the test is not "
            "hermetic and must fail."
        )


def _run(script_path: Path, env: dict) -> subprocess.CompletedProcess:
    """Run the script under bash, bounded, from the repo root so git resolves.

    The script is always run from ROOT (never the operator's arbitrary cwd):
    git rev-parse --show-toplevel works from any subdir, but a known cwd keeps
    the run deterministic. timeout=90 bounds the whole run; every scenario
    resolves in well under it.
    """
    return subprocess.run(
        ["bash", str(script_path)],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=90,
    )


def _log_lines(log_dir: Path, name: str) -> list[str]:
    """Read a stub log file as a list of stripped lines ([] when never written)."""
    p = log_dir / name
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _calls(log_dir: Path) -> list[str]:
    """Every gcloud/docker invocation, in order — the script's mutation trace."""
    return _log_lines(log_dir, "calls")


def _iam(log_dir: Path) -> list[str]:
    """The add/remove-iam-policy-binding log, in order ('added'/'removed' lines)."""
    return _log_lines(log_dir, "iam")


def _deploy_line(calls: list[str]) -> str:
    """The single gcloud run deploy invocation the script logged (fails if absent)."""
    deploy = [ln for ln in calls if ln.startswith("gcloud run deploy")]
    assert deploy, "the gcloud run deploy invocation was never logged"
    return deploy[0]


def _assert_deploy_uses_least_privilege_sa(deploy: str) -> None:
    """The deploy must pin the dedicated least-privilege runtime SA (S1).

    S1 (commit 888dede) added --service-account so the console runs as the
    dedicated `outbound-console-runtime@<project>.iam.gserviceaccount.com`
    identity — NOT the project's default COMPUTE service account, which GCP
    gives project-wide roles/editor by default (a blast radius the read-only
    console must not inherit). This is a RESULT-level guard for the same
    reason the rest of this file runs the real script instead of grepping it:
    deleting the --service-account line silently reverts the console to the
    default compute SA and every closure test above stays green — only the
    assembled gcloud invocation (captured by the stub) shows the regression.
    """
    # The flag itself must survive on the assembled gcloud run deploy line. A
    # future edit that drops it reverts the console to the default compute SA.
    assert "--service-account" in deploy, (
        "gcloud run deploy no longer carries --service-account — the console "
        "would run as the project's default COMPUTE service account "
        "(project-wide roles/editor) instead of the dedicated least-privilege "
        "runtime SA (S1)."
    )
    # And it must appear EXACTLY ONCE. gcloud's argument parsing is LAST-WINS
    # for a repeated --service-account flag, so a second occurrence silently
    # OVERRIDES the first. The exact scenario this catches: a careless merge or
    # copy-paste appends another --service-account pointing at the project's
    # default COMPUTE SA right after the dedicated one — the deploy then runs as
    # that over-privileged default identity, reverting precisely the blast
    # radius S1 was built to remove — while BOTH substring assertions above
    # still find their match on the line and pass, keeping the suite green. This
    # gap was found by adversarial review, not by the original S1 test design.
    assert deploy.count("--service-account") == 1, (
        "gcloud run deploy carries --service-account more than once. gcloud "
        "parses a repeated flag LAST-WINS, so a second --service-account "
        "(e.g. a merge/copy-paste appending the default COMPUTE SA) would "
        "silently override the dedicated least-privilege runtime SA and the "
        "console could end up running as an unintended identity — the exact "
        "blast radius S1 was built to remove. Keep exactly one --service-account."
    )
    # And it must pin THIS SA, not just any SA — a rewritten/typo'd value would
    # silently change the container's identity and its blast radius.
    assert "outbound-console-runtime@" in deploy, (
        "gcloud run deploy's --service-account is not the dedicated "
        "outbound-console-runtime@<project>.iam.gserviceaccount.com identity — "
        "the console would not run as its least-privilege runtime SA (S1)."
    )


def _assert_deploy_stays_closed(deploy: str) -> None:
    """The deploy call itself must never open the service — the core H14 contract.

    gcloud run deploy is NOT atomic: it applies its IAM flag BEFORE the revision
    succeeds, so a bare --allow-unauthenticated on this line would leave a
    failed deploy publicly reachable on the old image (the H14 incident). The
    deploy must always carry --no-allow-unauthenticated, and opening must be a
    SEPARATE later step. Also verifies the S1 least-privilege SA pin (see
    _assert_deploy_uses_least_privilege_sa) so every caller of this deploy-
    contract check guards both regressions.
    """
    assert "--no-allow-unauthenticated" in deploy, (
        "gcloud run deploy no longer carries --no-allow-unauthenticated — a "
        "failed deploy could then leave the service publicly reachable (H14)."
    )
    assert "--allow-unauthenticated" not in deploy, (
        "gcloud run deploy carries a BARE --allow-unauthenticated — gcloud "
        "applies it BEFORE the revision succeeds, so a failed deploy opens the "
        "service (the H14 incident). Opening must be a separate step."
    )
    # S1: every scenario that verifies the deploy contract must also verify the
    # console runs as the dedicated least-privilege runtime SA, never the
    # default compute SA.
    _assert_deploy_uses_least_privilege_sa(deploy)


@pytest.fixture()
def harness(tmp_path):
    """Per-test hermetic harness: stubs written to tmp_path, PATH verified, env ready."""
    stub_dir = tmp_path / "bin"
    log_dir = tmp_path / "log"
    _write_stubs(stub_dir, log_dir)
    env = _base_env(stub_dir, log_dir)
    _assert_stubs_resolve(stub_dir, env)  # fail now, not after a real binary ran
    return _Harness(stub_dir=stub_dir, log_dir=log_dir, env=env)


# ── The five lead-verified scenarios ───────────────────────────────────────────

def test_deploy_ok_amd64_allows_public_only_after_closure(harness):
    """Scenario 1 — happy path: exit 0, binding added exactly once, never removed.

    Pins the H14 happy path end-to-end: build -> push -> amd64 verify -> CLOSED
    deploy -> smoke A sees 403 -> SEPARATE public binding -> smoke B sees
    401/401/200 -> exit 0. The binding is added exactly once and never removed,
    and the deploy line itself never opens the service.
    """
    env = dict(harness.env)
    env["ALLOW_UNAUTH"] = "true"
    # The four codes consumed in order: smoke A GET / -> 403; then smoke B
    # GET / -> 401, POST /kill-switch -> 401, GET /_health -> 200.
    env["STUB_CODES"] = "403 401 401 200"
    proc = _run(DEPLOY_SCRIPT, env)
    assert proc.returncode == 0, "expected the happy-path deploy to exit 0"
    assert _iam(harness.log_dir) == ["added"]
    _assert_deploy_stays_closed(_deploy_line(_calls(harness.log_dir)))


def test_failed_deploy_never_adds_public_binding(harness):
    """Scenario 2 — deploy FAILS: exit 1, binding NEVER added.

    Pins today's incident: `gcloud run deploy` applies the IAM flag BEFORE the
    revision succeeds, so a deploy that fails mid-way used to leave the service
    publicly reachable on the old image. Because the deploy is always closed and
    the binding is a separate later step, a failed deploy aborts (set -e) before
    the binding can ever run — the service can only be MORE closed after a run.
    """
    env = dict(harness.env)
    env["ALLOW_UNAUTH"] = "true"
    env["STUB_DEPLOY_FAILS"] = "true"
    env["STUB_CODES"] = "403 401 401 200"
    proc = _run(DEPLOY_SCRIPT, env)
    assert proc.returncode == 1, "a failed deploy must exit non-zero"
    assert _iam(harness.log_dir) == [], (
        "a failed deploy must NEVER add the public binding (H14 incident)"
    )
    # The deploy WAS attempted, so its flag contract still has to hold.
    _assert_deploy_stays_closed(_deploy_line(_calls(harness.log_dir)))


def test_arm64_image_aborts_before_any_cloud_mutation(harness):
    """Scenario 3 — arm64 image: exit 1, gcloud run deploy NEVER called.

    Pins H14's ordering: the manifest check (docker buildx imagetools inspect)
    reads the REGISTRY image and refuses BEFORE any cloud mutation — the amd64
    check fires before gcloud run deploy, so a bad image can never be handed to
    gcloud at all (the previous bug only failed AT deploy time, AFTER gcloud had
    already applied the IAM flag).
    """
    env = dict(harness.env)
    env["STUB_PLATFORM"] = "linux/arm64"
    env["ALLOW_UNAUTH"] = "true"
    env["STUB_CODES"] = "403"
    proc = _run(DEPLOY_SCRIPT, env)
    assert proc.returncode == 1, "an arm64-only image must be rejected"
    calls = _calls(harness.log_dir)
    assert not any(ln.startswith("gcloud run deploy") for ln in calls), (
        "the amd64 platform check must abort BEFORE gcloud run deploy"
    )
    assert _iam(harness.log_dir) == []


def test_open_app_removes_the_just_added_binding(harness):
    """Scenario 4 — smoke B violation: exit 1, binding added AND then removed.

    Pins H14's safety net: when the post-binding smoke check proves the app is
    NOT protected (anonymous GET / -> 200 — the page served with no credential),
    the script removes the binding it added seconds ago — undoing its OWN
    mutation, strictly safer than leaving a known-open service up — and exits 1.
    """
    env = dict(harness.env)
    env["ALLOW_UNAUTH"] = "true"
    # Codes in order: smoke A GET / -> 403; smoke B GET / -> 200 (the
    # DEFINITIVE violation), POST /kill-switch -> 422, GET /_health -> 200.
    env["STUB_CODES"] = "403 200 422 200"
    proc = _run(DEPLOY_SCRIPT, env)
    assert proc.returncode == 1, "an exposed console must fail, never report success"
    assert _iam(harness.log_dir) == ["added", "removed"], (
        "a smoke-B violation must remove the binding this script just added"
    )
    _assert_deploy_stays_closed(_deploy_line(_calls(harness.log_dir)))


def test_no_public_binding_when_unauth_disallowed(harness):
    """Scenario 5 — ALLOW_UNAUTH=false: exit 0, binding NEVER added.

    Pins the closed-by-default contract: the service is deployed closed and
    stays closed. The add-iam-policy-binding step is the ONLY place a public URL
    can come from and it is gated on ALLOW_UNAUTH=true, so with it false the
    binding never runs.
    """
    env = dict(harness.env)
    env["ALLOW_UNAUTH"] = "false"
    env["STUB_CODES"] = "403"  # smoke A sees 403 (closed at the edge) on the first probe
    proc = _run(DEPLOY_SCRIPT, env)
    assert proc.returncode == 0, "expected a closed deploy with no public binding to exit 0"
    assert _iam(harness.log_dir) == []
    _assert_deploy_stays_closed(_deploy_line(_calls(harness.log_dir)))

def test_single_platform_amd64_manifest_is_accepted(harness):
    """A single-platform amd64 manifest must PASS the registry check.

    This pins a real false positive. `--provenance=false` (added by H14 so the
    push is not an OCI attestation index) makes docker publish ONE plain
    manifest -- and the default `imagetools inspect` output prints no platform
    line at all for that shape, because there is no list to enumerate. The
    first version of the check grepped that output for "linux/amd64" and so
    rejected every correctly built amd64 image, aborting a good deploy.

    It failed CLOSED, so it cost a re-run rather than an exposure. But the
    predicate was wrong, and nothing caught it until the operator ran the
    script -- the fourth execution-level defect in this script.

    This is the exact shape H14 produces in practice, so a regression to an
    index-only predicate fails here.
    """
    env = dict(harness.env)
    env["STUB_MANIFEST_SHAPE"] = "single"   # what --provenance=false produces
    env["STUB_PLATFORM"] = "linux/amd64"
    env["ALLOW_UNAUTH"] = "true"
    env["STUB_CODES"] = "403 401 401 200"
    proc = _run(DEPLOY_SCRIPT, env)
    assert proc.returncode == 0, (
        "a single-platform linux/amd64 manifest was rejected. The registry check must accept "
        "BOTH shapes: a single manifest (an object with architecture/os) and a multi-platform "
        f"index (a map keyed by platform). stderr:\n{proc.stderr}"
    )
    # It must have PASSED the check, not skipped the deploy entirely.
    assert any("run deploy" in c for c in _calls(harness.log_dir)), (
        "the script never reached gcloud run deploy"
    )


def test_multi_platform_index_containing_amd64_is_accepted(harness):
    """The other legal shape: an index that lists linux/amd64 among its platforms.

    Cloud Run accepts a multi-platform index as long as amd64/linux is in it
    (python:3.13-slim is exactly this). Fixing the single-manifest bug must not
    make the check single-manifest-ONLY -- that would just invert the same
    defect rather than remove it.
    """
    env = dict(harness.env)
    env["STUB_MANIFEST_SHAPE"] = "index"
    env["STUB_PLATFORM"] = "linux/amd64"
    env["ALLOW_UNAUTH"] = "true"
    env["STUB_CODES"] = "403 401 401 200"
    proc = _run(DEPLOY_SCRIPT, env)
    assert proc.returncode == 0, (
        f"a multi-platform index listing linux/amd64 was rejected. stderr:\n{proc.stderr}"
    )
