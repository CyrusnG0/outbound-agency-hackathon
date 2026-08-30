#!/usr/bin/env bash
#
# demo_replay.sh — launch the offline, prebuilt demo console (ticket D1).
#
# What this script IS: the operator's one command to open the console in
# "replay" mode — serving a REAL, already-completed local pipeline run
# (data/e2e_run2.db — 3 targets: suppressed / failed / routed, 216 rows; the
# same run scripts/restore_db.py copied into Cloud SQL for the deployed
# console) so a judge can click through the full audit trail with no network,
# no LLM call, no spend and no wall-clock dependence. It is deterministic
# enough to rehearse repeatedly and get the exact same result: the demo that
# cannot fail live (plan §5, D1).
#
# What this script is NOT:
#   - it is INERT until the operator runs it; the mutating/launching commands
#     are echoed by `set -x`, so the operator sees exactly what runs
#     (CLAUDE.md §3 — no hidden side effects);
#   - it NEVER serves data/e2e_run2.db directly. It makes a FRESH disposable
#     copy (data/replay_scratch.db, gitignored) on every launch and serves
#     THAT. This is the structural fix for the kill-switch hazard the lead
#     identified: the console has one write door that is not gated on target
#     state — the kill-switch toggle (POST /kill-switch, rendered on every
#     page) — and none of this run's targets are in awaiting_review, so the
#     review-decision form is inert but the toggle is not. Serving the pristine
#     source directly would let a single click mutate the ONE copy of this
#     hard-won run permanently, with no way to reconstruct it (regenerating it
#     needs live billable LLM calls against real targets — exactly what D1
#     exists to avoid). The copy-per-launch means any write during the demo
#     lands on the disposable scratch file, never the source.
#   - it also NEVER serves config/kill_switch.json directly. The kill-switch
#     toggle (POST /kill-switch, rendered on every page) writes to a FILE, not
#     the database: app/kill_switch.py resolves OUTBOUND_KILL_SWITCH_PATH
#     (default config/kill_switch.json — a REAL, git-TRACKED config file) and
#     write_kill_switch() is called with NO explicit path, so without a copy a
#     single click would silently flip the operator's REAL, live switch — worse
#     than the database hazard, because it affects every OTHER pipeline run
#     (real or demo) on this machine going forward. So the switch gets the SAME
#     copy-per-launch discipline: a FRESH disposable copy
#     (data/replay_kill_switch.json, gitignored) on every launch, and
#     OUTBOUND_KILL_SWITCH_PATH (step d) points the toggle at THAT, never the
#     tracked config file.
#   - it does NOT weaken or bypass the console's H11 auth (fail-closed: 503
#     with no key configured, NO disable/bypass flag of any kind). It only
#     supplies a REAL credential through the documented mechanism
#     (OUTBOUND_CONSOLE_API_KEY): reusing the operator's if one is already set,
#     generating one with `openssl rand -hex 32` (the exact method runbook §12
#     and deploy_console.sh use for the Cloud Run secret) otherwise.
#   - it does NOT touch scripts/restore_db.py, data/e2e_run2.db, the demo_seed
#     "DEMO DATA" banner, or any database query. OUTBOUND_REPLAY_MODE=1 exists
#     ONLY so the console's honest replay banner can render (it must never
#     affect _db_target(), _is_demo_database(), or any query).
#
# Usage:  scripts/demo_replay.sh        (Ctrl-C to stop the console — the same
#                                        way the manual uvicorn invocation in
#                                        README §6 is stopped)
#
# Env overrides (all optional — the deploy_console.sh convention):
#   OUTBOUND_CONSOLE_API_KEY  reuse the operator's key instead of generating one
#   PYTHON_BIN                the python interpreter to launch uvicorn with
#                             (default /opt/homebrew/bin/python3.14 — README's)
#   DEMO_REPLAY_SOURCE        the pristine run to replay (default
#                             <repo>/data/e2e_run2.db) — overridable for tests
#   REPLAY_SCRATCH_PATH       the disposable copy path (default
#                             <repo>/data/replay_scratch.db) — overridable for
#                             tests so they never write into the repo's data/
#   DEMO_REPLAY_KILL_SWITCH_SOURCE  the pristine switch state to mirror
#                             (default <repo>/config/kill_switch.json) —
#                             overridable for tests so they never read the
#                             real tracked config file
#   REPLAY_KILL_SWITCH_SCRATCH_PATH the disposable switch copy path (default
#                             <repo>/data/replay_kill_switch.json, gitignored)
#                             — overridable for tests so they never write into
#                             the repo's data/

set -euo pipefail
# -e: any failing command aborts — a half-copied source or a failed launch
#     must never look like success (CLAUDE.md §3: failures surface clearly).
# -u: referencing an unset variable is a bug, not a silent default.
# -o pipefail: a failing command inside a pipeline fails the pipeline.

# ── Configuration (overridable via env, the deploy_console.sh convention) ────
# The repo root, resolved from git so every path below is independent of the
# directory the operator happened to run this script from.
REPO_ROOT="$(git rev-parse --show-toplevel)"
# The pristine source of truth. DEMO_REPLAY_SOURCE lets a hermetic test point
# the script at a controlled file; the default is the restored run the ticket
# names. The script REFUSES loudly (never regenerates) if it is missing.
SOURCE="${DEMO_REPLAY_SOURCE:-${REPO_ROOT}/data/e2e_run2.db}"
# The disposable copy the console is served from. REPLAY_SCRATCH_PATH lets a
# test point it at a tmp file; the default data/replay_scratch.db matches
# .gitignore's `data/*.db`, so a stale scratch file is never committable and
# is clearly distinct from e2e_run2.db at a glance.
SCRATCH="${REPLAY_SCRATCH_PATH:-${REPO_ROOT}/data/replay_scratch.db}"
# The pristine source of the kill switch. DEMO_REPLAY_KILL_SWITCH_SOURCE lets a
# hermetic test point the script at a controlled file; the default is the
# operator's REAL, git-TRACKED config/kill_switch.json — this one genuinely IS
# the correct source to copy FROM, since it is the operator's real current
# switch state, not something that needs a separate "restored run" concept like
# the DB. (Unlike the DB, a missing switch is a setup error, not a fresh-clone
# absence — the file is tracked — so the script REFUSES loudly, see step b2.)
KILL_SWITCH_SOURCE="${DEMO_REPLAY_KILL_SWITCH_SOURCE:-${REPO_ROOT}/config/kill_switch.json}"
# The disposable copy the console's kill-switch toggle writes to.
# REPLAY_KILL_SWITCH_SCRATCH_PATH lets a test point it at a tmp file; the
# default data/replay_kill_switch.json matches the .gitignore entry added with
# this ticket, so a stale scratch switch is never committable and is clearly
# distinct from config/kill_switch.json at a glance.
KILL_SWITCH_SCRATCH="${REPLAY_KILL_SWITCH_SCRATCH_PATH:-${REPO_ROOT}/data/replay_kill_switch.json}"
# The interpreter the README documents for the console. PYTHON_BIN lets a test
# stub it; the default is the exact /opt/homebrew/bin/python3.14 README uses
# (a bare `python3.14` on PATH can resolve to the wrong interpreter on this
# machine — see docs/current_status.md — so the absolute path is the safe one).
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.14}"

# ── Step a: verify the source exists — refuse loudly, never regenerate ───────
# If data/e2e_run2.db is missing, exit 1 with a clear, actionable message.
# Do NOT attempt to regenerate it: that means live billable LLM calls against
# real targets, which is exactly what D1 exists to avoid. The file must be
# restored/obtained separately (re-run the pipeline and keep the output).
if [[ ! -r "${SOURCE}" ]]; then
    echo "ERROR: the replay source database '${SOURCE}' does not exist or is not readable." >&2
    echo "This must be a previously completed local pipeline run (data/e2e_run2.db)." >&2
    echo "It cannot be regenerated here — that needs live billable LLM calls against" >&2
    echo "real targets, which is exactly what this demo exists to avoid." >&2
    echo "Restore it separately (or re-run the pipeline and keep the output), then re-run." >&2
    exit 1
fi

# ── Step b: FRESH scratch copy on every launch — never serve the source ──────
# The kill-switch hazard (see the header): the console's one write door that is
# not gated on target state is POST /kill-switch, rendered on every page, and
# clicking it during a demo must never mutate the pristine source. Copying to a
# disposable path per launch means every write lands on the scratch file, never
# the source. The copy overwrites any stale scratch from a PRIOR run — that is
# harmless (disposable + gitignored) and deliberately left un-cleaned.
echo "Replay source verified: ${SOURCE}"
mkdir -p "$(dirname "${SCRATCH}")"
echo "Making a fresh disposable copy for this session..."
set -x
cp "${SOURCE}" "${SCRATCH}"
set +x
# Fail loud if the copy did not land byte-identical — a corrupt scratch DB
# would render a confusing demo instead of a clear error (CLAUDE.md §3). Being
# inside an `if !` keeps set -e from aborting on cmp's expected non-zero when
# the files differ; the branch then exits with the message.
if ! cmp -s "${SOURCE}" "${SCRATCH}"; then
    echo "ERROR: the scratch copy is not byte-identical to the source — aborting." >&2
    exit 1
fi
echo "Serving the console from a disposable copy: ${SCRATCH}"
echo "(the pristine source ${SOURCE} is never served or written.)"

# ── Step b2: FRESH scratch copy of the KILL SWITCH too — same discipline ─────
# The console's kill-switch toggle (POST /kill-switch,
# app/console/app.py::kill_switch_toggle) calls write_kill_switch() with NO
# explicit path, so it resolves OUTBOUND_KILL_SWITCH_PATH (app/kill_switch.py,
# default config/kill_switch.json) — a file COMPLETELY separate from the
# database.  The DB copy above protects the DB, but the toggle ALSO rewrites
# the switch file; without a copy here, a click during the demo would silently
# flip the operator's REAL, git-TRACKED config/kill_switch.json — worse than
# the DB hazard, because it affects every OTHER pipeline run (real or demo) on
# this machine.  So the switch file gets the exact same copy-per-launch
# treatment: source → disposable scratch (byte-identical, fail-loud), and
# OUTBOUND_KILL_SWITCH_PATH (step d) points the toggle at the scratch copy.
# First the refuse-loud-if-missing check, mirroring step a: the switch file's
# own reader (app/kill_switch.py) FAILS CLOSED for a missing file (reads as
# engaged), so launching anyway would be SAFE but would serve a demo whose
# kill switch reads as ENGAGED-fail-closed — a confusing, non-deterministic
# start (the toggle would also create the scratch file on its first click).
# A missing switch means there is no real operator state to mirror, which is a
# setup error the operator must fix — the same refuse-loudly doctrine as step a.
if [[ ! -r "${KILL_SWITCH_SOURCE}" ]]; then
    echo "ERROR: the kill-switch source '${KILL_SWITCH_SOURCE}' does not exist or is not readable." >&2
    echo "The console's kill-switch toggle rewrites this file (app/kill_switch.py);" >&2
    echo "serving a replay without a real switch to mirror would start the demo" >&2
    echo "with the switch ENGAGED (fail-closed), not the operator's state." >&2
    echo "Restore config/kill_switch.json (or point DEMO_REPLAY_KILL_SWITCH_SOURCE" >&2
    echo "at a real switch file), then re-run." >&2
    exit 1
fi
echo "Kill switch source verified: ${KILL_SWITCH_SOURCE}"
mkdir -p "$(dirname "${KILL_SWITCH_SCRATCH}")"
echo "Making a fresh disposable kill-switch copy for this session..."
set -x
cp "${KILL_SWITCH_SOURCE}" "${KILL_SWITCH_SCRATCH}"
set +x
# Fail loud if the copy did not land byte-identical — a corrupt scratch switch
# would render a confusing demo instead of a clear error (CLAUDE.md §3).  Same
# `if !` shape as the DB copy so set -e cannot abort on cmp's expected
# non-zero when the files differ; the branch then exits with the message.
if ! cmp -s "${KILL_SWITCH_SOURCE}" "${KILL_SWITCH_SCRATCH}"; then
    echo "ERROR: the kill-switch scratch copy is not byte-identical to the source — aborting." >&2
    exit 1
fi
echo "The kill-switch toggle will write to a disposable copy: ${KILL_SWITCH_SCRATCH}"
echo "(the real switch ${KILL_SWITCH_SOURCE} is never served or written.)"

# ── Step c: the console auth secret (H11's fail-closed contract, unchanged) ──
# The console fails closed without OUTBOUND_CONSOLE_API_KEY (every route 503).
# Reuse the operator's key when one is already set — never overwrite it
# silently; otherwise generate one with `openssl rand -hex 32`, the exact
# method runbook §12 / deploy_console.sh use, and PRINT it clearly and
# unmissably: the console uses HTTP Basic auth for the browser (username
# `operator`, password this key — app/console/auth.py::require_operator).
#
# Deliberately NOT under `set -x`: an xtrace of the assignment would echo the
# freshly generated key into the shell's captured output (a leak into logs);
# the key is printed exactly once, in the credentials block below.
if [[ -n "${OUTBOUND_CONSOLE_API_KEY:-}" ]]; then
    echo "Reusing existing OUTBOUND_CONSOLE_API_KEY (already set in the environment)."
else
    echo "Generating a fresh console API key (openssl rand -hex 32)..."
    OUTBOUND_CONSOLE_API_KEY="$(openssl rand -hex 32)"
    echo "======================================================================"
    echo "  Console credentials (HTTP Basic auth — the browser will prompt):"
    echo "    username: operator"
    echo "    password: ${OUTBOUND_CONSOLE_API_KEY}"
    echo "  (API clients may send header X-Internal-API-Key: ${OUTBOUND_CONSOLE_API_KEY})"
    echo "======================================================================"
fi
export OUTBOUND_CONSOLE_API_KEY

# ── Step d: the env vars the console reads for this run ──────────────────────
# OUTBOUND_DB_TARGET: the repo-wide convention for where the database lives
# (app/console/app.py::_db_target). Points at the DISPOSABLE COPY — never the
# source.
export OUTBOUND_DB_TARGET="${SCRATCH}"
# OUTBOUND_REPLAY_MODE=1: read by the console's honest replay banner ONLY
# (app/console/app.py::_replay_mode — a pure env read). It is NOT an auth
# bypass and NOT a behavior-changing flag anywhere except the banner text; it
# must never touch _db_target(), _is_demo_database(), or any query.
export OUTBOUND_REPLAY_MODE=1
# OUTBOUND_KILL_SWITCH_PATH: the env var app/kill_switch.py reads (at call
# time) to resolve the switch file. Points at the DISPOSABLE COPY — never the
# real config/kill_switch.json — so the console's toggle (write_kill_switch
# with no explicit path) and every read_kill_switch() land on the scratch
# copy, exactly as OUTBOUND_DB_TARGET does for the database.
export OUTBOUND_KILL_SWITCH_PATH="${KILL_SWITCH_SCRATCH}"

# ── Step e: the honest framing, printed BEFORE launching ─────────────────────
# The judge (and the operator) must never mistake a replay for a live run: the
# console says so, plainly, in the banner AND here at launch. Tone matches
# deploy_console.sh's own printed operator messages.
cat <<EOF

==== REPLAY MODE ==========================================================
This console is a REPLAY of a real, previously completed pipeline run.
Nothing on these pages is computing live. The data was restored from
  ${SOURCE}
and is being served from a disposable copy (${SCRATCH}).
The kill-switch toggle writes to a disposable copy too (${KILL_SWITCH_SCRATCH}).
No network, no LLM call, no spend.
Press Ctrl-C when the demo is over to stop the console.
============================================================================
EOF

# ── Step f: launch the console in the foreground ─────────────────────────────
# The exact README §6 invocation (python3.14 -m uvicorn app.console.app:app
# --port 8080), foregrounded so the operator Ctrl-Cs to stop. cd to the repo
# root first so the `app` package resolves regardless of where the script was
# run from. Echoed by set -x so the operator sees the exact launch command.
echo "Launching the console..."
set -x
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m uvicorn app.console.app:app --port 8080
