#!/usr/bin/env bash
#
# demo_live.sh — orchestrate the LIVE pipeline demo (ticket D2).
#
# What this script IS: the operator's one command to prove the pipeline works
# end to end on a company nobody pre-selected.  For a single target it drives
# the REAL stages — CSV import, real HTTP fetch of the target's real website,
# real Gemini research + scoring, the real policy gate, real draft generation,
# the real draft gate, a real human approval (through the console, clicked by
# the operator/judge), the real send gate (DRY_RUN — the only send that exists
# in this repo) — and then ONE simulated part: the reply, written by
# app/conversation_sim's scripted counterparty (no network, no model call)
# threaded against the DRY_RUN send's real Message-ID, after which
# app/reply_cli classifies it with a REAL Gemini call and the deterministic
# router acts on the result.
#
# The single most important demo beat (plan §5, D2): the pipeline STOPS at
# `awaiting_review` and WILL NOT PROCEED.  That is the product.  `setup` stops
# at draft and only prints instructions; `finish` REFUSES to send until the
# target is already `approved` by a human in the console.  This script NEVER
# auto-approves and NEVER bypasses the human gate.
#
# What this script is NOT:
#   - it is NOT a live-send path.  It calls `app.send_cli` exactly as-is, which
#     is structurally incapable of transmitting (its own module docstring: no
#     mode flag, no --live switch, no transport import; every "send" writes a
#     data/outbox/{message_id}.eml artifact and transitions approved ->
#     dry_run_sent).  No mail-transport library is imported anywhere, and this
#     script adds none.
#   - it does NOT approve anything.  `setup`'s job ends at printing the console
#     command and the finish instruction; `finish` verifies a recorded human
#     approval (target state `approved`) before it will even run the send.
#   - it NEVER touches the operator's real database (data/outbound.db) or the
#     D1/D3a seeded demo database (data/demo.db).  A real-database guard runs
#     BEFORE any DB I/O (see _guard_db below).
#   - it NEVER silently reuses ambiguous state.  `setup` refuses an existing
#     --db unless --fresh deletes it first, so a live run always starts from a
#     fresh file.
#
# Two subcommands:
#   setup  [--csv PATH] [--offer SLUG] [--db PATH] [--fresh]
#          Import one real target, research/score it, draft it, and STOP at
#          awaiting_review.  Prints the console launch command + credentials
#          and the exact finish command to run after a human approves.
#          Does NOT launch the console and does NOT approve anything.
#   finish [--db PATH] --target TARGET_ID --persona NAME
#          After a human approved in the console: DRY_RUN send, scripted reply
#          (conversation_sim), REAL classification (reply_cli), and a summary.
#          Refuses loudly if the target is not `approved`.
#   finish --list-personas
#          Convenience passthrough: prints the seven scripted counterparties.
#
# Env overrides (the deploy_console.sh / demo_replay.sh convention):
#   PYTHON_BIN            the python interpreter for every CLI (default
#                         /opt/homebrew/bin/python3.14 — README's)
#   DB_DEMO_LIVE          the default --db path (default
#                         <repo>/data/live_demo.db)
#   DEMO_LIVE_CSV         the default --csv path (default
#                         <repo>/data/demo_live_target.csv)
#   DEMO_LIVE_OFFER       the default --offer slug (default therapy-app)
#   OUTBOUND_CONSOLE_API_KEY  the console's Basic-auth key — reused if set,
#                         generated (openssl rand -hex 32) and printed if not.
#   DEMO_LIVE_SKIP_FRESH_GUARD  internal test hook: set to 1 to skip the
#                         "refuse an existing --db" check (kept out of the
#                         operator's path; tests use it so a --fresh-less test
#                         run can point at a pre-created tmp db).

set -euo pipefail
# -e: any failing command aborts — a failed research/draft/send stage must never
#     look like success (CLAUDE.md §3: failures surface clearly).
# -u: referencing an unset variable is a bug, not a silent default.
# -o pipefail: a failing command inside a pipeline fails the pipeline.

# ── Configuration (overridable via env, the deploy_console.sh convention) ────
# The repo root, resolved from git so every path below is independent of the
# directory the operator happened to run this script from.
REPO_ROOT="$(git rev-parse --show-toplevel)"
# Every CLI below runs `python -m app.*` and every python helper imports
# `app.*`, so the package must resolve: cd to the repo root up front, exactly
# as demo_replay.sh does before launching uvicorn.  This also makes the
# relative data/* paths in the python guards resolve against the repo root.
cd "${REPO_ROOT}"
# The interpreter the README documents for every pipeline stage.  PYTHON_BIN
# lets a hermetic test stub it; the default is the exact absolute path README
# uses (a bare `python3.14` on PATH can resolve to the wrong interpreter on
# this machine — see docs/current_status.md).
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.14}"
# The default scratch database for a live demo run.  NEVER data/outbound.db
# (the operator's real run data — _guard_db refuses it below) and NEVER
# data/demo.db (the D1/D3a seeded demo database — _guard_db refuses it too).
# A live run must start from a fresh file, which is why setup also refuses an
# existing --db unless --fresh deletes it first.
DB_DEFAULT="${DB_DEMO_LIVE:-${REPO_ROOT}/data/live_demo.db}"
# The default single-row CSV: exactly ONE real company (reused verbatim from
# data/hk_therapy_targets.csv — same company_name/domain, no invented company)
# whose contact_email is a plausible reserved-domain address (`.test`) the
# send gate accepts for a DRY_RUN send and conversation_sim can fabricate a
# reply sender from.  It is intentionally NOT data/hk_therapy_targets.csv
# (multiple rows, empty contact_email — that is the operator's real import).
CSV_DEFAULT="${DEMO_LIVE_CSV:-${REPO_ROOT}/data/demo_live_target.csv}"
# The offer YAML the pipeline syncs/uses — the one that already exists at
# config/offers/therapy-app.yaml (matches every row in hk_therapy_targets.csv).
OFFER_DEFAULT="${DEMO_LIVE_OFFER:-therapy-app}"

# ── _guard_db — the real-database guard, BEFORE any DB I/O ───────────────────
# Runs the D3a-verified guard (app.demo_seed._guard_violation — refuses
# data/outbound.db and an OUTBOUND_DB_TARGET pointing at it, on RESOLVED
# absolute paths so a relative spelling or symlink cannot sneak past) and adds
# the D2-specific refusal of data/demo.db (the seeded demo db — a live run
# must never silently reuse ambiguous seeded state).  Called first thing in
# BOTH subcommands, before any connect/CLI that could touch the file.
_guard_db() {
    local db="$1"  # the --db path the operator wants this demo to use
    # The guard is a python helper, not bash string math: it REUSES the exact
    # resolved-path comparison the D3a seed ships (one shared refusal), plus a
    # resolve() against data/demo.db — the same fail-loud, never-silent rule.
    "${PYTHON_BIN}" - "${db}" <<'PY'
import os
import sys
from pathlib import Path
# The D3a-verified guard: refuses data/outbound.db and an OUTBOUND_DB_TARGET
# pointing at it, comparing RESOLVED absolute paths (a symlink or relative
# spelling cannot sneak past).  Importing it means the refusal text and the
# comparison logic are the exact ones D3a and D1 already verified.
from app.demo_seed import _guard_violation
target = sys.argv[1]
violation = _guard_violation(target)
if violation is not None:
    print(f"ERROR: {violation}", file=sys.stderr)
    sys.exit(1)
# D2's own half: data/demo.db is the SEEDED demo database (demo_seed / the
# D1 replay source), not a live run's scratch file.  Reusing it would make the
# judge's live run silently build on synthetic state — refuse loudly instead.
def _is_seeded_demo(path):
    return Path(path).resolve() == Path("data/demo.db").resolve()
if _is_seeded_demo(target):
    print("ERROR: refusing to run the live demo against 'data/demo.db' — that is "
          "the seeded demo database (D1/D3a). Use --db data/live_demo.db "
          "(or another fresh scratch path) instead.", file=sys.stderr)
    sys.exit(1)
# Mirror the OUTBOUND_DB_TARGET half of _guard_violation for demo.db: the
# console's repo-wide convention (docs/gcp-setup.md §6) must never point a
# live demo at the seeded database either.  URL-shaped values are passed
# through untouched (they cannot resolve to a local seeded file).
env_target = os.environ.get("OUTBOUND_DB_TARGET")
if env_target and not env_target.startswith(("postgresql://", "postgres://", "cloudsql://")):
    if _is_seeded_demo(env_target):
        print("ERROR: refusing to run the live demo: OUTBOUND_DB_TARGET points at "
              "'data/demo.db' (the seeded demo database). Unset it or point it at "
              "the demo's own database first.", file=sys.stderr)
        sys.exit(1)
sys.exit(0)
PY
}

# ── _require_approved — the human-gate guard (the product beat) ──────────────
# finish's FIRST pipeline action.  A read-only query against the demo db; if
# the target is not already `approved` (i.e. a human clicked approve in the
# console, which writes a real review_decisions row AND transitions
# awaiting_review -> approved), the demo REFUSES loudly and exits non-zero.
# This is the guard that makes "the pipeline stops at awaiting_review and will
# not proceed" real even when THIS script is driving — nothing auto-approves.
_require_approved() {
    local db="$1" target="$2"  # the demo db and the target the operator named
    # A missing db file must be refused, not silently created: connect() would
    # otherwise open a brand-new empty sqlite file on a finish run that has no
    # setup behind it — a silent side effect that could never reach approved.
    if [[ "${db}" != postgresql://* && "${db}" != postgres://* && "${db}" != cloudsql://* && ! -f "${db}" ]]; then
        echo "ERROR: database '${db}' does not exist — run \`scripts/demo_live.sh setup\` first." >&2
        exit 1
    fi
    "${PYTHON_BIN}" - "${db}" "${target}" <<'PY'
import sys
from app.db import connect  # the repo's one connection helper — SELECT-only here
db, target = sys.argv[1], sys.argv[2]
conn = connect(db)
try:
    row = conn.execute("SELECT state FROM targets WHERE target_id=?;", (target,)).fetchone()
finally:
    conn.close()
if row is None:
    print(f"ERROR: target {target!r} does not exist in {db!r} — run "
          "`scripts/demo_live.sh setup` first, then approve it in the console.",
          file=sys.stderr)
    sys.exit(1)
if row["state"] != "approved":
    print(f"ERROR: target {target!r} is in state {row['state']!r}, not 'approved'. "
          "The D2 product beat is that the pipeline STOPS at awaiting_review and "
          "will not proceed. Open the console, review the reasoning (ticket U1), "
          "click approve, THEN re-run this command.", file=sys.stderr)
    sys.exit(1)
PY
}

# ── _target_id_state — read the single imported target's id + state ──────────
# setup prints these after phase1_cli + draft_cli.  Output is one
# `target_id|state` line (bash-3.2-safe, no mapfile) so setup can split it.
_target_id_state() {
    local db="$1"  # the demo db the pipeline just wrote
    "${PYTHON_BIN}" - "${db}" <<'PY'
import sys
from app.db import connect
conn = connect(sys.argv[1])
try:
    # The demo CSV has exactly one row, so the newest target IS the demo target
    # (ORDER BY created_at picks it deterministically if any row ever appears).
    row = conn.execute(
        "SELECT target_id, state FROM targets ORDER BY created_at ASC LIMIT 1;"
    ).fetchone()
finally:
    conn.close()
if row is None:
    print("ERROR: no targets in the database — phase1_cli imported none. "
          "Check the CSV and the phase1 trace.", file=sys.stderr)
    sys.exit(1)
print(f"{row['target_id']}|{row['state']}")
PY
}

# ── _reply_summary — read the classified reply + the target's final state ────
# finish prints these after reply_cli.  Output is one
# `classification|confidence|routed_action|target_state` line; a missing reply
# prints the sentinel `NO_REPLY|||` (an honest outcome, not an error — e.g. a
# persona turn that never classified).
_reply_summary() {
    local db="$1" target="$2"  # the demo db and the target whose reply to read
    "${PYTHON_BIN}" - "${db}" "${target}" <<'PY'
import sys
from app.db import connect
db, target = sys.argv[1], sys.argv[2]
conn = connect(db)
try:
    # The latest reply for this target, joined through messages to the target's
    # final state.  insert_seq DESC is the deterministic ordering key every
    # "latest row" read uses (ticket B5 — created_at is second-precision).
    row = conn.execute(
        "SELECT r.classification, r.confidence, r.routed_action, t.state AS target_state "
        "FROM replies r "
        "JOIN messages m ON r.message_id = m.message_id "
        "JOIN targets t ON m.target_id = t.target_id "
        "WHERE m.target_id=? "
        "ORDER BY r.insert_seq DESC, r.created_at DESC LIMIT 1;",
        (target,),
    ).fetchone()
finally:
    conn.close()
if row is None:
    print("NO_REPLY|||")
else:
    print(f"{row['classification']}|{row['confidence']}|{row['routed_action']}|{row['target_state']}")
PY
}

# ── _csv_data_row_count — count the data rows below a CSV's header ────────────
# setup's step-2b guard refuses a multi-row --csv, and this is the count it
# reads.  It prints ONE integer (the number of non-empty lines below the
# header — the same definition the demo CSV's own row-count test uses), and the
# caller captures it exactly the way setup captures _target_id_state's output
# (a heredoc inside a plain function whose stdout is captured by the caller —
# NOT a heredoc inside $(...), which bash 3.2 mis-parses when the python body
# contains a single quote).
_csv_data_row_count() {
    local csv="$1"  # the target CSV whose data rows to count
    "${PYTHON_BIN}" - "${csv}" <<'PY'
import csv
import sys
# data rows = every non-empty line below the header, matching the row-count
# test's definition of a data row (the stdlib csv module so a quoted comma
# cannot mislead the count).
with open(sys.argv[1], newline="", encoding="utf-8") as fh:
    rows = list(csv.reader(fh))
print(sum(1 for row in rows[1:] if any(cell.strip() for cell in row)))
PY
}

# ── setup — import + research + draft, then STOP at awaiting_review ──────────
setup() {
    # Parse setup's own args (--csv / --offer / --db / --fresh); the defaults
    # are the repo-root-relative constants declared at the top of the script.
    local db="${DB_DEFAULT}" csv="${CSV_DEFAULT}" offer="${OFFER_DEFAULT}" fresh=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --csv)   csv="$2"; shift 2;;    # the single-row target CSV
            --offer) offer="$2"; shift 2;;  # the offer slug (default therapy-app)
            --db)    db="$2"; shift 2;;     # the scratch db (default data/live_demo.db)
            --fresh) fresh=1; shift;;       # delete an existing --db before running
            *) echo "ERROR: unknown setup argument '$1'." >&2; exit 1;;
        esac
    done

    # 1. The real-database guard runs FIRST, before any DB I/O: an operator who
    #    passes --db data/outbound.db (or data/demo.db) by mistake is refused
    #    loudly before a single connect or CLI touches the file.
    _guard_db "${db}"

    # 2. The target CSV must exist and be readable — the real import reads it.
    #    A missing file is a setup error, never silently defaulted away.
    if [[ ! -r "${csv}" ]]; then
        echo "ERROR: the target CSV '${csv}' does not exist or is not readable." >&2
        echo "Default: ${CSV_DEFAULT}" >&2
        exit 1
    fi

    # 2b. The demo is deliberately single-target end to end (finish pins
    #    --limit 1, _target_id_state reads ONE target, the console review is
    #    per-target), so a --csv with more than one data row is refused HERE,
    #    before phase1_cli imports it and spends a billable Gemini call on a db
    #    the rest of the script cannot drive correctly.  The DEFAULT CSV is
    #    additionally pinned to exactly one row by an existing test; this guard
    #    makes single-target structural for ANY --csv.  A zero-row CSV is not
    #    refused here (phase1_cli imports nothing and _target_id_state errors
    #    loudly later) — the scope is the multi-target case this script's
    #    design cannot represent.  The count is read by the _csv_data_row_count
    #    helper below (same function-stdout pattern as _target_id_state).
    local data_rows
    data_rows="$(_csv_data_row_count "${csv}")"
    if [[ "${data_rows}" -gt 1 ]]; then
        echo "ERROR: '${csv}' has ${data_rows} data rows — the live demo is" >&2
        echo "deliberately single-target end to end. Use a CSV with exactly one" >&2
        echo "data row (the default ${CSV_DEFAULT} is single-row)." >&2
        exit 1
    fi

    # 3. Refuse an existing --db unless --fresh deletes it first.  A live run
    #    against a stale half-finished db from a previous rehearsal would be
    #    confusing (the demo's whole premise is a fresh start); --fresh is the
    #    explicit "yes, I mean to start over" door.  The guard above already
    #    ruled out data/outbound.db and data/demo.db, so the only file rm -f
    #    can ever reach here is a scratch path the operator chose.
    if [[ -e "${db}" && "${DEMO_LIVE_SKIP_FRESH_GUARD:-0}" != "1" ]]; then
        if [[ "${fresh}" == "1" ]]; then
            echo "Removing the existing demo database '${db}' (--fresh)..."
            set -x  # echo the mutating command so the operator sees exactly what runs
            rm -f "${db}"
            set +x
        else
            echo "ERROR: database '${db}' already exists. A live run must start from" >&2
            echo "a fresh file — pass --fresh to delete it first (never reuse a" >&2
            echo "half-finished rehearsal db)." >&2
            exit 1
        fi
    fi

    # 4. phase1_cli — the REAL research half.  Real HTTP fetch of the target's
    #    real website, real Gemini research + scoring, the real policy gate.
    #    set -x echoes the exact invocation; set +e is NOT used — a refused
    #    target must abort the script, never look like success.
    echo "Importing the target and running REAL research + scoring (phase1_cli)..."
    set -x
    "${PYTHON_BIN}" -m app.phase1_cli --csv "${csv}" --offer "${offer}" --db "${db}"
    set +x

    # 5. draft_cli — the REAL draft generation (real Gemini writer⇄critic loop,
    #    then the deterministic G2 draft gate writes the policy/injection
    #    verdict columns).  The target ends in awaiting_review — and stops.
    echo "Drafting outreach (draft_cli — real Gemini writer⇄critic)..."
    set -x
    "${PYTHON_BIN}" -m app.draft_cli --db "${db}"
    set +x

    # 6. Read the target's id + state so the operator knows exactly what to
    #    approve and what to hand to finish.
    local idstate target_id target_state
    idstate="$(_target_id_state "${db}")"
    target_id="${idstate%%|*}"   # split on the | separator — the target id
    target_state="${idstate##*|}"  # ... and the state after the | separator

    # 7. The console credentials — same fail-closed H11 contract as
    #    demo_replay.sh: reuse the operator's key when one is already set
    #    (never overwrite it silently), otherwise generate one with openssl
    #    rand -hex 32 and PRINT it clearly (HTTP Basic auth, username operator).
    #    Deliberately NOT under set -x: an xtrace of the assignment would echo
    #    the freshly generated key into captured output — a leak into logs.
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

    # 8. The instructions — setup's job ENDS here.  It does NOT launch the
    #    console and does NOT approve anything.  The operator must review in
    #    the console and click approve; only then may finish run.
    cat <<EOF

============================================================================
  Live demo target ready for human review.
    target_id: ${target_id}
    state:     ${target_state}  (should be awaiting_review — the pipeline STOPS here)
EOF
    if [[ "${target_state}" != "awaiting_review" ]]; then
        cat <<EOF
  WARNING: the target did not reach awaiting_review.  Inspect the trace in the
  console before continuing — the finish command below will refuse until a
  human approval is recorded.
EOF
    fi
    cat <<EOF

  1) Launch the console against THIS demo db:
       OUTBOUND_DB_TARGET="${db}" "${PYTHON_BIN}" -m uvicorn app.console.app:app --port 8080
     (the API key above is the Basic-auth password; username: operator)

  2) Review the target's reasoning (ticket U1), the live run view (ticket U2),
     and the pending-decision badge (ticket U3), then click APPROVE.

  3) THEN run:
       scripts/demo_live.sh finish --db "${db}" --target "${target_id}" --persona <name>

     Pick the persona that decides the ending (see \`scripts/demo_live.sh finish
     --list-personas\`): warms_up (positive -> meeting), pushes_back_then_leaves
     (objection -> unsubscribe), goes_legal (risky), stays_vague (unclear),
     negative, not_now, or wrong_person.

  The console is NOT launched by this script, and nothing is auto-approved.
============================================================================
EOF
}

# ── finish — DRY_RUN send + scripted reply + REAL classification ─────────────
finish() {
    # Parse finish's args; --list-personas is a passthrough that needs none of
    # --target/--persona, everything else requires both.
    local db="${DB_DEFAULT}" target="" persona="" list_personas=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --db) db="$2"; shift 2;;  # the same scratch db setup created
            --target) target="$2"; shift 2;;   # the target_id setup printed
            --persona) persona="$2"; shift 2;; # the scripted counterparty to play
            --list-personas) list_personas=1; shift;;  # roster passthrough
            *) echo "ERROR: unknown finish argument '$1'." >&2; exit 1;;
        esac
    done

    # --list-personas passthrough: the operator need not remember the
    # underlying module path to see the seven scripted counterparts.
    if [[ "${list_personas}" == "1" ]]; then
        echo "The seven scripted counterparts (each decides the demo's ending):"
        set -x  # echo the exact passthrough command
        "${PYTHON_BIN}" -m app.conversation_sim converse --list-personas
        set +x
        exit 0
    fi

    # Both operands are required to advance a thread — refuse with the usage
    # hint rather than letting an empty --target silently do nothing.
    if [[ -z "${target}" || -z "${persona}" ]]; then
        echo "ERROR: finish requires --target TARGET_ID and --persona NAME." >&2
        echo "(Use --list-personas to see the valid persona names.)" >&2
        exit 1
    fi

    # The real-database guard, exactly as in setup: data/outbound.db and
    # data/demo.db are refused before any DB I/O.
    _guard_db "${db}"

    # THE PRODUCT BEAT — the human-gate guard runs BEFORE any send.  If the
    # target is not already approved by a human in the console, finish refuses
    # loudly and exits non-zero.  Nothing here auto-approves or proceeds.
    _require_approved "${db}" "${target}"

    # 4. send_cli — the REAL send gate + DRY_RUN send.  This is ALWAYS the
    #    existing DRY_RUN path (writes data/outbox/{message_id}.eml, never
    #    transmits — send_cli has no other mode, verified in its docstring).
    #    --limit 1 is load-bearing: send_cli is a BATCH command (it dry-runs
    #    every `approved` target up to --limit; its own default is 10), but the
    #    gate above verified exactly ONE target.  The limit bounds a run to a
    #    single send even against a multi-target db.  Residual note — send_cli
    #    selects `state='approved' ORDER BY created_at`, not by the verified
    #    --target id, so on a multi-target db `--limit 1` sends the OLDEST
    #    approved target, which may differ from --target.  setup now refuses a
    #    multi-row --csv (setup step 2b), so a db built by this script is
    #    single-target by construction; the limit stays as belt-and-braces for
    #    a db that is not (a hand-made db, or one built before that guard
    #    landed).
    echo "Running the REAL send gate + DRY_RUN send (send_cli)..."
    set -x
    "${PYTHON_BIN}" -m app.send_cli --db "${db}" --limit 1
    set +x

    # 5. conversation_sim — the ONE simulated part.  The scripted counterparty
    #    writes a reserved-domain (.test) inbound .eml threaded for real against
    #    the DRY_RUN send's actual Message-ID.  No network, no model call.
    echo "Writing the scripted reply (conversation_sim, persona '${persona}')..."
    set -x
    "${PYTHON_BIN}" -m app.conversation_sim converse \
        --db "${db}" --persona "${persona}" --target "${target}" \
        --outbox "${REPO_ROOT}/data/outbox" --inbox "${REPO_ROOT}/data/inbox"
    set +x

    # 6. reply_cli — the ONE billable step in this phase: a REAL Gemini
    #    classification call, then the deterministic router acts on the result
    #    (queue follow-up / suppress / review-required / close / ...).
    echo "Classifying the reply (reply_cli — the one billable Gemini call)..."
    set -x
    "${PYTHON_BIN}" -m app.reply_cli --db "${db}"
    set +x

    # 7. The summary — the reply's classified class, the routed action, and the
    #    target's final state, plus where to see the full trace in the console.
    #    The helper returns one `classification|confidence|routed_action|state`
    #    line (or the NO_REPLY sentinel); IFS='|' read splits it into the four
    #    fields — none of these strings can contain a |, so the delimiter is
    #    unambiguous.
    local summary classification confidence routed_action target_state
    summary="$(_reply_summary "${db}" "${target}")"
    if [[ "${summary}" == "NO_REPLY|||" ]]; then
        echo "WARNING: no classified reply was found for target ${target}." >&2
        echo "Check the reply_cli output above and the trace in the console." >&2
        return 0  # an honest outcome — a persona that never classified is not a crash
    fi
    IFS='|' read -r classification confidence routed_action target_state <<< "${summary}"
    cat <<EOF

============================================================================
  Live demo reply processed.
    reply classification: ${classification}
    confidence:           ${confidence}
    routed action:        ${routed_action}
    target final state:   ${target_state}
  See the full trace in the console: /run/<run_id> (ticket U2) and
  /review/<target_id> (ticket U1).
============================================================================
EOF
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
# Exactly two subcommands; anything else is a usage error with the two commands
# spelled out so the operator never has to guess.
if [[ $# -lt 1 ]]; then
    cat >&2 <<EOF
Usage:
  scripts/demo_live.sh setup   [--csv PATH] [--offer SLUG] [--db PATH] [--fresh]
  scripts/demo_live.sh finish  [--db PATH] --target TARGET_ID --persona NAME
  scripts/demo_live.sh finish  --list-personas
EOF
    exit 1
fi
COMMAND="$1"  # setup or finish
shift
case "${COMMAND}" in
    setup)  setup "$@" ;;   # import + research + draft, then STOP at awaiting_review
    finish) finish "$@" ;;  # DRY_RUN send + scripted reply + real classification
    *) echo "ERROR: unknown command '${COMMAND}' — expected setup or finish." >&2; exit 1;;
esac
