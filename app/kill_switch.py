"""The kill-switch reader and writer (ticket B4a read; ticket B4b write).

WHAT THE KILL SWITCH IS — ``config/kill_switch.json`` is the single source
of truth for the global stop lever (docs/runbook.md §1, docs/policy-matrix.md
P6): setting ``"enabled": true`` halts all outbound actions immediately.
Until B4a nothing read that file, so the switch existed only on paper.  This
module holds the reader (app/agents/guardrail.py enforces it at agent entry
and app/policy.py enforces it as rule P6) and, since B4b, the writer
(``write_kill_switch`` — the console's toggle; the FILE remains the single
source of truth: the writer only rewrites it, no second channel is opened).

WHY FAIL CLOSED — a kill switch that fails open can be defeated by deleting
a file: remove ``config/kill_switch.json`` and every guard silently sees
"not engaged", which inverts the control into a liability.  A missing,
unreadable, or unparseable file — or a file whose ``enabled`` value is not a
JSON boolean — therefore reads as ``engaged=True`` with a ``reason`` naming
the specific fault.  This is the same doctrine the repo already applies
elsewhere: an unmapped action resolves to ``deny``
(docs/tool-registry.md), and a target with no policy row cannot draft
(B3's fail-closed precondition).

WHY NEVER CACHE — docs/runbook.md §1: "Every worker re-reads this file
immediately before any external write action."  The ONLY moment a kill
switch matters is a run already in progress when the operator flips it; a
cached read (``lru_cache`` or a module global) would make the flip
invisible until process restart, i.e. useless.  This module reads from disk
on every call and must never memoize — tests/test_kill_switch.py carries a
test that fails if a cache is added.

WHY STANDALONE — this module deliberately imports nothing from the app
package (no write_gate, state_machine, llm, agents, or tools).  It is read
by all of them — policy.py, the agent-entry guardrail, later B5's send
gate — and a dependency edge from here into any of those would be a
circular import at boot.

PATH RESOLUTION — the env var ``OUTBOUND_KILL_SWITCH_PATH``, when set,
overrides the path (the FILE is still the source of truth; the env var
only points at it — no env-says-off / file-says-on split-brain, which is
the state split runbook.md §1 forbids).  The default resolves relative to
the REPO ROOT (derived from ``__file__``), not the process cwd — the same
approach app/console/app.py uses for its templates — because the console
runs from arbitrary working directories and inside a container whose
WORKDIR is not guaranteed to be the repo root.
"""

import json  # the switch file is JSON — parsed strictly so a malformed file fails closed, never "maybe"
import os  # OUTBOUND_KILL_SWITCH_PATH — the operator-facing path override, read at call time
import tempfile  # the atomic-write pattern: write a sibling temp file, then os.replace over the real one
from datetime import datetime, timezone  # the writer's updated_at timestamp — UTC, second precision
from pathlib import Path  # path handling — the default is anchored to __file__, never to the cwd

from pydantic import BaseModel  # KillSwitchState: structured I/O for the one value every guard reads


# ── Path constants ────────────────────────────────────────────────────────────
# The default path is RESOLVED at import time relative to this file's
# location: app/kill_switch.py -> repo root -> config/kill_switch.json.
# A cwd-relative spelling (Path("config/kill_switch.json")) would resolve
# against wherever the process happens to be running — wrong inside the
# Cloud Run container (WORKDIR=/app, config lives at /app/config) and wrong
# for the console run from any directory.  This mirrors
# app/console/app.py's _TEMPLATES_DIR pattern exactly.
DEFAULT_KILL_SWITCH_PATH = str(
    Path(__file__).resolve().parent.parent / "config" / "kill_switch.json"
)

# The env var that overrides the path (not the state — see the module
# docstring).  Read at CALL time so a change takes effect on the next read
# without a restart — the same per-call resolution discipline as
# app.llm._resolve_model's env pins.
KILL_SWITCH_PATH_ENV_VAR = "OUTBOUND_KILL_SWITCH_PATH"

# Sentinel distinguishing "the key is absent" from "the key is present with
# a JSON null value" — both must fail closed, but the reason strings differ.
_MISSING = object()


class KillSwitchState(BaseModel):
    """The switch's current state as every consumer reads it.

    ``engaged`` is the operative bit; ``reason`` explains WHY the switch is
    in this state — including, on a fault, which fault forced the
    fail-closed engagement.  ``updated_at``/``updated_by`` are carried
    through from the file so a halt message can name who flipped the switch
    and when.
    """

    engaged: bool  # True = halt all outbound actions (by operator intent OR by fail-closed fault)
    updated_at: str  # the file's updated_at, or "" when the file was unreadable
    updated_by: str  # the file's updated_by, or "" when the file was unreadable
    reason: str  # why it is in this state; "" only when disengaged and healthy


def _fail_closed(path: str, reason: str) -> KillSwitchState:
    """Build the fail-closed state: engaged with a reason naming the fault.

    One shared builder so every fault path produces the same shape and the
    reason always names WHAT failed (asserted distinguishable by tests —
    "not just truthy" per the ticket).  updated_at/updated_by are empty
    because a file we could not read tells us nothing about who last
    touched it.
    """
    return KillSwitchState(engaged=True, updated_at="", updated_by="", reason=reason)


def read_kill_switch(path: str | None = None) -> KillSwitchState:
    """Read the kill-switch file from disk and return its state.

    Reads on EVERY call — never cached, never memoized (see the module
    docstring for why caching inverts the control).  Fails closed on every
    fault: a missing, unreadable, or unparseable file, a non-object JSON
    root, an absent ``enabled`` key, or an ``enabled`` value that is not a
    JSON boolean all return ``engaged=True`` with a reason naming the
    specific fault.

    Args:
        path: explicit file path (tests, future callers).  When None, the
            env var OUTBOUND_KILL_SWITCH_PATH is honoured if set, else the
            repo-root-anchored default.

    Returns:
        KillSwitchState — engaged=True on any fault; never raises for a bad
        file, because a raised reader would be caught-or-crash in ways that
        do not uniformly halt, and the switch must halt uniformly.
    """
    # Path resolution order: explicit argument, then env override, then the
    # repo-root default.  The env var is read at call time so tests and
    # operators can repoint the switch between reads without a restart.
    resolved = path or os.environ.get(KILL_SWITCH_PATH_ENV_VAR) or DEFAULT_KILL_SWITCH_PATH

    # ── Read the file.  FileNotFoundError is its own branch (the most
    # likely fault — nobody created the file yet) with a reason that names
    # the expected path, so the operator knows exactly what to create.
    try:
        raw = Path(resolved).read_text(encoding="utf-8")
    except FileNotFoundError:
        # No file = no switch = fail closed: a deleted switch file must
        # halt the harness, never disable the halt.  (Deleting the file is
        # the FIRST thing an attacker — or a bad deploy — would try.)
        return _fail_closed(
            resolved,
            f"kill switch file not found at {resolved} — failing closed (engaged)",
        )
    except OSError as exc:
        # The file exists but cannot be read (permissions, a directory at
        # that path, an I/O error).  Unreadable is indistinguishable from
        # missing for enforcement purposes — both fail closed.
        return _fail_closed(
            resolved,
            f"kill switch file at {resolved} is unreadable ({exc}) — failing closed (engaged)",
        )

    # ── Parse the JSON.  A file that exists but is not valid JSON is a
    # corrupt switch; guessing its intent would be exactly the kind of
    # invention this repo forbids, so it fails closed with the parser's
    # message to point at the corruption.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _fail_closed(
            resolved,
            f"kill switch file at {resolved} is not valid JSON ({exc}) — failing closed (engaged)",
        )

    # ── Shape check: the root must be a JSON OBJECT.  A JSON list or
    # scalar at the root has no "enabled" key to read; treating it as
    # disengaged would fail open on a file that is structurally not a
    # switch.
    if not isinstance(data, dict):
        return _fail_closed(
            resolved,
            f"kill switch file at {resolved} is not a JSON object (got "
            f"{type(data).__name__}) — failing closed (engaged)",
        )

    # ── The operative field.  Absent key and wrong type both fail closed:
    # the file's contract (runbook.md §1) is a JSON boolean named
    # "enabled", and anything else is a malformed switch.  The check is an
    # isinstance test, NOT a truthiness test, because of the string trap:
    # in Python the string "false" is truthy, so `if enabled:` would read
    # `"enabled": "false"` as ENGAGED — fail-closed-safe by accident, but a
    # misdiagnosis — and would read `"enabled": "0"` as engaged too.  Both
    # must be refused as MALFORMED (with a reason naming the type), never
    # guessed; isinstance(data.get("enabled"), bool) is the only correct
    # test, and it is also what distinguishes the JSON booleans true/false
    # from their string lookalikes.
    enabled = data.get("enabled", _MISSING)
    if enabled is _MISSING:
        return _fail_closed(
            resolved,
            f"kill switch file at {resolved} has no 'enabled' field — failing closed (engaged)",
        )
    if not isinstance(enabled, bool):
        return _fail_closed(
            resolved,
            f"kill switch file at {resolved} has 'enabled' = {enabled!r} "
            f"({type(enabled).__name__}), not a JSON boolean — failing closed (engaged)",
        )

    # ── Metadata.  updated_at/updated_by are informational (they name who
    # flipped the switch, for the halt message and the trace), NOT
    # safety-relevant: their absence does not change whether the switch
    # stops anything, so they degrade to "" rather than failing closed.
    # Extra keys beyond the documented three are tolerated for the same
    # reason — an operator may annotate the file without breaking the
    # switch's one operative field.
    updated_at = str(data.get("updated_at", ""))
    updated_by = str(data.get("updated_by", ""))

    # ── The two healthy states.  engaged=True is the operator's deliberate
    # halt — the reason states that plainly so a trace row can distinguish
    # "someone flipped the switch" from "the file is broken".
    if enabled:
        return KillSwitchState(
            engaged=True,
            updated_at=updated_at,
            updated_by=updated_by,
            reason=(
                f"kill switch engaged (enabled=true)"
                + (f", set by {updated_by} at {updated_at}" if updated_by else "")
            ),
        )
    # Disengaged and healthy: the normal state of a normal run — an empty
    # reason is the honest "no fault, no halt".
    return KillSwitchState(engaged=False, updated_at=updated_at, updated_by=updated_by, reason="")


def write_kill_switch(*, engaged: bool, updated_by: str, path: str | None = None) -> KillSwitchState:
    """Flip the kill switch by REWRITING the switch file, then return the
    new state as the reader sees it (ticket B4b — the console's toggle).

    The file — not this function's return value, and not any in-memory
    flag — remains the single source of truth (runbook.md §1): every guard
    reads the file uncached on each check, so rewriting it is the ONLY way
    to flip the switch, and no second channel (env var, module global) is
    opened.  This is the second of the two ways to flip the switch
    (runbook.md §1: editing the file by hand, or this function via the
    console toggle); both converge on the same file.

    WHY ATOMIC — the reader FAILS CLOSED: a truncated or half-written file
    (a crash mid-write) parses as invalid JSON and reads as ENGAGED, which
    halts the entire harness.  The operator's own toggle must therefore
    never be able to brick the harness: the JSON is written to a sibling
    temp file in the SAME directory and then moved over the real path with
    os.replace(), which is atomic on POSIX filesystems — a reader observes
    either the complete old file or the complete new one, never a partial
    write.

    WHY NO CACHING / NO READER CHANGE — this function only writes; it does
    not memoize anything and does not alter read_kill_switch's fail-closed
    behaviour (B4a's contract, still tested unchanged).  It returns the
    reader's view of the file it just wrote (a literal round-trip), so a
    caller can display the post-toggle state without a second call.

    Args:
        engaged: the new switch state — True halts all outbound actions.
        updated_by: who flipped it (the console passes "operator"; the
            field is carried in the file so a halt message can name the
            actor, exactly as runbook.md §1 documents).
        path: explicit file path (tests).  When None, the SAME resolution
            order as read_kill_switch applies — explicit arg, then
            OUTBOUND_KILL_SWITCH_PATH, then the repo-root default — so a
            process that repointed the reader via the env var toggles the
            same file it reads.

    Returns:
        KillSwitchState — read_kill_switch()'s view of the rewritten file
        (round-trip proof the write is parseable; if it were not, the
        fail-closed reader would report engaged, which is the honest
        signal).

    Raises:
        OSError — if the file cannot be written (unwritable directory,
        permission error).  Raised rather than swallowed: a toggle that
        silently failed would leave the operator believing the switch is
        in a state it is not — the exact split-brain runbook.md §1
        forbids.  The temp file is removed on failure.
    """
    # Path resolution mirrors read_kill_switch exactly (see its comment):
    # the writer and reader must agree on WHICH file is the switch, or a
    # toggle would write one file while every guard keeps reading another.
    resolved = path or os.environ.get(KILL_SWITCH_PATH_ENV_VAR) or DEFAULT_KILL_SWITCH_PATH

    # The UTC "now" the file will record — second precision with a Z
    # suffix, the same shape as the runbook.md §1 example
    # ("2026-07-30T00:00:00Z") so timestamps stay one format across
    # hand-edited and console-written flips.
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # The document is EXACTLY the three-field shape runbook.md §1
    # documents: enabled (the operative bit), updated_at, updated_by.  No
    # extra keys are invented — the reader tolerates extras, but the
    # writer stays minimal so the file's shape never drifts from its
    # documented contract.
    doc = {"enabled": engaged, "updated_at": updated_at, "updated_by": updated_by}

    target = Path(resolved)
    # Ensure the parent directory exists so a toggle works even when the
    # file was never created (the default config/ dir exists in the repo
    # and in the B4a-fixed container image; this covers a custom path
    # whose dir is missing — failing with ENOENT instead would force the
    # operator to hand-create the dir, which is noise, not safety).
    target.parent.mkdir(parents=True, exist_ok=True)

    # ── The atomic write ─────────────────────────────────────────────────
    # mkstemp creates the temp file IN the target's directory (same
    # filesystem, so os.replace stays a rename, which POSIX guarantees is
    # atomic — see WHY ATOMIC above).  The dot-prefix keeps it out of any
    # naive glob over the config dir.
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".kill_switch.", suffix=".tmp")
    try:
        # Write the JSON through the raw fd (os.fdopen adopts it, so the
        # with-block's close is also the fd's close), then atomically move
        # it over the real switch file.
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        os.replace(tmp_path, target)
    except BaseException:
        # Never leave the temp file behind on failure — a stray
        # .kill_switch.*.tmp is litter, and (worse) could confuse an
        # operator auditing the config dir.  unlink is best-effort: the
        # ORIGINAL error is what must propagate, so an unlink failure is
        # swallowed.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Round-trip through the reader: the returned state is exactly what
    # every guard will read from this file on its next uncached check —
    # proof the write produced a parseable switch, not merely that this
    # function returned.
    return read_kill_switch(resolved)
