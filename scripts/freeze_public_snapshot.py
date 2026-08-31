#!/usr/bin/env python3
"""scripts/freeze_public_snapshot.py — generate the PUBLIC, DB-free demo pages.

WHAT THIS DOES — a human runs this ONCE against a real, fully-populated
database (the operator's live Cloud SQL instance after the real 2026-08-29/30
batch) to pre-render the pages the console serves PUBLICLY with ZERO
credentials: /demo and /test-run and its fixed sub-paths (app/console/app.py
+ app/console/auth.py's PUBLIC_PATHS).  It writes into
app/console/static_snapshots/:

  - demo.html                      the one-screen demo page (the EXACT payload
                                   _fetch_demo_showcase builds — same data a
                                   live, authenticated /demo used to serve);
  - test_run_index.html            targets.html, filtered to ONLY the four
                                   showcase companies (_SHOWCASE_TARGETS) —
                                   never the full unfiltered `/` targets table;
  - test_run_target_<slug>.html    one target_detail.html per showcase target,
                                   slug = company name lowercased, non-alnum
                                   runs collapsed to one hyphen;
  - test_run_steps.json            the EXACT JSON GET /api/run/{run_id}/steps
                                   returns for the run tied to Solacetree's
                                   real scheduled meeting;
  - test_run_run.html              run.html for that same run, with
                                   static_steps_url set so the frozen page
                                   loads the frozen JSON once instead of
                                   polling the live DB-backed API.

WHY IT EXISTS — the public demo routes must be safe to leave reachable with
no API key indefinitely (judges click through them during the demo window).
A route that queried the real database per request would be an unauthenticated
read of pipeline data; a route that rendered templates per request would need
the DB for demo_data/target data.  Freezing the pages to disk ahead of time
makes every public route a pure static-file read: ZERO database connection,
ZERO write capability, ZERO template work at request time.  The app itself
never generates these files; this script does, once, by hand.

INERT UNTIL RUN — this module is NEVER imported or executed by the console
app or by the test suite automatically.  Like scripts/deploy_console.sh, it
provisions/writes NOTHING on import: all of the above happens inside
main()/its callees, and module level holds only imports and constants.  A
missing snapshot (this script never run in a fresh clone) is handled by the
app routes with a readable 503 naming this script (app/console/app.py's
_serve_static_snapshot) — the directory therefore does not need to exist in
git; the script creates it on first run, and the app fails closed with a
message instead of a 500 when a file is absent.

The database is read with app.db.connect() on the repo-wide OUTBOUND_DB_TARGET
convention (same helper and env var app/console/app.py's _db_target uses) —
never a newly invented DSN.  Renders reuse the console's OWN Jinja2Templates
instance (app.console.app.templates), never a forked template loader.
"""

import argparse  # stdlib argument parsing — --db / --output-dir overrides, defaulting to the repo conventions
import json  # serializing the frozen run-steps payload to the .json snapshot
import re  # the showcase company name -> slug transform for the fixed file names
import sys  # the sys.path bootstrap that makes `python scripts/freeze_public_snapshot.py` importable
from pathlib import Path  # resolving the repo root for the sys.path bootstrap, and the output paths

# ── sys.path bootstrap ───────────────────────────────────────────────────────
# This file lives in scripts/, but the code it imports lives in app/.  When
# run directly (`python scripts/freeze_public_snapshot.py`) Python puts
# scripts/ — not the repo root — at sys.path[0], so `import app` would fail.
# Inserting the repo root (scripts/'s parent) makes the script runnable the
# way the docs tell the operator to run it, regardless of cwd — the same
# bootstrap scripts/hypothesis_scoreboard.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Imports from the console app (the ALLOWLISTED modules) ───────────────────
# The script REUSES the console's own data-loading and rendering machinery
# instead of re-deriving its SQL or forking its template loading:
#   - _SHOWCASE_TARGETS / _fetch_demo_showcase  the exact four showcase names
#     and the exact payload the live /demo built;
#   - _fetch_target_detail / _pretty_json        the exact per-target audit
#     payload the live /targets/{id} route renders;
#   - _fetch_run_steps                           the exact function behind
#     GET /api/run/{run_id}/steps — its returned model IS the API's JSON;
#   - _is_demo_database / _replay_mode           the same banner flags the
#     live routes compute, so frozen pages look identical to live ones;
#   - _db_target                                 the OUTBOUND_DB_TARGET read;
#   - _SNAPSHOTS_DIR / templates                 where to write, and the
#     console's own Jinja2Templates instance to render with.
from app.console.app import (  # the console's own helpers and template instance — see the note above
    _SHOWCASE_TARGETS,
    _SNAPSHOTS_DIR,
    _db_target,
    _fetch_demo_showcase,
    _fetch_run_steps,
    _fetch_target_detail,
    _is_demo_database,
    _pretty_json,
    _replay_mode,
    templates,
)
from app.db import connect  # the dialect-agnostic connection (sqlite file path, or a postgresql:// / cloudsql:// URL)

# The fixed URL the frozen run.html fetches for its steps — served by the
# static route GET /test-run/run/steps.json (app/console/app.py), which reads
# test_run_steps.json off disk.  Absolute path so the frozen page works no
# matter where it is mounted.
_STATIC_STEPS_URL = "/test-run/run/steps.json"


def _slugify(name: str) -> str:
    """The deterministic file-name slug for one showcase company name.

    Company name lowercased, every run of non-alphanumeric characters
    collapsed to a single hyphen, no leading/trailing hyphen.  This is the
    contract the fixed /test-run/<slug> routes (app/console/app.py) and this
    script's file names both follow, so the two can never disagree:
      "MindnLife"                        -> mindnlife
      "Psychotherapy Counselling Clinic" -> psychotherapy-counselling-clinic
      "Focus2 Intelligent Therapy"       -> focus2-intelligent-therapy
      "Solacetree Counselling Limited"   -> solacetree-counselling-limited
    """
    # Lowercase first, then collapse every run of non-alphanumerics to one
    # hyphen, then strip any hyphen the transform left at either end.
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _find_target_id(conn, company_name: str) -> str | None:
    """Resolve one showcase company name to its target_id, or None.

    The same join _fetch_demo_showcase uses (targets joined to accounts on
    company_name) — a hardcoded target_id would silently 404 the moment the
    demo database is reseeded, while the name keeps resolving.  A name that
    does not resolve is a FAILURE for the freeze script (unlike the live
    /demo page, which mark-don't-drops it): a frozen page can only be built
    from a real target row, and freezing a partial set would look complete
    while missing a showcase moment.
    """
    row = conn.execute(
        """
        SELECT t.target_id
        FROM targets t
        JOIN accounts a ON t.account_id = a.account_id
        WHERE a.company_name = ?
        """,
        (company_name,),
    ).fetchone()
    return row["target_id"] if row is not None else None


def _render(template_name: str, context: dict) -> str:
    """Render one console template to a string with the console's own instance.

    Uses app.console.app.templates (the SAME Jinja2Templates the live routes
    use — never a forked loader) and renders to a string.  No Request object
    is threaded in: none of the frozen templates (base.html/demo.html/
    targets.html/target_detail.html/run.html) reference request or url_for
    (verified when this was written), so the rendered bytes are identical to
    what the live routes' TemplateResponse would produce.
    """
    # get_template + render bypasses starlette's TemplateResponse wrapper,
    # which only adds the request to the context — nothing these templates use.
    return templates.env.get_template(template_name).render(**context)


def _find_solacetree_run_id(conn) -> str:
    """The run_id tied to Solacetree Counselling Limited's real scheduled
    meeting — the run the frozen run.html and steps.json capture.

    The meeting row (app/tools/schedule_meeting.py) is the tie: its run_id
    names the run that reserved the real slot.  Fall back to the target's
    latest state_transition run_id (the same read _fetch_review_payload uses)
    so the freeze works even on a database whose meeting row predates run
    attribution.  Either way the run must exist — a frozen run view built
    from a nonexistent run would render an empty page that looks live.
    """
    solacetree_id = _find_target_id(conn, "Solacetree Counselling Limited")
    if solacetree_id is None:
        raise RuntimeError(
            "showcase target 'Solacetree Counselling Limited' not found in "
            "the database — cannot freeze the run view"
        )
    # 1. The meeting's own run_id — the run that reserved the real slot.
    meeting_row = conn.execute(
        "SELECT run_id FROM meetings WHERE target_id = ? "
        "ORDER BY created_at DESC LIMIT 1;",
        (solacetree_id,),
    ).fetchone()
    if meeting_row is not None and meeting_row["run_id"]:
        return meeting_row["run_id"]
    # 2. Fallback: the target's most recent transition's run (the same read
    #    _fetch_review_payload uses for its run grouping).
    transition_row = conn.execute(
        "SELECT run_id FROM state_transitions WHERE target_id=? "
        "ORDER BY created_at DESC LIMIT 1;",
        (solacetree_id,),
    ).fetchone()
    if transition_row is not None and transition_row["run_id"]:
        return transition_row["run_id"]
    raise RuntimeError(
        "Solacetree Counselling Limited has no scheduled meeting row and no "
        "state transition with a run_id — cannot identify the run to freeze"
    )


def _freeze_all(conn) -> dict[str, bytes]:
    """Render every snapshot into an in-memory {filename: bytes} map.

    Nothing is written here.  The caller writes the map only after this
    returns, so a data/render error at ANY point fails the whole run before a
    single file is touched — no stale mix of old and new snapshot files.
    """
    # The banner flags, computed exactly as the live routes compute them, so
    # the frozen pages render identically to the authenticated console.
    demo_data = _is_demo_database(conn)
    replay_mode = _replay_mode()

    files: dict[str, bytes] = {}

    # ── static_mode / static_target_slugs / static_frozen_run_id ─────────────
    # Every template below extends base.html and several of them (demo.html,
    # targets.html, target_detail.html) link to /targets/{id}, /review/{id},
    # or /run/{id} on the live console — all three are behind the operator
    # credential wall.  A judge clicking ANY of those links from a frozen
    # /demo or /test-run page would hit a login prompt, defeating the entire
    # point of a zero-credential public surface (found live, 2026-08-31: the
    # targets.html target-id link did exactly this).  static_mode=True is
    # passed to EVERY render below so base.html's nav (logo, review-queue
    # link, the review-queue poll script) and each page's own body links
    # switch to their frozen equivalents — or, where no frozen equivalent
    # exists (the live review-decision screen, a run_id that isn't the one
    # frozen run), omit the link entirely rather than point at a dead end.
    # static_target_slugs maps each showcase target's real target_id to its
    # /test-run/<slug> path, built ONCE here (not re-derived per template) so
    # every page that could link to a showcase target's audit trail agrees
    # on the exact same slug the file names on disk actually use.
    # target_id_by_name is kept alongside static_target_slugs (rather than
    # re-deriving it via _find_target_id a second time in step 3 below) so
    # the two dicts can never resolve a company name to two different ids —
    # one _find_target_id call per showcase company, total, for this whole
    # function.
    target_id_by_name: dict[str, str] = {}
    static_target_slugs: dict[str, str] = {}
    for company_name, _why in _SHOWCASE_TARGETS:
        target_id = _find_target_id(conn, company_name)
        if target_id is None:
            raise RuntimeError(
                f"showcase target {company_name!r} not found in the database "
                f"— cannot freeze its detail page (freezing nothing)"
            )
        target_id_by_name[company_name] = target_id
        static_target_slugs[target_id] = _slugify(company_name)
    # Resolved once, up front, because target_detail.html (step 3 below) also
    # needs it to decide which of a target's own run_ids (if any) is the one
    # actually frozen at /test-run/run — previously this was only computed
    # after the per-target loop (step 4), too late for the loop to use it.
    static_frozen_run_id = _find_solacetree_run_id(conn)

    # 1. demo.html — the EXACT payload _fetch_demo_showcase builds.
    payload = _fetch_demo_showcase(conn)
    files["demo.html"] = _render("demo.html", {
        "showcase": payload["showcase"],
        "meeting": payload["meeting"],
        "meeting_draft": payload["meeting_draft"],
        "demo_data": demo_data,
        "replay_mode": replay_mode,
        "static_mode": True,
        "static_target_slugs": static_target_slugs,
    }).encode("utf-8")

    # 2. test_run_index.html — targets.html filtered to ONLY the four
    #    showcase companies.  A NEW query written here (same SELECT/join shape
    #    as the index route, but WITH a WHERE filter on _SHOWCASE_TARGETS) —
    #    deliberately NOT the unfiltered index SQL, which would leak every
    #    target in the pipeline onto a public page.
    showcase_names = [name for name, _why in _SHOWCASE_TARGETS]
    placeholders = ", ".join("?" for _ in showcase_names)
    rows = conn.execute(
        f"""
        SELECT t.target_id, a.company_name, o.slug AS offer_slug,
               t.state, t.score, t.final_recommendation, t.updated_at
        FROM targets t
        JOIN accounts a ON t.account_id = a.account_id
        JOIN offers o ON t.offer_id = o.offer_id
        WHERE a.company_name IN ({placeholders})
        ORDER BY t.updated_at DESC
        """,
        tuple(showcase_names),
    ).fetchall()
    files["test_run_index.html"] = _render("targets.html", {
        "targets": [dict(row) for row in rows],
        "demo_data": demo_data,
        "replay_mode": replay_mode,
        "static_mode": True,
        "static_target_slugs": static_target_slugs,
    }).encode("utf-8")

    # 3. test_run_target_<slug>.html — one full audit-trail page per showcase
    #    target, rendered the EXACT way the live /targets/{id} route renders
    #    it (same _fetch_target_detail + _pretty_json display-step transform).
    for company_name, _why in _SHOWCASE_TARGETS:
        target_id = target_id_by_name[company_name]
        detail = _fetch_target_detail(conn, target_id)
        if detail is None:
            raise RuntimeError(
                f"could not load the audit trail for {company_name!r} "
                f"(target {target_id!r}) — freezing nothing"
            )
        # The pretty-print transform the live route applies before rendering.
        display_steps = [
            {
                **step,
                "input_json": _pretty_json(step["input_json"]),
                "output_json": _pretty_json(step["output_json"]),
            }
            for step in detail["steps"]
        ]
        files[f"test_run_target_{_slugify(company_name)}.html"] = _render(
            "target_detail.html",
            {
                **detail, "steps": display_steps, "demo_data": demo_data, "replay_mode": replay_mode,
                "static_mode": True,
                "static_target_slugs": static_target_slugs,
                "static_frozen_run_id": static_frozen_run_id,
            },
        ).encode("utf-8")

    # 4. The run tied to Solacetree's real scheduled meeting: dump the exact
    #    JSON the live API returns (via _fetch_run_steps — the function behind
    #    GET /api/run/{run_id}/steps), and render run.html for the same run
    #    with static_steps_url set so the frozen page loads the frozen JSON
    #    once instead of polling the DB-backed live API.  static_frozen_run_id
    #    IS this run_id (resolved once, above) — kept as a separate local
    #    name here only for readability at the call site.
    run_id = static_frozen_run_id
    run_steps = _fetch_run_steps(conn, run_id)
    files["test_run_steps.json"] = json.dumps(
        run_steps.model_dump(), indent=2, ensure_ascii=False
    ).encode("utf-8")
    files["test_run_run.html"] = _render("run.html", {
        "run_id": run_id,
        "demo_data": demo_data,
        "replay_mode": replay_mode,
        "static_steps_url": _STATIC_STEPS_URL,
        "static_mode": True,
    }).encode("utf-8")

    return files


def _write_all(files: dict[str, bytes], output_dir: Path) -> None:
    """Write every rendered snapshot, each atomically via a temp name.

    The whole map was rendered successfully before this is called (see
    _freeze_all), so the only failure left is an I/O error mid-write.  Each
    file is written to a sibling `*.tmp` name and renamed into place, so a
    crash cannot leave a half-written snapshot that the app would serve.
    """
    # The directory may not exist in a fresh clone (the app 503s, it does not
    # require the dir) — create it on first run.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        final = output_dir / name
        # Same-directory temp file + rename: atomic on the filesystem, so no
        # reader ever sees a partially-written snapshot.
        tmp = final.with_name(final.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(final)


def main(argv: list[str] | None = None) -> int:
    """Render every public snapshot from the database and write it to disk.

    Returns 0 on success (after printing the written files + byte counts),
    1 on any database/render/write error with a clear stderr message — never
    a silent partial write (rendering happens entirely in memory before the
    first file is touched).
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/freeze_public_snapshot.py"
    )
    # The same --db override convention every other repo CLI uses; the default
    # is the repo-wide OUTBOUND_DB_TARGET env-var read (_db_target) — never a
    # newly invented DSN.
    parser.add_argument(
        "--db", default=None,
        help="database target: sqlite file path or a postgresql:// / cloudsql:// "
        "URL (default: OUTBOUND_DB_TARGET, or data/outbound.db)",
    )
    # Testability/operator-flexibility override; the default is the console's
    # own static_snapshots directory (_SNAPSHOTS_DIR), so a plain run writes
    # exactly where the public routes read.
    parser.add_argument(
        "--output-dir", default=None,
        help="directory to write the snapshots into (default: "
        "app/console/static_snapshots/)",
    )
    args = parser.parse_args(argv)

    # Defaults resolved here (inside main), so import is inert and tests can
    # point --db / --output-dir at temp paths without touching the env.
    db_target = args.db or _db_target()
    output_dir = Path(args.output_dir) if args.output_dir else _SNAPSHOTS_DIR

    conn = connect(db_target)  # opens the dialect-agnostic connection
    try:
        # Everything is rendered into memory BEFORE the first write, so any
        # data/render failure below exits 1 with nothing written at all.
        files = _freeze_all(conn)
        _write_all(files, output_dir)
    except Exception as exc:
        # A clear stderr message with the real reason — never a silent
        # partial write (nothing was written if _freeze_all raised, and each
        # file write is atomic).
        print(f"ERROR: freeze failed — {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()  # explicit close, the repo CLI convention (open, read, close)

    # Success summary: every file path + byte count, so the operator can see
    # at a glance what the freeze produced.
    print(f"Wrote {len(files)} static snapshot file(s) to {output_dir}:")
    for name in sorted(files):
        print(f"  {output_dir / name}  ({len(files[name])} bytes)")
    return 0


# Guard so `python scripts/freeze_public_snapshot.py` works (not just imports
# from tests).  SystemExit keeps it testable (pytest can catch SystemExit).
if __name__ == "__main__":
    raise SystemExit(main())
