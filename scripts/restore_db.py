#!/usr/bin/env python3
"""Restore a completed local run into an EMPTY database, verbatim (ticket H17).

WHY THIS EXISTS — the deployed console reads Cloud SQL and is empty: every
pipeline CLI has only ever run against local SQLite, so a judge opening the
demo would see "No targets yet". The data worth showing is a completed,
already-verified end-to-end run (data/e2e_run2.db). This tool copies that run
row-for-row into the operator's hosted database so the demo shows the safety
machinery (suppressed / failed / routed targets, the send-gate and review
decisions, the full audit trail) instead of an empty screen. Re-running the
pipeline against Cloud SQL is the obvious alternative and is WORSE: it costs
live LLM calls and real HTTP fetches, and would very likely not reproduce the
suppressed and failed states that are exactly what the demo exists to show.

WHY THIS IS THE ONE CORRECT EXCEPTION TO CLAUDE.md §3's "core-table writes go
through the write gate" — the gate exists to make NEW pipeline actions
auditable (each write gets a fresh write_log row with its own run_id/step_id/
actor). This tool performs NO pipeline actions: it reproduces an audit trail
that ALREADY exists, including the write_log rows the gate itself wrote during
the original run. Routing a restore through the gate would fabricate NEW
provenance for OLD events — a second write_log row claiming a fresh run wrote
rows that an earlier run actually wrote — which is strictly worse than
bypassing it. The empty-destination guard below is what makes the bypass safe:
the destination must contain nothing, so a restore can never merge into (and
thereby fake) real history.

THE SAFETY PROPERTY — a non-empty destination is refused fail-closed, in BOTH
modes. A partial or duplicate load would corrupt the audit trail (write_log,
steps, state_transitions) in a way that cannot be distinguished from real
history afterwards. Refusing is always recoverable (drop the destination
database); a bad merge is not. There is no --force.

Usage (dry run is the default — nothing is written without --confirm):

    python scripts/restore_db.py --source data/e2e_run2.db --dest "cloudsql://..."
    python scripts/restore_db.py --source data/e2e_run2.db --dest "cloudsql://..." --confirm

Both --source and --dest accept any target string app.db.connect() understands
(a sqlite file path, a postgresql:// URL, or a cloudsql:// sentinel); the
connection layer is reused, never reimplemented.

The copy is verbatim: ids, timestamps, JSON payloads and NULLs are copied
exactly as stored, via parameterised inserts through the Conn wrapper (never
string interpolation), and it deliberately does NOT go through
write_gate.commit() or state_machine.transition() — see above. It DOES respect
foreign keys (tables are copied in an FK-safe order derived from the DDL),
convert SQLite 0/1 to real booleans for any BOOLEAN destination column
(determined from the destination catalog, not guessed — none exist in the
current DDL), and wrap the whole copy in one conn.begin_write() transaction so
a mid-way failure leaves the destination EMPTY rather than half-loaded (the
H4b lesson: bare execute() calls autocommit on both dialects, and a partial
write there destroyed a table).

COVERAGE BOUNDARY — the SQLite→Postgres dialect path (boolean conversion, the
information_schema catalog reads) is exercised by the operator's real run
against Cloud SQL, NOT by the automated suite. The tests in
tests/test_restore_db.py are hermetic SQLite→SQLite round-trips; the
dialect-specific code paths are deliberately small and mirror the already
tested app/db.py / _migrate_suppressions patterns.
"""

import argparse  # stdlib argument parsing — no new dependency for the operator
import re  # extracting table names and REFERENCES edges from the DDL
import sys  # stderr for refusal messages, argv for the default None sentinel
from pathlib import Path  # resolving the repo root and the missing-source check

# ── sys.path bootstrap ───────────────────────────────────────────────────────
# Same pattern as scripts/add_suppression.py: this file lives in scripts/, the
# code lives in app/, and running `python scripts/restore_db.py` puts scripts/
# (not the repo root) at sys.path[0]. Inserting the repo root makes the import
# below work regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# _DDL and _split_statements are the schema's single source of truth: the same
# constant apply_schema() executes and the same quote/comment-aware splitter
# the postgres path uses. Importing them (rather than copying the DDL) is what
# makes the FK-safe table order below automatically track schema changes.
from app.db import _DDL, _split_statements, apply_schema, connect

# The Conn type alias — used in signatures so the reader knows what flows in.
from app.db import Conn


# ── DDL-derived table order ───────────────────────────────────────────────────


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL -- line and /* block */ comments from one DDL statement so
    identifier extraction below never matches comment prose.

    The sources table's comment text contains the literal phrase
    'REFERENCES targets(target_id)' even though the table has no such
    constraint — without stripping, a naive regex would invent a fake
    dependency (harmless for ordering here, but wrong). The DDL contains no
    string literals that include '--' or '/*', so this simple scanner is exact
    for its input; it does not need to be a full SQL tokenizer.
    """
    out: list[str] = []  # the comment-free statement is accumulated here
    i, n = 0, len(sql)  # i is the scan position; n the statement length
    while i < n:
        if sql.startswith("--", i):  # a -- line comment: drop to end of line
            eol = sql.find("\n", i)
            if eol == -1:
                break  # comment runs to end of statement — done
            i = eol + 1  # resume after the newline
        elif sql.startswith("/*", i):  # a /* block comment */: drop to its close
            end = sql.find("*/", i + 2)
            i = end + 2 if end != -1 else n  # unterminated → rest is comment
        else:
            out.append(sql[i])  # ordinary character — keep it
            i += 1
    return "".join(out)


def _table_order() -> list[str]:
    """Return the table copy order: every table in apply_schema's DDL,
    topologically sorted so each table's FOREIGN KEY targets come first.

    Deriving from the real _DDL constant (rather than hardcoding a list) means
    a table added to the schema is picked up automatically, and the FK-safe
    order means the copy never violates a foreign key on the fresh destination
    (sqlite connect() sets PRAGMA foreign_keys=ON; Postgres always enforces).
    tests/test_restore_db.py pins that every DDL table is present and unique,
    so a future accidental hardcoded order cannot silently skip a table.
    """
    statements = _split_statements(_DDL)  # quote/comment-aware statement split
    names: list[str] = []  # table names in DDL order (the fallback order)
    refs: dict[str, set[str]] = {}  # table -> set of tables it REFERENCES
    for stmt in statements:
        clean = _strip_sql_comments(stmt)  # see _strip_sql_comments — no prose matches
        m = re.search(
            r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)", clean, re.IGNORECASE
        )
        if m is None:
            continue  # a DDL statement that creates no table — nothing to order
        name = m.group(1)
        names.append(name)
        # Every 'REFERENCES <table> (' in the (comment-stripped) statement is a
        # real foreign-key edge. Non-table names are impossible here (a DDL
        # REFERENCES always names a table), but the sort guards against it.
        refs[name] = set(re.findall(r"REFERENCES\s+(\w+)\s*\(", clean, re.IGNORECASE))

    # Depth-first topological sort: a table is emitted only after every table
    # it references has been emitted. On a cycle (none exist in this DDL) the
    # DFS simply emits one node of the cycle first — still a valid copy order,
    # and it never hangs.
    ordered: list[str] = []  # the FK-safe output order
    visited: set[str] = set()  # tables already emitted
    visiting: set[str] = set()  # tables on the current DFS path (cycle guard)

    def _visit(table: str) -> None:
        if table in visited or table in visiting:
            return  # already emitted, or in a cycle — do nothing further
        visiting.add(table)  # mark on-path so a back-edge cannot recurse forever
        for dep in refs.get(table, ()):  # every table this table REFERENCES
            if dep in refs:  # only real tables constrain the order
                _visit(dep)  # recurse: the dependency must come first
        visiting.discard(table)  # done exploring this path
        visited.add(table)
        ordered.append(table)  # all dependencies emitted — safe to copy now

    for name in names:  # the outer loop reaches every table, even an orphan
        _visit(name)
    return ordered


# ── Dialect-aware catalog helpers ─────────────────────────────────────────────


def _existing_tables(conn: Conn) -> set[str]:
    """The set of table names present in `conn`'s database, dialect-aware.
    Used to fail closed on a schema-mismatched source (every DDL table must
    exist there) before any row is read or written."""
    if conn.dialect == "sqlite":
        return {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
    return {
        row["table_name"]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public';"
        ).fetchall()
    }


def _column_names(conn: Conn, table: str) -> list[str]:
    """The column names of `table` in definition order, dialect-aware.
    Table names here are code-derived from the DDL (never user input), so
    interpolating them into PRAGMA is safe — the same precedent as
    app/db.py::_ensure_column. The postgres path uses information_schema, the
    same catalog every other dialect-aware read in app/db.py uses."""
    if conn.dialect == "sqlite":
        return [
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
        ]
    return [
        row["column_name"]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=? "
            "ORDER BY ordinal_position;",
            (table,),
        ).fetchall()
    ]


def _column_types(conn: Conn, table: str) -> dict[str, str]:
    """Map column name → declared type (upper-cased) for `table`.
    Used to find BOOLEAN destination columns. The current DDL types every
    boolean-ish column as INTEGER, so the resulting set is empty today — but
    it is determined from the catalog, never guessed, so a future BOOLEAN
    column is converted automatically (see _sqlite_bool_to_pg)."""
    if conn.dialect == "sqlite":
        return {
            row["name"]: (row["type"] or "").upper()  # PRAGMA type may be NULL
            for row in conn.execute(f"PRAGMA table_info({table});").fetchall()
        }
    return {
        row["column_name"]: (row["data_type"] or "").upper()
        for row in conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=? "
            "ORDER BY ordinal_position;",
            (table,),
        ).fetchall()
    }


def _bool_columns(conn: Conn, table: str) -> set[str]:
    """The set of BOOLEAN-typed columns on `table` (upper-cased type check).
    Empty for the current schema — see _column_types — but computed from the
    destination's real catalog so the conversion never relies on a hardcoded
    column list that could drift from the DDL."""
    return {
        name for name, typ in _column_types(conn, table).items() if typ == "BOOLEAN"
    }


def _table_row_count(conn: Conn, table: str) -> int:
    """The number of rows in `table`. Used three ways: the emptiness check on
    the destination, the dry-run copy plan, and the post-restore verification
    (the re-count that confirms the copy actually landed)."""
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table};").fetchone()
    return int(row["n"])


# ── The copy ──────────────────────────────────────────────────────────────────


def _sqlite_bool_to_pg(v):
    """Map a SQLite 0/1 boolean (stored as INTEGER) to a Postgres boolean.
    None stays None; anything that is not a clear 0/1 is refused loudly — a
    genuinely corrupt value in a boolean column should fail the restore, not
    be silently coerced. Fires only for BOOLEAN destination columns, none of
    which exist in the current DDL (see _bool_columns)."""
    if v is None:
        return None  # NULL is NULL in both dialects — never invented
    if isinstance(v, bool):
        return v  # already a real boolean — pass through untouched
    if v in (0, 1):
        return bool(v)  # the sqlite integer spelling: 0 → False, 1 → True
    if v in ("0", "1"):
        return v == "1"  # a text spelling sqlite could hold via loose typing
    raise ValueError(f"expected 0/1/None for a boolean column, got {v!r}")


def _copy_table(
    source: Conn, dest: Conn, table: str, *, bool_columns: set[str]
) -> int:
    """Copy every row of `table` from source to dest, verbatim. Returns the
    number of rows copied. Raises on any failure — the caller's transaction
    rolls the whole restore back, so a partial table never survives.

    The copy reads the source's ACTUAL columns (not a hardcoded list), selects
    exactly those, and inserts exactly those with parameterised placeholders —
    ids, timestamps, JSON payloads and NULLs round-trip unchanged. It never
    routes through write_gate.commit(): this tool reproduces an audit trail
    that already exists, and the gate would fabricate NEW provenance for OLD
    events (see the module docstring).
    """
    cols = _column_names(source, table)  # the source's real columns
    if not cols:
        return 0  # an empty table has no columns to select or insert
    # The destination was just created by apply_schema, so it must contain
    # every source column; if it does not, the schemas have diverged and a
    # silent column drop would corrupt the audit trail — refuse loudly.
    dest_cols = _column_names(dest, table)
    missing = [c for c in cols if c not in dest_cols]
    if missing:
        raise RuntimeError(
            f"table {table}: destination is missing source columns {missing} — "
            f"schemas have diverged; refusing to copy"
        )
    # The index positions of BOOLEAN destination columns, computed once per
    # table. SQLite stores 0/1 as INTEGER; Postgres BOOLEAN columns need real
    # booleans, so those values are converted per row below.
    bool_idx = {i for i, c in enumerate(cols) if c in bool_columns}
    # Parameterised SQL on BOTH dialects: `?` placeholders flow through the
    # Conn wrapper (translated to %s on postgres) — never string interpolation
    # of values. Column/table names are code-derived literals, so interpolating
    # THOSE is safe (the same precedent the repo uses throughout).
    select_sql = f"SELECT {', '.join(cols)} FROM {table};"
    insert_sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(['?'] * len(cols))});"
    )
    copied = 0  # the per-table counter the caller reports and re-verifies
    for row in source.execute(select_sql):  # stream rows — no full fetch needed
        # Positional access keeps the value order aligned with `cols` on both
        # dialects (sqlite3.Row and the postgres mapping row both support it).
        values = [row[i] for i in range(len(cols))]
        if bool_idx:  # only when this table actually has a BOOLEAN column
            values = [
                _sqlite_bool_to_pg(values[i]) if i in bool_idx else values[i]
                for i in range(len(cols))
            ]
        dest.execute(insert_sql, values)  # the verbatim insert, in-transaction
        copied += 1
    return copied


def _restore(
    source: Conn,
    dest: Conn,
    tables: list[str],
    *,
    bool_columns_by_table: dict[str, set[str]],
) -> dict[str, int]:
    """Copy every table into the destination inside ONE transaction. Returns
    {table: copied_count}. A failure at any point rolls the WHOLE restore back,
    leaving the destination EMPTY rather than half-loaded — the H4b lesson:
    bare execute() calls autocommit on both dialects, and a partial write there
    destroyed a table. The COMMIT is the single point of no return."""
    dest.begin_write()  # sqlite BEGIN IMMEDIATE / postgres BEGIN — see app/db.py
    try:
        copied: dict[str, int] = {}  # per-table counts for the report + re-count
        for table in tables:  # FK-safe order, so no insert violates a constraint
            copied[table] = _copy_table(
                source, dest, table,
                bool_columns=bool_columns_by_table.get(table, set()),
            )
        dest.execute("COMMIT")  # only after EVERY table copied does it land
        return copied
    except Exception:
        # Any failure — a constraint collision, a corrupt value, a connection
        # drop — rolls back to the untouched empty destination. Re-raise so
        # main() reports the failure instead of mistaking it for success.
        dest.execute("ROLLBACK")
        raise


# ── The CLI ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """The CLI entry point. Returns the process exit code: 0 on a successful
    dry-run plan or a completed (and verified) restore; 1 on any refusal or
    failure. Errors are printed to stderr; every step is printed so the tool
    never acts silently (Golden Rule: never skip logs)."""
    parser = argparse.ArgumentParser(
        prog="scripts/restore_db.py",
        description=(
            "Copy every row of a completed local run into an EMPTY database "
            "verbatim. Dry run is the default; --confirm performs the copy. "
            "A non-empty destination is refused fail-closed."
        ),
    )
    parser.add_argument(
        "--source", required=True,
        help="the source database — a sqlite file path or a postgresql:// / "
             "cloudsql:// URL",
    )
    parser.add_argument(
        "--dest", required=True,
        help="the EMPTY destination database — a sqlite file path or a "
             "postgresql:// / cloudsql:// URL",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="perform the copy. WITHOUT this flag the tool only connects, "
             "checks the destination is empty, and prints the copy plan — it "
             "writes nothing.",
    )
    args = parser.parse_args(argv)

    source_conn = None  # opened lazily below; the finally block closes it
    dest_conn = None  # opened lazily below; the finally block closes it
    try:
        # The source must exist BEFORE connect(): sqlite3.connect() silently
        # CREATES a missing file, and a tool that invented an empty source
        # would "successfully" copy nothing — a hidden side effect.
        if not args.source.startswith(("postgresql://", "postgres://", "cloudsql://")):
            if not Path(args.source).exists():
                print(
                    f"ERROR: source database {args.source!r} does not exist — "
                    f"nothing to restore.",
                    file=sys.stderr,
                )
                return 1

        # Reuse the ONE connection layer — connect() decides the dialect from
        # the target string. No second connection implementation exists here.
        source_conn = connect(args.source)
        dest_conn = connect(args.dest)
        # apply_schema() is idempotent (CREATE IF NOT EXISTS + the in-place
        # migrations), so calling it on the destination first makes a fresh
        # database work and never disturbs a provisioned one. It is what makes
        # the per-table row counts below meaningful.
        apply_schema(dest_conn)

        tables = _table_order()  # the FK-safe copy order derived from the DDL

        # Fail closed on a schema-mismatched source: every DDL table must exist
        # in the source, or the FK-safe order cannot be trusted (a present
        # table might reference a missing one and the destination's enforced
        # FKs would reject the copy mid-way).
        source_tables = _existing_tables(source_conn)
        missing_tables = [t for t in tables if t not in source_tables]
        if missing_tables:
            print(
                f"ERROR: source {args.source!r} is missing tables "
                f"{missing_tables} — the source predates the current schema; "
                f"refusing.",
                file=sys.stderr,
            )
            return 1

        # The counts BOTH modes report: what the source holds (the copy plan)
        # and what the destination holds (the emptiness check).
        source_counts = {t: _table_row_count(source_conn, t) for t in tables}
        dest_counts = {t: _table_row_count(dest_conn, t) for t in tables}

        # ── THE EMPTY-DESTINATION GUARD ─────────────────────────────────────
        # This is the substance of the ticket. The destination is the
        # operator's real Cloud SQL database; a partial or duplicate load
        # would corrupt the audit trail (write_log, steps, state_transitions)
        # in a way that cannot be distinguished from real history afterwards.
        # Refusing is always recoverable (drop the destination database); a
        # bad merge is not. There is no --force. The guard fires in BOTH
        # modes — even a dry run against a non-empty destination is refused,
        # because pointing this tool at a database that already has history is
        # a configuration error that must be surfaced, never "planned around".
        nonempty = [f"{t}={n}" for t, n in dest_counts.items() if n > 0]
        if nonempty:
            print(
                "ERROR: destination is NOT empty — refusing to restore into it. "
                "Found rows in: " + ", ".join(nonempty) + ". "
                "This tool's destination must be empty: a partial or duplicate "
                "load would corrupt the audit trail in a way that cannot be "
                "distinguished from real history afterwards. Refusing is always "
                "recoverable (drop the destination database); a bad merge is "
                "not. There is no --force.",
                file=sys.stderr,
            )
            return 1

        if not args.confirm:
            # ── DRY RUN (the default) — report the plan, write nothing ─────
            print(f"source: {args.source} (dialect={source_conn.dialect})")
            print(f"dest:   {args.dest} (dialect={dest_conn.dialect})")
            print("DRY RUN — nothing will be written. Re-run with --confirm to copy.")
            print("destination is empty — safe to restore.")
            total = 0
            for t in tables:
                # Per-table report of BOTH sides: what would be copied (the
                # source's rows) and what the destination currently holds (the
                # emptiness check, all zero here — the guard already refused a
                # non-empty destination above).
                print(
                    f"  {t}: would copy {source_counts[t]} row(s), "
                    f"destination currently has {dest_counts[t]}"
                )
                total += source_counts[t]
            print(f"TOTAL: {total} row(s) would be copied.")
            return 0

        # ── CONFIRM: perform the copy ──────────────────────────────────────
        print(f"source: {args.source} (dialect={source_conn.dialect})")
        print(f"dest:   {args.dest} (dialect={dest_conn.dialect})")
        print("destination is empty — safe to restore.")
        # The BOOLEAN destination columns per table, read from the destination
        # catalog (freshly created by apply_schema, so this IS the DDL's
        # types). None exist in the current DDL — the conversion path is
        # dormant but correct and future-proof.
        bool_columns_by_table = {t: _bool_columns(dest_conn, t) for t in tables}
        try:
            copied = _restore(
                source_conn, dest_conn, tables,
                bool_columns_by_table=bool_columns_by_table,
            )
        except Exception as exc:
            # The transaction already rolled back inside _restore — say so, so
            # the operator knows the destination is empty and can retry after
            # fixing the cause (a real failure must never look like success).
            print(
                f"ERROR: restore FAILED — the destination was rolled back to "
                f"empty: {exc}",
                file=sys.stderr,
            )
            return 1

        # ── Report + post-check ────────────────────────────────────────────
        total = 0
        for t in tables:  # per-table copied counts, then the running total
            total += copied[t]
            print(f"  copied {t}: {copied[t]} row(s)")
        print(f"TOTAL: {total} row(s) copied.")
        # The post-check is the point: do NOT report success from the fact that
        # the inserts did not raise — re-count the destination and confirm the
        # numbers match what was copied.
        re_counted = {t: _table_row_count(dest_conn, t) for t in tables}
        mismatches = [
            f"{t}: copied={copied[t]} dest={re_counted[t]}"
            for t in tables
            if re_counted[t] != copied[t]
        ]
        if mismatches:
            print(
                "ERROR: post-restore re-count does not match what was copied: "
                + ", ".join(mismatches),
                file=sys.stderr,
            )
            return 1
        print("post-restore verification: all destination counts match the copy.")
        return 0
    finally:
        # Explicit close on every path — the same hygiene every CLI keeps, and
        # it releases the Cloud SQL connector's background thread if a
        # cloudsql:// destination was used.
        if source_conn is not None:
            source_conn.close()
        if dest_conn is not None:
            dest_conn.close()


# Guard so `python scripts/restore_db.py` works, not just `python -m
# scripts.restore_db` (both are supported — this file carries the sys.path
# bootstrap). Uses SystemExit instead of sys.exit() to stay testable (pytest
# can catch SystemExit), the same pattern as every stage CLI.
if __name__ == "__main__":
    raise SystemExit(main())
