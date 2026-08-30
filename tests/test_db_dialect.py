"""
Tests for the dialect-aware db layer (Google port plan row A2) — SQLite side.

The postgres side is covered by test_db_postgres.py, which only runs when
Cloud SQL env config is present (a passing SQLite suite cannot prove the
postgres path works). These tests pin the wrapper contract that BOTH dialects
must honor, using SQLite temp files:

- connect() returns a Conn whose .dialect reports the engine
- rows are dict-like (row["column_name"]) on sqlite
- begin_write() issues BEGIN IMMEDIATE on sqlite (write-lock-up-front)
- _translate_sql() rewrites ? → %s and datetime('now') → a to_char(now()…) form
  that yields SQLite's exact timestamp string
  without touching quoted literals or comments
- app.db.IntegrityError catches the sqlite variant
"""

import sqlite3

import pytest

from app.db import (
    _PG_DATETIME_NOW,
    Conn,
    IntegrityError,
    _translate_sql,
    apply_schema,
    connect,
)


def test_connect_returns_conn_with_sqlite_dialect(tmp_path):
    # connect() must return the wrapper (not a raw sqlite3.Connection) so
    # callers can read .dialect and get begin_write()/executescript().
    conn = connect(str(tmp_path / "t.db"))
    assert isinstance(conn, Conn)
    assert conn.dialect == "sqlite"
    conn.close()


def test_rows_are_dict_like_on_sqlite(tmp_path):
    # The wrapper's sqlite path must keep sqlite3.Row's dict-like access —
    # every module reads rows via row["column_name"].
    conn = connect(str(tmp_path / "t.db"))
    apply_schema(conn)
    conn.execute(
        "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        ("off_1", "dict-like", 1),
    )
    row = conn.execute(
        "SELECT slug, active FROM offers WHERE offer_id = ?", ("off_1",)
    ).fetchone()
    assert row["slug"] == "dict-like"  # name access — the shape every module relies on
    assert row["active"] == 1
    assert row[0] == "dict-like"  # index access still works too
    conn.close()


def test_begin_write_issues_begin_immediate_on_sqlite(tmp_path):
    # begin_write() must keep the pre-A2 lock-up-front semantics on sqlite:
    # while connection A holds the write transaction, connection B's
    # begin_write() must fail cleanly instead of silently upgrading a read
    # lock — the same property test_db.py's
    # test_write_transaction_uses_begin_immediate_semantics proves.
    db_path = str(tmp_path / "t.db")
    conn_a = connect(db_path)
    apply_schema(conn_a)
    conn_b = connect(db_path)

    conn_a.begin_write()
    conn_a.execute(
        "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        ("off_1", "writer-a", 1),
    )
    with pytest.raises(sqlite3.OperationalError):
        conn_b.begin_write()
    conn_a.commit()
    conn_a.close()
    conn_b.close()


# ── _translate_sql unit tests — the postgres-side rewrite ────────────────────


def test_translate_question_mark_placeholder():
    # The core rewrite: every bare ? placeholder becomes %s (pg8000's native
    # paramstyle).
    sql = "SELECT * FROM t WHERE a = ? AND b = ?"
    assert _translate_sql(sql) == "SELECT * FROM t WHERE a = %s AND b = %s"


def test_translate_skips_question_mark_inside_string_literals():
    # A literal ? inside quotes is DATA, not a placeholder — the scanner must
    # skip quoted regions rather than blind-replace (a str.replace would
    # corrupt this query into a broken %s).
    sql = "INSERT INTO t (a, b) VALUES (?, 'what?')"
    assert _translate_sql(sql) == "INSERT INTO t (a, b) VALUES (%s, 'what?')"


def test_translate_skips_escaped_quotes_inside_strings():
    # '' inside a string is an escaped quote, not the end of the string — a ?
    # after it must stay untouched or the scanner would treat the remainder
    # of the string as SQL.
    sql = "SELECT 'it''s a ?' AS q, ? AS p"
    assert _translate_sql(sql) == "SELECT 'it''s a ?' AS q, %s AS p"


def test_translate_skips_double_quoted_identifiers_and_comments():
    # "double-quoted identifiers", -- line comments and /* block comments */
    # are all copied verbatim — a ? inside any of them is not a placeholder.
    sql = 'SELECT "col?" AS x, ? AS y -- trailing ? comment\nFROM t /* ? */'
    assert _translate_sql(sql) == (
        'SELECT "col?" AS x, %s AS y -- trailing ? comment\nFROM t /* ? */'
    )


def test_translate_datetime_now_becomes_current_timestamp():
    # datetime('now') is SQLite's UTC timestamp function; Postgres has no
    # datetime() at all. It becomes the to_char(now() AT TIME ZONE 'UTC', …)
    # form rather than CURRENT_TIMESTAMP, so the string stored in these TEXT
    # columns is byte-identical to SQLite's — see _PG_DATETIME_NOW in app/db.py.
    sql = "INSERT INTO t (id, created_at) VALUES (?, datetime('now'))"
    assert _translate_sql(sql) == (
        "INSERT INTO t (id, created_at) VALUES (%s, " + _PG_DATETIME_NOW + ")"
    )


def test_translate_leaves_datetime_now_inside_string_untouched():
    # datetime('now') appearing as DATA (inside quotes) must not be rewritten
    # — only the function call in code position is.
    sql = "SELECT 'datetime(''now'')' AS literal, datetime('now') AS ts"
    assert _translate_sql(sql) == (
        "SELECT 'datetime(''now'')' AS literal, " + _PG_DATETIME_NOW + " AS ts"
    )


def test_translate_dollar_quoted_section_untouched():
    # Dollar-quoted strings are a postgres-only quoting form — the scanner
    # honors them so postgres-authored SQL passes through untranslated.
    sql = "SELECT $$a ? b$$ AS d, ? AS p"
    assert _translate_sql(sql) == "SELECT $$a ? b$$ AS d, %s AS p"


def test_cloudsql_target_with_missing_password_raises(monkeypatch):
    # "Failures surfaced clearly": a cloudsql:// target without
    # OUTBOUND_DB_PASSWORD must raise naming the missing variable — never
    # silently fall back to sqlite and split the operator's data across two
    # engines without anyone noticing.
    monkeypatch.delenv("OUTBOUND_DB_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="OUTBOUND_DB_PASSWORD"):
        connect("cloudsql://proj:region:inst/db")


def test_malformed_cloudsql_target_raises(monkeypatch):
    # The sentinel must be cloudsql://<instance-connection-name>/<database> —
    # anything else raises before any network attempt (password is stubbed so
    # the shape check, which runs first, is what this test exercises).
    monkeypatch.setenv("OUTBOUND_DB_PASSWORD", "stub")
    with pytest.raises(ValueError, match="malformed"):
        connect("cloudsql://no-database-here")


def test_postgresql_url_missing_parts_raises():
    # A postgresql:// URL missing host/user/database must raise naming what
    # is missing, before any connection attempt — no guessing defaults.
    with pytest.raises(ValueError, match="missing"):
        connect("postgresql://localhost")


def test_integrity_error_tuple_catches_sqlite_variant(tmp_path):
    # app.db.IntegrityError is a tuple of both dialects' exception classes;
    # on sqlite it must catch sqlite3.IntegrityError. This is the exact catch
    # shape detect_signals uses to dedupe signals, so if this regresses the
    # dedup path breaks.
    conn = connect(str(tmp_path / "t.db"))
    apply_schema(conn)
    conn.execute(
        "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        ("off_1", "unique-slug", 1),
    )
    caught = None
    try:
        # Duplicate slug violates the UNIQUE constraint on offers.slug.
        conn.execute(
            "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
            ("off_2", "unique-slug", 1),
        )
    except IntegrityError as exc:
        caught = exc
    assert caught is not None
    assert isinstance(caught, sqlite3.IntegrityError)
    conn.close()
