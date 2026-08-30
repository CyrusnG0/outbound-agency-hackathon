"""
Live Postgres (Cloud SQL) tests for the dialect-aware db layer (plan row A2).

These tests SKIP unless the operator has wired Postgres access, because they
need a real instance — a passing SQLite suite cannot prove the postgres path
works. To run them against a SCRATCH database:

    export OUTBOUND_TEST_DB_TARGET="cloudsql://outbound-agency-devpost:us-central1:outbound-db/outbound_scratch"
    export OUTBOUND_DB_PASSWORD="$(gcloud secrets versions access latest --secret=outbound-db-password --project=outbound-agency-devpost)"
    pytest tests/test_db_postgres.py -v

WHY OUTBOUND_TEST_DB_TARGET, NOT THE PRODUCTION VARIABLE — THE 2026-08-27/28
INCIDENT:
This module's tests are destructive: the shared scratch_db_target fixture
empties its target before every test, and
test_reset_scratch_database_empties_live_postgres drops the whole public
schema. Until ticket S2 this file read the PRODUCTION database variable —
the one every console and CLI reads (tests/conftest.py's TEST_DB_TARGET_ENV
comment names it and explains why the scratch fixture deliberately uses a
different one) — and its own docstring instructed pointing it at the live
Cloud SQL instance "outbound". Running the suite as documented wrote 13 real
offers (named pg-live-test-* / pg-live-dup-*) and a partial agent_registry
(2 of 7 rows) into production; they were found and manually cleared before a
real demo run could be restored into it. Do NOT "helpfully" revert this to
the production variable: the scratch_db_target fixture refuses (fail-closed,
app/db.py::scratch_target_violation) any target whose database name lacks
'scratch' or 'test', so the production instance can never be this suite's
target again.

Acceptance gate: apply_schema() must create all 20 tables from
docs/db-schema.md on the live instance — including the A3 migration pass
adding agent_id to the already-provisioned steps and write_log tables — a
write_gate.commit() must round-trip its data row AND its write_log audit row
(with agent attribution), the agent registry must seed idempotently, and
app.db.IntegrityError must catch the postgres integrity exception class.
"""

import json
import os
import re
import uuid

import pytest

# TEST_DB_TARGET_ENV is the shared scratch-target variable name
# (OUTBOUND_TEST_DB_TARGET), defined ONCE in tests/conftest.py — importing the
# constant (rather than re-spelling the string) keeps this module and the
# fixture unable to drift apart (S2).
from conftest import TEST_DB_TARGET_ENV
from app.agents_registry import seed_agent_registry
from app.db import IntegrityError, apply_schema, connect, reset_scratch_database
from app.write_gate import commit

# The target comes from OUTBOUND_TEST_DB_TARGET (the S2 correction) — the
# SEPARATE destructive-scratch variable, deliberately never the production
# database variable (see conftest.py's TEST_DB_TARGET_ENV comment for that
# name and why the two must never be the same).  The skipif below is the only
# use of this module-level value; every test gets its real target from the
# scratch_db_target fixture instead.
TARGET = os.environ.get(TEST_DB_TARGET_ENV)
PASSWORD = os.environ.get("OUTBOUND_DB_PASSWORD")

pytestmark = pytest.mark.skipif(
    not (TARGET and PASSWORD),
    reason="Cloud SQL not configured: set OUTBOUND_TEST_DB_TARGET and "
    "OUTBOUND_DB_PASSWORD (password via `gcloud secrets versions access "
    "latest --secret=outbound-db-password`)",
)

# The complete table list from docs/db-schema.md — the same set test_db.py
# asserts on SQLite, so a schema drift between dialects fails one of the two
# suites. (20 tables as of plan task A3, which adds agent_registry.)
REQUIRED_TABLES = {
    "accounts", "signals", "contacts", "offers", "targets", "messages",
    "message_draft_versions", "replies", "steps", "suppressions",
    "write_log", "policy_decisions", "state_transitions",
    "send_gate_decisions", "review_decisions", "signal_outcome_link",
    "signal_weights", "candidate_fields", "enrichment_runs",
    "agent_registry",
}


@pytest.fixture
def conn(scratch_db_target):
    # One connection per test, closed after — connect() opens a fresh Cloud
    # SQL connector each time and close() releases its refresh thread.
    # The target is the shared scratch_db_target fixture: it honours
    # OUTBOUND_TEST_DB_TARGET (Postgres) else the SQLite tmp default, and
    # when configured it has ALREADY emptied the target and refused any URL
    # whose database name lacks 'scratch'/'test' (fail-closed) — so this
    # fixture can never reach production (S2).
    c = connect(scratch_db_target)
    # Apply the full DDL so every table exists: the scratch reset leaves an
    # EMPTY schema, and several tests below write (offers, write_log) or seed
    # the agent registry — without this they would die on "relation does not
    # exist" the first time they run against a fresh target.  Idempotent.
    apply_schema(c)
    yield c
    c.close()


def test_connect_reports_postgres_dialect(conn):
    # Dialect detection must route the cloudsql:// sentinel to postgres.
    assert conn.dialect == "postgres"


def test_apply_schema_creates_all_documented_tables(conn):
    # The acceptance gate: the full schema must exist on the live instance.
    # Idempotent (CREATE TABLE IF NOT EXISTS), so re-running the suite against
    # the shared instance is safe.
    apply_schema(conn)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public';"
        ).fetchall()
    }
    missing = REQUIRED_TABLES - tables
    assert not missing, f"missing tables after apply_schema: {sorted(missing)}"


def test_placeholders_and_datetime_translate_against_live_postgres(conn):
    # One statement exercising all three translation behaviors at once:
    # a ? placeholder (→ %s), datetime('now') (→ the to_char(now()…) form), and a
    # quoted '?' literal that must survive untranslated as data.
    row = conn.execute(
        "SELECT '?' AS literal_q, datetime('now') AS ts, ? AS bound",
        ("bound-value",),
    ).fetchone()
    assert row["literal_q"] == "?"
    # The translated datetime('now') must yield SQLite's exact string shape:
    # 'YYYY-MM-DD HH:MM:SS', UTC, second precision, no offset and no
    # microseconds. A bare CURRENT_TIMESTAMP would give
    # '2026-08-19 07:39:38.620064+00' here and silently diverge from dev.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(row["ts"])), row["ts"]
    assert row["bound"] == "bound-value"


def test_write_gate_roundtrip_writes_data_and_audit_row(conn):
    # A full write_gate.commit() must work end-to-end on postgres: the data
    # row AND the append-only write_log audit row, atomically. Unique ids per
    # run so repeated suite runs against the shared instance never collide.
    #
    # Seed first (plan A3): commit() refuses writes from unregistered agents,
    # and the shared live instance's registry may be empty on first contact.
    # The upsert seeder makes this safe to run on every suite run.
    seed_agent_registry(conn, run_id="pg_live_test", step_id="pg_live_seed_step")
    offer_id = "off_" + uuid.uuid4().hex[:8]
    slug = f"pg-live-test-{uuid.uuid4().hex[:8]}"
    write_id = commit(
        conn,
        action="insert_offer",
        table_name="offers",
        record_id=offer_id,
        payload={"slug": slug},
        run_id="pg_live_test",
        step_id="pg_live_step",
        actor="system",
        agent_id="system",  # The seeded deterministic principal.
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=(offer_id, slug, 1),
    )

    # The data row landed, with dict-like row access on the postgres path.
    row = conn.execute(
        "SELECT slug, active FROM offers WHERE offer_id = ?", (offer_id,)
    ).fetchone()
    assert row is not None
    assert row["slug"] == slug
    assert row["active"] == 1

    # The audit row landed under the returned write_id — the "never skip
    # logging" guarantee, on postgres — and names the writing agent.
    audit = conn.execute(
        "SELECT action, table_name, record_id, agent_id, payload_json "
        "FROM write_log WHERE write_id = ?",
        (write_id,),
    ).fetchone()
    assert audit is not None
    assert audit["action"] == "insert_offer"
    assert audit["table_name"] == "offers"
    assert audit["record_id"] == offer_id
    assert audit["agent_id"] == "system"
    assert json.loads(audit["payload_json"]) == {"slug": slug}


def test_agent_id_columns_exist_on_steps_and_write_log(conn):
    # The A3 migration hazard test: the live instance's steps and write_log
    # tables existed BEFORE agent_id was added to the DDL, so this proves
    # apply_schema()'s ALTER pass added the columns to the real, already-
    # provisioned tables (not just to fresh ones).
    apply_schema(conn)
    for table in ("steps", "write_log"):
        col = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=? AND column_name='agent_id';",
            (table,),
        ).fetchone()
        assert col is not None, f"agent_id column missing on live {table}"


def test_seed_agent_registry_on_live_is_idempotent(conn):
    # The registry itself must exist on the live instance and seeding must
    # round-trip: two runs, still exactly the two existing principals.
    apply_schema(conn)
    seed_agent_registry(conn, run_id="pg_a3_seed_1", step_id="pg_a3_seed_step_1")
    seed_agent_registry(conn, run_id="pg_a3_seed_2", step_id="pg_a3_seed_step_2")
    rows = conn.execute(
        "SELECT agent_id, enabled, allowed_actions FROM agent_registry "
        "WHERE agent_id IN ('system','operator');"
    ).fetchall()
    by_id = {r["agent_id"]: r for r in rows}
    assert set(by_id) == {"system", "operator"}
    for row in rows:
        # Both seeded principals are enabled and carry the full action set.
        assert row["enabled"] == 1
        assert "insert_offer" in json.loads(row["allowed_actions"])


def test_integrity_error_catches_postgres_unique_violation(conn):
    # app.db.IntegrityError must catch the postgres variant too — this is the
    # detect_signals dedup path, and if it regressed, duplicate signals would
    # crash the node instead of being skipped.
    slug = f"pg-live-dup-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        ("off_" + uuid.uuid4().hex[:8], slug, 1),
    )
    caught = None
    try:
        # Second insert with the same slug violates offers.slug's UNIQUE
        # constraint — Postgres SQLSTATE 23505 — which Conn.execute()
        # re-raises as pg8000's IntegrityError class.
        conn.execute(
            "INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
            ("off_" + uuid.uuid4().hex[:8], slug, 1),
        )
    except IntegrityError as exc:
        caught = exc
    assert caught is not None


def test_reset_scratch_database_empties_live_postgres(scratch_db_target):
    # This test is destructive by design — it wipes the whole public schema.
    # It is safe ONLY because it runs against the scratch target the shared
    # scratch_db_target fixture provides: that fixture has already refused
    # (fail-closed, app/db.py::scratch_target_violation) any URL whose
    # database name lacks 'scratch' or 'test', and emptied the target.  The
    # manual scratch_target_violation check this test used to do is gone
    # because the fixture now enforces the identical guard before every test
    # (S2).
    conn = connect(scratch_db_target)
    try:
        apply_schema(conn)
    finally:
        conn.close()

    # Wipe and recreate public through the exact dialect-aware reset path.
    reset_scratch_database(scratch_db_target)

    # Prove public is empty: zero user tables remain after the reset.
    conn = connect(scratch_db_target)
    try:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public';"
        ).fetchall()
        assert tables == [], f"expected zero tables after reset, got {len(tables)}"
    finally:
        conn.close()
