"""Tests for ticket G1: making ``contacts.email_verified`` settable from a CSV.

Covers the operator-asserted ``email_verified`` column end to end:
  - the full truthy/falsy vocabulary (and a visible error for an
    unrecognised value, so ``ture`` is caught at import, not at the gate);
  - the backward-compatible absent-column path (still 0);
  - the hard syntactic gate — a malformed address stores 0 even when the CSV
    asserts otherwise (each rejection form gets its own assertion);
  - a verified contact no longer trips send-gate check 3;
  - provenance of the flag is recoverable from write_log and steps;
  - re-import creates NEW rows and never downgrades an existing verified
    contact (the current no-upsert behaviour, asserted as-is).

No real addresses appear here — every address is on an RFC 2606 reserved
domain (``.test``), matching the repo's standing rule that test fixtures
never fabricate deliverable addresses for real companies.
"""

import csv as csv_module
import json
from pathlib import Path

import pytest

from app.agents_registry import seed_agent_registry
from app.config import sync_offers_table
from app.db import IntegrityError, apply_schema, connect  # IntegrityError is app.db's dialect-agnostic tuple (sqlite3 + pg8000), so a raises-clause written against it catches the constraint violation on BOTH engines
from app.kill_switch import write_kill_switch
from app.send_gate import evaluate_send_gate
from app.tools.get_targets import EmailVerifiedValueError, import_csv

# The repo root, used to import the REAL operator CSV (data/hk_therapy_targets.csv)
# and prove its new 7-column shape still imports cleanly with every flag 0.
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def conn_with_offers(tmp_path, scratch_db_target):
    """Fresh DB with the schema, seeded principals, and both offer slugs the
    tests need (acme-offer for synthetic CSVs, therapy-app for the real CSV)."""
    offers_dir = tmp_path / "offers"
    offers_dir.mkdir()
    # One offer per slug the tests reference.  from_address is a reserved-domain
    # placeholder — offer YAML content is not exercised by import_csv.
    for slug in ("acme-offer", "therapy-app"):
        (offers_dir / f"{slug}.yaml").write_text(
            "pitch: p\npersona_hint: h\ntemplate: t\nfrom_address: a@b.test\n"
        )
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    seed_agent_registry(c, run_id="r0", step_id="s0")
    sync_offers_table(c, str(offers_dir), run_id="r0", step_id="s0")
    yield c
    c.close()


def write_csv(path, rows, fieldnames):
    """Write a CSV the same way the operator's spreadsheet export would."""
    with open(path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _import(conn, csv_path, *, run_id="r1", step_id="s1"):
    return import_csv(
        conn, csv_path=str(csv_path), cli_offer_slug=None,
        run_id=run_id, step_id=step_id,
    )


# ── Vocabulary and the fail-closed default ───────────────────────────────────


def test_absent_email_verified_column_defaults_unverified(conn_with_offers, tmp_path):
    """Backward compatibility: a CSV with no email_verified column imports
    the contact with email_verified=0, exactly as before G1."""
    csv_path = tmp_path / "absent.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email"],
    )
    target_ids = _import(conn_with_offers, csv_path)
    assert len(target_ids) == 1
    contact = conn_with_offers.execute("SELECT email, email_verified FROM contacts;").fetchone()
    assert contact["email"] == "jane@acme.test"
    assert contact["email_verified"] == 0


@pytest.mark.parametrize("token", ["1", "true", "yes", "y", "verified", "on", " TRUE ", "Yes"])
def test_truthy_vocabulary_sets_verified(conn_with_offers, tmp_path, token):
    """Every documented truthy token (case/whitespace-insensitive) asserts
    verification and, with a syntactically valid address, stores 1."""
    csv_path = tmp_path / "truthy.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test", "email_verified": token}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email", "email_verified"],
    )
    _import(conn_with_offers, csv_path)
    contact = conn_with_offers.execute("SELECT email_verified FROM contacts;").fetchone()
    assert contact["email_verified"] == 1


@pytest.mark.parametrize("token", ["0", "false", "no", "n", "unverified", "off", "", " False "])
def test_falsy_vocabulary_stays_unverified(conn_with_offers, tmp_path, token):
    """Every documented falsy token (including blank) stores 0 — fail closed."""
    csv_path = tmp_path / "falsy.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test", "email_verified": token}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email", "email_verified"],
    )
    _import(conn_with_offers, csv_path)
    contact = conn_with_offers.execute("SELECT email_verified FROM contacts;").fetchone()
    assert contact["email_verified"] == 0


def test_unrecognised_email_verified_value_is_a_visible_error(conn_with_offers, tmp_path):
    """A typo (``ture``) must stop the import loudly, not become a silent
    unverified contact that the send gate refuses three stages later."""
    csv_path = tmp_path / "typo.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test", "email_verified": "ture"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email", "email_verified"],
    )
    with pytest.raises(EmailVerifiedValueError):
        _import(conn_with_offers, csv_path)


def test_truthy_assertion_without_contact_email_is_a_visible_error(conn_with_offers, tmp_path):
    """Asserting an address is verified when no address is present is a data
    contradiction — surfaced at import, never stored as a verified NULL email."""
    csv_path = tmp_path / "noemail.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "email_verified": "true"}],
        fieldnames=["company_name", "domain", "offer_id", "email_verified"],
    )
    with pytest.raises(EmailVerifiedValueError):
        _import(conn_with_offers, csv_path)


# ── The hard syntactic gate (G1 §2.2) ────────────────────────────────────────


@pytest.mark.parametrize("address", [
    "no-at-sign.test",          # no @
    "a@b@acme.test",            # more than one @
    "@acme.test",               # empty local part
    "user@",                    # empty domain
    "user@localhost",           # domain with no dot
    "user name@acme.test",      # whitespace inside the address
    ".user@acme.test",          # leading dot in local part
    "user.@acme.test",          # trailing dot in local part
    "user@.acme.test",          # leading dot in domain
    "user@acme.test.",          # trailing dot in domain
])
def test_syntax_rejection_overrides_truthy_assertion(conn_with_offers, tmp_path, address):
    """Each syntactic rejection form stores email_verified=0 DESPITE a truthy
    CSV assertion — the operator can vouch for a real address, not a malformed
    string (ticket G1 §2.2)."""
    csv_path = tmp_path / "malformed.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": address, "email_verified": "true"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email", "email_verified"],
    )
    _import(conn_with_offers, csv_path)
    contact = conn_with_offers.execute("SELECT email, email_verified FROM contacts;").fetchone()
    assert contact["email"] == address
    assert contact["email_verified"] == 0


# ── Send-gate integration ────────────────────────────────────────────────────


def test_verified_contact_passes_send_gate_check_3(conn_with_offers, tmp_path, monkeypatch):
    """The test that proves the ticket worked: drive a verified contact through
    the real evaluate_send_gate and show check 3 no longer appears in
    missing_requirements (the other checks may still refuse — only check 3 is
    under test)."""
    switch = tmp_path / "kill_switch.json"
    write_kill_switch(engaged=False, updated_by="fixture", path=str(switch))
    monkeypatch.setenv("OUTBOUND_KILL_SWITCH_PATH", str(switch))

    csv_path = tmp_path / "verified.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test", "email_verified": "true"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email", "email_verified"],
    )
    target_ids = _import(conn_with_offers, csv_path)
    decision = evaluate_send_gate(
        conn_with_offers, target_id=target_ids[0], run_id="r1", step_id="s9",
    )
    assert "contact.email_verified == true" not in decision.missing_requirements


# ── Provenance (G1 §2.3) ─────────────────────────────────────────────────────


def test_verification_provenance_is_recoverable(conn_with_offers, tmp_path):
    """The origin of the verified flag must be auditable: the per-contact
    write_log payload carries email_verified AND its source, and the import
    step carries the run-level list of operator-verified contacts."""
    csv_path = tmp_path / "provenance.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test", "email_verified": "true"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email", "email_verified"],
    )
    target_ids = _import(conn_with_offers, csv_path)

    contact = conn_with_offers.execute("SELECT contact_id FROM contacts;").fetchone()
    write_row = conn_with_offers.execute(
        "SELECT payload_json FROM write_log WHERE action='insert_contact';"
    ).fetchone()
    payload = json.loads(write_row["payload_json"])
    assert payload["email_verified"] == 1
    assert payload["email_verified_source"] == "operator_asserted_csv"

    step = conn_with_offers.execute(
        "SELECT output_json FROM steps WHERE tool_name='get_targets';"
    ).fetchone()
    out = json.loads(step["output_json"])
    assert out["target_ids"] == target_ids
    assert out["operator_verified_contact_ids"] == [contact["contact_id"]]


# ── Re-import behaviour (G1 §3) ──────────────────────────────────────────────


def test_reimport_same_domain_fails_loudly_and_never_downgrades(conn_with_offers, tmp_path):
    """Establish the actual no-upsert behaviour: import_csv NEVER updates an
    existing contact.  Re-importing the SAME normalized_domain hits the
    accounts UNIQUE constraint and raises IntegrityError BEFORE any contact is
    written — a visible failure, not a silent downgrade of the existing
    verified contact."""
    first = tmp_path / "first.csv"
    write_csv(
        first,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test", "email_verified": "true"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email", "email_verified"],
    )
    _import(conn_with_offers, first, run_id="r1", step_id="s1")
    original = conn_with_offers.execute("SELECT contact_id, email_verified FROM contacts;").fetchone()
    assert original["email_verified"] == 1

    second = tmp_path / "second.csv"
    # Same domain but NO email_verified column — would import unverified if it
    # could reach the contacts write, but it must not: the accounts UNIQUE
    # constraint stops the row first.
    write_csv(
        second,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email"],
    )
    # IntegrityError is app.db's dialect-agnostic tuple (sqlite3.IntegrityError
    # + pg8000_dbapi.IntegrityError), so this raises-clause catches the accounts
    # UNIQUE constraint violation on BOTH engines — on Postgres the driver
    # raises the pg8000 class, which the old sqlite3-only clause missed (H4a).
    with pytest.raises(IntegrityError):
        _import(conn_with_offers, second, run_id="r2", step_id="s2")

    # The original verified contact is untouched and no second contact exists.
    rows = conn_with_offers.execute("SELECT contact_id, email_verified FROM contacts;").fetchall()
    assert len(rows) == 1
    assert rows[0]["contact_id"] == original["contact_id"]
    assert rows[0]["email_verified"] == 1


def test_reimport_under_a_different_domain_never_touches_existing_verified(conn_with_offers, tmp_path):
    """When the new CSV uses a DIFFERENT domain, import_csv creates a brand-new
    account + contact (unverified, because the column is absent) and leaves the
    existing verified contact untouched — there is no in-place UPDATE path."""
    first = tmp_path / "first.csv"
    write_csv(
        first,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test", "email_verified": "true"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email", "email_verified"],
    )
    _import(conn_with_offers, first, run_id="r1", step_id="s1")
    original = conn_with_offers.execute("SELECT contact_id, email_verified FROM contacts;").fetchone()
    assert original["email_verified"] == 1

    second = tmp_path / "second.csv"
    write_csv(
        second,
        [{"company_name": "Other Co", "domain": "other.test", "offer_id": "acme-offer",
          "contact_email": "jane@acme.test"}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email"],
    )
    _import(conn_with_offers, second, run_id="r2", step_id="s2")

    rows = conn_with_offers.execute("SELECT contact_id, email_verified FROM contacts;").fetchall()
    assert len(rows) == 2  # a new contact row under the new account
    by_id = {r["contact_id"]: r["email_verified"] for r in rows}
    assert by_id[original["contact_id"]] == 1  # untouched
    assert sorted(by_id.values()) == [0, 1]  # the new row is unverified by default


# ── The real operator CSV imports cleanly ────────────────────────────────────


def test_real_hk_therapy_csv_shape_imports_cleanly(conn_with_offers):
    """The real data/hk_therapy_targets.csv (now with empty contact_email and
    email_verified headers) must import all 10 rows with every stored flag 0
    — no fabricated addresses, no accidental verification."""
    csv_path = REPO_ROOT / "data" / "hk_therapy_targets.csv"
    target_ids = _import(conn_with_offers, csv_path)
    assert len(target_ids) == 10
    contacts = conn_with_offers.execute("SELECT email, email_verified FROM contacts;").fetchall()
    assert contacts, "rows with a contact_name must create a contact"
    for contact in contacts:
        assert contact["email"] is None  # no real address was fabricated
        assert contact["email_verified"] == 0  # empty assertion → fail closed
