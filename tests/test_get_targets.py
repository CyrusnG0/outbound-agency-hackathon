import csv as csv_module

import pytest

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.config import sync_offers_table
from app.tools.get_targets import (
    import_csv, MissingOfferIdError, UnknownOfferError, OfferConflictError,
)


@pytest.fixture
def conn_with_offers(tmp_path, scratch_db_target):
    offers_dir = tmp_path / "offers"
    offers_dir.mkdir()
    (offers_dir / "acme-offer.yaml").write_text(
        "pitch: p\npersona_hint: h\ntemplate: t\nfrom_address: a@b.test\n"
    )
    # scratch_db_target honours OUTBOUND_TEST_DB_TARGET (Postgres) else SQLite.
    c = connect(scratch_db_target)
    apply_schema(c)
    # Register the system agent (plan A3) — sync_offers_table writes through
    # the gate, which refuses unregistered agents.
    seed_agent_registry(c, run_id="r0", step_id="s0")
    sync_offers_table(c, str(offers_dir), run_id="r0", step_id="s0")
    yield c
    c.close()


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_import_creates_target_and_account_rows(conn_with_offers, tmp_path):
    csv_path = tmp_path / "targets.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme Inc", "domain": "https://www.acme.test/", "offer_id": "acme-offer"}],
        fieldnames=["company_name", "domain", "offer_id"],
    )
    target_ids = import_csv(
        conn_with_offers, csv_path=str(csv_path), cli_offer_slug=None,
        run_id="r1", step_id="s1",
    )
    assert len(target_ids) == 1

    account = conn_with_offers.execute("SELECT * FROM accounts;").fetchone()
    assert account["domain"] == "acme.test"  # normalized: scheme + www. stripped
    assert account["company_name"] == "Acme Inc"

    target = conn_with_offers.execute("SELECT * FROM targets;").fetchone()
    assert target["state"] == "new"


def test_missing_offer_id_column_and_no_cli_flag_raises(conn_with_offers, tmp_path):
    csv_path = tmp_path / "targets.csv"
    write_csv(csv_path, [{"company_name": "Acme", "domain": "acme.test"}], fieldnames=["company_name", "domain"])
    with pytest.raises(MissingOfferIdError):
        import_csv(conn_with_offers, csv_path=str(csv_path), cli_offer_slug=None, run_id="r1", step_id="s1")


def test_cli_offer_flag_fills_missing_column(conn_with_offers, tmp_path):
    csv_path = tmp_path / "targets.csv"
    write_csv(csv_path, [{"company_name": "Acme", "domain": "acme.test"}], fieldnames=["company_name", "domain"])
    target_ids = import_csv(
        conn_with_offers, csv_path=str(csv_path), cli_offer_slug="acme-offer",
        run_id="r1", step_id="s1",
    )
    assert len(target_ids) == 1
    target = conn_with_offers.execute("SELECT offer_id FROM targets;").fetchone()
    offer = conn_with_offers.execute(
        "SELECT slug FROM offers WHERE offer_id=?;", (target["offer_id"],)
    ).fetchone()
    assert offer["slug"] == "acme-offer"


def test_unknown_offer_slug_raises(conn_with_offers, tmp_path):
    csv_path = tmp_path / "targets.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "does-not-exist"}],
        fieldnames=["company_name", "domain", "offer_id"],
    )
    with pytest.raises(UnknownOfferError):
        import_csv(conn_with_offers, csv_path=str(csv_path), cli_offer_slug=None, run_id="r1", step_id="s1")


def test_offer_id_path_traversal_is_rejected_as_unknown(conn_with_offers, tmp_path):
    csv_path = tmp_path / "targets.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "../../etc/passwd"}],
        fieldnames=["company_name", "domain", "offer_id"],
    )
    with pytest.raises(UnknownOfferError):
        import_csv(conn_with_offers, csv_path=str(csv_path), cli_offer_slug=None, run_id="r1", step_id="s1")


def test_row_offer_id_conflicting_with_cli_flag_raises(conn_with_offers, tmp_path):
    csv_path = tmp_path / "targets.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer"}],
        fieldnames=["company_name", "domain", "offer_id"],
    )
    with pytest.raises(OfferConflictError):
        import_csv(
            conn_with_offers, csv_path=str(csv_path), cli_offer_slug="a-different-offer-slug",
            run_id="r1", step_id="s1",
        )


def test_missing_contact_email_does_not_block_import(conn_with_offers, tmp_path):
    csv_path = tmp_path / "targets.csv"
    write_csv(
        csv_path,
        [{"company_name": "Acme", "domain": "acme.test", "offer_id": "acme-offer", "contact_email": ""}],
        fieldnames=["company_name", "domain", "offer_id", "contact_email"],
    )
    target_ids = import_csv(
        conn_with_offers, csv_path=str(csv_path), cli_offer_slug=None, run_id="r1", step_id="s1",
    )
    assert len(target_ids) == 1
    target = conn_with_offers.execute("SELECT contact_id FROM targets;").fetchone()
    assert target["contact_id"] is None
