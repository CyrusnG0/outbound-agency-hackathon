"""
Tests for app.config — offer config loading and offers-table sync.

These tests verify that:
1. load_offer_configs() reads every .yaml file in a directory and returns
   a {slug: content} dict keyed by filename-minus-extension.
2. sync_offers_table() is idempotent — running it twice against the same YAML
   directory does not create duplicate rows in the offers table.

Both functions are critical for the "no hidden side effects" golden rule:
the operator edits YAML files on disk, and the system syncs them into SQLite
on each pipeline run. Idempotency means the same YAML file on a second run
doesn't corrupt the table with a duplicate row.
"""

from app.agents_registry import seed_agent_registry
from app.db import connect, apply_schema
from app.config import load_offer_configs, sync_offers_table


def test_load_offer_configs_reads_yaml_by_slug(tmp_path):
    # Arrange: Create a temp offers directory with one .yaml file named "acme.yaml".
    # The slug should be "acme" (filename minus ".yaml" extension).
    offers_dir = tmp_path / "offers"
    offers_dir.mkdir()
    (offers_dir / "acme.yaml").write_text(
        "pitch: test pitch\npersona_hint: test persona\ntemplate: t1\nfrom_address: a@b.test\n"
    )

    # Act: call load_offer_configs with the temp directory path.
    configs = load_offer_configs(str(offers_dir))

    # Assert: the returned dict has exactly one key, "acme", and its value
    # is the parsed YAML content with all four required fields.
    assert configs == {
        "acme": {
            "pitch": "test pitch",
            "persona_hint": "test persona",
            "template": "t1",
            "from_address": "a@b.test",
        }
    }


def test_sync_offers_table_is_idempotent(tmp_path):
    # Arrange: Create a temp offers directory with one YAML file and a fresh
    # SQLite database with the full schema applied.
    offers_dir = tmp_path / "offers"
    offers_dir.mkdir()
    (offers_dir / "acme.yaml").write_text(
        "pitch: p\npersona_hint: h\ntemplate: t\nfrom_address: a@b.test\n"
    )
    db_path = str(tmp_path / "test.db")
    conn = connect(db_path)
    apply_schema(conn)
    # Register the system agent (plan A3) — sync_offers_table writes through
    # the gate, which refuses unregistered agents.
    seed_agent_registry(conn, run_id="run_0", step_id="step_0")

    # Act: sync twice with different run/step IDs — the second call should see
    # that "acme" already exists and skip the INSERT.
    sync_offers_table(conn, str(offers_dir), run_id="run_1", step_id="step_1")
    sync_offers_table(conn, str(offers_dir), run_id="run_2", step_id="step_2")

    # Assert: the offers table has exactly one row with slug "acme" — no duplicate.
    rows = conn.execute("SELECT slug FROM offers;").fetchall()
    assert [r["slug"] for r in rows] == ["acme"]
    conn.close()
