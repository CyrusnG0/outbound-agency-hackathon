"""
Offer config loading and offers-table sync.

This module is the bridge between on-disk YAML offer definitions and the
offers table. The operator defines offers by editing .yaml files in
config/offers/ — one file per offer, with the filename (minus extension)
serving as the unique slug. On each pipeline run, sync_offers_table() reads
every .yaml file and inserts any offers it hasn't seen before into the offers
table, using the write gate for every INSERT so each new row gets an audit
entry in write_log.

Idempotency is the key design constraint: running sync twice against the same
YAML directory must not produce duplicate rows. The idempotency mechanism is a
SELECT of all existing slugs BEFORE any INSERT — slugs found in that set are
skipped. This is checked by test_sync_offers_table_is_idempotent.
"""

import os

import yaml

from app.ids import new_id
from app.write_gate import commit as write_gate_commit


def load_offer_configs(offers_dir: str) -> dict[str, dict]:
    """Read every config/offers/*.yaml file into {slug: content}.

    Each .yaml file is parsed with PyYAML's safe_load (never yaml.load — config
    files come from the operator's disk and must not be allowed to construct
    arbitrary Python objects). The filename stem becomes the offer slug, which
    is the unique key used to detect already-synced offers in sync_offers_table.

    Files are sorted so discovery order is deterministic across OSes and
    filesystems — otherwise os.listdir could return files in any order, making
    sync behavior non-reproducible between runs.
    """
    configs: dict[str, dict] = {}
    # Walk every entry in the offers directory, sorted for deterministic ordering.
    # Sorting ensures the same set of YAML files always produces the same dict
    # iteration order on every OS and filesystem — important for reproducible
    # sync behavior and predictable test assertions.
    for filename in sorted(os.listdir(offers_dir)):
        # Only process .yaml files. Anything else (README.md, .DS_Store, editor
        # backup files) is silently skipped so the operator can keep notes or
        # other artifacts alongside offer configs without breaking the loader.
        if not filename.endswith(".yaml"):
            continue
        # Derive the slug from the filename by stripping the ".yaml" suffix.
        # This enforces a 1:1 mapping between filenames and slugs — no slug
        # field inside the YAML itself — so the operator can rename an offer
        # by renaming the file and the system picks it up on the next sync.
        # Using slicing rather than .removesuffix() for clarity: we know the
        # filename ends with ".yaml" because of the check above.
        slug = filename[: -len(".yaml")]
        # Open and parse the file. yaml.safe_load is used over yaml.load
        # because the latter can construct arbitrary Python objects from YAML
        # tags — config files are operator-authored and "untrusted-ish" in the
        # sense that a mistake or malicious YAML tag should never be able to
        # execute code or instantiate unexpected objects.
        with open(os.path.join(offers_dir, filename)) as f:
            configs[slug] = yaml.safe_load(f)
    return configs


def sync_offers_table(conn, offers_dir: str, run_id: str, step_id: str) -> None:
    """Populate the offers table from discovered YAML files. Idempotent.

    This is called once per pipeline run (or at startup) to ensure the offers
    table reflects whatever YAML files are on disk. Only new offers (slugs not
    already in the table) are inserted — existing rows are never updated or
    deleted. This is by design: the operator adds a new offer by creating a
    new .yaml file; they remove an offer by setting active=0 manually (future
    admin tooling), not by deleting the file.

    Every INSERT goes through the write gate (not a raw conn.execute) so each
    new offer row gets a corresponding audit record in write_log. The action
    "insert_offer" must be registered in write_gate.KNOWN_ACTIONS (it is, as
    of this writing).
    """
    # Load all offer configs from the YAML directory. This returns {slug: {...}}
    # for every .yaml file found.
    configs = load_offer_configs(offers_dir)

    # Build a set of slugs already present in the offers table. This SELECT
    # runs fresh on every call — it's what makes the function idempotent. If
    # we cached this set across calls, a second sync wouldn't see the rows
    # inserted by the first call and would attempt duplicate INSERTs (which
    # would fail on the UNIQUE constraint on slug).
    existing = {
        row["slug"] for row in conn.execute("SELECT slug FROM offers;").fetchall()
    }

    # For each discovered offer, check whether its slug already exists in the
    # database. Slugs found in the existing set are skipped — no UPDATE, no
    # re-INSERT. This is the idempotency mechanism that
    # test_sync_offers_table_is_idempotent verifies: running sync twice inserts
    # on the first call and skips on the second.
    for slug in configs:
        if slug in existing:
            continue  # Already synced — skip to avoid duplicate INSERT.
        # Generate a unique primary key for the new offers row. The "off"
        # prefix makes offer ids self-describing in write_log rows.
        offer_id = new_id("off")
        # Write through the gate, not a raw conn.execute. This is the "never
        # skip logging" golden rule: every core-table mutation must produce a
        # write_log row so the audit trail is complete. The gate also enforces
        # that "insert_offer" is a known action type, "system" is a valid
        # actor, and agent "system" is registered with this action allowed —
        # if any check fails, WriteGateRefused is raised before any SQL executes.
        write_gate_commit(
            conn,
            action="insert_offer",         # Must be in write_gate.KNOWN_ACTIONS.
            table_name="offers",           # The target table for the INSERT.
            record_id=offer_id,            # The new row's primary key.
            payload={"slug": slug},        # Audit payload: what was written.
            run_id=run_id,                 # Groups this write with other steps in this run.
            step_id=step_id,               # Identifies the specific step within the run.
            actor="system",                # Deterministic pipeline code — allowed actor.
            agent_id="system",             # Which registered agent writes: the deterministic principal.
            sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
            params=(offer_id, slug, 1),    # active=1: new offers default to active.
        )
