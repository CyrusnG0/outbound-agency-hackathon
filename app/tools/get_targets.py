"""
Target import — the entry point of the outbound-agency pipeline.

This module's sole public function, ``import_csv()``, reads a CSV of prospect
companies and writes each row into the database as:
  - an ``accounts`` row (the company/domain),
  - an optional ``contacts`` row (a named person at that company, if the CSV
    provides contact_name or contact_email),
  - a ``targets`` row (the unit of work that moves through the state machine,
    starting in state ``"new"`` per app/state_machine.py).

Ticket G1 adds an operator-asserted ``email_verified`` CSV column: a truthy
value means the OPERATOR vouches that ``contact_email`` is deliverable, and
the contact row's ``email_verified`` is then set to 1 — but only if the
address also passes ``app.db.email_syntax_valid`` (syntactic validation is a
hard gate over the assertion; a malformed address stores 0 even when the CSV
asserts otherwise). Absent/blank/falsy values keep the historical default of
0, so a CSV without the column imports exactly as before.

Every INSERT goes through ``app.write_gate.commit`` (Task 4) so each row gets
its ``write_log`` audit entry. The whole import is logged once via
``app.tools.log_step.log_step`` (Task 6), not once per row — the import
operation itself is the step.

Per-row offer resolution enforces three distinct failure modes:
  1. No row ``offer_id`` AND no ``--offer`` CLI flag → MissingOfferIdError
  2. Row ``offer_id`` present but not found in ``offers`` table → UnknownOfferError
  3. Row ``offer_id`` AND CLI flag present AND they disagree → OfferConflictError

Domain normalization strips URL scheme, ``www.`` prefix, trailing slash, and
lowercases so ``https://www.acme.test/`` and ``acme.test`` both collapse to the
same ``normalized_domain`` value for account lookups elsewhere in the pipeline.
"""

import csv
from urllib.parse import urlparse

from app.db import email_syntax_valid  # G1: THE shared syntactic validator (lives beside normalize_email — one address parser, no drift)
from app.ids import new_id
from app.tools.log_step import log_step
from app.write_gate import commit as write_gate_commit


# ── Exception types ──────────────────────────────────────────────────────────
# Each of these maps to one specific offer-resolution failure branch in the
# per-row offer_id logic inside import_csv(). They are separate types (not one
# generic error) so the caller can programmatically distinguish "you forgot to
# provide any offer" from "the offer you gave doesn't exist" from "the CSV and
# the CLI flag disagree" — important for a CLI that might print different help
# text for each case.


class MissingOfferIdError(Exception):
    """Raised when a CSV row has no offer_id column value AND no --offer CLI
    flag was passed — the row provides nothing to resolve an offer from, so the
    import cannot proceed for this row."""


class UnknownOfferError(Exception):
    """Raised when the resolved offer_slug (from row or CLI) does not match any
    slug in the ``offers`` table. This also safely rejects path-traversal-
    lookalike strings (e.g. ``"../../etc/passwd"``) because they are treated as
    a slug lookup against a parameterized SQL query — not as a filesystem path.

    The slug is only ever used as a parameterized value (``WHERE slug = ?``),
    never interpolated into a file path or command, so a traversal string is
    naturally rejected the same way any other unknown slug is."""


class OfferConflictError(Exception):
    """Raised when the CSV row has an ``offer_id`` value AND the CLI passed
    ``--offer`` AND those two values disagree. A silent mismatch between the
    file's own declared offer and the operator's CLI intent must be surfaced —
    it is never silently resolved either way.  When both are present AND they
    agree, the row value wins (which is the value the CLI also confirmed)."""


class EmailVerifiedValueError(Exception):
    """Raised when the ``email_verified`` CSV column holds a value that is
    neither a recognised truthy nor a recognised falsy token.

    This is a VISIBLE import error, not a silent fallback to unverified: a
    typo like ``ture`` must stop the import at the row where it appears,
    rather than becoming an unverified contact that the send gate quietly
    refuses three stages later (ticket G1 §2.1)."""


# ── email_verified vocabulary (ticket G1 §2.1) ───────────────────────────────
# The operator is the verifier; these are the exact tokens a CSV may use to
# ASSERT that a contact_email is deliverable.  Matching is case-insensitive
# and whitespace-trimmed.  Absent column, empty string, or any falsy token
# → 0 (fail closed); any truthy token → asserted; anything else raises
# EmailVerifiedValueError.  Kept as two sets so adding a token is one edit
# and the vocabulary stays grep-able in docs and tests.
_EMAIL_VERIFIED_TRUTHY = {"1", "true", "yes", "y", "verified", "on"}
_EMAIL_VERIFIED_FALSY = {"0", "false", "no", "n", "unverified", "off"}


def _parse_email_verified_flag(raw: str | None) -> bool:
    """Map one CSV ``email_verified`` cell to a boolean assertion.

    Absent column (None) and blank values are falsy — the backward-compatible
    path for CSVs that predate the column, and the fail-closed default.  A
    truthy token returns True; a falsy token returns False; any other
    non-empty value raises EmailVerifiedValueError so a typo is a visible
    import failure, never a silent "unverified".
    """
    value = (raw or "").strip().lower()  # Tolerate spreadsheet padding/case; empty or missing is the falsy default.
    if not value:
        return False  # No assertion — never default to verified.
    if value in _EMAIL_VERIFIED_TRUTHY:
        return True  # The operator asserts this address is deliverable.
    if value in _EMAIL_VERIFIED_FALSY:
        return False  # Explicitly not asserted — same stored flag as absent.
    raise EmailVerifiedValueError(  # A value outside both vocabularies is a data error, surfaced immediately.
        f"unrecognised email_verified value {raw!r} — expected one of "
        f"truthy {sorted(_EMAIL_VERIFIED_TRUTHY)} or falsy "
        f"{sorted(_EMAIL_VERIFIED_FALSY)}"
    )


# ── Domain normalization ─────────────────────────────────────────────────────


def _normalize_domain(raw: str) -> str:
    """Collapse domain strings into a canonical form for account matching.

    Why each transformation exists (and why they're in this order):
      1. Strip leading/trailing whitespace — CSV columns often carry stray
         spaces from copy-paste or spreadsheet export.
      2. Strip URL scheme (https://, http://) via urlparse — the operator may
         paste a full URL from their browser's address bar. We keep only the
         netloc (domain + optional port). If urlparse can't find a netloc
         (malformed input), we fall back to the original stripped value so the
         rest of the pipeline still gets something to work with.
      3. Strip leading "www." — many domains are entered as www.example.com
         but the DNS apex example.com is the canonical form for dedup.
      4. Strip trailing slash — URLs often end with "/" but domains never do;
         this catches the case where urlparse left a trailing path separator.
      5. Lowercase — domains are case-insensitive per DNS, so "ACME.test" and
         "acme.test" must be treated as the same domain.
    """
    # Step 1: strip leading/trailing whitespace from copy-paste CSV values.
    raw = raw.strip()

    # Step 2: if the input contains "://", it's a URL — extract the netloc.
    # If urlparse can't parse a netloc (e.g. "://bad" → netloc is empty string),
    # fall back to raw so the remaining steps still get a value to work with.
    if "://" in raw:
        raw = urlparse(raw).netloc or raw

    # Step 3: strip leading "www." so www.acme.test and acme.test collapse to
    # the same canonical form for account dedup.
    if raw.startswith("www."):
        raw = raw[len("www.") :]

    # Steps 4-5: strip trailing slash (if any), then lowercase.
    return raw.rstrip("/").lower()


# ── Main import function ─────────────────────────────────────────────────────


def import_csv(
    conn,
    *,
    csv_path: str,
    cli_offer_slug: str | None,
    run_id: str,
    step_id: str,
) -> list[str]:
    """Load targets from a CSV file into accounts, contacts, and targets tables.

    Per docs/tool-registry.md's get_targets CSV column schema, the expected
    columns are: company_name, domain, offer_id (optional when --offer CLI flag
    is provided), contact_name, contact_email, contact_title, and (G1) an
    operator-asserted email_verified flag.

    Every row writes 1-3 rows through the write gate and returns the list of
    created target_ids. The entire import is logged as one step via log_step.

    Args:
        conn: app.db.Conn from app.db.connect() (sqlite or postgres).
        csv_path: Filesystem path to the CSV file to import.
        cli_offer_slug: The --offer CLI flag value, or None if not passed.
            When provided, this fills in a missing offer_id column for rows
            that don't specify one. When a row has its own offer_id AND this
            flag is set, the two values must agree — a mismatch raises
            OfferConflictError.
        run_id: Pipeline run identifier, groups all writes and the step log
            together under one execution.
        step_id: Specific step identifier within the run for this import.

    Returns:
        List of created target_id strings, one per successfully-imported row.
    """
    # ── Load known offer slugs once, up front ────────────────────────────────
    # Why load all slugs before iterating rows? Because the same SELECT would
    # otherwise run once per CSV row. In a 10,000-row import, that's 10,000
    # redundant queries. One SELECT at the start is O(1) regardless of CSV
    # size. The set is built from the slug column of the offers table — these
    # are the only offer identifiers the system recognizes.
    known_offer_slugs = {
        row["slug"] for row in conn.execute("SELECT slug FROM offers;").fetchall()
    }

    # Accumulates the target_id of every successfully-imported row. This list
    # is returned to the caller AND stored in the log_step output for the
    # audit trail, so a future operator can answer "what targets came from
    # this CSV?" by reading the steps table.
    created_target_ids: list[str] = []

    # Accumulates the contact_id of every contact whose email_verified flag
    # the operator ASSERTED in this CSV (G1). Stored in the same log_step
    # output so the provenance of each verified flag is recoverable from the
    # import step alone, without joining write_log (which also carries the
    # per-contact source in its payload).
    operator_verified_contact_ids: list[str] = []

    # Open the CSV file and iterate. newline="" is required by the csv module:
    # without it, quoted fields containing newlines would be mangled on some
    # platforms. DictReader uses the header row as field names, so rows are
    # accessible by column name rather than positional index.
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # ── Extract and normalize company/domain fields ─────────────────
            # company_name is required by the accounts table's NOT NULL
            # constraint. If a CSV has an empty company_name, the write gate
            # will reject it at INSERT time with a constraint error — the
            # error surfaces clearly rather than silently importing garbage.
            company_name = row["company_name"].strip()

            # Normalize the domain so https://www.acme.test/ and acme.test
            # both become "acme.test" — they represent the same company.
            domain = _normalize_domain(row["domain"])

            # ── Offer resolution ───────────────────────────────────────────
            # Three branches, exactly one of which executes per row:
            #
            # Branch A: row has an offer_id value. If the CLI also passed
            # --offer and the values disagree, raise OfferConflictError — a
            # silent mismatch between the file's intent and the operator's CLI
            # intent must not be silently resolved. If they agree (or CLI
            # wasn't passed), the row value wins. Row value always takes
            # precedence over CLI flag when both are present and agree.
            #
            # Branch B: row has no offer_id, but CLI passed --offer. Use the
            # CLI flag value. This is the common case for CSVs that don't
            # declare an offer column — the operator says "import this CSV for
            # the acme-offer campaign."
            #
            # Branch C: row has no offer_id AND no CLI flag. There's nothing
            # to resolve — raise MissingOfferIdError. The operator must either
            # add an offer_id column to the CSV or pass --offer.
            row_offer_id = (row.get("offer_id") or "").strip()
            if row_offer_id:
                # Branch A: row provides an offer_id value.
                if cli_offer_slug and row_offer_id != cli_offer_slug:
                    # CLI flag disagrees with the row — surface the conflict.
                    # Both values are shown in the error message so the
                    # operator can decide which one is correct.
                    raise OfferConflictError(
                        f"row offer_id '{row_offer_id}' conflicts with --offer '{cli_offer_slug}'"
                    )
                offer_slug = row_offer_id
            elif cli_offer_slug:
                # Branch B: no row value, CLI flag provides the offer.
                offer_slug = cli_offer_slug
            else:
                # Branch C: neither row nor CLI provides an offer — fail.
                raise MissingOfferIdError(
                    "row has no offer_id column value and no --offer flag was passed"
                )

            # ── Validate the resolved slug against known offers ───────────
            # This is the single check that enforces "only known offers can be
            # used." Because offer_slug is only ever used as a parameterized
            # SQL value (WHERE slug = ?), this is also what safely rejects a
            # path-traversal-lookalike string like "../../etc/passwd" — it's
            # just an unrecognized slug, never a filesystem path. The next
            # query after this check uses the offer_id (a UUID-style string),
            # not the slug, so the slug is never concatenated into SQL either.
            if offer_slug not in known_offer_slugs:
                raise UnknownOfferError(
                    f"offer_id '{offer_slug}' does not match any known offer "
                    f"(known: {sorted(known_offer_slugs)})"
                )

            # ── Resolve offer_slug to offer_id ─────────────────────────────
            # The offers table maps slug → offer_id (the primary key). We need
            # the offer_id for the targets.offers_id foreign key. The slug's
            # presence was already validated above, so this SELECT is
            # guaranteed to return a row.
            offer_row = conn.execute(
                "SELECT offer_id FROM offers WHERE slug = ?;", (offer_slug,)
            ).fetchone()
            offer_id = offer_row["offer_id"]

            # ── Write the accounts row ─────────────────────────────────────
            # Every CSV row gets its own new account_id. Phase 1 does NOT
            # attempt de-duplication against existing accounts by normalized
            # domain — that's a known simplification. A future enrichment or
            # dedup task may collapse duplicate domains, but for now each
            # import row is an independent account. The "acc" prefix makes
            # account ids self-describing in logs and join output.
            account_id = new_id("acc")
            # Write through the gate — never a raw conn.execute — so this
            # INSERT produces a write_log audit row and is subject to the
            # known-action and valid-actor gate checks. actor="system" because
            # this is deterministic pipeline code, not a human operator
            # clicking a button.
            write_gate_commit(
                conn,
                action="insert_account",
                table_name="accounts",
                record_id=account_id,
                payload={"company_name": company_name, "domain": domain},
                run_id=run_id,
                step_id=step_id,
                actor="system",
                agent_id="system",  # Deterministic import code — the registered system agent.
                sql="""
                    INSERT INTO accounts
                        (account_id, company_name, domain, normalized_domain, created_at, updated_at)
                    VALUES (?,?,?,?, datetime('now'), datetime('now'))
                """,
                params=(account_id, company_name, domain, domain),
            )

            # ── Write the optional contacts row ────────────────────────────
            # A contact row is only created when the CSV row actually provides
            # something to store (contact_email or contact_name). A completely
            # blank contact is valid — enrichment may discover a contact later
            # and link it to the target. target.contact_id remains NULL in
            # that case, which is allowed by the targets table's foreign key
            # (REFERENCES contacts(contact_id) with no NOT NULL constraint).
            contact_id = None
            contact_email = (row.get("contact_email") or "").strip()
            contact_name = (row.get("contact_name") or "").strip()

            # ── Operator-asserted verification (G1) ───────────────────────
            # Parse the CSV's email_verified column FIRST. Absent/blank/falsy
            # → not asserted; truthy → the operator vouches the address; any
            # unrecognised value raises immediately (a visible import error,
            # never a silent fallback to unverified three stages later).
            email_verified_asserted = _parse_email_verified_flag(
                row.get("email_verified")
            )

            # The stored flag always starts fail-closed at 0. It only rises
            # to 1 when BOTH the operator asserted it AND the address passes
            # syntactic validation — the operator can vouch for a real
            # address, never for a malformed string (ticket G1 §2.2).
            email_verified = 0
            email_verified_source = "unverified_default"
            if email_verified_asserted:
                if not contact_email:
                    # Asserting verification on a row with no address is a
                    # contradiction — surface it rather than store a verified
                    # NULL email.
                    raise EmailVerifiedValueError(
                        "row asserts email_verified but has no contact_email — "
                        "there is no address to verify"
                    )
                syntax_reason = email_syntax_valid(contact_email)
                if syntax_reason is None:
                    # Honoured: syntactically sound AND operator-asserted.
                    email_verified = 1
                    email_verified_source = "operator_asserted_csv"
                else:
                    # Syntax fails — the hard gate OVERRIDES the assertion,
                    # so a malformed address can never reach the send gate as
                    # verified regardless of what the operator typed.
                    email_verified = 0
                    email_verified_source = "syntax_invalid"

            if contact_email or contact_name:
                # At least one contact field is non-empty — create a contact
                # row. The "con" prefix makes contact ids self-describing in
                # logs and join output.
                contact_id = new_id("con")
                # Write through the gate so the contact INSERT is audited.
                write_gate_commit(
                    conn,
                    action="insert_contact",
                    table_name="contacts",
                    record_id=contact_id,
                    # The payload now carries the verification verdict AND its
                    # provenance (G1 §2.3): the write_log payload_json records
                    # why email_verified is 1/0 — operator_asserted_csv,
                    # syntax_invalid, or unverified_default — so the origin of
                    # the flag is recoverable from the audit trail alone.
                    payload={
                        "full_name": contact_name,
                        "email": contact_email,
                        "email_verified": email_verified,
                        "email_verified_source": email_verified_source,
                    },
                    run_id=run_id,
                    step_id=step_id,
                    actor="system",
                    agent_id="system",  # Same deterministic principal as the accounts write above.
                    sql="""
                        INSERT INTO contacts
                            (contact_id, account_id, full_name, title, email,
                             email_verified, created_at, updated_at)
                        VALUES (?,?,?,?,?,?, datetime('now'), datetime('now'))
                    """,
                    params=(
                        contact_id,
                        account_id,
                        contact_name or None,
                        (row.get("contact_title") or "").strip() or None,
                        contact_email or None,
                        email_verified,  # 0 by default; 1 only when operator-asserted AND syntactically valid (G1)
                    ),
                )

                # Record the contact in the run-level provenance list so the
                # import step (steps.output_json) can answer "which contacts
                # did this CSV's operator assert as verified?" without joining
                # write_log. The per-contact source lives in the payload above.
                if email_verified:
                    operator_verified_contact_ids.append(contact_id)

            # ── Write the targets row ──────────────────────────────────────
            # The target is the unit of work that moves through the state
            # machine. Every target starts in state="new" — this is the
            # initial state defined in app/state_machine.py's VALID_TRANSITIONS
            # table; from "new", the only valid transition is to "researched".
            # The "tgt" prefix makes target ids self-describing in logs.
            # source="csv_import" records how this target entered the system.
            target_id = new_id("tgt")
            write_gate_commit(
                conn,
                action="insert_target",
                table_name="targets",
                record_id=target_id,
                payload={"offer_id": offer_id, "source": "csv_import"},
                run_id=run_id,
                step_id=step_id,
                actor="system",
                agent_id="system",  # Same deterministic principal as the accounts/contacts writes above.
                sql="""
                    INSERT INTO targets
                        (target_id, account_id, contact_id, offer_id, source, state,
                         created_at, updated_at)
                    VALUES (?,?,?,?,?,?, datetime('now'), datetime('now'))
                """,
                params=(
                    target_id,
                    account_id,
                    contact_id,       # May be None — valid when no contact info in CSV.
                    offer_id,
                    "csv_import",
                    "new",            # Initial state per app/state_machine.py.
                ),
            )

            # Accumulate the created target_id so it's returned to the caller
            # and included in the log_step output. This allows the operator to
            # inspect which specific targets were created by this import run.
            created_target_ids.append(target_id)

    # ── Log the import as one step ──────────────────────────────────────────
    # Log once for the whole CSV import — not once per row. This step
    # represents the import operation itself. target_id=None because this
    # step is not scoped to a single target; it's the batch import. A future
    # operator can read this steps row to answer "what CSV was imported, with
    # which CLI flags, and what target IDs were created?"
    log_step(
        conn,
        run_id=run_id,
        step_id=step_id,
        target_id=None,  # Batch operation — not scoped to a single target.
        tool_name="get_targets",
        agent_id="system",  # Deterministic import code — the registered system agent.
        input_data={"csv_path": csv_path, "cli_offer_slug": cli_offer_slug},
        output_data={
            "target_ids": created_target_ids,
            # G1 provenance: the run-level list of contacts whose verified
            # flag this import's operator asserted (per-contact source is in
            # each insert_contact write_log payload).
            "operator_verified_contact_ids": operator_verified_contact_ids,
        },
        status="success",
    )

    # Return the list of created target_ids so the caller (CLI, test, or
    # pipeline orchestrator) can immediately act on the imported targets —
    # e.g. by queueing them for research in the next pipeline stage.
    return created_target_ids
