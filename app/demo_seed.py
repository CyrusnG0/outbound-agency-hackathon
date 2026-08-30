"""The demo seed (ticket D3a): make the full loop reachable with clearly
labelled placeholder data.

WHY THIS MODULE EXISTS — the DRY_RUN send (`app/send_cli.py`) is
unreachable on real data: `contacts.email_verified` is hardcoded 0 at CSV
import, no email-verification step exists, and nothing ever writes the
draft revision's `policy_check_passed` / `injection_scan_passed` columns
non-NULL, so the send gate's 19-item preflight refuses every real target
today.  The demo (deliverable D3) needs a judge to watch the REAL gates
execute, not a mocked pipeline — so this module seeds the preconditions
the real gate checks, and nothing downstream of them.

THE HONESTY CONTRACT (ticket §3.3) — every value this module writes that
stands in for a result no real step produced is (a) a PLACEHOLDER with an
in-line comment saying so at its write site, (b) written only onto RFC
2606 reserved, non-routable domains (`.test` / `.invalid` / `.example` —
enforced by RESERVED_TLDS and by tests), and (c) detectable afterwards: the
console shows an unmissable "DEMO DATA" banner whenever the database
contains a `steps` row with tool_name='demo_seed' (the marker this module
logs).  The gates themselves are NOT seeded and NOT weakened: the send
gate, the policy gate, the review gate and the state machine all run for
real against the seeded rows — they would refuse without them.

WHAT IS DELIBERATELY NOT HERE — no model call of any kind (the seed writes
its own placeholder text), no mail transport (nothing in this file or any
app/ module may import one — tests/test_send_gate.py walks this file like
every other), no raw core-table write (every mutation goes through
`write_gate.commit`, every state change through `state_machine.transition`,
verified by an AST test in tests/test_demo_seed.py), and no new
KNOWN_ACTIONS entry or agent principal (system / operator / draft_writer /
icp_judge are the existing registered principals and are used as-is).

SAFETY GUARDS (both subcommands) — refuse to run against the operator's
real database: exit non-zero if the resolved `--db` path equals
`data/outbound.db`, or if the OUTBOUND_DB_TARGET env var (the console's
repo-wide convention) resolves there.  The guard runs BEFORE the database
is opened, so the real file is never even connected to.

Subcommands:
  seed     — build the demo dataset: 3 targets walked new → researched →
             scored → drafted → awaiting_review → approved through real
             transitions, with placeholder contacts (reserved domains),
             placeholder gate columns on the draft revision, a real
             policy-gate allow row, and a real recorded operator approval.
  replies  — run AFTER send_cli: read data/outbox/*.eml, take each
             message's real Message-ID, and write data/inbox/demo_reply_*.eml
             whose In-Reply-To actually threads against the recorded
             messages rows (the committed data/inbox/01..05 fixtures carry
             placeholder In-Reply-To values that thread against nothing —
             that is why this subcommand exists).  One reply per outbound,
             cycling through the interested / unsubscribe / risky classes
             so the reply router's behaviour is visible.
"""

import argparse  # stdlib argument parsing — no new dependency for the operator
import json  # serializing the critique JSON stored on the seeded draft revision
import os  # reading OUTBOUND_DB_TARGET for the real-database guard
import sys  # stderr for refusal messages, argv for the default None sentinel
from datetime import datetime, timedelta, timezone  # now(UTC) fallback, the +1h reply-Date shift, and UTC normalization for naive dates
from email import policy as email_policy  # the stdlib parsing policy for reading outbox .eml files
from email.parser import BytesParser  # RFC-5322 parsing of outbox .eml bytes — PARSING ONLY, never transport
from email.utils import format_datetime, parsedate_to_datetime  # Date-header arithmetic for the generated replies
from pathlib import Path  # path resolution for the real-database guard and the outbox/inbox dirs

from app.agents_registry import seed_agent_registry  # registers the principals the write gate checks before any seed write
from app.db import apply_schema, connect  # opens the demo DB and applies the DDL (idempotent)
from app.ids import new_id  # unique prefixed ids: one per seeded row, one run id, one step id per stage
from app.policy import policy_check_phase1  # THE real policy gate — the seed's allow row is produced by it, not hand-written
from app.review import ReviewDecisionRequest, record_review_decision  # THE real review gate — the seeded approval is a real recorded decision
from app.schemas import CompanyProfile, ICPAssessment, Signal  # structured inputs for the real policy gate (CLAUDE.md §7)
from app.state_machine import transition  # THE single state-change path — the seeded target walks its hops through it
from app.tools.log_step import log_step  # steps-trace writer — every seeded stage lands in the trace (Golden Rule)
from app.write_gate import commit as write_gate_commit  # THE core-table write path — every seeded row goes through it

# ── Constants ─────────────────────────────────────────────────────────────────

# The RFC 2606 reserved, non-routable TLDs every seeded email address must
# use (ticket §3.1, a hard requirement enforced by tests).  `.test`,
# `.invalid` and `.example` can never resolve in DNS, so a seeded address
# can never collide with a real inbox — the same convention the committed
# data/inbox/*.eml fixtures already use.
RESERVED_TLDS = ("test", "invalid", "example")

# The targets.source value every seeded target carries.  It is (a) the
# seed's idempotency sentinel (a re-run finds it and skips), and (b) the
# recognisable field value the console could detect — in practice the
# console detects the steps marker below, but source stays honest in the
# audit UI: these rows did not come from a CSV.
DEMO_SOURCE = "demo_seed"

# The steps.tool_name every seed step row carries — distinct from every
# pipeline tool so the trace shows "the demo seed ran here" at a glance,
# and the console's DEMO DATA banner detects it with one SELECT.
DEMO_TOOL_NAME = "demo_seed"

# The offer slug the seeded targets attach to.  Reusing the real committed
# offer (config/offers/therapy-app.yaml) means the DRY_RUN send's From
# address resolves through the same config path the real pipeline uses —
# its from_address is itself a documented placeholder
# ("outreach@REPLACE-ME-BEFORE-SENDING.test"), which is exactly right for
# a demo.
DEMO_OFFER_SLUG = "therapy-app"

# The operator's real database — the path both subcommands refuse to touch.
# Deliberately the same relative spelling every CLI default uses; the
# guard compares RESOLVED absolute paths so it holds regardless of cwd.
REAL_DB_PATH = "data/outbound.db"

# The inbox filename prefix the replies subcommand writes.  Its own
# namespace: re-running deletes only files with this prefix (idempotency)
# and never touches the committed 01..05 example fixtures.
DEMO_REPLY_PREFIX = "demo_reply_"

# The three reply classes the replies subcommand cycles through — the
# minimum spread the ticket requires so the reply router's behaviour is
# visible: an interested reply (positive → follow-up draft), an
# unsubscribe (auto_suppress — the one auto-action that exists), and a
# risky one (freeze_target + human review required).
REPLY_CLASSES = ("interested", "unsubscribe", "risky")

# The seeded draft's compliance footer, verbatim B3's deterministic
# format.  The send gate's unsubscribe-token check looks for
# "[unsubscribe:" inside the footer — the token is the honest placeholder
# (no real unsubscribe URL exists anywhere in the repo, and none is
# invented here).
DEMO_FOOTER = "[unsubscribe: {UNSUBSCRIBE_URL}]"


# ── The real-database guard ───────────────────────────────────────────────────


class SeedAborted(Exception):
    """Raised when a REAL gate the seed drives refuses the seeded inputs
    (e.g. the policy gate denies, or the review gate refuses the approval).
    main() catches it, prints the gate's own reason, and returns exit code
    1 — a seed that silently continued past a live deny would be lying.
    Not a subclass of anything app-specific: it exists to turn a gate
    refusal into a clean CLI outcome, nothing more."""


def _guard_violation(db_target: str) -> str | None:
    """Return a refusal message when ``db_target`` (or the OUTBOUND_DB_TARGET
    env var) points at the operator's real database, else None.

    Runs BEFORE any connection is opened, so the real file is never even
    touched (no WAL pragma, no schema, no read).  Two checks, both against
    RESOLVED absolute paths so a relative spelling or a symlink cannot
    sneak past:
      1. the --db argument itself, and
      2. OUTBOUND_DB_TARGET — the console's repo-wide convention
         (docs/gcp-setup.md §6); if the operator's environment points at
         the real DB, the seed must refuse even when --db says otherwise,
         because a demo tool must have zero ways to reach production data.
    URL-shaped targets (postgresql:// / cloudsql://) cannot "resolve to
    data/outbound.db" and are passed through untouched.
    """
    # The real database's absolute identity, resolved once — the thing
    # both checks compare against.
    real_db = Path(REAL_DB_PATH).resolve()
    # Check 1: the --db argument.  resolve() works on nonexistent paths
    # too (strict=False by default), so a typo'd path cannot crash the
    # guard.
    if Path(db_target).resolve() == real_db:
        return (
            f"refusing to run the demo seed against {REAL_DB_PATH!r} — "
            f"that file holds the operator's real run data. "
            f"Use --db data/demo.db (or a scratch path) instead."
        )
    # Check 2: the environment convention.  Only file-shaped values are
    # compared — a cloud URL is not a path and resolves nowhere.
    env_target = os.environ.get("OUTBOUND_DB_TARGET")
    if env_target and not env_target.startswith(
        ("postgresql://", "postgres://", "cloudsql://")
    ):
        if Path(env_target).resolve() == real_db:
            return (
                f"refusing to run the demo seed: OUTBOUND_DB_TARGET points "
                f"at {REAL_DB_PATH!r} (the operator's real run data). "
                f"Unset it or point it at a demo database first."
            )
    # Neither check fired — the target is not the real database.
    return None


# ── Shared CLI plumbing ───────────────────────────────────────────────────────


def _open_demo_db(db_target: str):
    """Run the guard, open the database, apply the schema, and seed the
    agent registry — the same startup sequence every stage CLI uses, so
    the write gate accepts the seed's writes (it refuses unregistered
    agents).  The guard fires first: the real database is never opened.
    """
    # The guard runs before connect() — see _guard_violation's docstring
    # for why the order is load-bearing, not cosmetic.
    violation = _guard_violation(db_target)
    if violation is not None:
        print(f"ERROR: {violation}", file=sys.stderr)
        return None
    # Open the (safe) target and make sure every table exists — the demo
    # database is created by this module, exactly like the CLIs create
    # theirs.
    conn = connect(db_target)
    apply_schema(conn)
    # The registry seed is idempotent (upsert) and is what makes
    # agent_id="system" / "operator" / "draft_writer" / "icp_judge" valid
    # for the write gate in the writes below.
    seed_agent_registry(conn, run_id=new_id("run"), step_id=new_id("step"))
    return conn


# ── Subcommand: seed ──────────────────────────────────────────────────────────


def _write_account(conn, spec: dict, *, run_id: str, step_id: str) -> str:
    """Create one demo account row through the write gate (action
    insert_account — the same action get_targets uses for real imports)
    and return its account_id.

    ``spec`` is one entry of _DEMO_TARGETS below: a plain data block, so
    the write itself is one commented unit — the columns are the same
    NOT NULL set a CSV import writes (company_name, domain,
    normalized_domain); everything else is filled by the stage-shaped
    updates that follow, mirroring the real pipeline's order.
    """
    account_id = new_id("acc")  # "acc" prefix — the repo-wide account id vocabulary
    write_gate_commit(
        conn,
        action="insert_account",  # the existing import action — no new KNOWN_ACTIONS entry needed
        table_name="accounts",
        record_id=account_id,
        payload={"company_name": spec["company_name"], "domain": spec["domain"]},
        run_id=run_id,
        step_id=step_id,
        actor="system",  # deterministic seed code — the actor allowlist's system principal
        agent_id="system",  # attributed to the registered deterministic principal
        sql="""
            INSERT INTO accounts
                (account_id, company_name, domain, normalized_domain, created_at, updated_at)
            VALUES (?,?,?,?, datetime('now'), datetime('now'))
        """,
        params=(account_id, spec["company_name"], spec["domain"], spec["domain"]),
    )
    return account_id


def _write_research_profile(conn, spec: dict, account_id: str, *, run_id: str, step_id: str) -> None:
    """Persist the research-shaped profile columns (industry, estimated
    size, geo, one-line summary) via update_account_profile — the same
    action the summarize stage uses — so the console's company section
    reads like a researched account.

    EVERY TEXT VALUE HERE IS A SEEDED PLACEHOLDER: no research agent ran,
    no page was fetched, and the summary says so in its own words.  The
    send gate does not read these columns; they exist so the demo's audit
    views are populated, not to fake a verification result.
    """
    write_gate_commit(
        conn,
        action="update_account_profile",  # the summarize stage's own action — reused, not reinvented
        table_name="accounts",
        record_id=account_id,
        payload={"industry": spec["industry"], "estimated_size": spec["estimated_size"], "geo": spec["geo"]},
        run_id=run_id,
        step_id=step_id,
        actor="system",
        agent_id="system",
        sql="""
            UPDATE accounts SET industry=?, estimated_size=?, geo=?,
                company_summary=?, updated_at=datetime('now')
            WHERE account_id=?
        """,
        params=(
            spec["industry"],
            spec["estimated_size"],
            spec["geo"],
            # PLACEHOLDER (demo seed): this summary was written by this
            # script, not by the research agent — no model call happened.
            f"[DEMO SEED PLACEHOLDER] {spec['company_name']} is a Hong Kong "
            f"therapy practice seeded for the demo. No research ran to "
            f"produce this summary.",
            account_id,
        ),
    )


def _write_icp_score(conn, spec: dict, account_id: str, target_id: str, *, run_id: str, step_id: str) -> None:
    """Persist the deterministic ICP score and the judge's verdict — the
    two account column groups the send gate reads (icp_fit_label /
    icp_fit_score) and the console renders (judge_*).

    The VALUES are seeded placeholders (fit_score >= the P4 floor of 60,
    label strong_fit — chosen so the REAL gates pass, not because a
    formula computed them), but the WRITES use the pipeline's own actions
    and attribution: update_account_score as system (the formula's
    principal) and update_account_icp_verdict as icp_judge (the judge's
    principal), so write_log distinguishes them exactly as it would on a
    real run.
    """
    # The deterministic score write — mirrors score_lead's shape.
    write_gate_commit(
        conn,
        action="update_account_score",  # score_lead's own action
        table_name="accounts",
        record_id=account_id,
        payload={"icp_fit_label": spec["fit_label"], "icp_fit_score": spec["fit_score"]},
        run_id=run_id,
        step_id=step_id,
        actor="system",
        agent_id="system",
        sql="""
            UPDATE accounts SET icp_fit_label=?, icp_fit_score=?,
                icp_fit_reasons=?, icp_non_fit_reasons=?, updated_at=datetime('now')
            WHERE account_id=?
        """,
        params=(
            # PLACEHOLDER (demo seed): strong_fit and a >=60 score are
            # seeded so the REAL send-gate fit checks pass; no scoring
            # formula ran.
            spec["fit_label"],
            spec["fit_score"],
            json.dumps(["seeded demo score — no scoring formula ran"]),
            json.dumps([]),
            account_id,
        ),
    )
    # The judge's verdict write — mirrors judge_icp's shape and
    # attribution (the judge principal owns the verdict columns).
    write_gate_commit(
        conn,
        action="update_account_icp_verdict",  # judge_icp's own action (ticket B2c)
        table_name="accounts",
        record_id=account_id,
        payload={"judge_fit_label": spec["fit_label"]},
        run_id=run_id,
        step_id=step_id,
        actor="system",  # deterministic code executes the write…
        agent_id="icp_judge",  # …but the verdict is the judge's — the B2c attribution pattern
        sql="""
            UPDATE accounts SET judge_fit_label=?, judge_rationale=?,
                judge_divergence_justification=?, updated_at=datetime('now')
            WHERE account_id=?
        """,
        params=(
            # PLACEHOLDER (demo seed): the judge agrees with the seeded
            # deterministic label; no judge model ran.
            spec["fit_label"],
            "[DEMO SEED PLACEHOLDER] No ICP judge ran — this verdict is seeded demo data.",
            None,  # no divergence: the seeded verdict equals the seeded deterministic label
            account_id,
        ),
    )
    # The target's own score column — score_lead's second write, mirrored
    # so the console's target table shows the same number.
    write_gate_commit(
        conn,
        action="update_target_score",  # score_lead's own action
        table_name="targets",
        record_id=target_id,
        payload={"score": spec["fit_score"]},
        run_id=run_id,
        step_id=step_id,
        actor="system",
        agent_id="system",
        sql="UPDATE targets SET score=? WHERE target_id=?;",
        params=(spec["fit_score"], target_id),
    )


def _write_contact(conn, spec: dict, account_id: str, *, run_id: str, step_id: str) -> str:
    """Create the demo contact row and return its contact_id.

    THE PLACEHOLDER THAT MATTERS MOST (ticket §3.1): ``email`` is a
    reserved-domain address and ``email_verified=1`` is SEEDED — no
    verification service ran, none exists in this repo.  The send gate's
    "contact.email_verified == true" check is GENUINE and would refuse
    this row if the flag were 0 (which is what real imports write); the
    seed sets it to 1 so the demo can show the gate PASSING, and the
    console banner plus this comment say plainly that no verification
    happened.  Setting it to 1 is a fixture practice (the send-gate tests
    do the same), not a weakening of the gate.
    """
    contact_id = new_id("con")  # "con" prefix — the repo-wide contact id vocabulary
    write_gate_commit(
        conn,
        action="insert_contact",  # get_targets' own import action
        table_name="contacts",
        record_id=contact_id,
        payload={"full_name": spec["contact_name"], "email": spec["contact_email"]},
        run_id=run_id,
        step_id=step_id,
        actor="system",
        agent_id="system",
        sql="""
            INSERT INTO contacts
                (contact_id, account_id, full_name, title, email,
                 email_verified, created_at, updated_at)
            VALUES (?,?,?,?,?,?, datetime('now'), datetime('now'))
        """,
        params=(
            contact_id,
            account_id,
            spec["contact_name"],
            spec["contact_title"],
            # PLACEHOLDER (demo seed): reserved non-routable domain — this
            # address can never receive mail; it exists so the real gate
            # has an address to check.
            spec["contact_email"],
            # PLACEHOLDER (demo seed): email_verified=1 is SEEDED — no
            # verification ran.  The gate's check is genuine and refuses
            # real imports (which hardcode 0) until a verification path
            # exists; the demo seeds the pass.
            1,
        ),
    )
    return contact_id


def _write_target(conn, account_id: str, contact_id: str, offer_id: str, *, run_id: str, step_id: str) -> str:
    """Create the demo target row (state 'new', source 'demo_seed') and
    return its target_id.  The state walk below moves it through every
    hop via state_machine.transition — the row starts where every real
    target starts.
    """
    target_id = new_id("tgt")  # "tgt" prefix — the repo-wide target id vocabulary
    write_gate_commit(
        conn,
        action="insert_target",  # get_targets' own import action
        table_name="targets",
        record_id=target_id,
        payload={"offer_id": offer_id, "source": DEMO_SOURCE},
        run_id=run_id,
        step_id=step_id,
        actor="system",
        agent_id="system",
        sql="""
            INSERT INTO targets
                (target_id, account_id, contact_id, offer_id, source, state,
                 created_at, updated_at)
            VALUES (?,?,?,?,?,?, datetime('now'), datetime('now'))
        """,
        params=(
            target_id,
            account_id,
            contact_id,
            offer_id,
            DEMO_SOURCE,  # the idempotency sentinel AND the honest provenance label
            "new",  # the state machine's initial state — the walk below leaves it
        ),
    )
    return target_id


def _write_signals(conn, spec: dict, target_id: str, *, run_id: str, step_id: str) -> list[Signal]:
    """Write the target's two seeded signals through the write gate and
    return the Signal models for the real policy gate.

    The send gate requires at least one signal with strength >= 0.6 on the
    latest research run — the first of each pair is seeded above that
    floor so the REAL check passes.  Every value is a placeholder (the
    evidence_quote says so in its own words, evidence_tier='unverified'
    is B2b's honest "appears in no fetched text" verdict), and the write
    uses detect_signals' own action and system attribution.
    """
    models: list[Signal] = []  # the structured inputs the policy gate will consume
    for sig in spec["signals"]:
        # One signal row per entry of the data block — the UNIQUE
        # (target_id, run_id, signal_type, signal_value) key keeps a
        # re-seed from duplicating even without the sentinel.
        signal_id = new_id("sig")  # "sig" prefix — the repo-wide signal id vocabulary
        write_gate_commit(
            conn,
            action="insert_signal",  # detect_signals' own action
            table_name="signals",
            record_id=signal_id,
            payload={"signal_type": sig["type"], "signal_strength": sig["strength"]},
            run_id=run_id,
            step_id=step_id,
            actor="system",
            agent_id="system",  # detect_signals attributes to system — mirrored
            sql="""
                INSERT INTO signals
                    (signal_id, run_id, target_id, signal_type, signal_value,
                     signal_strength, source_url, source_confidence,
                     evidence_quote, evidence_verified, evidence_tier, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
            """,
            params=(
                signal_id,
                run_id,
                target_id,
                sig["type"],
                sig["value"],
                sig["strength"],
                None,  # source_url: no source was fetched — NULL is the honest "no URL exists"
                None,  # source_confidence: same — no measurement happened
                # PLACEHOLDER (demo seed): the quote is invented text and
                # says so — no research ran, nothing was fetched.
                "[DEMO SEED PLACEHOLDER] No research ran; this evidence quote is invented demo data.",
                0,  # evidence_verified=0: honestly unverified — B2b's invariant (verified iff tier='source')
                "unverified",  # evidence_tier='unverified': B2b's vocabulary for "appears in no fetched text"
            ),
        )
        # Build the structured model for the policy gate with the same
        # values the row just received — the two can never disagree.
        models.append(
            Signal(
                signal_type=sig["type"],
                signal_value=sig["value"],
                signal_strength=sig["strength"],
                evidence_quote="[DEMO SEED PLACEHOLDER] No research ran; this evidence quote is invented demo data.",
            )
        )
    return models


def _write_policy_allow(conn, spec: dict, target_id: str, signals: list[Signal], *, run_id: str, step_id: str) -> None:
    """Produce the policy_decisions allow row by RUNNING THE REAL POLICY
    GATE (policy_check_phase1) — not by hand-writing a row.  The send
    gate requires the target's LATEST policy decision to be "allow"; here
    the real P6/P3a/P4 checks execute against the seeded profile/score/
    signals and record their own verdict, so the demo shows the policy
    gate working, not a faked result.  If the gate denies (engaged kill
    switch, or a seeded value below a floor), the seed aborts loudly —
    a demo that seeds around a live deny would be lying.
    """
    # The real gate call.  Its inputs are the seeded placeholders; its
    # logic, its P6 kill-switch read, and its persisted row are genuine.
    decision = policy_check_phase1(
        conn,
        company_profile=CompanyProfile(
            # PLACEHOLDER (demo seed): see _write_research_profile.
            one_line_summary=f"[DEMO SEED PLACEHOLDER] {spec['company_name']} — seeded demo summary, no research ran.",
            industry=spec["industry"],
            estimated_size=spec["estimated_size"],
            geo=spec["geo"],
            confidence=0.9,  # placeholder confidence — the gate does not read it in Phase 1 scope
        ),
        icp_assessment=ICPAssessment(
            fit_label=spec["fit_label"],
            fit_score=spec["fit_score"],
            fit_reasons=["seeded demo score — no scoring formula ran"],
            non_fit_reasons=[],
        ),
        signals=signals,
        target_id=target_id,
        run_id=run_id,
        step_id=step_id,
    )
    # The seed's contract with the demo: a deny here (P6 kill switch, or
    # a floor) means the preconditions are NOT set — abort rather than
    # continue and let the send gate refuse confusingly later.
    if decision.decision != "allow":
        raise SeedAborted(
            f"demo seed aborted: the real policy gate denied target "
            f"{target_id!r} ({decision.decision}) — "
            f"reasons: {decision.reasons}. The demo needs the kill switch "
            f"disengaged (config/kill_switch.json) and the seeded score "
            f"above the P4 floor."
        )


def _write_draft_revision(conn, spec: dict, target_id: str, *, run_id: str, step_id: str) -> None:
    """Write the seeded draft revision through the write gate (B3's own
    action, attributed to draft_writer as B3 attributes it).

    THE TWO GATE COLUMNS ARE THE TICKET'S OTHER PLACEHOLDERS:
    policy_check_passed=1 and injection_scan_passed=1 are SEEDED — no
    draft-content policy runner and no prompt-injection scanner exist in
    this repo, and on real data both columns stay NULL, which the send
    gate correctly treats as "no check has run → refuse".  The seed sets
    them to 1 (fixture practice, exactly like the send-gate tests) so the
    REAL checks pass; the checks themselves are untouched and would
    refuse NULL.  send_gate_passed stays NULL: that column is the send
    gate's own (B3-Z3 — the drafter never sets its own gates) and the
    demo's whole point is watching the send gate fill it for real.
    """
    # The draft's placeholder text — written by this script, no model
    # call.  The body says it is placeholder text in its own words so the
    # .eml artifact can never be mistaken for an LLM draft.
    body = (
        f"[DEMO SEED PLACEHOLDER DRAFT] Hi {spec['contact_name'].split()[-1]},\n\n"
        f"This is seeded demo text written by app/demo_seed.py — no draft "
        f"agent produced it. In a real run this would be the writer⇄critic "
        f"output approved by the operator.\n\n"
        f"Best regards"
    )
    draft_version_id = new_id("dv")  # "dv" prefix — B3's revision id vocabulary
    write_gate_commit(
        conn,
        action="insert_message_draft_version",  # B3's own action — draft writes are audited distinctly
        table_name="message_draft_versions",
        record_id=draft_version_id,
        payload={
            "revision_number": 1,
            "edited_by": "draft_writer",
            "gate_columns_seeded": True,  # the audit payload says it plainly: the gate columns below are seeded
        },
        run_id=run_id,
        step_id=step_id,
        actor="system",  # deterministic seed code executes the write…
        agent_id="draft_writer",  # …but the revision is the writer's — B3's attribution, mirrored
        sql="""
            INSERT INTO message_draft_versions
                (draft_version_id, target_id, message_id, revision_number,
                 subject, body, footer, edited_by, policy_check_passed,
                 injection_scan_passed, send_gate_passed, critique_passed,
                 critique_json, insert_seq, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,
                    (SELECT COALESCE(MAX(insert_seq),0)+1 FROM message_draft_versions),
                    datetime('now'))
        """,
        params=(
            draft_version_id,
            target_id,
            None,  # message_id: no messages row exists until send_cli sends — NULL is the honest "not a message yet"
            1,  # revision_number 1: the first and only revision — the send gate's "approved revision is latest" check passes by construction
            spec["subject"],
            body,
            DEMO_FOOTER,  # the deterministic compliance footer with the unsubscribe token the send gate checks
            "draft_writer",  # edited_by: agent-authored revision, B3's vocabulary
            # PLACEHOLDER (demo seed): policy_check_passed=1 is SEEDED — no
            # draft-content policy runner exists; NULL on real data, which
            # the send gate refuses (fail closed).  The check itself is
            # genuine and runs live at send time.
            1,
            # PLACEHOLDER (demo seed): injection_scan_passed=1 is SEEDED —
            # the Guardrails AI scanner (open-questions.md item 8) is not
            # implemented; NULL on real data, refused by the gate.  The
            # gate's check is genuine and runs live at send time.
            1,
            None,  # send_gate_passed: the send gate's own column (B3-Z3) — left NULL; send_cli fills the verdict for real
            # The critic's verdict on this revision — seeded as a passing
            # critique so the console's draft-diff view reads like a
            # completed writer⇄critic loop (placeholder: no critic ran).
            1,
            json.dumps({"passed": True, "issues": [], "required_changes": "", "severity": "none"}),
        ),
    )


def _record_operator_approval(conn, target_id: str, *, run_id: str) -> None:
    """Record the seeded operator approval by RUNNING THE REAL REVIEW GATE
    (record_review_decision, decision='approve') — the same door the
    console's approval button uses.  The review_decisions row and the
    awaiting_review → approved transition are both written by the gate
    itself, attributed to the operator principal, so the send gate's
    "a recorded human approval exists AND took effect" check passes on a
    genuine row, not a hand-written one.  A refusal (engaged kill switch,
    wrong state) aborts the seed loudly — the demo must not paper over a
    live deny.
    """
    outcome = record_review_decision(
        conn,
        request=ReviewDecisionRequest(
            target_id=target_id,
            decision="approve",  # the plain approval — the seeded draft is approved as written
            # The operator's recorded reasoning, honestly worded: the
            # operator approves SEEDED placeholder text for demo purposes.
            reason="demo seed: operator approves the seeded placeholder draft (no model produced this text)",
        ),
        run_id=run_id,
    )
    # A refusal is the real gate saying no — surface its reason and stop,
    # never continue with a target the gate refused.
    if outcome.refused:
        raise SeedAborted(
            f"demo seed aborted: the real review gate refused the seeded "
            f"approval for target {target_id!r}: {outcome.refusal_reason}"
        )


def _seed_one_target(conn, spec: dict, offer_id: str, *, run_id: str) -> None:
    """Seed one demo target end to end: account, research profile, score +
    judge verdict, contact, target row, the four-transition state walk,
    signals, the real policy-gate allow row, the draft revision, and the
    real recorded approval.  One step id per stage-group so the target's
    audit entries hang together like a real run's.
    """
    # Stage-shaped step ids: the seed mirrors the pipeline's
    # one-step-per-stage shape so the trace reads like a real run.
    step_research = new_id("step")
    step_score = new_id("step")
    step_draft = new_id("step")
    step_review = new_id("step")

    # ── Stage 1 (research): account + profile + contact + target + walk ──
    account_id = _write_account(conn, spec, run_id=run_id, step_id=step_research)
    _write_research_profile(conn, spec, account_id, run_id=run_id, step_id=step_research)
    contact_id = _write_contact(conn, spec, account_id, run_id=run_id, step_id=step_research)
    target_id = _write_target(conn, account_id, contact_id, offer_id, run_id=run_id, step_id=step_research)
    # new → researched: the first hop happens as part of the research stage.
    transition(
        conn, target_id=target_id, from_state="new", to_state="researched",
        reason="research_complete_no_enrichment", actor="system",
        run_id=run_id, step_id=step_research,
    )
    log_step(
        conn, run_id=run_id, step_id=step_research, target_id=target_id,
        tool_name=DEMO_TOOL_NAME, agent_id="system",
        input_data={"stage": "demo_seed_research", "simulated": True},
        output_data={"account_id": account_id, "contact_id": contact_id, "target_id": target_id},
        status="success",
    )

    # ── Stage 2 (score + judge): the two account verdict groups + the
    # researched → scored hop.
    _write_icp_score(conn, spec, account_id, target_id, run_id=run_id, step_id=step_score)
    transition(
        conn, target_id=target_id, from_state="researched", to_state="scored",
        reason="scoring_complete", actor="system",
        run_id=run_id, step_id=step_score,
    )
    log_step(
        conn, run_id=run_id, step_id=step_score, target_id=target_id,
        tool_name=DEMO_TOOL_NAME, agent_id="system",
        input_data={"stage": "demo_seed_score", "simulated": True},
        output_data={"fit_label": spec["fit_label"], "fit_score": spec["fit_score"]},
        status="success",
    )

    # ── Stage 3 (draft): signals + real policy gate + revision + the two
    # draft hops (scored → drafted → awaiting_review).
    signal_models = _write_signals(conn, spec, target_id, run_id=run_id, step_id=step_draft)
    _write_policy_allow(conn, spec, target_id, signal_models, run_id=run_id, step_id=step_draft)
    _write_draft_revision(conn, spec, target_id, run_id=run_id, step_id=step_draft)
    # The two draft hops, attributed to draft_writer exactly as
    # app/agents/draft.py attributes them.
    transition(
        conn, target_id=target_id, from_state="scored", to_state="drafted",
        reason="policy_allows_draft", actor="system", agent_id="draft_writer",
        run_id=run_id, step_id=step_draft,
    )
    transition(
        conn, target_id=target_id, from_state="drafted", to_state="awaiting_review",
        reason="draft_complete", actor="system", agent_id="draft_writer",
        run_id=run_id, step_id=step_draft,
    )
    log_step(
        conn, run_id=run_id, step_id=step_draft, target_id=target_id,
        tool_name=DEMO_TOOL_NAME, agent_id="system",
        input_data={"stage": "demo_seed_draft", "simulated": True},
        output_data={"revision_number": 1, "gate_columns_seeded": True},
        status="success",
    )

    # ── Stage 4 (review): the REAL review gate records the operator's
    # approval and moves the target to approved.
    _record_operator_approval(conn, target_id, run_id=run_id)
    log_step(
        conn, run_id=run_id, step_id=step_review, target_id=target_id,
        tool_name=DEMO_TOOL_NAME, agent_id="operator",  # the approval step is the operator's
        input_data={"stage": "demo_seed_review", "simulated": True},
        output_data={"new_state": "approved", "decision": "approve"},
        status="success",
    )


# ── The demo data block ───────────────────────────────────────────────────────
# One entry per demo target.  THIS IS A GENERATED DATA BLOCK: one comment
# per logical unit (per target), not per field — the field names are
# self-explanatory.  Every company/contact is fictional, every domain is
# an RFC 2606 reserved non-routable TLD (.test), and every score/label is
# chosen so the REAL gates pass — none of it was computed by the pipeline.
_DEMO_TARGETS = (
    # Target 1 — the interested-reply demo target.  Its draft subject
    # mirrors the committed data/inbox fixtures' "Re:" lines so the
    # simulated conversation reads coherently.
    {
        "company_name": "Serenity Clinic",
        "domain": "serenity-clinic.test",
        "industry": "Healthcare",
        "estimated_size": "11-50",
        "geo": "Hong Kong",
        "fit_label": "strong_fit",
        "fit_score": 72,
        "contact_name": "Dr. Chan Mei-Ling",
        "contact_title": "Clinical Director",
        "contact_email": "dr.chan@serenity-clinic.test",
        "subject": "A question about your intake admin workload",
        "signals": (
            {"type": "hiring_relevant_role", "value": "Hiring 2 front-desk coordinators", "strength": 0.8},
            {"type": "workflow_complexity_evidence", "value": "Manual booking across two locations", "strength": 0.55},
        ),
    },
    # Target 2 — the unsubscribe-reply demo target.
    {
        "company_name": "Clearwater Physiotherapy",
        "domain": "clearwater-clinic.test",
        "industry": "Healthcare",
        "estimated_size": "11-50",
        "geo": "Hong Kong",
        "fit_label": "strong_fit",
        "fit_score": 81,
        "contact_name": "Chris Lee",
        "contact_title": "Operations Manager",
        "contact_email": "chris.lee@clearwater-clinic.test",
        "subject": "Your intake scheduling stack",
        "signals": (
            {"type": "hiring_relevant_role", "value": "Hiring a clinic administrator", "strength": 0.75},
            {"type": "recent_launch_or_expansion", "value": "Opened a second branch in Kowloon", "strength": 0.6},
        ),
    },
    # Target 3 — the risky-reply demo target.
    {
        "company_name": "Harbour View Wellness Centre",
        "domain": "harbour-view-wellness.test",
        "industry": "Healthcare",
        "estimated_size": "51-200",
        "geo": "Hong Kong",
        "fit_label": "strong_fit",
        "fit_score": 65,
        "contact_name": "Emily Wong",
        "contact_title": "Practice Manager",
        "contact_email": "emily.wong@harbour-view-wellness.test",
        "subject": "A question about your patient intake process",
        "signals": (
            {"type": "product_or_ops_change", "value": "Switching to a new EMR system", "strength": 0.7},
            {"type": "workflow_complexity_evidence", "value": "Paper intake forms still in use", "strength": 0.5},
        ),
    },
)


def _ensure_offer(conn, *, run_id: str, step_id: str) -> str:
    """Return the offer_id for DEMO_OFFER_SLUG, inserting the offer row
    through the write gate when the demo database does not have one yet.

    The slug is the real committed therapy-app offer — its YAML lives in
    config/offers/, so send_cli's From-address resolution works against
    the same config the real pipeline reads.
    """
    # Reuse an existing row (idempotent across re-seeds and shared with
    # any other data in the demo database).
    existing = conn.execute(
        "SELECT offer_id FROM offers WHERE slug=?;", (DEMO_OFFER_SLUG,)
    ).fetchone()
    if existing is not None:
        return existing["offer_id"]
    # No row yet — insert it through the gate, the same way the operator's
    # offer would be recorded.
    offer_id = new_id("off")  # "off" prefix — the repo-wide offer id vocabulary
    write_gate_commit(
        conn,
        action="insert_offer",  # the existing offer action
        table_name="offers",
        record_id=offer_id,
        payload={"slug": DEMO_OFFER_SLUG},
        run_id=run_id,
        step_id=step_id,
        actor="system",
        agent_id="system",
        sql="INSERT INTO offers (offer_id, slug, active, created_at) VALUES (?,?,?,datetime('now'))",
        params=(offer_id, DEMO_OFFER_SLUG, 1),
    )
    return offer_id


def seed_demo_data(conn, *, run_id: str) -> int:
    """Build the demo dataset.  Returns the number of targets seeded (0
    when the dataset already exists — the idempotency sentinel).

    THE SENTINEL (idempotency, ticket §3.1): if any target already
    carries source='demo_seed', a previous seed ran — skip everything and
    report, so re-running never duplicates rows and never crashes.  (A
    half-finished seed from a killed process is recovered the blunt way:
    delete the demo database and re-run — documented in runbook.md.)
    """
    # The sentinel read — one SELECT, the whole idempotency mechanism.
    already = conn.execute(
        "SELECT 1 FROM targets WHERE source=? LIMIT 1;", (DEMO_SOURCE,)
    ).fetchone()
    if already is not None:
        # A previous seed is present — doing nothing IS the correct
        # behaviour on re-run (row counts must stay stable).
        print(
            "demo data already present (targets.source='demo_seed' found) — "
            "skipping; delete the demo database and re-run to rebuild."
        )
        return 0
    # One shared step id for the offer write; each target then gets its
    # own stage-shaped steps (see _seed_one_target).
    step_id = new_id("step")
    offer_id = _ensure_offer(conn, run_id=run_id, step_id=step_id)
    seeded = 0
    for spec in _DEMO_TARGETS:
        # Per-target isolation: one entry's failure aborts loudly rather
        # than leaving a half-seeded target silently in place.
        _seed_one_target(conn, spec, offer_id, run_id=run_id)
        seeded += 1
    # The batch summary step — one trace row naming what the seed wrote,
    # and the marker row the console's DEMO DATA banner detects.
    log_step(
        conn, run_id=run_id, step_id=step_id, target_id=None,
        tool_name=DEMO_TOOL_NAME, agent_id="system",
        input_data={"stage": "demo_seed", "simulated": True},
        output_data={"targets_seeded": seeded, "source": DEMO_SOURCE},
        status="success",
    )
    return seeded


# ── Subcommand: replies ───────────────────────────────────────────────────────


def _reserved_domain_of(email: str) -> str | None:
    """Return the lowercased domain of ``email`` when it is on an RFC 2606
    reserved TLD, else None.  The replies subcommand refuses to fabricate
    a sender on any other domain — a generated reply From a real-looking
    address would be a fake real person, which this repo never invents.
    """
    # The domain half is what the reserved-TLD rule applies to; the local
    # part is irrelevant to routability.
    domain = email.split("@", 1)[-1].lower() if "@" in email else ""
    # The TLD is the final label — compare against the reserved set.
    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    return domain if tld in RESERVED_TLDS else None


# The three placeholder reply bodies, one per class — GENERATED DATA
# BLOCK: one comment per logical unit.  Written by this script (no model
# call); each is worded so the real reply classifier has an unambiguous
# class to find, mirroring the tone of the committed data/inbox fixtures.
_REPLY_BODIES = {
    # interested — warm, asks for more detail: the classifier's
    # "positive" shape, which routes to queue_follow_up_draft.
    "interested": (
        "Hi,\n\n"
        "Thanks for reaching out — this is actually quite relevant to us "
        "right now. Could you send over a bit more detail on how the "
        "automation would work, and what the setup looks like for a "
        "practice of our size?\n\n"
        "[This reply is seeded demo data generated by app/demo_seed.py — "
        "no real person wrote it.]"
    ),
    # unsubscribe — a clear removal demand: the classifier's
    # "unsubscribe" shape, the ONE class whose auto-action
    # (auto_suppress) is executed in v1 — the demo's visible suppression.
    "unsubscribe": (
        "Please stop contacting me.\n\n"
        "Remove this address from your mailing list and do not send any "
        "further messages to anyone at this practice. We are not "
        "interested and will not be in the future.\n\n"
        "[This reply is seeded demo data generated by app/demo_seed.py.]"
    ),
    # risky — legal-toned demands: the classifier's "risky" shape, which
    # routes to freeze_target + mandatory human review (P5) — the demo's
    # visible review-bound path.
    "risky": (
        "To whom it may concern,\n\n"
        "I represent the practice you contacted. Your unsolicited email "
        "raises concerns under data-protection law and we require that "
        "you cease all further contact immediately and confirm in writing "
        "that you have deleted any data you hold about the practice.\n\n"
        "We reserve all rights.\n\n"
        "[This reply is seeded demo data generated by app/demo_seed.py.]"
    ),
}


def _compose_reply_bytes(*, from_name: str, from_email: str, to_address: str, subject: str, date_str: str, in_reply_to: str, message_id_token: str, body: str) -> bytes:
    """Compose the reply as RFC-5322 bytes (CRLF line endings), using the
    STDLIB email package for FORMATTING ONLY — the bytes go to a file,
    never to a transport (there is no transport anywhere in this repo).
    All inputs are deterministic (derived from the outbox message and the
    fixed templates), so re-running produces byte-identical files.
    """
    # The header block — In-Reply-To carries the outbound message's REAL
    # Message-ID token verbatim, which is exactly what
    # fetch_inbox._match_by_headers matches on (the messages-row id is
    # embedded inside the token by send_email's make_msgid call).
    header = (
        f"From: {from_name} <{from_email}>\r\n"
        f"To: {to_address}\r\n"
        f"Subject: Re: {subject}\r\n"
        # The reply's Date is the outbound Date + 1 hour (a reply arrives
        # after the send) — computed by the caller, deterministic rather
        # than wall-clock, so re-runs are identical.
        f"Date: {date_str}\r\n"
        # A deterministic Message-ID: it embeds the outbound messages-row
        # id so the reply's own identity is traceable back to the send.
        f"Message-ID: <{message_id_token}>\r\n"
        # The outbound token arrives WITH its angle brackets (msg
        # ["Message-ID"] includes them) — write it verbatim, never
        # re-wrap, or the header would carry a doubled <<...>> pair.
        f"In-Reply-To: {in_reply_to}\r\n"
        "\r\n"
        f"{body}\r\n"
    )
    return header.encode("utf-8")


def _generate_replies(conn, *, outbox_dir: str, inbox_dir: str) -> tuple[list[str], list[str]]:
    """Generate one threaded reply per outbox message, cycling through the
    interested / unsubscribe / risky classes, and return (generated,
    skipped) filename lists.

    For each data/outbox/*.eml: parse the real Message-ID, find the
    recorded outbound messages row whose id is embedded in the token (the
    same substring rule fetch_inbox uses to thread inbound mail), read
    the contact's seeded email as the reply's sender, REFUSE anything not
    on a reserved domain, and write data/inbox/demo_reply_*.eml with an
    In-Reply-To that threads for real.  Idempotent: the demo_reply_*
    namespace is cleared and rewritten on every run, and the committed
    01..05 example fixtures are never touched.
    """
    # The outbox may not exist yet (run before send_cli) — that is "no
    # sends yet", not an error.
    outbox = Path(outbox_dir)
    if not outbox.is_dir():
        print(f"no outbox directory at {outbox_dir!r} — run send_cli first.")
        return [], []
    # Sorted filenames make the class↔message mapping deterministic
    # run-to-run (the same sweep order fetch_inbox uses).
    files = sorted(outbox.glob("*.eml"))
    if not files:
        print(f"no .eml files in {outbox_dir!r} — run send_cli first.")
        return [], []

    # The recorded outbound rows, read once — the threading source of
    # truth (a reply must answer a RECORDED send, never a stray file).
    message_rows = conn.execute(
        "SELECT message_id, contact_id FROM messages WHERE direction='outbound';"
    ).fetchall()
    # Index the rows by id so the per-file token match is a dict lookup.
    by_id = {row["message_id"]: row["contact_id"] for row in message_rows}

    # Idempotency: clear the previous run's generated files (and ONLY
    # those — the demo_reply_ prefix is this module's namespace).
    inbox = Path(inbox_dir)
    inbox.mkdir(parents=True, exist_ok=True)
    for stale in inbox.glob(f"{DEMO_REPLY_PREFIX}*.eml"):
        stale.unlink()  # removing only files this module itself wrote

    generated: list[str] = []
    skipped: list[str] = []
    parser = BytesParser(policy=email_policy.default)  # stdlib parser — one instance reused for the batch
    for index, path in enumerate(files):
        # Per-file isolation: one unparseable file is reported and
        # skipped, never aborts the batch (the B1f rule re-applied).
        try:
            msg = parser.parsebytes(path.read_bytes())
        except Exception as exc:
            skipped.append(f"{path.name}: unparseable ({type(exc).__name__})")
            continue
        # The outbound message's real Message-ID token — the value the
        # reply's In-Reply-To must echo verbatim.
        outbound_token = msg["Message-ID"]
        if not outbound_token:
            skipped.append(f"{path.name}: no Message-ID header")
            continue
        # The messages-row id embedded in the token — the substring rule
        # fetch_inbox._match_by_headers uses (send_email's make_msgid
        # embeds the row id inside the angle-bracket token).
        matched_id = next(
            (mid for mid in by_id if mid in outbound_token), None
        )
        if matched_id is None:
            # The file does not correspond to a recorded send — never
            # fabricate a reply for it.
            skipped.append(f"{path.name}: no recorded outbound messages row matches its Message-ID")
            continue
        # The seeded contact the outbound was sent to — the reply's
        # sender (a reply comes FROM the person who received the send).
        contact_row = conn.execute(
            "SELECT full_name, email FROM contacts WHERE contact_id=?;",
            (by_id[matched_id],),
        ).fetchone()
        if contact_row is None or not contact_row["email"]:
            skipped.append(f"{path.name}: contact row missing or has no email")
            continue
        sender_email = contact_row["email"]
        # THE RESERVED-DOMAIN RULE: refuse to write a reply From any
        # address that could look real — a fabricated sender on a real
        # domain would be a fake real person.
        sender_domain = _reserved_domain_of(sender_email)
        if sender_domain is None:
            skipped.append(
                f"{path.name}: sender {sender_email!r} is not on a reserved domain "
                f"({', '.join(RESERVED_TLDS)}) — refusing to fabricate a reply from it"
            )
            continue
        # The class for this message: cycle through the three-class
        # spread by sorted-file index, so a 3-target demo shows all three
        # router behaviours.
        reply_class = REPLY_CLASSES[index % len(REPLY_CLASSES)]
        # The reply's Date: the outbound Date + 1 hour, parsed from the
        # artifact.  A naive parsed date (send_email writes localtime
        # dates with no zone) is assumed UTC — the same convention
        # fetch_inbox's _parse_reply_date uses — so the +1h arithmetic
        # and the written header are unambiguous.  An unparseable Date
        # degrades to now, documented at the check site below.
        try:
            outbound_dt = parsedate_to_datetime(msg["Date"])
            if outbound_dt.tzinfo is None:
                # Naive date — pin it to UTC before the arithmetic, or
                # the +1h shift would depend on the operator's timezone.
                outbound_dt = outbound_dt.replace(tzinfo=timezone.utc)
            reply_dt = outbound_dt + timedelta(hours=1)  # a reply arrives after the send
            date_str = format_datetime(reply_dt)  # RFC-5322 Date header text
        except (TypeError, ValueError):
            # Absent/malformed outbound Date — degrade to now rather than
            # skip an otherwise threadable reply (fetch_inbox's convention).
            date_str = format_datetime(datetime.now(timezone.utc))
        # Compose the artifact: headers + the class's placeholder body.
        raw = _compose_reply_bytes(
            from_name=contact_row["full_name"] or sender_email,
            from_email=sender_email,
            to_address=msg["From"] or "dry-run@outbound-agency.invalid",
            subject=msg["Subject"] or "(no subject)",
            date_str=date_str,
            in_reply_to=outbound_token,
            message_id_token=f"{matched_id}.demo-reply@{sender_domain}",
            body=_REPLY_BODIES[reply_class],
        )
        out_path = inbox / f"{DEMO_REPLY_PREFIX}{index + 1:02d}_{reply_class}.eml"
        out_path.write_bytes(raw)  # the file write IS the simulated inbound mail
        generated.append(out_path.name)
    return generated, skipped


# ── The CLI ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Dispatch the two subcommands.  Returns the process exit code (0 on
    success, 1 on a refusal or crash) — the same shape every stage CLI
    uses.
    """
    # The subcommand name is the first positional; the flags match the
    # stage CLIs' vocabulary (--db everywhere, --outbox/--inbox where
    # relevant).
    parser = argparse.ArgumentParser(prog="python -m app.demo_seed")
    sub = parser.add_subparsers(dest="subcommand", required=True)
    seed_p = sub.add_parser("seed", help="build the demo dataset (3 targets walked to approved)")
    seed_p.add_argument("--db", default="data/demo.db")  # the demo database — never data/outbound.db (guarded)
    replies_p = sub.add_parser("replies", help="generate threaded demo replies from data/outbox/*.eml")
    replies_p.add_argument("--db", default="data/demo.db")  # read-only here, but guarded identically
    replies_p.add_argument("--outbox", default="data/outbox")
    replies_p.add_argument("--inbox", default="data/inbox")
    args = parser.parse_args(argv)

    # ── The real-database guard runs FIRST, before any connection ───────
    # Both subcommands share it: the demo tool must have zero ways to
    # reach the operator's real run data.
    violation = _guard_violation(args.db)
    if violation is not None:
        print(f"ERROR: {violation}", file=sys.stderr)
        return 1

    if args.subcommand == "replies":
        # The replies subcommand only READS the database (messages →
        # contacts, both SELECTs) and writes .eml files to the inbox.
        # Refuse a missing sqlite file rather than letting connect()
        # silently create an empty one — replies answers recorded sends,
        # and an empty database has none.
        if not args.db.startswith(("postgresql://", "postgres://", "cloudsql://")) and not Path(args.db).exists():
            print(
                f"ERROR: demo database {args.db!r} does not exist — run "
                f"`python -m app.demo_seed seed --db {args.db}` and "
                f"`python -m app.send_cli --db {args.db}` first.",
                file=sys.stderr,
            )
            return 1
        conn = connect(args.db)  # read-only usage: no apply_schema, no registry seed
        generated, skipped = _generate_replies(
            conn, outbox_dir=args.outbox, inbox_dir=args.inbox
        )
        conn.close()
        # The summary — generated files, then every refusal reason, so
        # nothing is silently dropped.
        print(
            f"demo replies: {len(generated)} generated into {args.inbox}/ "
            f"({len(skipped)} skipped)."
        )
        for name in generated:
            print(f"  {name}")
        for reason in skipped:
            print(f"  skipped: {reason}")
        return 0

    # ── The seed subcommand ─────────────────────────────────────────────
    conn = _open_demo_db(args.db)
    if conn is None:
        # The guard refused (or the open failed) — the message was
        # already printed to stderr.
        return 1
    run_id = new_id("run")  # one run id groups every write of this seed invocation
    try:
        seeded = seed_demo_data(conn, run_id=run_id)
    except SeedAborted as exc:
        # A REAL gate refused the seeded inputs (kill switch engaged, a
        # floor missed) — report its reason and exit non-zero.  The
        # finally below still closes the connection.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()  # explicit close — the same hygiene every CLI keeps
    if seeded == 0:
        print("demo seed: nothing to do (already seeded).")
    else:
        print(
            f"demo seed complete: {seeded} targets seeded to 'approved' "
            f"(placeholder contacts and gate results — no verification or "
            f"scan ran). Next: python -m app.send_cli --db {args.db}"
        )
    return 0


# Guard so `python app/demo_seed.py` also works, not just `python -m
# app.demo_seed`.  Uses SystemExit instead of sys.exit() to stay testable
# (pytest can catch SystemExit) — the same pattern as every stage CLI.
if __name__ == "__main__":
    raise SystemExit(main())
