"""The draft gate runner (ticket G2) — the deterministic code that closes
the two remaining send-gate blockers by actually WRITING the two gate columns
the drafting agent may not write itself.

WHY THIS MODULE EXISTS — ``app/send_gate.py`` is a pure reader: it refuses a
revision whose ``policy_check_passed`` / ``injection_scan_passed`` columns are
not both 1 (NULL means "no check has run", which is not "passed" — fail
closed).  But until G2 NOTHING wrote those two columns on a real revision:
the drafting agent writes all three gate columns NULL on purpose (B3-Z3, so
an LLM can never set its own gates), and ``send_gate.py`` deliberately does
not evaluate them.  The result was the last two structural gaps that made a
real run produce zero sends.  This module is the missing evaluator: it runs
two deterministic checks over a persisted revision and writes exactly the two
columns it owns — never ``send_gate_passed``, which stays the send gate's own.

THE GOVERNANCE SPLIT (same shape as every other stage): the drafting agent
PRODUCES TEXT; this deterministic runner JUDGES that text and performs the
gated write.  An LLM never sets its own pass/fail.

FAIL CLOSED, ALWAYS — a runner that crashes, or a revision it never evaluated,
leaves the columns NULL, and NULL still refuses at the send gate.  The only
values ever written are 0 (a named, machine-readable failure) or 1 (a full
pass); the error path writes nothing and logs a failed step, so a refusal can
never masquerade as a pass and a crash can never silently clear the gate.
"""

import re  # the detection rules are regexes: template tokens, role markers, banned phrases

from pydantic import BaseModel  # DraftGateResult — the structured verdict the callers and tests consume (CLAUDE.md §7)

from app.ids import new_id  # one fresh step id per evaluation, matching the repo's per-step trace pattern
from app.send_gate import UNSUBSCRIBE_TOKEN  # THE one footer token constant — imported, never redefined, so the runner and the gate cannot drift apart
from app.tools.log_step import log_step  # steps-trace writer — every evaluation, pass, fail, and crash lands in the trace
from app.write_gate import commit as write_gate_commit  # THE core-table write path — the column UPDATE is written through it, never a raw UPDATE

# ── Machine-readable reason strings ─────────────────────────────────────────
# Each failure names itself with one of these stable identifiers, the way
# SendGateDecision.missing_requirements names unmet checklist items.  Tests
# import and assert these exact strings; a boolean with no reasons would be
# unauditable (ticket §2.1).
REASON_SUBJECT_LENGTH = "content_policy:subject_length_out_of_bounds"
REASON_BODY_LENGTH = "content_policy:body_too_short"
REASON_FOOTER_UNSUBSCRIBE = "content_policy:footer_missing_unsubscribe"
REASON_UNRESOLVED_TOKEN = "content_policy:unresolved_template_token"
REASON_BANNED_PHRASE = "content_policy:banned_phrase"

REASON_INSTRUCTION_OVERRIDE = "injection:instruction_override"
REASON_ROLE_MARKER = "injection:role_marker"
REASON_AGENT_DIRECTIVE = "injection:agent_directive"
REASON_SEND_DIRECTIVE = "injection:embedded_send_directive"

# ── Content-policy bounds (state them; the runner is the gate, so it re-checks
# rather than assuming EmailDraft's schema validation already ran) ──────────
# EmailDraft already enforces subject 3–120 and body ≥80 at the schema layer;
# those same numbers are the gate here, spelled out so the doc and the
# enforcement cannot drift without a visible edit.
SUBJECT_MIN_LENGTH = 3
SUBJECT_MAX_LENGTH = 120
BODY_MIN_LENGTH = 80

# ── Obfuscation tolerance ────────────────────────────────────────────────────
# A scanner defeated by "i g n o r e" (or "i.g.n.o.r.e", or zero-width
# characters stuffed between letters) is theatre.  Two normalisations cover it:
#  - _normalized: lowercases, strips zero-width chars, collapses whitespace —
#    for phrase/role-marker matching where word boundaries must survive.
#  - _canonical: lowercases, strips zero-width chars, and removes ALL
#    non-alphanumerics — for instruction-shaped patterns that must survive
#    punctuation or spacing inserted BETWEEN letters.
# The zero-width characters an attacker can hide between letters: ZERO WIDTH
# SPACE, ZERO WIDTH NON-JOINER, ZERO WIDTH JOINER, ZERO WIDTH NO-BREAK SPACE
# (BOM), and WORD JOINER.  Written as escapes so the set is visible in code.
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff\u2060]")


def _strip_zero_width(text: str) -> str:
    """Remove zero-width characters, which are invisible to a human reader
    but split a token as far as a naive substring match is concerned."""
    return _ZERO_WIDTH_RE.sub("", text)


def _normalized(text: str) -> str:
    """Lowercase, strip zero-width chars, collapse every whitespace run to a
    single space.  Word boundaries survive, so phrase rules can use \\b."""
    return re.sub(r"\s+", " ", _strip_zero_width(text).lower()).strip()


def _canonical(text: str) -> str:
    """Lowercase, strip zero-width chars, then delete every non-alphanumeric
    character.  'i g n o r e' and 'i.g.n.o.r.e' both fold to 'ignore'."""
    return re.sub(r"[^a-z0-9]", "", _strip_zero_width(text).lower())


# ── Content-policy rule set ─────────────────────────────────────────────────
# The wording library (docs/wording_library.md) is a general concepts primer,
# not a banned-phrase list, so this small, defensible set is taken from the
# draft writer's own rule 5 (no pressure tactics: no fake urgency, no fake
# scarcity, no "limited time" language).  It is deliberately NOT elaborate.
_BANNED_PHRASES = (
    "limited time",   # the writer rule names this one verbatim — the classic fake-scarcity phrase
    "act now",        # fake-urgency directive aimed at the recipient
    "last chance",    # fake-scarcity pressure
    "don't miss out", # fake-urgency/social-pressure phrasing
    "expires soon",   # fake-scarcity deadline that does not exist
)

# Precompile each banned phrase as a word-bounded regex.  The negative
# lookarounds matter: "unlimited time" contains "limited time" as a substring
# and must NOT flag, because unlimited time off is a legitimate benefit a
# cold email might reference; the lookbehind on the first letter and lookahead
# on the last letter keep "limited time" from matching inside a longer word.
_BANNED_PHRASE_RES = tuple(
    re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])")
    for phrase in _BANNED_PHRASES
)

# Unresolved template-token sentinels, matched case-insensitively against the
# lowercased drafted text.  "replace-me" is the SHARED vocabulary: it is a
# substring of app/tools/send_email.py's own
# "outreach@REPLACE-ME-BEFORE-SENDING.test" sentinel, so the runner and the
# send tool agree rather than inventing a second vocabulary (ticket §2.1).
_UNRESOLVED_LITERALS = ("todo", "placeholder", "replace-me")

# Braced template tokens — {{double}} or {single_identifier} — that would leak
# an unsubstituted variable into a real message.  Scoped to subject/body only:
# the deterministic footer legitimately carries {UNSUBSCRIBE_URL} (B3-Z1), so
# scanning the footer here would flag every correct draft.
_TEMPLATE_TOKEN_RE = re.compile(r"\{\{.*?\}\}|\{[A-Za-z_][A-Za-z0-9_.\-]*\}")

# ── Square-bracket mail-merge tokens (ticket H9) ─────────────────────────────
# The first real end-to-end run produced three drafts that ALL opened with an
# unresolved mail-merge placeholder in the SQUARE-BRACKET convention
# ("Hi [Name],", "Hi [First Name],") and the content gate passed all three,
# because _TEMPLATE_TOKEN_RE above only matched {braced} tokens.  This is a
# SIBLING rule to that regex, not a widening of it: the shapes it matches are
# different enough (a bracket group naming a field, versus any {identifier})
# that two rules read more honestly than one sprawling alternation.
#
# COVERAGE BOUNDARY — which bracket shapes are caught and which are NOT, so a
# future widening knows the deliberate edge (the same convention as H7's
# COVERAGE BOUNDARY comment in tests/test_dialect_coverage.py):
#   CAUGHT:   a [...] group whose NORMALIZED content is one of the KNOWN
#             mail-merge field names below — the standard contact/company
#             vocabulary a drafter reaches for when personalizing.  The four
#             shapes that actually slipped through in the real run are all
#             here: [Name], [First Name], [Company], [CLIENT].
#   NOT CAUGHT (deliberately): bracketed verb-phrase call-to-actions
#             ("[Book a call]", "[Click here]", "[Sign up]"), bracketed
#             values/abbreviations ("[HK]", "[2024]", "[Google]"), and
#             bracketed inline annotations ("[sic]") — these are legitimate
#             prose shapes in a cold email, and flagging them would be a
#             false-positive defect.  Also not caught: bracket content
#             containing sentence punctuation (a bracket can carry a full
#             parenthetical sentence).  A model that invents a NON-standard
#             field name ("[Prospect Name]") is not caught HERE — that is the
#             writer brief and critic's job (app/agents/draft.py), which
#             prevent placeholders from being authored at all; this gate is
#             defense-in-depth for the known vocabulary, not the primary fix.
_MAIL_MERGE_FIELD_NAMES = frozenset({
    "name", "first name", "last name", "full name",        # the person
    "company", "company name", "client", "prospect",       # the organization
    "title", "role", "position", "email", "address",       # the contact record
    "industry", "account", "organization",
})

# The bracket-group matcher: [...] with no nested brackets.  The content is
# normalized (lowercase, whitespace-collapsed) and compared against the
# vocabulary above, so "[First Name]" and "[first name]" both match.
_MAIL_MERGE_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def evaluate_content_policy(subject: str, body: str, footer: str) -> list[str]:
    """Run every content-policy check over one persisted revision and return
    the machine-readable reasons it failed.  Empty list == passed.

    Checks: length bounds (subject 3–120, body ≥80 — the gate re-checks rather
    than assuming EmailDraft validated them), the deterministic footer carrying
    the unsubscribe affordance, no unresolved template tokens, and no
    banned/pressure phrasing.
    """
    reasons: list[str] = []  # accumulated failures; empty at the end means a clean pass

    # ── Length bounds ─────────────────────────────────────────────────────
    # The raw lengths are checked, exactly the way EmailDraft's Field
    # constraints see them — the runner is the gate, so it does not trust
    # that schema validation already rejected an out-of-bounds value.
    if not (SUBJECT_MIN_LENGTH <= len(subject) <= SUBJECT_MAX_LENGTH):
        reasons.append(REASON_SUBJECT_LENGTH)
    if len(body) < BODY_MIN_LENGTH:
        reasons.append(REASON_BODY_LENGTH)

    # ── Compliance footer ─────────────────────────────────────────────────
    # The footer is code-composed (B3-Z1), so its presence is structural — but
    # the gate re-checks the composed artifact rather than assuming, and a
    # mangled/missing unsubscribe affordance is a hard refusal.
    if UNSUBSCRIBE_TOKEN not in footer:
        reasons.append(REASON_FOOTER_UNSUBSCRIBE)

    # ── Unresolved template tokens ────────────────────────────────────────
    # Scoped to the drafted subject/body (never the footer — see the regex
    # comment).  A braced token or a TODO/PLACEHOLDER/REPLACE-ME sentinel that
    # survives to send time is a defect the recipient would see verbatim.
    drafted_text = f"{subject}\n{body}"
    lowered = drafted_text.lower()
    if _TEMPLATE_TOKEN_RE.search(drafted_text) or any(
        literal in lowered for literal in _UNRESOLVED_LITERALS
    ):
        reasons.append(REASON_UNRESOLVED_TOKEN)
    # ── Square-bracket mail-merge tokens (ticket H9) — the sibling rule ────
    # The braced check above could not see the real run's defect: "Hi [Name],"
    # / "Hi [First Name]," sailed through because the square-bracket
    # convention was unguarded.  This loop catches the KNOWN field-name
    # vocabulary (see _MAIL_MERGE_FIELD_NAMES' COVERAGE BOUNDARY comment for
    # the shapes deliberately NOT caught).  Same reason string as the braced
    # rule — one unresolved-token vocabulary, not two — so the audit trail
    # reads "which class of defect" without parsing which bracket it used.
    for match in _MAIL_MERGE_BRACKET_RE.finditer(drafted_text):
        # Normalized content (lowercase, whitespace-collapsed) compared
        # against the vocabulary — "[First Name]" and "[first name]" both hit.
        if _normalized(match.group(1)) in _MAIL_MERGE_FIELD_NAMES:
            reasons.append(REASON_UNRESOLVED_TOKEN)
            break  # one hit names the class; the audit row stays small

    # ── Banned/pressure phrasing ──────────────────────────────────────────
    # The one deliberate content judgement: the writer's own rule 5 forbids
    # fake urgency/scarcity, and these phrases are what that rule means in
    # code.  Only the first hit is recorded — the reason is the rule name,
    # not a list of every offending phrase, so the audit row stays small.
    normalized = _normalized(drafted_text)
    for phrase_res in _BANNED_PHRASE_RES:
        if phrase_res.search(normalized):
            reasons.append(REASON_BANNED_PHRASE)
            break
    return reasons


# ── Injection rule set ──────────────────────────────────────────────────────
# The threat: attacker-controlled text reaches the draft (scraped research, and
# since E1 a reply body).  This scanner checks the DRAFT WE ARE ABOUT TO SEND
# for smuggled instruction-shaped content.  Each group below is one class of
# attack; the canonical patterns are matched against the punctuation-stripped
# text so simple obfuscation does not defeat them.

# Instruction-override phrasings — "ignore previous instructions" and friends.
# Canonical (no separators) so "i g n o r e" / "i.g.n.o.r.e" still match.
_INSTRUCTION_OVERRIDE_PATTERNS = (
    "ignorepreviousinstructions",
    "ignorepriorinstructions",
    "ignoreallpreviousinstructions",
    "ignoretheabove",
    "ignoreaboveinstructions",
    "disregardtheabove",
    "disregardpreviousinstructions",
    "disregardpriorinstructions",
    "forgettheabove",
)

# Directives aimed at an agent — a role/personality reassignment smuggled into
# prose.  Recall-biased on purpose: these phrasings are out-of-distribution for
# a cold email to a prospect, so a false positive is less costly than a missed
# takeover (the trade-off is documented in docs/threat-model.md).
_AGENT_DIRECTIVE_PATTERNS = (
    "youarenow",
    "yournewtaskis",
)

# Embedded approve/send directives — the exact actions the attacker wants the
# harness to take without a human.  Multi-word on purpose: bare "send" or
# "approve" appears in legitimate prose ("send the details", "I'd love your
# approval"), so only the agent-directed combinations flag.
_SEND_DIRECTIVE_PATTERNS = (
    "sendthisemail",
    "sendtheemail",
    "sendthisemailnow",
    "sendnow",
    "approvethis",
    "approvethedraft",
    "approvethisemail",
    "bypassreview",
    "bypassapproval",
    "skipreview",
    "skipapproval",
    "donotrequireapproval",
    "sendwithoutapproval",
    "sendwithoutreview",
    "overridetheapproval",
    "overridereview",
)

# Role/system markers injected into prose.  Matched on the NORMALIZED text
# (word boundaries and the colon survive) rather than the canonical text, so
# a legitimate "system" inside a word does not flag as "SYSTEM:".
_ROLE_MARKER_RE = re.compile(r"\b(system|assistant)\s*:", re.IGNORECASE)
# <|...|> markers (e.g. <|im_start|>, <|endofprompt|>) — the tokenizer-style
# injection that asks the model to switch roles.
_ANGLE_BRACE_MARKER_RE = re.compile(r"<\|[^|>]*\|>")


def evaluate_injection_scan(subject: str, body: str) -> list[str]:
    """Run every injection rule over the drafted subject/body and return the
    machine-readable reasons it failed.  Empty list == passed.

    Scans subject and body only — the footer is deterministic code output, not
    attacker-controlled, and scanning it would add false-positive surface for
    no threat.
    """
    reasons: list[str] = []  # accumulated failures; empty means clean
    haystack = f"{subject}\n{body}"
    canonical = _canonical(haystack)  # obfuscation-tolerant form for the canonical patterns
    normalized = _normalized(haystack)  # word-boundary form for the role markers

    # ── Instruction-override phrasing ─────────────────────────────────────
    for pattern in _INSTRUCTION_OVERRIDE_PATTERNS:
        if pattern in canonical:
            reasons.append(REASON_INSTRUCTION_OVERRIDE)
            break  # one hit names the class; listing every variant adds noise

    # ── Agent-directed role reassignment ──────────────────────────────────
    for pattern in _AGENT_DIRECTIVE_PATTERNS:
        if pattern in canonical:
            reasons.append(REASON_AGENT_DIRECTIVE)
            break

    # ── Embedded approve/send directives ──────────────────────────────────
    for pattern in _SEND_DIRECTIVE_PATTERNS:
        if pattern in canonical:
            reasons.append(REASON_SEND_DIRECTIVE)
            break

    # ── Role/system markers ───────────────────────────────────────────────
    if _ROLE_MARKER_RE.search(normalized) or _ANGLE_BRACE_MARKER_RE.search(normalized):
        reasons.append(REASON_ROLE_MARKER)
    return reasons


class DraftGateResult(BaseModel):
    """The runner's verdict on one revision — the structured output callers
    and tests consume.

    ``evaluated`` is the fail-closed flag: False on a crash or a missing
    revision, in which case NOTHING was written and the columns stay NULL.
    ``policy_check_passed`` / ``injection_scan_passed`` are only meaningful
    when ``evaluated`` is True; ``policy_reasons`` / ``injection_reasons``
    name the specific failures for the audit trail.
    """

    draft_version_id: str  # which revision was evaluated
    evaluated: bool  # True only when the checks ran and the columns were written
    policy_check_passed: bool  # True when no content-policy reason was produced
    injection_scan_passed: bool  # True when no injection reason was produced
    policy_reasons: list[str]  # machine-readable content-policy failures (empty on pass)
    injection_reasons: list[str]  # machine-readable injection failures (empty on pass)
    error: str | None = None  # set only on the crash/missing-revision path


def run_draft_gate(conn, *, draft_version_id: str, run_id: str) -> DraftGateResult:
    """Evaluate one persisted revision and write exactly the two gate columns
    this runner owns (``policy_check_passed`` / ``injection_scan_passed``).

    Flow: read the revision -> run both deterministic checks -> write 1/0 for
    each column through the write gate -> log a step (success on a clean pass,
    failed on a named failure).  On an unexpected exception the columns are
    deliberately NOT written (left NULL, which still refuses at the send gate)
    and a failed step is logged — never a bare 1 on an error path (ticket §2.4).
    """
    # One fresh step id — the runner is its own pipeline step, exactly like the
    # draft persist node and the review gate each own their own steps.
    step_id = new_id("step")

    # ── Read the revision the runner is asked to evaluate ────────────────
    row = conn.execute(
        "SELECT subject, body, footer, target_id FROM message_draft_versions "
        "WHERE draft_version_id=?;",
        (draft_version_id,),
    ).fetchone()
    if row is None:
        # A missing revision is an integrity anomaly, not a pass.  Log the
        # refusal (never skip logs) and leave the columns untouched — nothing
        # was written, so the send gate still fails closed on NULL.
        log_step(
            conn, run_id=run_id, step_id=step_id, target_id=None,
            tool_name="draft_gate", agent_id="system",
            input_data={"stage": "draft_gate", "draft_version_id": draft_version_id},
            output_data={"error": "missing revision"},
            status="failed",
        )
        return DraftGateResult(
            draft_version_id=draft_version_id, evaluated=False,
            policy_check_passed=False, injection_scan_passed=False,
            policy_reasons=[], injection_reasons=[], error="missing revision",
        )

    try:
        # ── Run both checks over the persisted artifact ───────────────────
        policy_reasons = evaluate_content_policy(row["subject"], row["body"], row["footer"])
        injection_reasons = evaluate_injection_scan(row["subject"], row["body"])
        # A check passes iff its reason list is empty — the list is the truth.
        policy_passed = not policy_reasons
        injection_passed = not injection_reasons

        # ── Write the two columns through the write gate ──────────────────
        # A single gated UPDATE, so the audit row and the data row commit
        # atomically.  record_id is the revision id; agent_id="system" is the
        # registered deterministic principal.  send_gate_passed is NOT touched
        # — that column stays the send gate's own (ticket §2.3).
        write_gate_commit(
            conn,
            action="update_draft_gate_columns",  # G2's new KNOWN_ACTION — the runner's write is audited distinctly
            table_name="message_draft_versions",
            record_id=draft_version_id,
            payload={
                "policy_check_passed": 1 if policy_passed else 0,
                "injection_scan_passed": 1 if injection_passed else 0,
                "policy_reasons": policy_reasons,
                "injection_reasons": injection_reasons,
            },
            run_id=run_id,
            step_id=step_id,
            actor="system",   # deterministic code performs the write
            agent_id="system",  # attributed to the registered system principal
            sql="UPDATE message_draft_versions "
                "SET policy_check_passed = ?, injection_scan_passed = ? "
                "WHERE draft_version_id = ?",
            params=(
                1 if policy_passed else 0,
                1 if injection_passed else 0,
                draft_version_id,
            ),
        )

        # ── Log the verdict (never skip logs — including a refusal) ───────
        # status="failed" on any named failure so the trace distinguishes a
        # refused draft from a clean one; the reasons travel in output_data so
        # an auditor sees WHY without re-running the checks.
        log_step(
            conn, run_id=run_id, step_id=step_id, target_id=row["target_id"],
            tool_name="draft_gate", agent_id="system",
            input_data={"stage": "draft_gate", "draft_version_id": draft_version_id},
            output_data={
                "policy_check_passed": 1 if policy_passed else 0,
                "injection_scan_passed": 1 if injection_passed else 0,
                "policy_reasons": policy_reasons,
                "injection_reasons": injection_reasons,
            },
            status="success" if (policy_passed and injection_passed) else "failed",
        )
        return DraftGateResult(
            draft_version_id=draft_version_id, evaluated=True,
            policy_check_passed=policy_passed, injection_scan_passed=injection_passed,
            policy_reasons=policy_reasons, injection_reasons=injection_reasons,
            error=None,
        )
    except Exception as exc:
        # ── THE CRASH PATH (fail closed, §2.4) ────────────────────────────
        # An unexpected exception must NEVER write 1 — and must not write 0
        # either, because a crash is not a verdict.  Nothing is written here,
        # so the columns stay NULL and the send gate refuses.  The failed step
        # below is what makes the crash auditable rather than silent.
        log_step(
            conn, run_id=run_id, step_id=step_id, target_id=row["target_id"],
            tool_name="draft_gate", agent_id="system",
            input_data={"stage": "draft_gate", "draft_version_id": draft_version_id},
            output_data={"error_type": type(exc).__name__, "error": str(exc)},
            status="failed",
        )
        return DraftGateResult(
            draft_version_id=draft_version_id, evaluated=False,
            policy_check_passed=False, injection_scan_passed=False,
            policy_reasons=[], injection_reasons=[],
            error=f"{type(exc).__name__}: {exc}",
        )
