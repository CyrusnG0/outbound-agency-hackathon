"""
detect_signals — the second LLM-backed node in the Phase 1 pipeline.

Takes the combined extracted_text from normalize_sources (Task 9) and asks the
LLM to produce a list of Signal objects (buying/trigger signals like "hiring a
relevant role" or "recent product launch").  Each signal is persisted to the
``signals`` table and later consumed by ``score_lead`` via its
``signal_strength`` field.

This is a research-extraction tool, not a creative one — the system prompt
explicitly forbids inventing signals not present in the source text.  An empty
list is a valid, successful result (the company genuinely has no detectable
signals), not a failure.

Retry discipline (docs/state-machine.md §7b): if the LLM's output is empty or
fails schema validation, exactly one retry is attempted with a stricter
re-prompt appended.  If both attempts fail, the target transitions to "failed"
with reason "llm_output_invalid_phase1" — the same reason string Task 10 uses,
because both nodes fail for the same underlying reason (the LLM couldn't
produce valid structured output).

Transport discipline (added with LLMTransportError): a provider transport
failure (429/5xx/connection drop) is a third failure mode with its own reason
string, "llm_transport_error_phase1" — same as Task 10's, because both nodes
fail for the same underlying reason here too (the provider was unreachable).
Retryable transport errors consume the same bounded retry, after a fixed pause
(TRANSPORT_RETRY_SLEEP_SECONDS) so an immediate retry doesn't just hit the same
rate-limit window again; non-retryable ones (400/401/403/404) fail immediately
without burning the second attempt.  Both paths still log every failed attempt
and still end in the same any -> failed transition.

Dedup by rerun: the ``signals`` table's UNIQUE (target_id, run_id, signal_type,
signal_value) constraint (defined in app/db.py) prevents duplicate rows when
this node runs twice for the same target_id + run_id.  The resulting
IntegrityError is caught and silently ignored per signal — it's the
intended dedup mechanism, not a bug.

Evidence verification (B2b): B2a checked each ``evidence_quote`` against the
``extracted_text`` this tool was GIVEN — which, in the ADK pipeline, is the
research agent's own prose (its ``output_key``), not the fetched pages.  So
``evidence_verified=1`` meant only "the signal agent quoted the research
agent faithfully", never "the claim is true of the company".  B2b closes
that hole: the raw fetched pages and the research findings are now persisted
to the ``sources`` table (see app/tools/fetch_sources.py and
app/agents/phase1.py), and every quote is checked against persisted RAW
source text first, the findings second, else ``unverified`` — a three-way
verdict recorded as ``signals.evidence_tier`` (see ``_evidence_tier`` for
why the ordering is load-bearing).  The ``findings`` tier is NOT a failure:
the research agent's google_search/url_context results resolve server-side
and can never be captured, so a search-derived quote is legitimately
checkable only against the agent's prose.
"""

import time

from app.db import IntegrityError
from app.ids import new_id
from app.llm import (
    call_structured,
    hash_call,
    LLMEmptyResponseError,
    LLMSchemaValidationError,
    LLMTransportError,
    TRANSPORT_RETRY_SLEEP_SECONDS,
)
from app.schemas import Signal
from app.state_machine import transition
# The shared source_type constant that marks agent prose in the sources table
# — imported (not redefined) so the exclusion filter here can never drift from
# the value ResearchBookkeepingNode writes (see FINDINGS_SOURCE_TYPE's comment
# in fetch_sources.py for why three modules must agree on it).
from app.tools.fetch_sources import FINDINGS_SOURCE_TYPE
from app.tools.log_step import log_step
from app.write_gate import commit as write_gate_commit


# ── System prompt ─────────────────────────────────────────────────────────────
# The prompt explicitly enumerates the four exact valid signal_type values —
# "hiring_relevant_role", "product_or_ops_change", "recent_launch_or_expansion",
# "workflow_complexity_evidence" — because these are the only strings that
# Signal.signal_type (typed as the SignalType Literal in app/schemas.py) will
# accept.  If the LLM invents a fifth category like "funding_event", Pydantic
# validation rejects the whole response as LLMSchemaValidationError — so
# constraining the taxonomy here keeps the LLM inside the scoring formula's
# known dimensions.  The prompt also explicitly states that an empty list is a
# valid result ("do not invent one") — this is what makes the empty-list-is-
# success test case reliable rather than a fluke of the current model's behavior.
#
# B2a extension: the prompt now also demands evidence_quote — a span copied
# exactly, character for character, from the provided text.  Telling the model
# up front that a quote which does not appear verbatim will be "recorded as
# unverified" (not rejected, not rewritten) makes the downstream containment
# check (see _evidence_quote_verified below) a stated contract rather than a
# hidden trap — the model is warned that fabrication is detectable, and the
# pipeline keeps the unverified signal for the operator to judge instead of
# pretending it never happened.
#
# B2b note: the prompt string itself is UNCHANGED by B2b (no prompt-behavior
# change, so no prompt-doc update).  Its "recorded as unverified" warning
# still names the fabrication case exactly; B2b only ADDS an upgrade path
# the model never needs to know about — a quote that also appears in a
# persisted fetched page is recorded as the stronger 'source' tier.
_SYSTEM_PROMPT = (
    "Extract buying/trigger signals from the company text. Valid signal_type "
    "values: hiring_relevant_role, product_or_ops_change, "
    "recent_launch_or_expansion, workflow_complexity_evidence. Every signal "
    "must include evidence_quote: a span copied exactly, character for "
    "character, from the provided text — not paraphrased, not summarised, "
    "not reconstructed. A quote that does not appear verbatim in the "
    "provided text will be recorded as unverified. An empty list is a valid "
    "result if no signals are present — do not invent one."
)


def _call_detect_signals(system_prompt: str, extracted_text: str) -> list[Signal]:
    """Call the LLM and return a validated list of Signal objects.

    This is a separate function (not inlined in detect_signals) so tests can
    mock it directly via ``patch("app.tools.detect_signals._call_detect_signals")``
    — the same isolation pattern used by Task 8's ``_fetch_static_page`` and
    Task 10's ``call_structured`` mock in ``test_summarize_company.py``.

    Uses an inline ``_Wrapper(BaseModel)`` class with a single ``signals:
    list[Signal]`` field because ``call_structured``'s contract (per Task 10)
    is "one Pydantic model in, one instance out" — it does NOT support a bare
    ``list[Signal]`` as the top-level response_schema.  Wrapping the list in a
    single-field container lets this node get a list back through the exact
    same contract, and then unwrap it before returning to the caller.
    """
    from pydantic import BaseModel

    # A throwaway wrapper: call_structured validates against this model (which
    # has one field, "signals", a list[Signal]), then we extract just the list
    # before returning.  This keeps call_structured's API simple (always one
    # model class) while still letting this node produce a list.
    class _Wrapper(BaseModel):
        signals: list[Signal]

    # Call the LLM via the same structured-output path every other LLM node
    # uses.  The model_alias "research_model" is resolved to a pinned model
    # string via config/models.yaml — no hardcoded model names in app code.
    result = call_structured(
        model_alias="research_model",
        system_prompt=system_prompt,
        user_content=extracted_text,
        response_schema=_Wrapper,
    )
    # Unwrap the validated _Wrapper instance back to the plain list[Signal]
    # that the caller (and the rest of this module) expects — the wrapper
    # exists only to bridge call_structured's single-model contract.
    return result.signals


def _normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run (spaces, tabs, newlines) to one space.

    Why this is needed: models reflow line breaks and collapse runs of spaces
    even when quoting a source faithfully, so a byte-for-byte ``in`` check
    would flag honest quotes as fabricated — a whitespace-only difference is
    formatting, not invention.  Why it stops there: str.split() with no
    separator splits on ANY whitespace run, and nothing else — no
    lowercasing (changing case is an edit, not formatting), no fuzzy or
    substring-of-substring matching (that would let a paraphrase slip
    through, which is exactly the fabrication this check exists to catch).
    The check must stay strict enough to mean something.
    """
    # split() with no argument is the entire normalisation: it discards
    # leading/trailing whitespace and treats every run of any whitespace
    # character as a single separator, then join() reassembles with single
    # spaces — so "paper  and\npen" and "paper and pen" become identical.
    return " ".join(text.split())


def _evidence_quote_verified(evidence_quote: str, extracted_text: str) -> bool:
    """Deterministically check that the quote literally appears in the source.

    Deliberately NOT an LLM re-reading the text to judge whether the signal
    is true — that is circular verification: the same model that hallucinated
    the claim can hallucinate agreement with it, producing confidence without
    information.  A model that invented a signal cannot produce a quote that
    literally appears in text it never read, so plain containment (after the
    whitespace normalisation above) is the check.  URL matching is
    deliberately NOT used either: ADK returns grounding URLs as opaque Vertex
    redirects, near-useless for attribution — text containment is the
    reliable signal.
    """
    # Substring containment, one comparison per side normalisation — the
    # whole check.  No lowercasing, no fuzzy ratio, no "quote trimmed to a
    # fragment that matches" escape hatch.
    return _normalize_whitespace(evidence_quote) in _normalize_whitespace(extracted_text)


def _load_raw_source_texts(conn, target_id: str, run_id: str) -> list[str]:
    """Load the persisted RAW source texts for (target, run) — the strongest
    evidence tier's ground truth (ticket B2b).

    Reads the ``sources`` table for this target+run, EXCLUDING the
    research_findings rows: those are the agent's own prose, and checking a
    quote against them proves only that the signal agent quoted the research
    agent faithfully — the exact circularity B2b exists to break.  Raw rows
    (company_website today) are text we actually fetched and persisted, so a
    quote found there is attributable to a stored page.

    Returns possibly-empty: a run whose fetch_page calls all failed (or a
    legacy call path with no persistence) has no raw rows, and every signal
    then falls through to the findings/unverified tiers — the honest
    downgrade, never an error.
    """
    rows = conn.execute(
        # The != filter is the load-bearing distinction: raw pages qualify,
        # agent prose does not.  Uses the shared constant, never a literal,
        # so a renamed findings marker cannot silently start counting as raw
        # evidence (see FINDINGS_SOURCE_TYPE's comment in fetch_sources.py).
        "SELECT extracted_text FROM sources "
        "WHERE target_id=? AND run_id=? AND source_type != ?;",
        (target_id, run_id, FINDINGS_SOURCE_TYPE),
    ).fetchall()
    return [row["extracted_text"] for row in rows]


def _evidence_tier(evidence_quote: str, raw_texts: list[str], findings_text: str) -> str:
    """The three-way evidence verdict (ticket B2b): 'source', 'findings', or
    'unverified'.

    WHY THREE TIERS, AND WHY THE ORDER IS source → findings → unverified —
    a future reader must NOT "simplify" this back to B2a's boolean:

    - ``source``: the quote appears in a persisted raw text we actually
      fetched.  Strongest: attributable to a stored page.  Checked FIRST so
      the strongest attribution always wins — a quote present in both raw
      text and findings must record as source, never the weaker findings.
    - ``findings``: the quote appears in the findings text the signal agent
      was given, but in NO stored raw text.  The research agent asserted it,
      plausibly from google_search / url_context — ADK built-ins that resolve
      SERVER-SIDE, so their text never passes through this process and can
      never be persisted (measured on the real run: 4 searches for
      otandp.com).  This tier is NOT a failure and must not be treated as
      one: it means "trust the research agent — we cannot independently
      check", which is a legitimate state for server-side research, not
      evidence of fabrication.  Collapsing findings into unverified would
      smear every search-derived signal with the fabrication label and
      destroy the very distinction this check exists to create.
    - ``unverified``: the quote appears in NEITHER — the signal agent
      produced text that is in no source it was given.  This is the
      fabrication signal.

    The check itself delegates to _evidence_quote_verified unchanged: the
    same strict verbatim containment after whitespace normalisation, per
    text — no lowercasing, no fuzzy matching, no LLM re-reading.
    """
    for raw in raw_texts:
        # Raw pages first: any match here is the strongest attribution, so
        # the loop returns immediately rather than letting a later findings
        # match downgrade it.
        if _evidence_quote_verified(evidence_quote, raw):
            return "source"
    # No raw page contains the quote — fall back to the findings text the
    # signal agent actually read (the same string ResearchBookkeepingNode
    # persisted as a research_findings row, so this tier stays checkable
    # after the run).
    if _evidence_quote_verified(evidence_quote, findings_text):
        return "findings"
    # In neither — the quote was produced by the signal agent itself.
    return "unverified"


def detect_signals(
    conn,
    *,
    extracted_text: str,
    target_id: str,
    run_id: str,
    step_id: str,
    actor: str = "system",
) -> list[Signal] | None:
    """Detect buying/trigger signals from company research text.

    Returns a list of Signal objects (possibly empty — that's a valid success
    outcome, not a failure) on success, or None if both the initial call and
    exactly one retry failed — in that case the target has already been
    transitioned to "failed" and there are no signals to use downstream.
    """
    # Compute the call hash once up front, before the retry loop, using the
    # *original* system prompt (not the retry's stricter variant).  This way
    # both the success-on-first-try and success-on-retry log rows share the
    # same hash that identifies "this extracted_text was analyzed for signals"
    # — not which specific attempt succeeded.  An operator can still tell which
    # attempt worked by looking at the status field ("success" vs. "retried").
    call_hash = hash_call(_SYSTEM_PROMPT, extracted_text)

    # The retry prompt appends a stricter instruction to the original system
    # prompt, mirroring Task 10's pattern.  Computed once here so both
    # the enumerate() list and the call_hash are built from the same inputs.
    stricter_prompt = _SYSTEM_PROMPT + " Return ONLY valid structured output matching the schema exactly."

    # Track the failure category across loop iterations so the transition
    # below writes the reason that matches what actually happened.  Defaults
    # to the output-invalid reason; only a transport failure overwrites it.
    last_failure = "llm_output_invalid_phase1"

    # Bounded two-attempt retry loop — exactly the same discipline as Task 10's
    # summarize_company: first attempt uses the standard prompt, second attempt
    # uses the stricter re-prompt.  After two failures the loop falls through
    # to the transition+return below.  No third attempt, no exponential backoff.
    for attempt, prompt in enumerate([_SYSTEM_PROMPT, stricter_prompt]):
        # Each attempt gets a unique step_id suffix (e.g. "s1_a0", "s1_a1")
        # so both the failure log (on attempt 0) and the success/retry log
        # (on attempt 1) can coexist in the steps table without a UNIQUE
        # constraint violation on step_id.  This follows Task 10's pattern
        # exactly — the external caller's original step_id is preserved as
        # a prefix so the rows are still traceable.
        attempt_step_id = f"{step_id}_a{attempt}"
        try:
            # Call the LLM via the isolated helper.  _call_detect_signals
            # handles the SDK call, tool-use extraction, Pydantic validation,
            # and wrapper unwrapping — it either returns a validated
            # list[Signal] or raises one of the two documented exception types.
            signals = _call_detect_signals(prompt, extracted_text)

            # Three-tier tallies for the step log, computed over the model's
            # OUTPUT (not over inserted rows): the split describes the
            # evidence quality in what the model produced, so a signal that
            # later dedupes against an existing row still counts here — it
            # was still produced, and still either quoted a stored page, the
            # findings, or nothing at all.
            source_count = 0
            findings_count = 0
            unverified_count = 0

            # Load the persisted raw source texts ONCE, before the per-signal
            # loop — they do not change between signals or attempts, and
            # loading per signal would repeat the same query needlessly.
            # (B2b: this is the ground truth the ticket adds — before B2b
            # the ONLY checkable text was the findings prose, which made
            # "verified" mean "quoted the research agent faithfully" instead
            # of "true of the company".)
            raw_texts = _load_raw_source_texts(conn, target_id, run_id)

            # Persist each detected signal to the signals table.  An empty
            # signals list means this loop body runs zero times — nothing is
            # written, which is the intended behavior for "no signals found."
            for sig in signals:
                # Generate a fresh signal_id for this row.  "sig" prefix makes
                # ids self-describing in logs and join output.
                signal_id = new_id("sig")
                # Compute the three-way evidence verdict BEFORE persisting:
                # the quote is checked against persisted RAW source texts
                # first, then the findings text this tool was given, else
                # unverified — see _evidence_tier for why that order is
                # load-bearing and must not be collapsed back to a boolean.
                # Mark, don't delete: a signal in ANY tier — including
                # unverified — is still persisted, because DROPPING it would
                # hide the fabrication — deleting turns a detectable lie into
                # an invisible one, while marking creates the audit trail the
                # operator (and later the ICPJudge) can use to decide what to
                # trust.  No signal is dropped, skipped, or silently
                # rewritten based on this check — it is recorded only.  This
                # mark-don't-delete decision is the load-bearing choice of
                # ticket B2a and stands unchanged under B2b's three tiers.
                tier = _evidence_tier(sig.evidence_quote, raw_texts, extracted_text)
                # The B2b invariant: evidence_verified = 1 if and only if
                # tier = 'source' — "verified" now means verified against
                # persisted raw text, not merely found in the agent's prose.
                # Both columns are written from this ONE computation, so the
                # two can never disagree (the documented contract in
                # docs/db-schema.md and app/db.py's DDL comment).
                verified = 1 if tier == "source" else 0
                if tier == "source":
                    source_count += 1
                elif tier == "findings":
                    findings_count += 1
                else:
                    unverified_count += 1
                try:
                    # Write through the gate — never a raw conn.execute() for
                    # core-table mutations.  The payload is sig.model_dump() so
                    # every model-produced field (signal_type, signal_value,
                    # evidence_quote, signal_strength, source_url,
                    # source_confidence) maps directly from the validated
                    # Signal instance — no manual field-picking that could go
                    # out of sync with the schema.  evidence_verified and
                    # evidence_tier are NOT in the payload because they are
                    # not model output — they are this tool's verdict, passed
                    # as explicit columns/params so the model can never spoof
                    # its own verification.
                    write_gate_commit(
                        conn,
                        action="insert_signal",
                        table_name="signals",
                        record_id=signal_id,
                        payload=sig.model_dump(),
                        run_id=run_id,
                        step_id=step_id,
                        actor=actor,
                        # agent_id mirrors actor: the two seeded principals
                        # are 1:1 with the actor allowlist today. Plan A4's
                        # ResearchAgent will pass its own id here.
                        agent_id=actor,
                        sql="""
                            INSERT INTO signals
                                (signal_id, run_id, target_id, signal_type, signal_value,
                                 signal_strength, source_url, source_confidence,
                                 evidence_quote, evidence_verified, evidence_tier,
                                 created_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?, datetime('now'))
                        """,
                        params=(
                            signal_id, run_id, target_id, sig.signal_type, sig.signal_value,
                            sig.signal_strength, sig.source_url, sig.source_confidence,
                            sig.evidence_quote, verified, tier,
                        ),
                    )
                except IntegrityError:
                    # app.db.IntegrityError is a tuple of both dialects'
                    # exception classes — sqlite3.IntegrityError on SQLite,
                    # pg8000's IntegrityError on Postgres — so this one catch
                    # handles the dedup on either engine.
                    #
                    # Dedup by rerun: the signals table has a UNIQUE constraint
                    # on (target_id, run_id, signal_type, signal_value), defined
                    # in app/db.py.  If the same signal is detected again for
                    # the same target within the same run (e.g. a retried
                    # pipeline step), the INSERT raises IntegrityError.  We
                    # catch and silently pass — this is NOT an error.  It means
                    # the exact same signal was already recorded, which is the
                    # intended outcome of a rerun.  No manual "have I seen this
                    # signal before" check is needed in application code because
                    # the database constraint IS that check — safe to rely on
                    # without any in-memory dedup state that could be lost
                    # across process boundaries.
                    #
                    # This is safe per-signal because write_gate.commit() wraps
                    # each call in its own begin_write()/COMMIT/ROLLBACK
                    # transaction (confirmed in app/write_gate.py) — a rollback
                    # for signal #3 doesn't undo signals #1 and #2.
                    pass

            # Log the successful call.  output_data reports the THREE-tier
            # split alongside the total (B2b, extending B2a's
            # verified/unverified pair) so an operator scanning the steps
            # trace sees evidence quality per target without querying the
            # signals table — a run where most quotes are `source` is well
            # grounded, `findings`-heavy means server-side research we
            # cannot check, and `unverified`-heavy means fabrication
            # pressure.  The full signal payloads (including each
            # evidence_quote, evidence_verified and evidence_tier) are in
            # the signals table.
            # status="success" if the first attempt worked cleanly, "retried"
            # if it succeeded on the retry — same distinction Task 10 makes.
            log_step(
                conn, run_id=run_id, step_id=attempt_step_id, target_id=target_id,
                tool_name="detect_signals",
                agent_id=actor,  # Which registered agent ran this step — mirrors actor today (see write_gate_commit above).
                input_data={"text_len": len(extracted_text)},
                output_data={
                    "signal_count": len(signals),
                    "source_count": source_count,
                    "findings_count": findings_count,
                    "unverified_count": unverified_count,
                },
                status="success" if attempt == 0 else "retried", model_call_hash=call_hash,
            )
            # Return the signals list immediately on success — this skips the
            # second loop iteration entirely.  If we're here on attempt 0, the
            # retry prompt never runs; if on attempt 1, the loop ends.  This
            # return is reached even when signals == [], which is exactly what
            # makes the empty-list case a success rather than falling through
            # to the failure branch below.
            return signals
        except (LLMEmptyResponseError, LLMSchemaValidationError, LLMTransportError) as exc:
            # Only catch the three documented exception types that
            # call_structured is documented to raise.  Any other error is a
            # genuine bug that should propagate, not something the retry loop
            # should silently rescue.
            #
            # Build the failure log row FIRST (Golden Rule: never skip
            # logging) — even the non-retryable break path below leaves a
            # steps row behind.
            output_data = {"error": str(exc), "error_type": type(exc).__name__}
            if isinstance(exc, LLMTransportError):
                # For transport failures the steps row must also record
                # whether a retry could plausibly help and what the HTTP
                # status was, so an operator can tell "429, retried" apart
                # from "401, fatal" without opening the SDK.
                output_data["retryable"] = exc.retryable
                output_data["status_code"] = exc.status_code
            # Log the failure with the attempt number so the trace log shows
            # which of the two attempts failed — useful when only the retry
            # fails or only the first attempt fails.  This is identical to
            # Task 10's failure-path logging.
            log_step(
                conn, run_id=run_id, step_id=attempt_step_id, target_id=target_id,
                tool_name="detect_signals",
                agent_id=actor,  # Same mirroring as the success-path log above.
                input_data={"text_len": len(extracted_text), "attempt": attempt},
                output_data=output_data,
                status="failed", model_call_hash=call_hash,
            )
            if isinstance(exc, LLMTransportError):
                # Transport failure — a different failure category from
                # "model output invalid", so it gets its own machine-readable
                # reason in state_transitions (same string as Task 10's).
                last_failure = "llm_transport_error_phase1"
                if not exc.retryable:
                    # A 400/401/403/404 cannot be fixed by an identical second
                    # call — the request itself is wrong, not the provider's
                    # momentary state.  Break immediately instead of burning
                    # the second attempt on a doomed round trip.  (The log
                    # above runs FIRST, the break SECOND — deliberate order:
                    # the failed attempt still leaves a steps row.)
                    break
                if attempt < 1:
                    # Fixed pause before the retry (NOT exponential backoff):
                    # a 429 retried with zero delay is almost certain to 429
                    # again, and the retry count stays bounded at the existing
                    # two attempts.  Only sleep when another attempt actually
                    # remains — sleeping after the final failure is pure dead
                    # wall-clock.
                    time.sleep(TRANSPORT_RETRY_SLEEP_SECONDS)
            else:
                # Output was empty or failed schema validation — the stricter
                # re-prompt on the next attempt is the fix, and the reason
                # string stays the original one (unchanged behaviour).
                last_failure = "llm_output_invalid_phase1"

    # Both attempts failed (or a non-retryable transport error broke out of
    # the loop) — the for-loop finished without returning.  Transition the
    # target to "failed".
    #
    # from_state="researched" is the same known Task-scope simplification used
    # in Tasks 9 and 10 — in Phase 1's researched→scored flow this is the
    # expected state.  Task 14 will generalize this when wiring the full graph.
    #
    # reason is whichever failure category the loop recorded.  Both reason
    # strings are reused verbatim from Task 10 rather than getting
    # per-tool-distinct strings (e.g. "signal_detection_failed") because both
    # nodes fail for the exact same underlying reasons: the LLM couldn't
    # produce valid structured output, or the provider was unreachable.  If a
    # future operator needs to distinguish "summarize_company failed" from
    # "detect_signals failed" they can join on steps.tool_name — the reason
    # string describes the failure category, not the tool that produced it.
    # Keeping the vocabulary small keeps the audit trail meaningful rather
    # than fragmenting it per-tool.
    transition(
        conn, target_id=target_id, from_state="researched", to_state="failed",
        reason=last_failure, actor=actor, run_id=run_id, step_id=step_id,
    )
    # Return None to signal to the caller that the target has already been
    # failed — there is no signal list to use downstream, and the caller
    # should not attempt further processing on this target.  Same convention
    # as Task 10's summarize_company returning None on final failure.
    return None
