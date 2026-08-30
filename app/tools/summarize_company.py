"""
summarize_company — the first LLM-backed node in the Phase 1 pipeline.

Takes the combined text from normalize_sources (Task 9) and asks the LLM to
produce a structured CompanyProfile via call_structured().  This is a
research-summary tool, not a creative one — the system prompt explicitly
forbids inventing facts not present in the source text.

Retry discipline (docs/state-machine.md §7b): if the LLM's output is empty or
fails schema validation, exactly one retry is attempted with a stricter
re-prompt appended.  If both attempts fail, the target transitions to "failed"
with reason "llm_output_invalid_phase1" — a distinct machine-readable reason
so the audit trail always shows *why* a target failed, not just *that* it
failed.

Transport discipline (added with LLMTransportError): a provider transport
failure (429/5xx/connection drop) is a third failure mode with its own reason
string, "llm_transport_error_phase1".  Retryable transport errors consume the
same bounded retry, after a fixed pause (TRANSPORT_RETRY_SLEEP_SECONDS) so an
immediate retry doesn't just hit the same rate-limit window again;
non-retryable ones (400/401/403/404) fail immediately without burning the
second attempt.  Both paths still log every failed attempt and still end in
the same any -> failed transition.

Persistence (ticket A8): on the success path only, the validated profile is
written to the account row (company_summary / industry / estimated_size /
geo) through write_gate.commit — the data update and its write_log audit row
land atomically.  confidence deliberately has no accounts column (schema
changes are out of scope); it stays in steps.output_json via log_step.
"""

import time

from app.llm import (
    call_structured,
    hash_call,
    LLMEmptyResponseError,
    LLMSchemaValidationError,
    LLMTransportError,
    TRANSPORT_RETRY_SLEEP_SECONDS,
)
from app.schemas import CompanyProfile
from app.state_machine import transition
from app.tools.log_step import log_step
from app.write_gate import commit as write_gate_commit

# ── System prompt ─────────────────────────────────────────────────────────────
# The prompt explicitly instructs the model never to invent facts not present in
# the source text — this is a research-summary tool, not a creative one.
# Hallucinated company facts (a made-up industry, a fabricated employee count)
# would corrupt scoring and outreach downstream, so the model is directed to
# signal uncertainty via the confidence field rather than guessing.
_SYSTEM_PROMPT = (
    "Convert the provided company text into a structured company profile. "
    "Never invent facts not present in the source text. If confidence is "
    "low, say so via the confidence field rather than guessing."
)


def summarize_company(
    conn,
    *,
    extracted_text: str,
    target_id: str,
    run_id: str,
    step_id: str,
    actor: str = "system",
) -> CompanyProfile | None:
    """Summarize extracted company text into a structured CompanyProfile.

    Returns a CompanyProfile on success, or None if both the initial call and
    exactly one retry failed — in that case the target has already been
    transitioned to "failed" and there is no profile to use downstream.
    """
    # Compute the call hash once up front, before the retry loop, using the
    # *original* system prompt (not the retry's modified one).  This way both
    # the success-on-first-try and success-on-retry cases log the same hash
    # that identifies "this extracted_text was summarized" — not "which
    # specific attempt succeeded."  An operator can still tell which attempt
    # worked by looking at the status field ("success" vs. "retried").
    call_hash = hash_call(_SYSTEM_PROMPT, extracted_text)

    # Track the failure category across loop iterations so the transition
    # below writes the reason that matches what actually happened.  Defaults
    # to the output-invalid reason; only a transport failure overwrites it.
    last_failure = "llm_output_invalid_phase1"

    # The retry loop runs at most twice — exactly two prompts: the original,
    # then the original plus a stricter suffix.  After two failures the loop
    # falls through to the transition+return below.  This is a bounded retry,
    # not an unbounded loop — no third attempt, no exponential backoff.
    for attempt, prompt in enumerate([
        _SYSTEM_PROMPT,
        _SYSTEM_PROMPT + " Return ONLY valid structured output matching the schema exactly.",
    ]):
        # Each attempt gets a unique step_id suffix (e.g. "s1_a0", "s1_a1")
        # so both the failure log (on attempt 0) and the success/retry log
        # (on attempt 1) can coexist in the steps table without a UNIQUE
        # constraint violation on step_id.  The external caller's original
        # step_id is preserved as a prefix so the rows are still traceable.
        attempt_step_id = f"{step_id}_a{attempt}"
        try:
            # Call the LLM with the current prompt.  call_structured handles
            # the SDK call, tool-use extraction, and Pydantic validation —
            # it either returns a validated CompanyProfile or raises one of
            # the three documented exception types.
            profile = call_structured(
                model_alias="research_model",
                system_prompt=prompt,
                user_content=extracted_text,
                response_schema=CompanyProfile,
            )
            # Log the successful call.  status="success" if the first attempt
            # worked cleanly, "retried" if it succeeded on the retry — this
            # distinguishes the two paths in the trace log without needing a
            # separate boolean field.
            log_step(
                conn, run_id=run_id, step_id=attempt_step_id, target_id=target_id,
                tool_name="summarize_company",
                agent_id=actor,  # Which registered agent ran this step — mirrors actor today (see module note below).
                input_data={"text_len": len(extracted_text)},
                output_data=profile.model_dump(),
                status="success" if attempt == 0 else "retried",
                model_call_hash=call_hash,
            )
            # ── Persist the profile to the account row (ticket A8) ──────────
            # Placed INSIDE the success branch, between the step log above and
            # the return below, for two deliberate reasons:
            #   1. Log first, write second: if this write fails, the steps row
            #      above still records exactly what the LLM produced — the
            #      audit trail keeps the successful call even when the
            #      persistence doesn't land.
            #   2. Write before return: the profile can never reach the caller
            #      unpersisted. If the gate refuses or the SQL fails, the
            #      exception propagates — it is NOT one of the three LLM
            #      exception types in the except clause below — so the caller
            #      sees a hard failure instead of silently continuing with a
            #      profile that exists only in memory.
            # A profile that never validated never reaches this code: every
            # failure path below transitions the target to "failed" and
            # returns None first.
            #
            # Resolve account_id from targets rather than adding a parameter:
            # the signature is fixed by app/agents/phase1.py's SummarizeNode
            # (out of scope for this ticket), and the lookup mirrors
            # score_lead's precedent for account-scoped writes, so the
            # target→account mapping stays consistent across tools.
            account_id = conn.execute(
                "SELECT account_id FROM targets WHERE target_id=?;", (target_id,)
            ).fetchone()["account_id"]
            # All core-table writes go through write_gate.commit — a raw
            # conn.execute here would be invisible to the write_log audit
            # trail, which is exactly what the gate exists to prevent.
            write_gate_commit(
                conn,
                action="update_account_profile",  # New action — requires the one-line addition to KNOWN_ACTIONS made in this ticket.
                table_name="accounts",
                record_id=account_id,
                # The audit payload carries exactly the four mapped columns —
                # confidence is deliberately excluded: it has no accounts
                # column (schema changes are out of scope) and lives only in
                # steps.output_json, where the log_step above already recorded
                # the full profile.model_dump().
                payload={
                    "company_summary": profile.one_line_summary,
                    "industry": profile.industry,
                    "estimated_size": profile.estimated_size,
                    "geo": profile.geo,
                },
                run_id=run_id,
                # attempt_step_id, not the caller's bare step_id: the write
                # belongs to the specific attempt that produced this profile,
                # and only the suffixed attempt rows exist in steps — the bare
                # step_id would reference a steps row that doesn't exist.
                step_id=attempt_step_id,
                actor=actor,
                # agent_id mirrors actor — same 1:1 mapping as score_lead's
                # account write; the two seeded principals are 1:1 with the
                # actor allowlist today, and plan A4's agents will pass their
                # own id.
                agent_id=actor,
                sql="""
                    UPDATE accounts SET company_summary=?, industry=?, estimated_size=?, geo=?,
                        updated_at=datetime('now')
                    WHERE account_id=?
                """,
                params=(
                    profile.one_line_summary, profile.industry,
                    profile.estimated_size, profile.geo, account_id,
                ),
            )
            # Return immediately on success — this skips the second loop
            # iteration entirely.  If we're here on attempt 0, the retry
            # prompt never runs; if we're here on attempt 1, the loop ends.
            return profile
        except (LLMEmptyResponseError, LLMSchemaValidationError, LLMTransportError) as exc:
            # Only catch the three documented exception types that
            # call_structured is documented to raise.  Anything else is a
            # genuine bug that should propagate, not be silently rescued by
            # the retry loop.
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
            # fails or only the first attempt fails.
            log_step(
                conn, run_id=run_id, step_id=attempt_step_id, target_id=target_id,
                tool_name="summarize_company",
                agent_id=actor,  # Same mirroring as the success-path log above.
                input_data={"text_len": len(extracted_text), "attempt": attempt},
                output_data=output_data,
                status="failed", model_call_hash=call_hash,
            )
            if isinstance(exc, LLMTransportError):
                # Transport failure — a different failure category from
                # "model output invalid", so it gets its own machine-readable
                # reason in state_transitions.
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
    # the loop) — transition the target to "failed".  This only runs if the
    # for-loop finished without returning.
    # from_state="researched" is a known simplification for this task's scope
    # (Task 10) — in Phase 1's researched→scored flow this is the expected
    # state.  Will be generalized later when Task 14 wires the full graph.
    # reason is whichever failure category the loop recorded: "llm_output_invalid_phase1"
    # for empty/schema-invalid output, "llm_transport_error_phase1" for
    # provider transport failures.  Both are distinct from Task 9's
    # "no_sources_available" — an operator can tell "no data" apart from "the
    # LLM couldn't produce valid output" apart from "the provider was
    # unreachable" just by reading state_transitions.reason.
    transition(
        conn, target_id=target_id, from_state="researched", to_state="failed",
        reason=last_failure, actor=actor, run_id=run_id, step_id=step_id,
    )
    # Return None to signal to the caller that the target has already been
    # failed — there is no CompanyProfile to use downstream, and the caller
    # should not attempt further processing on this target.
    return None
