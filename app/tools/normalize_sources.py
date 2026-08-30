"""
Normalize sources — combine every acquisition tool's extracted text into one
LLM-ready text blob, or route a target straight to ``failed`` when there is
zero usable source data.

This tool sits between the acquisition tools (Task 8's ``fetch_sources``, and
later Playwright/Tavily variants — all producing ``NormalizedSource`` objects)
and the LLM nodes (Task 10's ``summarize_company``, a later task). Its job is
narrow: combine every source's ``extracted_text`` into one text blob the LLM
can read, OR — if there were zero usable sources — recognize this as an
upstream data failure and route the target straight to the ``failed`` state
via ``app.state_machine.transition`` (Task 5, already built), **without ever
calling an LLM on nothing**.

This zero-sources case matters because it's a different failure category
than "the LLM produced a bad summary" — it's "there was never any data to
summarize in the first place," and per ``docs/state-machine.md`` §7c that
distinction is enforced structurally here, not left for the LLM node to
discover and handle.  "Zero usable sources" covers two upstream shapes:
an empty sources list (nothing fetched at all) and fetched sources whose
combined text strips to nothing (a JS-only page whose static HTML is an
empty shell) — both route to ``failed`` with reason="no_sources_available"
before any LLM call, so an empty prompt can never reach the model.
"""

from app.state_machine import transition
from app.tools.fetch_sources import NormalizedSource
from app.tools.log_step import log_step


def normalize_sources(
    conn,
    *,
    sources: list[NormalizedSource],
    target_id: str,
    run_id: str,
    step_id: str,
    actor: str = "system",
) -> str | None:
    """Combine every source's extracted_text into one LLM-ready blob.

    Args:
        conn: An app.db.Conn from app.db.connect() (sqlite or postgres).
        sources: The list of NormalizedSource objects produced by acquisition
                 tools (fetch_sources and future Playwright/Tavily variants).
        target_id: The target these sources belong to.
        run_id: The pipeline run this normalization is part of.
        step_id: The specific step within the run that triggered this call.
        actor: Passed through to transition() and log_step(); defaults to
               "system" since this is deterministic pipeline code. Also
               passed as log_step's agent_id — the two seeded principals
               ("system"/"operator") map 1:1 to the actor allowlist today.

    Returns:
        A combined text string ready for LLM summarization, OR None if there
        were zero usable sources — an empty list, or fetched sources whose
        combined text strips to nothing (in which case the target has
        already been transitioned to "failed" with
        reason="no_sources_available").
    """

    # ── Combine sources first ───────────────────────────────────────────────
    # Each source's extracted_text is joined with a double newline ("\n\n").
    # This keeps each source visually and structurally separate for the LLM —
    # rather than running them together into one undifferentiated block where
    # the LLM can't tell where one source ends and the next begins.  A
    # paragraph break between sources gives the LLM the same structural cue a
    # human reader would get.
    #
    # We read s.extracted_text specifically — that's the only field of
    # NormalizedSource this tool cares about.  We do NOT use
    # source_confidence or source_priority for ordering or filtering in this
    # task's scope (a later task may add priority-based ordering, but for now
    # sources are joined in whatever order the acquisition tools produced them).
    #
    # The join is computed BEFORE the guard below because the guard needs the
    # actual combined text to decide whether anything usable exists — "the
    # sources list is non-empty" alone no longer proves there is text to
    # summarize, since a source can fetch fine and still extract to nothing.
    combined = "\n\n".join(s.extracted_text for s in sources)

    # ── Zero-usable-text guard ──────────────────────────────────────────────
    # Two distinct upstream situations land in this ONE branch, and both mean
    # the same thing per docs/state-machine.md §7c: there is zero usable
    # source text, so summarize_company/detect_signals must never be called
    # (nothing to summarize).  This is NOT an LLM failure (the LLM was never
    # called) — it's an upstream data failure, and §7c says that distinction
    # is enforced structurally here in the normalization layer, not left for
    # the LLM node to discover.  The guard must happen before anything
    # LLM-related — the test_zero_sources_does_not_call_any_llm test exists
    # specifically to document and enforce this invariant so no future change
    # can silently slip an LLM call before or instead of this branch.
    #
    # Case (a): ``not sources`` — every acquisition tool upstream either
    # failed or produced nothing usable; the world gave us zero sources.
    #
    # Case (b): sources exist but ``combined.strip()`` is empty — pages
    # fetched successfully (HTTP 200) yet yielded no extractable text (the
    # common shape is a JS-only page whose static HTML is an empty shell,
    # but a page that extracts to pure whitespace like "\n\n   \t\n" is just
    # as empty to the model).  The .strip() is what makes case (b) reachable:
    # without it, whitespace-only text passes the guard, flows downstream
    # into the LLM prompt, and Vertex rejects the call with
    # 400 INVALID_ARGUMENT "Model input cannot be empty" — a wasted,
    # guaranteed-to-fail API call that also mislabels the failure as
    # llm_transport_error_phase1 ("provider unreachable") when the truth is
    # "we had no text to summarize".  Failing here instead keeps the reason
    # honest: reason="no_sources_available" per §7c, because text that
    # strips to nothing IS zero usable sources.
    if not sources or not combined.strip():
        # Transition the target straight to "failed" with a machine-readable
        # reason string.  from_state="new" is a known, intentional
        # simplification for this task's scope (Task 9) — the test fixture's
        # target starts in state "new".  A later task (Task 14, graph wiring)
        # will pass the target's actual current state instead of a hardcoded
        # value when this is wired into the full pipeline (a Google ADK
        # SequentialAgent since task A4a).  Do not
        # try to fix or generalize this in this task.
        #
        # to_state="failed" is allowed from ANY state (it's in
        # ANY_TARGET_TRANSITIONS in state_machine.py), so this transition is
        # valid regardless of what state the target is actually in.
        #
        # reason="no_sources_available" is the specific machine-readable
        # string that makes this failure category visible in the audit trail's
        # state_transitions.reason column — an operator scanning the log can
        # distinguish "failed because no sources existed" from "failed because
        # the LLM returned invalid output" without opening any other file.
        # It is shared verbatim by cases (a) and (b): §7c defines it for
        # "zero usable sources", and empty-after-strip text is zero usable
        # sources, so NO new reason string is invented for case (b).
        transition(
            conn,
            target_id=target_id,
            from_state="new",
            to_state="failed",
            reason="no_sources_available",
            actor=actor,
            run_id=run_id,
            step_id=step_id,
        )

        if not sources:
            # Case (a) log — nothing fetched.  output_data=None is
            # deliberate: with zero sources there is literally nothing to
            # report — zero chars, zero source_count, zero everything.  NULL
            # in the steps table's output_json column is semantically "no
            # output was produced", which is exactly what happened.
            # status="failed" records that this normalization attempt did
            # not succeed — the target is now in a terminal failure state
            # and will not proceed to summarization.
            log_step(
                conn,
                run_id=run_id,
                step_id=step_id,
                target_id=target_id,
                tool_name="normalize_sources",
                agent_id=actor,  # Which registered agent ran this step — mirrors actor (see docstring).
                input_data={"source_count": 0},
                output_data=None,
                status="failed",
            )
        else:
            # Case (b) log — fetched but unextractable.  Both cases share
            # reason="no_sources_available", but they are different upstream
            # situations, so the steps row must let an operator tell them
            # apart: input source_count >= 1 proves sources WERE fetched
            # (case (a) always logs 0), and output_data={"chars":
            # len(combined)} records how much raw text arrived that stripped
            # to nothing — 0 when every source extracted to "" and >0 for
            # whitespace-only shells.  A dict (not NULL) also mirrors the
            # happy path's output shape so log consumers see a dict either
            # way.  status="failed" as above: terminal failure, no
            # summarization.
            log_step(
                conn,
                run_id=run_id,
                step_id=step_id,
                target_id=target_id,
                tool_name="normalize_sources",
                agent_id=actor,  # Same mirroring as the case-(a) log above.
                input_data={"source_count": len(sources)},
                output_data={"chars": len(combined)},
                status="failed",
            )

        # Return None to signal to the caller (summarize_company, a later
        # task — Task 10) that there is no text to summarize.  The caller is
        # expected to check for None and skip the LLM call entirely — the
        # target has already been routed to "failed" by this function, so
        # there is nothing left for the summarization node to do with this
        # target.
        return None

    # ── Happy path: log the combined blob and return it ────────────────────
    # Log the successful normalization.  output_data={"chars": len(combined)}
    # is a lightweight trace signal — same reasoning as Task 8's
    # chars_extracted field: it tells an operator scanning the trace log how
    # much combined text was produced without duplicating the full combined
    # text into the log (which could be thousands of chars for multiple
    # sources).  status="success" records that normalization completed
    # normally and the target will proceed to summarization.
    log_step(
        conn,
        run_id=run_id,
        step_id=step_id,
        target_id=target_id,
        tool_name="normalize_sources",
        agent_id=actor,  # Same mirroring as the failure-path log above.
        input_data={"source_count": len(sources)},
        output_data={"chars": len(combined)},
        status="success",
    )

    # Return the combined text blob.  The caller (summarize_company, Task 10)
    # passes this string directly into the LLM summarization prompt — it
    # becomes the "here is what we know about this company" context that the
    # LLM reads and summarizes into a structured company profile.
    return combined
