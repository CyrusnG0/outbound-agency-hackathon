"""
Static page fetch tool — the first acquisition tool in the research phase.

Given a target's company ``domain``, this tool fetches the company's website
over HTTP, strips it down to plain text, and wraps the result in a
``NormalizedSource`` — the one shape every acquisition tool (this one now;
``scrape_dynamic_page``/Playwright and ``search_web``/Tavily later) must
produce, per ``docs/data-flow.md`` §5's normalization contract.  Nothing
downstream ever touches raw HTML — only ``NormalizedSource`` objects.

Critical behavioral contract (per ``docs/state-machine.md`` §7c): any
failure fetching or parsing a single source (timeout, HTTP error/block, or
any other unexpected exception) is caught, logged as a failed ``steps`` row
via ``log_step``, and skipped — the function still returns normally, with
that source simply absent from the result list.  An empty list is a valid
(if degraded) result.

Evidence persistence (ticket B2b): every SUCCESSFULLY fetched source's raw
text is now persisted to the ``sources`` table through the write gate
(``persist_source_row`` below) — that raw text is the only ground truth a
signal's ``evidence_quote`` can be fact-checked against, so it must outlive
the process.  Deliberately NARROWED contract: the never-raise rescue above
covers fetch/parse failures only.  A failure of the persistence write itself
PROPAGATES (raises) instead of being rescued — if the raw text were lost
silently, every downstream signal legitimately derived from it would be
mis-tiered as ``unverified`` (the fabrication label), which is a worse
outcome than failing the target loudly.  See docs/data-flow.md §9i.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from app.ids import new_id
from app.tools.log_step import log_step
from app.write_gate import commit as write_gate_commit


@dataclass
class NormalizedSource:
    # Every acquisition tool (this static fetcher, the future Playwright
    # dynamic scraper, and the Tavily web-search tool) MUST produce this
    # exact shape.  Downstream consumers (normalize_sources, summarize_company,
    # score_lead) branch on source_priority / source_confidence, not on which
    # tool produced the source — so adding a new tool never requires
    # downstream changes.

    source_type: str
    # Human-readable label for this source category (e.g. "company_website"
    # for a direct fetch, "web_search" for a Tavily result).  Used only for
    # trace readability; no pipeline logic depends on this string.

    source_url: str
    # The exact URL this source was fetched from.  For the static path this is
    # ``https://{domain}``; for other tools it may be a search-result URL or
    # a specific sub-page.

    extracted_text: str
    # The full plain-text content extracted from the page — no HTML, no
    # scripts, no styles, just prose.  This is what downstream summarization
    # and scoring tools consume.

    extracted_at: str
    # ISO-8601 UTC timestamp of when the fetch completed.  Used for
    # freshness checks downstream (e.g. "is this source older than N days?").

    source_confidence: float
    # 0.0–1.0 confidence that this source's content is actually relevant and
    # trustworthy.  Static fetch of the company's own site gets 0.8 (generally
    # reliable but self-reported — not independently verified).  Search results
    # would get lower confidence; direct data feeds would get higher.

    source_priority: int
    # Lower = preferred.  Priority 1 (static company site) is the primary
    # choice; priority 2–3 are fallbacks (search results, third-party data).
    # Downstream tools use this to order sources for summarization.

    extraction_method: str
    # How the text was obtained: "static" for this tool, "dynamic" for
    # Playwright, "search" for Tavily.  Purely diagnostic — no pipeline
    # logic branches on this, but it helps an operator debugging a run
    # understand why a source looks a certain way.


def _fetch_static_page(url: str) -> str:
    # Thin isolation wrapper around the network call so tests can mock this
    # function directly (see tests/test_fetch_sources.py's ``patch`` calls)
    # without hitting the real network.  Keeping it separate from
    # ``_extract_text`` also means the caller's ``try/except`` can distinguish
    # a network failure from a parsing failure — though both are rescued, the
    # logged ``error_type`` will tell an operator which layer broke.
    resp = requests.get(url, timeout=10)
    # raise_for_status() turns every non-2xx response (403, 404, 500, …) into
    # a Python HTTPError so the caller's broad ``except Exception`` catches it
    # and logs it — rather than silently returning error-page HTML as if it
    # were legitimate company content.
    resp.raise_for_status()
    return resp.text


def _extract_text(html: str) -> str:
    # Parse the raw HTML into a BeautifulSoup tree so we can strip non-content
    # elements before text extraction.
    soup = BeautifulSoup(html, "html.parser")
    # <script> and <style> tags contain JS/CSS code, not human-readable page
    # content.  If we left them in, ``get_text()`` would dump minified JS and
    # CSS selectors into the extracted text, polluting the summary downstream.
    for tag in soup(["script", "style"]):
        tag.decompose()
    # ``get_text(separator=" ")`` joins text nodes with a single space.
    # ``.split()`` breaks on any whitespace and ``" ".join(...)`` re-joins
    # with single spaces — this collapses all the whitespace/newlines that
    # HTML's indentation and block-structure produce into clean, contiguous
    # prose that downstream summarization tools can consume directly.
    return " ".join(soup.get_text(separator=" ").split())


# The source_type value ResearchBookkeepingNode (app/agents/phase1.py) uses
# when it persists the research agent's consolidated findings.  It is a
# module-level constant — not a string literal at the call site — because
# THREE modules agree on it: phase1.py WRITES rows with it, detect_signals
# EXCLUDES these rows when loading raw source texts, and this module's
# persist_source_row PERSISTS them.  A renamed literal at any one site would
# silently break the tier check (findings rows would start counting as raw
# sources), so the name is shared.  The value marks a row as "agent prose,
# not a fetched page" — the load-bearing distinction ticket B2b's three-way
# verdict is built on.
FINDINGS_SOURCE_TYPE = "research_findings"


def persist_source_row(
    conn,
    *,
    source_id: str,
    run_id: str,
    target_id: str,
    step_id: str,
    source_type: str,
    source_url: str | None,
    extracted_text: str,
    source_confidence: float | None,
    source_priority: int | None,
    extraction_method: str,
    agent_id: str = "system",
) -> str:
    """Persist ONE evidence row to the ``sources`` table through the write gate.

    This is THE single write path for sources rows (ticket B2b, Golden Rule:
    core-table writes go through write_gate.commit, never a raw
    conn.execute).  Two call sites share it — fetch_sources on every
    successful fetch (raw page text, the ``source`` tier's ground truth) and
    ResearchBookkeepingNode on the research agent's usable findings (the
    ``findings`` tier's post-run copy) — so the INSERT SQL and the audit
    payload shape exist in exactly one place and cannot drift apart.

    DELIBERATE CONTRACT: this function does NOT catch exceptions.  A
    persistence failure must surface loudly (see the module docstring) —
    rescuing it here would let the pipeline continue with evidence missing
    from storage, and detect_signals would then mark signals derived from
    that text ``unverified`` (fabrication) when they are actually the
    strongest kind of evidence.  Failing the target is the honest outcome.

    ``extracted_at`` is written with datetime('now') rather than the
    dataclass's ISO string on purpose: in this pipeline the row is persisted
    in the same instant the text was extracted (the dataclass travels only
    in-memory for milliseconds), and datetime('now') keeps every timestamp
    column in the table byte-identical in format to the rest of the schema
    (see app/db.py's _PG_DATETIME_NOW note).  The column stays distinct from
    created_at because a future acquisition tool may extract and persist
    asynchronously.
    """
    # The audit payload mirrors the row's data columns one-for-one, built
    # from the caller's values so the write_log row is self-describing
    # without a join.  ids/attribution are separate params (the same split
    # detect_signals uses for insert_signal): the model-independent fields
    # are never part of a payload a caller could accidentally fabricate.
    payload = {
        "source_type": source_type,
        "source_url": source_url,
        "extracted_text": extracted_text,
        "source_confidence": source_confidence,
        "source_priority": source_priority,
        "extraction_method": extraction_method,
    }
    # The write goes through the gate — the gate checks the action allowlist,
    # the actor, and the agent's registered capabilities, then writes the
    # sources row and its write_log audit row in ONE transaction.
    write_gate_commit(
        conn,
        action="insert_source",  # Registered in KNOWN_ACTIONS by ticket B2b.
        table_name="sources",
        record_id=source_id,  # the sources row's PK doubles as the audit row's record_id
        payload=payload,
        run_id=run_id,
        step_id=step_id,  # links the evidence row to the step that produced it in the trace
        actor="system",  # deterministic pipeline code — passes the actor allowlist
        agent_id=agent_id,  # both call sites are deterministic code today; the param exists for a future agent writer
        sql="""
            INSERT INTO sources
                (source_id, run_id, target_id, source_type, source_url,
                 extracted_text, extracted_at, source_confidence,
                 source_priority, extraction_method, created_at)
            VALUES (?,?,?,?,?,?,datetime('now'),?,?,?,datetime('now'))
        """,
        params=(
            source_id, run_id, target_id, source_type, source_url,
            extracted_text, source_confidence, source_priority, extraction_method,
        ),
    )
    # Return the row's PK so the caller can link its step log to the evidence
    # (fetch_sources puts it in the step row's output_data).
    return source_id


def fetch_sources(conn, *, domain: str, target_id: str, run_id: str, step_id: str) -> list[NormalizedSource]:
    # Always assume HTTPS for Phase 1 — no HTTP fallback yet.  The vast
    # majority of company websites redirect HTTP→HTTPS anyway, and adding
    # a fallback-with-retry adds complexity this early stage doesn't need.
    url = f"https://{domain}"
    sources: list[NormalizedSource] = []
    try:
        # 1. Fetch the raw HTML over the network — may raise on timeout,
        #    DNS failure, or non-2xx status.
        html = _fetch_static_page(url)
        # 2. Strip the HTML down to plain prose text — no JS, no CSS, no
        #    DOM whitespace.  Runs only if _fetch_static_page succeeded.
        text = _extract_text(html)
    except Exception as exc:
        # Broad ``Exception`` catch — not only TimeoutError/PermissionError.
        # Per the function's never-raise-for-FETCH contract, ANY unexpected
        # fetch/parse error (ValueError from a mock, OSError from DNS,
        # HTTPError from a 5xx, etc.) must be rescued and logged, not
        # propagated.  The caller (normalize_sources) decides whether an
        # empty result list warrants routing the target to a failed state.
        # NOTE this rescue covers fetch/parse only: the persistence write
        # below sits OUTSIDE it, so an evidence-loss failure raises instead
        # of being swallowed (see the module docstring, ticket B2b).
        log_step(
            conn, run_id=run_id, step_id=step_id, target_id=target_id,
            tool_name="fetch_company_page",
            agent_id="system",  # Same deterministic principal as the success path.
            input_data={"domain": domain},
            # Log both the human-readable error message AND the exception's
            # Python class name.  Together they let an operator distinguish
            # "TimeoutError: timed out" (network) from "HTTPError: 403
            # Forbidden" (blocked) from "ValueError: unexpected" (bug) — all
            # without needing access to the original exception object.
            output_data={"error": str(exc), "error_type": type(exc).__name__},
            status="failed",
        )
        # Nothing was fetched, so there is nothing to persist — return the
        # empty list immediately; the failure row above is the whole audit
        # record for this attempt.
        return sources
    # 3. Build the normalization wrapper.  source_confidence=0.8 reflects
    #    that a company's own site is generally reliable but self-reported
    #    (not cross-verified against third-party data).  source_priority=1
    #    marks this as the primary/first-choice source type — downstream
    #    tools prefer lower numbers.  extraction_method="static" is a
    #    diagnostic label (vs. "dynamic"/"search" for future tools).
    source = NormalizedSource(
        source_type="company_website",
        source_url=url,
        extracted_text=text,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        source_confidence=0.8,
        source_priority=1,
        extraction_method="static",
    )
    # 4. Persist the raw text BEFORE anything downstream can consume it
    #    (ticket B2b): this text is the ground truth detect_signals will
    #    fact-check evidence_quote against, and it must be in the sources
    #    table by the time that check runs.  OUTSIDE the fetch rescue on
    #    purpose — see persist_source_row's contract.  The dataclass fields
    #    are passed explicitly (extracted_at is NOT: the persisted column is
    #    written as datetime('now') — persist time IS extract time here, see
    #    persist_source_row's docstring), so the sources table mirrors the
    #    dataclass one-for-one without an accidental **-spread.
    source_id = persist_source_row(
        conn,
        source_id=new_id("src"),  # "src" prefix makes sources ids self-describing in join output
        run_id=run_id,
        target_id=target_id,
        step_id=step_id,
        source_type=source.source_type,
        source_url=source.source_url,
        extracted_text=source.extracted_text,
        source_confidence=source.source_confidence,
        source_priority=source.source_priority,
        extraction_method=source.extraction_method,
    )
    # 5. Log the successful fetch.  output_data carries chars_extracted
    #    as a lightweight signal of how much content was actually pulled
    #    (so an operator scanning the trace log can spot an empty or
    #    near-empty page without digging through the full extracted text)
    #    plus persisted_source_id so the trace row links straight to the
    #    evidence row the ticket added.  The full text lives in the
    #    NormalizedSource object AND now in the sources table, not the log.
    log_step(
        conn, run_id=run_id, step_id=step_id, target_id=target_id,
        tool_name="fetch_company_page",
        agent_id="system",  # Deterministic fetch code — the registered system agent.
        input_data={"domain": domain},
        output_data={"chars_extracted": len(text), "persisted_source_id": source_id},
        status="success",
    )
    sources.append(source)
    # Return whatever sources were successfully collected.  An empty list is
    # a valid, non-exceptional outcome — it means every attempted fetch failed
    # and was individually logged, and it's the caller's job to decide whether
    # that warrants a state transition or a retry.
    return sources
