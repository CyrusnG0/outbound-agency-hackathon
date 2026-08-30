"""
Trace-log writer for the outbound-agency pipeline.

Every tool invocation in the pipeline (fetch_sources, summarize_company,
score_lead, etc.) calls log_step() to persist one row in the ``steps`` table
— the backbone of every run's audit trail. Per docs/tool-registry.md, a
pipeline run is not complete until every step it took is logged.

Important design note: log_step writes DIRECTLY to ``steps`` via
``conn.execute(...)`` — it does NOT route through the write gate.  This is
intentional: ``steps`` is the trace log itself, not a core business/audit
table the write gate governs (the same reasoning docs/gates.md §1.2 uses to
exclude an orchestrator's own bookkeeping tables — the LangGraph checkpointer
when that reasoning was written, ADK's session store since task A4a).  Keeping ``steps`` outside
the write-gated set means a step can always be logged even if the write gate
refuses the underlying business write — so failures are never silently
dropped from the trace.
"""

import json

# A lightweight type hint for status values, purely for reader clarity.
# The actual set of valid values ("success", "failed", "retried") is NOT
# enforced in Python — it is enforced by the ``steps`` table's CHECK
# constraint in SQLite. This alias just signals intent; it does not validate.
Status = str  # "success" | "failed" | "retried" — enforced by the steps table CHECK constraint


def log_step(
    conn,
    *,
    run_id: str,
    step_id: str,
    target_id: str | None,
    tool_name: str,
    agent_id: str,
    input_data: dict,
    output_data: dict | None = None,
    status: Status,
    model_call_hash: str | None = None,
) -> None:
    """Persist a trace step. Per docs/tool-registry.md, a run is not
    complete until every step is logged — call this after every tool
    invocation, success or failure.

    ``agent_id`` (required, plan task A3) records WHICH registered agent
    performed the step — the same id the write gate checks in
    agent_registry — so the trace log stays attributable once multiple
    agents write concurrently.
    """
    conn.execute(
        """
        INSERT INTO steps
            (step_id, run_id, target_id, tool_name, agent_id, input_json,
             output_json, model_call_hash, status, created_at)
        VALUES (?,?,?,?,?,?,?,?,?, datetime('now'))
        """,
        (
            step_id,
            run_id,
            target_id,
            tool_name,
            # Which registered agent performed this step — written verbatim;
            # the write gate is what enforces the id is registered for
            # core-table writes, while steps rows are the trace itself.
            agent_id,
            # Serialize the input dict to a JSON string for persistence.
            # Every pipeline tool receives its inputs as a Python dict; we
            # store it as TEXT so a human (or replay.py) can inspect exactly
            # what was passed in without reconstructing it from code.
            json.dumps(input_data),
            # output_data can legitimately be None — for example, a tool
            # that threw an exception before producing any result, or a best-
            # effort enrichment that found nothing.  We must map None to SQL
            # NULL (not the JSON string "null"), because the steps table
            # schema uses ``output_json TEXT`` with no NOT NULL constraint
            # — NULL means "no output was produced", which is semantically
            # different from a JSON null literal.
            json.dumps(output_data) if output_data is not None else None,
            model_call_hash,
            status,
        ),
    )
    # Committing directly here (not through the write gate's BEGIN IMMEDIATE
    # / rollback pattern) is deliberate: there is no paired audit-log write
    # to keep atomic with.  The write gate pairs every business write with a
    # write_log row inside one transaction; steps is the trace log itself and
    # does not need a corresponding write_log entry — it IS the log.  This
    # also guarantees a step can always be persisted even if the write gate
    # refuses the underlying business write (see module docstring).
    conn.commit()
