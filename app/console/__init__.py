"""
Read-only operator console (ticket A5a) — FastAPI + Jinja2, local only.

This package is the hosted surface for the hackathon demo: it makes the
pipeline's audit trail visible (targets, scores, ICP verdicts, signals,
policy decisions, state transitions and the steps trace log). It is
structurally read-only — see app/console/app.py for the guarantees and
tests/test_console.py for the tests that enforce them. Cloud Run deployment
is ticket A5b; the approval gate and kill switch are ticket B4, and neither
lives here.
"""
