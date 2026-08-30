# app/agents/__init__.py — ADK agent builders for the outbound harness.
#
# Task A4a replaced the LangGraph StateGraph (app/graph.py) with Google ADK
# agents.  Re-export the Phase 1 builder so callers can use
# `from app.agents import build_phase1_agent` as the stable public entry point.
from app.agents.phase1 import build_phase1_agent

__all__ = ["build_phase1_agent"]
