# A5b — container image for the read-only operator console
# (app/console/app.py).
#
# Two contracts this file must keep holding:
# - `pip install .` / `pip install -e .` CANNOT work here: pyproject.toml
#   deliberately has no [build-system] section (docs/current_status.md — a
#   documented non-fix, not an oversight). Always install from
#   requirements-console.txt.
# - The image must contain ONLY the console's runtime closure. The console
#   imports fastapi/jinja2/pydantic/app.db and never llm/agents/tools; the
#   dependency list is enforced by tests/test_deploy_artifacts.py.

# Base image: python 3.13 satisfies the repo's requires-python >=3.11 and
# is a current stable; slim keeps the image small. Everything the console
# needs installs as prebuilt wheels (pg8000, cloud-sql-python-connector,
# uvicorn[standard], pydantic), so no compiler or distro build tools are
# required.
FROM python:3.13-slim

# Unbuffered stdout/stderr: Cloud Run captures container stdout as Cloud
# Logging entries, and Python's default block buffering would hold log
# lines in a buffer (indefinitely, for a quiet long-lived process), so the
# audit trail would vanish exactly when the container is killed.
# CLAUDE.md §3: never skip logs.
ENV PYTHONUNBUFFERED=1

# Every later command runs from /app, and the app package is copied to
# /app/app below, so `uvicorn app.console.app:app` resolves.
WORKDIR /app

# Dependency layer FIRST, copied alone: pip install is the slow step, and
# Docker caches layers by file content — requirements change rarely, code
# changes constantly, so installing before any app code is copied keeps
# code-only rebuilds cheap (the install layer is reused).
COPY requirements-console.txt ./
# --no-cache-dir keeps pip's wheel cache (tens of MB) out of the layer;
# the image never runs pip again, so the cache has zero runtime value.
RUN pip install --no-cache-dir -r requirements-console.txt

# Only what the console needs at runtime: the app/ package, which includes
# app/console/templates/ — the Jinja2 templates are resolved relative to
# app/console/app.py, so they must ship inside the package. tests/, docs/
# and data/ are never copied: the image contains no tests, no docs, no
# database, and (via .dockerignore) no .env keys.
COPY app/ ./app/
# config/ ships too (ticket B4a): app/kill_switch.py resolves
# config/kill_switch.json relative to the REPO ROOT, and its reader FAILS
# CLOSED — an image without the file would read the switch as engaged and
# halt every pipeline entry on boot.  config/ holds no secrets (model
# aliases and offer YAMLs only; .env* is already excluded by
# .dockerignore), so copying it is safe, and the committed file has
# enabled=false so a normal run is unaffected.
COPY config/ ./config/

# Non-root runtime user — defense in depth, not convenience. The console's own
# module issues only SELECTs (tests/test_console.py), but it has three write
# routes and, since H11, an auth layer — so "read-only" was never the reason
# this line exists (ticket H12 removed that stale claim). If a route bug ever
# allowed code execution, a root process would turn it into root on the
# container filesystem. An unprivileged user caps that blast radius, and
# uid 10001 avoids colliding with host uid ranges under Cloud Run.
RUN useradd --create-home --uid 10001 console
# USER fires for the CMD below (and any later RUN): the server process
# itself never runs as root.
USER console

# The serving command.
# --host 0.0.0.0 is mandatory: Cloud Run routes external traffic to $PORT
# on all interfaces; uvicorn's default 127.0.0.1 bind is unreachable from
# the router, so the revision would fail its health check and never serve
# traffic.
# Shell form (explicit sh -c) is mandatory for the port: Cloud Run injects
# $PORT at container start, and ${PORT:-8080} must be expanded by a shell
# AT RUNTIME. A pure exec-form array (CMD ["uvicorn", "--port",
# "${PORT:-8080}"]) would hand uvicorn the literal string "${PORT:-8080}"
# (exec form has no shell, so no expansion) and uvicorn would crash parsing
# it. The 8080 fallback exists only for local `docker run` without -e PORT;
# in Cloud Run $PORT is always set.
# `exec` replaces sh with uvicorn so uvicorn is PID 1 and receives Cloud
# Run's SIGTERM for graceful shutdown.
# One process, no --workers and no --reload: Cloud Run scales by instances,
# not in-process workers, and --reload belongs in a dev shell, not in a
# production container.
CMD ["sh", "-c", "exec uvicorn app.console.app:app --host 0.0.0.0 --port \"${PORT:-8080}\""]
