"""
Console authentication (ticket H11) — the one decision the console may not
outsource and the one secret it may read.

The console's display queries are SELECT-only by construction (see the module
docstring of app/console/app.py and tests/test_console.py), but three routes
are NOT read-only: the two review-approval POSTs and the kill-switch toggle.
Before H11 those were protected by nothing — the service was deployed to
Cloud Run with `--allow-unauthenticated`, and a well-formed anonymous POST
would have been accepted (the live probes all answered 422, FastAPI's
*validation* error, proving the request reached the handler).  This module is
the whole auth decision, kept pure so the console keeps its "no write path of
its own" character: it reads one environment variable and compares
credentials; it has NO database, NO SQL, and NO import of any other app.*
module (that last property is enforced by tests/test_console.py — this module
is on the console's import allowlist precisely because it is pure).

Fail-closed doctrine (CLAUDE.md §3):
  - no secret configured  -> 503, the console refuses to serve anything
  - ambiguous credential  -> 401, never 500
  - /_health              -> the ONE carve-out, because Cloud Run's health
                             check must work and that route touches no data.
                             Named /_health, NOT /healthz, because Google's
                             Cloud Run frontend intercepts the exact path
                             /healthz and it never reaches the container
                             (ticket H16)
  - NO disable/bypass flag of any kind.  The only way to run the console is
    with OUTBOUND_CONSOLE_API_KEY set.  A bypass flag is exactly the sort of
    thing that reaches production by accident, and this repo has already
    shipped one accident this week.
"""

import base64
import hmac
import os

from fastapi import HTTPException, Request


def console_auth_secret() -> str | None:
    """The console's API key, or None when unset/blank.

    Returns None for a missing OR empty/whitespace-only value: an empty
    secret must never authenticate anyone (fail closed).  Read per request
    (not at import time) so tests can repoint it via monkeypatch and Cloud
    Run can inject it without a code change — the same convention
    app/console/app.py's _db_target() uses for OUTBOUND_DB_TARGET.
    """
    value = os.environ.get("OUTBOUND_CONSOLE_API_KEY")
    if not value or not value.strip():
        # Unset, or set to a string that is only whitespace — both mean "no
        # secret configured" and must fail closed upstream (503), because
        # an empty/blank secret would otherwise authenticate nobody while
        # looking configured.
        return None
    # Normalise ONLY the CONFIGURED secret, never the submitted credential
    # (H15).  Secret stores and shell pipelines routinely append a trailing
    # newline — a bash herestring (`<<<`), `echo` without -n, a text editor's
    # final newline, `gcloud secrets versions add --data-file=-` — and
    # surrounding whitespace in a credential is never meaningful.  Stripping
    # here means the 64-char key the operator types still matches a stored
    # value that carries a trailing "\n": the exact H15 defect, where the
    # runbook's `<<<` appended one, Secret Manager stored 65 bytes, Cloud Run
    # injected them verbatim, and every login 401'd.  The value that must
    # stay STRICT is the client-submitted credential, which is compared
    # byte-for-byte AFTER this normalisation — stripping what the client
    # sends would let "key" and "key " both authenticate, widening the
    # accepted set for no benefit.  The stored secret is trusted input being
    # cleaned; the client's is untrusted input being checked.
    return value.strip()


def _constant_time_equal(left: str, right: str) -> bool:
    """Constant-time string comparison (hmac.compare_digest), never ``==``.

    ``==`` on strings short-circuits at the first differing byte, so the
    elapsed time leaks how much of the key matches — enough for an attacker
    to recover it byte by byte.  compare_digest takes time proportional to
    the longer input regardless of content, so a wrong guess is
    indistinguishable from a right one by timing.

    Both sides are UTF-8 encoded to bytes first: compare_digest accepts str
    only when it is ASCII-only, and a hostile client could send a non-ASCII
    header value that would otherwise raise TypeError -> 500.  Encoding
    makes every input comparable, so every wrong or malformed credential is
    a clean False -> 401, never a 500.
    """
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def require_operator(request: Request) -> None:
    """FastAPI dependency: refuse every request that is not the operator's.

    Wired GLOBALLY in create_app() (app/console/app.py), not per-route — a
    route added to this console in six months is protected automatically,
    while per-route decoration fails open for anything a future edit
    forgets.  This function is the entire decision; see the module docstring
    for the fail-closed rules.
    """
    # The ONE carve-out.  Cloud Run's health check must reach /_health
    # without credentials (ticket A5b), and that route touches no database
    # and returns no data, so allowing it leaks nothing.  Anything else with
    # this exact path is still protected — there is no wildcard here.
    # The path is /_health, not /healthz (ticket H16): Google's Cloud Run
    # frontend intercepts the exact path /healthz and it never reaches the
    # container, so carving out /healthz would carve out a route that is
    # unreachable in production.
    if request.url.path == "/_health":
        return

    secret = console_auth_secret()
    if secret is None:
        # Fail closed: with no secret configured the console serves nothing.
        # A deployed-but-misconfigured console must refuse loudly (503)
        # rather than silently run unauthenticated — the exact hole H11
        # exists to close.
        raise HTTPException(
            status_code=503,
            detail="OUTBOUND_CONSOLE_API_KEY is not set — refusing to run the "
            "console without authentication.",
        )

    # Credential 1: the documented header (docs/api.md §1), for curl/JSON
    # clients.  A present-but-empty header is just a wrong key — it falls
    # through to the 401 below, never authenticates.
    header_key = request.headers.get("X-Internal-API-Key")
    if header_key is not None and _constant_time_equal(header_key, secret):
        return

    # Credential 2: HTTP Basic auth — a browser cannot set a custom header,
    # so without this the HTML console is unusable.  The username must be
    # exactly "operator" (the ticket's contract); the password is the same
    # secret, compared in constant time.
    auth_header = request.headers.get("Authorization")
    if auth_header is not None:
        try:
            # Split "Basic <base64>" — partition on the first space so a
            # malformed header with extra spaces still parses to a scheme
            # and some credentials rather than crashing.
            scheme, _, credentials = auth_header.partition(" ")
            if scheme.lower() == "basic" and credentials:
                # validate=False discards non-base64 characters instead of
                # raising, so even "!!!!" decodes (to empty bytes) and is
                # safely rejected by the compare below rather than 500ing.
                decoded = base64.b64decode(credentials, validate=False).decode("utf-8")
                # partition on the FIRST colon — a password containing a
                # colon is a legal basic-auth value, and the username is
                # fixed to "operator" so only the first field matters.
                username, _, password = decoded.partition(":")
                if username == "operator" and _constant_time_equal(password, secret):
                    return
        except Exception:
            # Malformed Authorization (bad base64 padding, non-UTF-8
            # payload, ...) — a 401, never a 500.  Decode defensively; every
            # ambiguous case denies (fail closed).
            pass

    # No valid credential was supplied — 401 with a WWW-Authenticate
    # challenge so a browser shows a login prompt.  The challenge is also
    # what makes the 401 useful to an API client, which can read it to learn
    # which scheme is accepted.
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": 'Basic realm="outbound-console"'},
    )
