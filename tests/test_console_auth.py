"""
Tests for the console authentication gate (ticket H11).

The live defect this file pins: the deployed console had NO auth at all — a
well-formed anonymous POST to any of the three write routes
(`/kill-switch`, `/review/decision`, `/review/{id}/decision`) would have been
accepted; the live probes got 422, FastAPI's *validation* error, which proves
the request reached the handler and was refused only for payload shape.

The single most important test here is therefore:

    an unauthenticated POST to each write route returns 401, NOT 422.

Auth must fire BEFORE body validation.  If you see a 422, the dependency is
not global and the fix is wrong.  The rest of the file pins the fail-closed
doctrine: 503 with no secret, 401 for every wrong/malformed credential, the
/_health carve-out, and two structural guards (every route is covered; the
auth module stays pure and reads exactly one env var).

The module-level `app` (app/console/app.py) reads OUTBOUND_CONSOLE_API_KEY
per request, so monkeypatch controls the secret without rebuilding the app.
Auth fires before any database access, so the 401/503 tests need no DB; the
"valid credential" tests use a minimal seeded DB so the handler can actually
run.
"""

import ast
import base64
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.agents_registry import seed_agent_registry
from app.console.app import app
from app.console.auth import PUBLIC_PATHS, console_auth_secret, require_operator
from app.db import apply_schema, connect

# The secret the "valid credential" tests use.  Chosen to be distinct from any
# real key and long enough that a constant-time comparison is meaningful.
TEST_SECRET = "test-console-secret-123"

# The three write routes — the routes that, before H11, were anonymously
# reachable with a well-formed body.  The paths use tgt_1 as a stand-in; auth
# fires before the handler, so the target's existence is irrelevant.
WRITE_ROUTES = [
    ("post", "/kill-switch"),
    ("post", "/review/decision"),
    ("post", "/review/tgt_1/decision"),
]

# The read routes — every GET the console serves.  Auth protects these too.
READ_ROUTES = [
    ("get", "/"),
    ("get", "/targets/tgt_1"),
    ("get", "/api/targets/tgt_1"),
    ("get", "/review/queue"),
    ("get", "/api/review/queue"),
    ("get", "/review/tgt_1"),
    # /demo is NOT here: it moved to the PUBLIC_PATHS carve-out (the
    # static-snapshot ticket) and is tested separately as a public route.
]

# ── The public carve-out set (static-snapshot ticket) ─────────────────────────
# The routes reachable with ZERO credentials and ZERO database connection.
# This mirrors app/console/auth.py's PUBLIC_PATHS exactly — /rules, /demo,
# /test-run and its six fixed sub-paths — so a drift between this test list
# and the auth constant is a deliberate, visible edit.
PUBLIC_ROUTES = [
    "/rules",
    "/demo",
    "/test-run",
    "/test-run/mindnlife",
    "/test-run/psychotherapy-counselling-clinic",
    "/test-run/focus2-intelligent-therapy",
    "/test-run/solacetree-counselling-limited",
    "/test-run/run",
    "/test-run/run/steps.json",
]

# Every file the static routes serve off disk (app/console/app.py's
# _serve_static_snapshot).  The snapshot fixture writes one minimal file per
# name so the 200 path is exercised — never just the 503 fallback.
SNAPSHOT_FILES = [
    "demo.html",
    "test_run_index.html",
    "test_run_target_mindnlife.html",
    "test_run_target_psychotherapy-counselling-clinic.html",
    "test_run_target_focus2-intelligent-therapy.html",
    "test_run_target_solacetree-counselling-limited.html",
    "test_run_run.html",
    "test_run_steps.json",
]

# The HTML routes in the public set — the JSON mirror (steps.json) is not
# HTML and is excluded from the no-action-surface content checks.
PUBLIC_HTML_ROUTES = [
    "/rules",
    "/demo",
    "/test-run",
    "/test-run/mindnlife",
    "/test-run/psychotherapy-counselling-clinic",
    "/test-run/focus2-intelligent-therapy",
    "/test-run/solacetree-counselling-limited",
    "/test-run/run",
]


def _header_key() -> dict[str, str]:
    # The documented credential (docs/api.md §1): X-Internal-API-Key.
    return {"X-Internal-API-Key": TEST_SECRET}


def _basic_auth(username: str, password: str) -> dict[str, str]:
    # HTTP Basic credential: base64("username:password").
    raw = f"{username}:{password}".encode("utf-8")
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}


@pytest.fixture
def console_db(tmp_path):
    """A minimal valid console database (schema + agent registry) so that
    AUTHENTICATED requests can reach the handlers and return 200 (empty
    tables render fine) instead of a 500 from a missing schema.  The auth
    layer fires before any database access, so the unauthenticated tests need
    no database at all."""
    path = str(tmp_path / "auth_test.db")
    conn = connect(path)
    apply_schema(conn)
    seed_agent_registry(conn, run_id="r0", step_id="s0")
    conn.close()
    return path


@pytest.fixture
def client(console_db, monkeypatch):
    """A TestClient with a valid secret AND a valid database — the only
    fixture whose requests can actually run a handler.  Used by the
    "valid credential is not 401" tests."""
    monkeypatch.setenv("OUTBOUND_DB_TARGET", console_db)
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", TEST_SECRET)
    return TestClient(app)


@pytest.fixture
def authd_client(monkeypatch, tmp_path):
    """A TestClient with the secret CONFIGURED but no database.  Enough for
    every test whose assertion is about the auth layer firing (401), which
    happens before any database access.  The DB target points at a
    non-existent path as defense in depth — even a bug that reached the
    handler could never touch the operator's real database."""
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", TEST_SECRET)
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    return TestClient(app)


@pytest.fixture
def no_secret_client(monkeypatch, tmp_path):
    """A TestClient with NO secret configured (the env var is deleted) —
    every route must fail closed with 503."""
    monkeypatch.delenv("OUTBOUND_CONSOLE_API_KEY", raising=False)
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    return TestClient(app)


@pytest.fixture
def snapshot_client(monkeypatch, tmp_path):
    """A TestClient with NO secret, NO database, and a tmp snapshots directory
    holding minimal fixture files for every static route.

    This is the fixture that proves the PUBLIC carve-out end to end: the
    routes reachable with ZERO credentials must ALSO open ZERO database
    connections, so the client is deliberately configured with
    OUTBOUND_CONSOLE_API_KEY deleted and OUTBOUND_DB_TARGET pointed at a path
    whose parent directory does not exist (any connect attempt — even
    sqlite's implicit file creation — would fail).  The routes serve the
    fixture files from the tmp snapshots dir (monkeypatched over
    app.console.app._SNAPSHOTS_DIR) so the 200 path is genuinely exercised,
    not just the 503 fallback.  /rules renders a template and needs no
    snapshot file; it is covered here too for the same public/no-DB proof.
    """
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    for name in SNAPSHOT_FILES:
        # Minimal content — deliberately NO <form>, no /review/decision, no
        # /kill-switch — so the no-action-surface content checks can run
        # against these same fixtures.
        (snap_dir / name).write_text(
            f"<html><body><h1>FROZEN-{name}</h1></body></html>", encoding="utf-8"
        )
    monkeypatch.setattr("app.console.app._SNAPSHOTS_DIR", snap_dir)
    monkeypatch.delenv("OUTBOUND_CONSOLE_API_KEY", raising=False)
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    monkeypatch.delenv("OUTBOUND_REPLAY_MODE", raising=False)
    return TestClient(app)


# ── The H11 regression, first and loudest ─────────────────────────────────────


@pytest.mark.parametrize("method,path", WRITE_ROUTES)
def test_unauth_write_route_returns_401_not_422(authd_client, method, path):
    """THE H11 REGRESSION.  Before the fix, an anonymous POST to any write
    route reached the handler and came back 422 (FastAPI validation) — the
    request was refused for payload shape, not identity.  The fix wires auth
    GLOBALLY so it fires BEFORE body validation, so the same anonymous POST
    must now be 401.  Asserting != 422 explicitly names the regression: if
    the dependency stops being global, an empty-body POST starts 422ing again
    and this test fails."""
    resp = getattr(authd_client, method)(path)
    assert resp.status_code == 401
    assert resp.status_code != 422, (
        f"anonymous {method.upper()} {path} got 422 — the auth dependency is "
        f"not firing before body validation; the fix is not global (H11)"
    )


@pytest.mark.parametrize("method,path", READ_ROUTES)
def test_unauth_read_route_returns_401(authd_client, method, path):
    """Every read route is equally protected — the console is not
    "read-only means public"; the display data is the operator's pipeline
    audit trail."""
    resp = getattr(authd_client, method)(path)
    assert resp.status_code == 401


# ── The public carve-out behaviour (static-snapshot ticket) ───────────────────
# The routes in PUBLIC_PATHS are reachable with ZERO credentials and ZERO
# database connection.  The snapshot_client fixture proves both halves at
# once: it sends no credential (OUTBOUND_CONSOLE_API_KEY deleted) and its DB
# target points at a nonexistent parent dir (any connect attempt — even
# sqlite's implicit file creation — would fail).  These tests pin that the
# public set stays public, stays action-free, stays GET-only, and — the
# regression half — that EVERYTHING ELSE stays behind the gate.

@pytest.mark.parametrize("path", PUBLIC_ROUTES)
def test_public_routes_reachable_with_zero_credentials(snapshot_client, path):
    """Every PUBLIC_PATHS route answers 200 to an anonymous, DB-less request.

    The snapshot_client fixture is the whole proof: no credential is sent
    (OUTBOUND_CONSOLE_API_KEY is deleted) and the DB target points at a
    nonexistent parent directory, so a route that opened a connection would
    fail even sqlite's implicit file creation.  A 200 here means the request
    passed the PUBLIC_PATHS carve-out AND never touched a database."""
    resp = snapshot_client.get(path)
    # The 200-with-no-DB proof is built into the fixture (see its docstring);
    # assert it explicitly per path so the public set cannot silently regress
    # to a 503 (missing secret) or a 500 (DB connect attempt).
    assert resp.status_code == 200


@pytest.mark.parametrize("path", PUBLIC_HTML_ROUTES)
def test_public_html_routes_expose_no_action_surface(snapshot_client, path):
    """A zero-credential page must not reach an action: no <form>, no
    review-decision POST URL, no kill-switch POST URL.

    The fixture files are deliberately free of these strings (see the
    snapshot_client fixture), so this is a regression guard for FUTURE edits
    to the real templates / frozen files — the moment someone adds a decision
    or toggle form to a public page, it fails here."""
    resp = snapshot_client.get(path)
    assert resp.status_code == 200
    body = resp.text
    # The literal substrings, exactly as the real review/decision and
    # kill-switch routes are spelled — a public page must not even link to
    # them, let alone host a form that POSTs to them.
    assert "<form" not in body
    assert "/review/decision" not in body
    assert "/kill-switch" not in body


def test_public_paths_map_to_get_only_routes():
    """Every path in PUBLIC_PATHS names a route registered as GET-only.

    A public route that also accepted POST/PUT/DELETE would be an
    unauthenticated mutation vector — the exact class H11 closes.  This walks
    app.routes and asserts that no path in the public set accepts any method
    outside {GET}."""
    # Build {path: set-of-HTTP-methods} from the app's routes.  APIRoute
    # carries .methods as a set; route types without .methods (none today)
    # are skipped here — a future Mount would fail the /_health-coverage
    # guard (test_every_route_except_health_is_covered_by_global_auth)
    # instead, which is the right place for that failure.
    methods_by_path: dict[str, set[str]] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is None or methods is None:
            continue
        methods_by_path.setdefault(path, set()).update(methods)
    # Every public path must exist AND accept nothing but GET.
    for path in PUBLIC_PATHS:
        assert path in methods_by_path, (
            f"{path!r} is in PUBLIC_PATHS but no console route is registered "
            f"at that exact path"
        )
        assert methods_by_path[path] <= {"GET"}, (
            f"{path!r} is public but accepts {methods_by_path[path]} — public "
            f"routes must be GET-only (H11)"
        )


def test_routes_outside_public_paths_still_require_auth(authd_client):
    """The carve-out is EXACT: everything NOT in PUBLIC_PATHS stays 401.

    A public route that grew a sibling (a typo'd prefix, a new /test-run-*
    path) would silently widen the carve-out; this pins that the ordinary
    routes the console serves are still behind the gate.  Uses authd_client
    (a valid secret configured, no credential sent) — the same convention as
    test_unauth_read_route_returns_401 — so a 401 is the auth layer firing,
    not a 503 from a missing secret."""
    for path in ("/", "/review/queue", "/run/run_1", "/targets/tgt_1"):
        resp = authd_client.get(path)
        assert resp.status_code == 401


# ── Valid credentials are accepted ────────────────────────────────────────────


def test_valid_header_key_authenticates(client):
    """The documented X-Internal-API-Key header (docs/api.md §1) is accepted."""
    resp = client.get("/", headers=_header_key())
    assert resp.status_code != 401
    assert resp.status_code == 200


def test_valid_header_key_missing_target_is_404_not_401(client):
    """A valid key plus an unknown target is a legitimate 404, not a 401 —
    proof the request passed auth and reached the handler's own 404 path."""
    resp = client.get("/targets/no_such_target", headers=_header_key())
    assert resp.status_code != 401
    assert resp.status_code == 404


def test_valid_basic_auth_authenticates(client):
    """HTTP Basic (username exactly 'operator') is accepted — the credential
    a browser can present, which is why it must exist for the HTML console."""
    resp = client.get("/", headers=_basic_auth("operator", TEST_SECRET))
    assert resp.status_code != 401
    assert resp.status_code == 200


# ── Wrong / malformed credentials are 401, never 500 ──────────────────────────


def test_wrong_header_key_is_401(authd_client):
    resp = authd_client.get("/", headers={"X-Internal-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_empty_header_key_is_401(authd_client):
    """A present-but-empty key is not a valid credential — it must not
    authenticate anyone."""
    resp = authd_client.get("/", headers={"X-Internal-API-Key": ""})
    assert resp.status_code == 401


def test_wrong_basic_username_is_401(authd_client):
    """The username must be exactly 'operator' — any other identity is
    rejected even with the correct secret as the password."""
    resp = authd_client.get("/", headers=_basic_auth("admin", TEST_SECRET))
    assert resp.status_code == 401


def test_wrong_basic_password_is_401(authd_client):
    resp = authd_client.get("/", headers=_basic_auth("operator", "wrong-password"))
    assert resp.status_code == 401


def test_empty_basic_password_is_401(authd_client):
    """base64('operator:') is a legal-looking Basic header whose password is
    empty — it must be rejected, never authenticate."""
    resp = authd_client.get("/", headers=_basic_auth("operator", ""))
    assert resp.status_code == 401


@pytest.mark.parametrize("auth_value", [
    "Basic !!!!",   # not valid base64 at all
    "Basic abc",    # valid chars, invalid padding — base64 decode raises
    "Bearer x",     # a different scheme entirely
    "Basic",        # the word 'Basic' with no credentials
])
def test_malformed_authorization_is_401_not_500(authd_client, auth_value):
    """Every malformed Authorization must be a clean 401, never a 500 — a
    hostile client must not be able to turn a malformed header into a
    server error (the 'decode defensively' rule)."""
    resp = authd_client.get("/", headers={"Authorization": auth_value})
    assert resp.status_code == 401
    assert resp.status_code != 500


def test_401_carries_www_authenticate_challenge(authd_client):
    """The 401 must carry WWW-Authenticate: Basic realm=... so a browser
    shows a login prompt instead of a bare error page — the header is what
    makes the HTML console usable with a credential a browser can send."""
    resp = authd_client.get("/", headers=_basic_auth("admin", TEST_SECRET))
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == 'Basic realm="outbound-console"'


# ── The H15 regression: the CONFIGURED secret may carry surrounding whitespace ─
# A bash herestring (`<<<`), `echo` without -n, a text editor's final newline,
# or `gcloud secrets versions add --data-file=-` all append a trailing "\n" to
# the stored secret.  Secret Manager stores those 65 bytes, Cloud Run injects
# them verbatim into OUTBOUND_CONSOLE_API_KEY, and — before H15 —
# console_auth_secret() returned the value UNstripped, so every comparison was
# against "<key>\n" and the clean key the operator typed never matched (401).
# There was no workaround: a newline is illegal in an HTTP header value, so
# the byte-exact secret could not even be sent.  The fix normalises ONLY the
# CONFIGURED secret; the SUBMITTED credential stays strict, compared
# byte-for-byte after the stored secret is cleaned.
#
# COVERAGE BOUNDARY — a real httpx/curl client CANNOT send a header value
# containing \n or \r (the transport refuses), so the negative
# newline-suffixed HEADER cases below only exist because TestClient's in-process
# ASGI transport is lenient.  They still pin the byte-for-byte comparison (401,
# not 200) — they do NOT imply an operator can send such a header as a
# workaround; they cannot.

# The secret these tests use; deliberately distinct from TEST_SECRET so a
# cross-test environment leak cannot accidentally mask a failure.
H15_SECRET = "h15-secret-key-456"


@pytest.mark.parametrize("configured_suffix", [
    "\n",   # the exact H15 case: a bash herestring stores key + "\n"
    " \n",
    "\r\n",
    "  ",   # leading/trailing spaces
    "\t",
])
def test_clean_key_authenticates_when_configured_secret_has_whitespace(
    console_db, monkeypatch, configured_suffix,
):
    """H15 REGRESSION.  With OUTBOUND_CONSOLE_API_KEY set to "<key>\n" — the
    secret as Secret Manager actually stores it after a herestring — the CLEAN
    key must authenticate via BOTH the header and HTTP Basic.  Before H15 the
    configured secret was returned unstripped, so the clean key was compared
    against "<key>\n" and never matched (401 on every login, with no workaround
    because a newline is illegal in an HTTP header value).  The fix strips the
    CONFIGURED secret only, so the clean key now matches.  The other suffixes
    (spaces, \r\n, tabs) behave the same way."""
    monkeypatch.setenv("OUTBOUND_DB_TARGET", console_db)
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", H15_SECRET + configured_suffix)
    client = TestClient(app)
    assert client.get("/", headers={"X-Internal-API-Key": H15_SECRET}).status_code == 200
    assert client.get("/", headers=_basic_auth("operator", H15_SECRET)).status_code == 200


def test_console_auth_secret_strips_configured_secret(monkeypatch):
    """Direct pin of the one-line behaviour change: console_auth_secret()
    returns the CONFIGURED secret with surrounding whitespace removed, so the
    value handed to require_operator is always the clean key regardless of how
    the secret was stored."""
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", H15_SECRET + "\n")
    assert console_auth_secret() == H15_SECRET
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", "  " + H15_SECRET + "  ")
    assert console_auth_secret() == H15_SECRET


@pytest.mark.parametrize("submitted_suffix", [" ", "\t", "\n", " \n", "\r\n"])
def test_submitted_credential_with_surrounding_whitespace_is_rejected(
    authd_client, submitted_suffix,
):
    """The OTHER half of H15 — the client's credential is NEVER normalised.
    Only the configured secret gets stripped; a clean secret is configured
    (authd_client) and a submitted credential carrying a trailing space or
    newline must be rejected 401.  If the submitted input were stripped too,
    "key " and "key" would both authenticate, widening the accepted set for no
    benefit.  This pins the exact line between the two halves of the fix."""
    resp = authd_client.get("/", headers={"X-Internal-API-Key": TEST_SECRET + submitted_suffix})
    assert resp.status_code == 401
    resp = authd_client.get("/", headers=_basic_auth("operator", TEST_SECRET + submitted_suffix))
    assert resp.status_code == 401


# ── Fail closed: no secret configured → 503 ───────────────────────────────────


@pytest.mark.parametrize("method,path", WRITE_ROUTES + READ_ROUTES)
def test_no_secret_fails_closed_503(no_secret_client, method, path):
    """With OUTBOUND_CONSOLE_API_KEY unset, EVERY route (/_health excepted) is
    503 — the console refuses to serve anything rather than run
    unauthenticated.  This is the fail-closed doctrine: a misconfigured
    deploy must be visibly broken, never silently open."""
    resp = getattr(no_secret_client, method)(path)
    assert resp.status_code == 503


@pytest.mark.parametrize("method,path", [("get", "/"), ("post", "/kill-switch")])
def test_whitespace_only_secret_fails_closed(monkeypatch, tmp_path, method, path):
    """F4: a secret that is only whitespace is 'not set'.  The fail-closed
    behaviour is implemented (console_auth_secret returns None for a blank
    value) but was UNCOVERED — deleting `or not value.strip()` from
    console_auth_secret passes every other test.  This pins it: with
    OUTBOUND_CONSOLE_API_KEY set to spaces, console_auth_secret() returns
    None and every protected route is 503, exactly as if the var were unset.
    One read route and one write route are asserted; the write route proves
    auth still fires before body validation (503, not 422)."""
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", "   ")
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    # The blank secret must be treated as unset at the source, not merely
    # rejected downstream.
    assert console_auth_secret() is None
    resp = getattr(TestClient(app), method)(path)
    assert resp.status_code == 503


# ── The /_health carve-out ────────────────────────────────────────────────────


def test_health_allowed_without_secret(no_secret_client):
    """Cloud Run's health check must work even with no secret configured
    (A5b): /_health is the ONE carve-out.  It touches no database and returns
    no data, so allowing it leaks nothing.  Named /_health, not /healthz,
    because Google's Cloud Run frontend intercepts the exact path /healthz
    before it reaches the container (ticket H16)."""
    assert no_secret_client.get("/_health").status_code == 200


def test_health_allowed_with_wrong_secret(monkeypatch, tmp_path):
    """The carve-out is path-based, not credential-based: even a request with
    a WRONG (or absent) secret reaches /_health."""
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", "wrong")
    monkeypatch.setenv("OUTBOUND_DB_TARGET", str(tmp_path / "no_such_dir" / "auth.db"))
    client = TestClient(app)
    assert client.get("/_health").status_code == 200


def test_health_allowed_with_right_secret(client):
    assert client.get("/_health").status_code == 200


# ── The /_health carve-out is EXACT (ticket H11, finding F5) ──────────────────
# The carve-out lives INSIDE require_operator as `request.url.path ==
# "/_health"`.  Widening it to `startswith("/_health")` (or the old
# `startswith("/health")`) would silently unauthenticate every future
# /_health* route (e.g. "/_health/details" showing DB stats) while every test
# stays green — nothing else today shares the prefix.  Timing of a real route
# is not the point: the comparison itself must be pinned, so we unit-call
# require_operator with a stub request whose path we control.  No real route
# is added to the app (the ticket's preference) — the stub is a
# faithful-enough stand-in because require_operator reads exactly two things
# off the request: url.path and headers.get(...).


class _StubUrl:
    # Minimal stand-in for starlette's URL: require_operator reads only
    # request.url.path, so a stub with a settable path pins the carve-out's
    # exactness without spinning up a server or registering a route.
    def __init__(self, path: str) -> None:
        self.path = path


class _StubHeaders:
    # Minimal stand-in for starlette's Headers: require_operator calls
    # request.headers.get(...) for the two credentials; a stub that returns
    # None (no credential present) exercises the challenge path (401) or the
    # fail-closed path (503) rather than ever authenticating.
    def get(self, key: str, default=None):
        return default


class _StubRequest:
    # The two attributes require_operator touches.  If require_operator ever
    # reads something else off the request, this stub breaks loudly and the
    # test forces the question — that is the point.
    def __init__(self, path: str) -> None:
        self.url = _StubUrl(path)
        self.headers = _StubHeaders()


@pytest.mark.parametrize("path", [
    # /health* never was the carve-out — still challenged.
    "/health",
    "/health/",
    "/health/details",
    # The OLD carve-out name is now an ordinary path — still challenged.  (In
    # production Google's edge intercepts it anyway, so it is doubly dead.)
    "/healthz",
    "/healthz/",
    "/healthz/extra",
    "/healthzz",
    "/healthz-extra",
    "/healthzfoo",
    # /_health-prefixed paths must NOT be carved out — the exact-match
    # regression: a startswith("/_health") widening would let these through.
    "/_health/",
    "/_health/details",
    "/_healthz",
    "/_health-extra",
    "/_healthfoo",
])
def test_health_carveout_is_exact_other_paths_are_challenged(path, monkeypatch):
    """F5: any path that merely shares the /_health (or /health) prefix must
    still be challenged.  With a valid secret configured and NO credential on
    the stub request, require_operator must raise 401 — proving the path was
    NOT carved out.  A `startswith("/_health")` widening would return early
    for every one of these and this test would fail (no HTTPException
    raised)."""
    monkeypatch.setenv("OUTBOUND_CONSOLE_API_KEY", TEST_SECRET)
    with pytest.raises(HTTPException) as exc_info:
        require_operator(_StubRequest(path))
    assert exc_info.value.status_code == 401


@pytest.mark.parametrize("path", [
    "/health",
    "/health/details",
    "/_health/details",
    "/_healthz",
])
def test_health_carveout_is_exact_other_paths_fail_closed_without_secret(path, monkeypatch):
    """F5, fail-closed half: with NO secret configured, a non-exact path must
    still reach the 503 branch — not be silently allowed by a widened
    carve-out.  A `startswith("/_health")` widening returns before the secret
    is even read, so this test would fail (no HTTPException raised)."""
    monkeypatch.delenv("OUTBOUND_CONSOLE_API_KEY", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        require_operator(_StubRequest(path))
    assert exc_info.value.status_code == 503


def test_health_carveout_is_exact_exact_path_allowed_without_secret(monkeypatch):
    """The carve-out is the ONE literal path: with no secret configured,
    exactly /_health returns None (allowed) while every other path 503s.
    This is the positive half of the exactness pin — the widening sabotage
    makes the negative tests above fail, and removing the carve-out entirely
    makes this one fail."""
    monkeypatch.delenv("OUTBOUND_CONSOLE_API_KEY", raising=False)
    assert require_operator(_StubRequest("/_health")) is None


def test_no_route_named_healthz_exists():
    """H16 regression: the app must NOT contain a route at /healthz.

    Google's Cloud Run frontend intercepts the exact, case-sensitive path
    /healthz and answers it itself (an HTML 404 with no `server: Google
    Frontend` header) before the request ever reaches the container —
    measured against the live service on 2026-08-28.  A /healthz route here
    would be dead code that looks alive: every test would pass against it
    locally while the deployed health check silently stopped working.  The
    health endpoint is /_health (see test_health_allowed_without_secret).  If
    someone "restores the conventional name", this guard fails."""
    for route in app.routes:
        assert getattr(route, "path", None) != "/healthz", (
            "a route at /healthz exists on the console app — Google's Cloud "
            "Run frontend intercepts that exact path and it never reaches "
            "the container (ticket H16), so the route would be dead code "
            "that looks alive. The health endpoint is /_health."
        )


# ── Structural guards ─────────────────────────────────────────────────────────


def test_every_route_except_health_is_covered_by_global_auth():
    """A test that fails when someone adds an unprotected route.  The auth
    dependency is wired GLOBALLY (FastAPI(dependencies=[Depends(...)])) in
    create_app(), so every APIRoute on the app carries it as a sub-dependency.
    Walking app.routes and asserting that property pins the global wiring:
    a future route added without the dependency (e.g. on a separate router,
    or by bypassing create_app) fails here.  /_health is skipped because it
    is the deliberate carve-out — the dependency still runs on it, it just
    allows it.

    F2: the check is INVERTED from the pre-H11-hardening version.  The old
    guard first filtered to routes with `.dependant` and only asserted on
    those — which silently skipped every route type that does NOT run the
    app-level dependencies, and a Mount is exactly that.  FastAPI's
    app-level dependencies protect Route objects; a `Mount("/x", sub_app)`
    delegates straight to the sub-app's ASGI callable and never runs the
    parent's dependencies, so a mounted sub-app (e.g. StaticFiles for CSS)
    would ship unauthenticated with a fully green suite.  Unknown route types
    must FAIL, never be skipped: every route on this app must carry
    `.dependant`, and every dependant-carrying route except /_health must
    run require_operator."""
    assert app.routes, "no routes found on the console app — did the app change shape?"
    for route in app.routes:
        # Route types without `.dependant` (Mount, WebSocketRoute, ...) do
        # NOT run the app-level dependencies.  A future Mount must wrap its
        # sub-app in its OWN require_operator — the message tells the next
        # developer exactly that instead of silently passing them.
        assert hasattr(route, "dependant"), (
            f"route {getattr(route, 'path', route)!r} (type "
            f"{type(route).__name__}) has no `.dependant` — the app-level "
            f"auth dependency does not run for it, so it would be served "
            f"unauthenticated (H11/F2). Wrap the sub-app in its own "
            f"require_operator instead of relying on the app-level "
            f"dependencies."
        )
        if route.path == "/_health":
            continue
        assert any(
            sub.call is require_operator for sub in route.dependant.dependencies
        ), (
            f"route {route.path!r} is not covered by the global auth "
            f"dependency (H11) — a route added to the console must be "
            f"protected automatically by the app-level dependencies"
        )


def _is_literal_str(node) -> bool:
    # True when the AST node is a plain string literal (`ast.Constant` with a
    # str value).  Only a literal key can be statically proven to name one
    # specific environment variable; any other expression is unprovable and
    # must be treated as a bypass.
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_environ_attr(node) -> bool:
    # True when the AST node is an Attribute named "environ" — covers
    # `os.environ` and (conservatively) any `X.environ`.  The base object is
    # NOT checked: in this tiny module the only .environ attribute is os's,
    # and a false positive (flagging an unrelated .environ) is the safe
    # direction.
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _is_compare_digest_call(node) -> bool:
    # True when the AST node is a call to hmac.compare_digest — matched as any
    # Attribute named "compare_digest" (`hmac.compare_digest`, `os.…`, etc.).
    # The base module is not pinned to "hmac": the only constant-time
    # comparison primitive this repo uses is compare_digest, and matching the
    # attribute name is what catches a swap for `==`.
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compare_digest"
    )


def _is_console_auth_secret_call(node) -> bool:
    # True when the AST node is a call to console_auth_secret() — the secret
    # source.  Matched as a bare Name call so an inline
    # `console_auth_secret() == x` in require_operator is treated exactly like
    # comparing the local `secret` variable.
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "console_auth_secret"
    )


def _node_refs_secret(node) -> bool:
    # True when the expression tree contains the secret VALUE: the local
    # variable `secret` (assigned from console_auth_secret()) or a direct
    # call to console_auth_secret().  Walking the subtree catches wrapped
    # forms too (e.g. `(secret)` or a ternary containing secret).
    if isinstance(node, ast.Name) and node.id == "secret":
        return True
    if _is_console_auth_secret_call(node):
        return True
    return any(_node_refs_secret(child) for child in ast.iter_child_nodes(node))


def test_auth_module_reads_only_the_documented_env_var():
    """The one safety valve this ticket refuses to allow: a disable/bypass
    flag (AUTH_DISABLED, DEBUG, a localhost exemption, ...) reaching
    production by accident.  This pins that app/console/auth.py's ONLY
    environment-variable read is OUTBOUND_CONSOLE_API_KEY — any other
    os.environ / os.getenv access fails the test, so a bypass flag cannot be
    added silently.

    F1 hardening: the pre-hardening guard only matched env-var reads whose
    KEY was an ast.Constant string, so a sabotage like
        _BYPASS_VAR_NAME = "AUTH_DISABLED"
        os.environ.get(_BYPASS_VAR_NAME)   # key is an ast.Name
    slipped through silently — the read was real and the guard never saw it.
    The guard now FAILS on any env read whose key is not the literal
    "OUTBOUND_CONSOLE_API_KEY".  A non-literal key cannot be statically
    proven safe, so fail closed: the sabotage above is caught because
    `_BYPASS_VAR_NAME` (an ast.Name) is not a literal."""
    auth_path = Path(__file__).resolve().parent.parent / "app" / "console" / "auth.py"
    assert auth_path.is_file(), "app/console/auth.py missing"
    tree = ast.parse(auth_path.read_text(encoding="utf-8"), filename=str(auth_path))

    # First pass: find local aliases of os.environ (`env = os.environ`) so a
    # read through the alias is caught too.  Only a direct assignment of an
    # environ Attribute to a single Name is tracked; a name later reassigned
    # to something else stays in the set (conservative — fail closed).
    env_aliases: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_environ_attr(node.value)
        ):
            env_aliases.add(node.targets[0].id)

    # Every environment-variable read: (lineno, key-expression, is_literal).
    # The key expression is recorded REGARDLESS of its shape — the bug in the
    # old guard was matching only Constant keys, which silently dropped every
    # other key form (Name, Attribute, f-string JoinedStr, ...) from the scan
    # instead of failing on it.
    env_reads: list[tuple[int, object, bool]] = []
    for node in ast.walk(tree):
        # os.environ[KEY] / env_alias[KEY] — a subscript read.
        if isinstance(node, ast.Subscript):
            if _is_environ_attr(node.value) or (
                isinstance(node.value, ast.Name) and node.value.id in env_aliases
            ):
                env_reads.append((node.lineno, node.slice, _is_literal_str(node.slice)))
        # os.environ.get(KEY, ...) / env_alias.get(KEY, ...) — a .get call.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        ):
            func_value = node.func.value
            if _is_environ_attr(func_value) or (
                isinstance(func_value, ast.Name) and func_value.id in env_aliases
            ):
                key = node.args[0] if node.args else None
                env_reads.append((node.lineno, key, _is_literal_str(key)))
        # os.getenv(KEY, ...) — a bare getenv call (`from os import getenv`).
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getenv"
        ):
            key = node.args[0] if node.args else None
            env_reads.append((node.lineno, key, _is_literal_str(key)))

    assert env_reads, (
        "no env-var read found in app/console/auth.py — the bypass guard "
        "expects exactly one (OUTBOUND_CONSOLE_API_KEY); did auth stop reading "
        "it?"
    )
    for lineno, key, is_literal in env_reads:
        # A non-literal key cannot be proven to name the one allowed variable
        # — fail closed, per the repo doctrine ("cannot statically prove this
        # is safe" is a FAILURE, never a skip).  This is the F1 fix: the old
        # guard never even saw these reads.
        assert is_literal, (
            f"app/console/auth.py line {lineno} reads an environment variable "
            f"with a NON-LITERAL key — its value cannot be statically proven "
            f"to be OUTBOUND_CONSOLE_API_KEY, and a disable/bypass flag must "
            f"never exist (H11/F1). Use the literal "
            f"'OUTBOUND_CONSOLE_API_KEY'."
        )
        # `key` is an ast.Constant here (is_literal guarantees it); pull out
        # the actual string to compare against the one allowed name.
        literal_key = key.value
        assert literal_key == "OUTBOUND_CONSOLE_API_KEY", (
            f"app/console/auth.py line {lineno} reads env var {literal_key!r} "
            f"— the ONLY env var the console auth may read is "
            f"OUTBOUND_CONSOLE_API_KEY; a disable/bypass flag must never "
            f"exist (H11)"
        )


# F1 COVERAGE BOUNDARY — what the env-var walk does and does not catch, so the
# next reader trusts it only as far as it goes (the same style as
# tests/test_console.py and tests/test_deploy_artifacts.py).
#
# CAUGHT:
#   - os.environ["KEY"], os.environ.get("KEY", ...), os.getenv("KEY", ...)
#     where KEY is ANY expression — literal or not.  A non-literal key FAILS
#     outright (cannot be statically proven to be OUTBOUND_CONSOLE_API_KEY).
#   - A read through a directly aliased name: `env = os.environ` then
#     env["KEY"] / env.get("KEY", ...).
#   - `X.environ` for ANY base X (the base object is not pinned to "os"), and
#     `X.environ.get(...)` / `X.environ[KEY]` — a false positive (flagging an
#     unrelated module's .environ) is the safe direction.
#   - A bare `from os import getenv` + `getenv("KEY", ...)` call.
#
# NOT caught (fail-open directions, recorded so they are a known boundary, not
# a silent hole):
#   - `from os import getenv as ge; ge("KEY")` — the call's func is a Name
#     other than "getenv", invisible to a walk that matches the literal id.
#     (Still caught if it goes through os.environ/os.getenv in the canonical
#     spellings.)
#   - An alias assigned through an intermediate expression rather than a
#     direct os.environ assignment — e.g. `env = _os_environ()` where
#     `_os_environ` returns os.environ.  Only direct assignments are tracked.
#   - A name assigned os.environ and later REASSIGNED to something else: the
#     name stays in the alias set, so a later `.get(...)` on the reassigned
#     value is a false positive (conservative — forces a deliberate edit).
#   - The guard proves the KEY is the literal string; it does not prove that
#     the value read is only used safely.  A bypass that reads
#     OUTBOUND_CONSOLE_API_KEY itself and inverts on it (e.g.
#     `os.environ.get("OUTBOUND_CONSOLE_API_KEY") == "DISABLE"`) is outside
#     this guard's scope — that would be a behavioural hole caught by review,
#     not by this static walk.


def test_secret_is_compared_only_in_constant_time():
    """F3: pin the constant-time comparison statically.

    Timing is not deterministically testable in CI, so a sabotage that swaps
    `hmac.compare_digest(left, right)` for `left == right` in
    _constant_time_equal — or compares the secret with a bare `==` in
    require_operator — passes every behavioural test.  Both are timing
    side-channels (``==`` short-circuits at the first differing byte, leaking
    how much of the key matched), so the CALL itself must be asserted, the
    same way F1's guard asserts the env-var read.

    Two halves:
    1. _constant_time_equal's body must call hmac.compare_digest.
    2. require_operator must compare the secret ONLY through that helper: no
       bare ==/!= where either side is the secret value, and no direct
       hmac.compare_digest call (the helper is the single choke point).

    Comparisons of NON-secret values (request.url.path == "/_health",
    scheme.lower() == "basic", username == "operator") stay allowed — none of
    those reference the secret, so the walk below never trips on them."""
    auth_path = Path(__file__).resolve().parent.parent / "app" / "console" / "auth.py"
    assert auth_path.is_file(), "app/console/auth.py missing"
    tree = ast.parse(auth_path.read_text(encoding="utf-8"), filename=str(auth_path))
    funcs = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    # Half 1 — the helper must actually be constant-time.  A `left == right`
    # swap is statically invisible to every behavioural test and is exactly
    # the F3 sabotage.
    helper = funcs.get("_constant_time_equal")
    assert helper is not None, (
        "app/console/auth.py lost _constant_time_equal — the constant-time "
        "comparison helper must exist (H11/F3)"
    )
    assert any(
        _is_compare_digest_call(node) for node in ast.walk(helper)
    ), (
        "app/console/auth.py _constant_time_equal does not call "
        "hmac.compare_digest — the constant-time comparison was replaced with "
        "a non-constant-time one (timing side-channel, H11/F3)"
    )

    # Half 2 — require_operator compares the secret only through the helper.
    require = funcs.get("require_operator")
    assert require is not None, (
        "app/console/auth.py lost require_operator — the auth dependency must "
        "exist (H11/F3)"
    )
    for node in ast.walk(require):
        # A bare ==/!= where either side references the secret value.
        if isinstance(node, ast.Compare):
            # ops and comparators are PARALLEL lists (a == b == c has
            # ops=[Eq,Eq], comparators=[b,c]); the left value is the running
            # first operand.  For the secret check, what matters is that a
            # value comparison exists AND that any operand (left or any
            # comparator) is the secret — so check every operand against any
            # Eq/NotEq op.  (The old zip() here was a bug: it stopped at the
            # shorter list, so the RIGHT-hand operand of a binary == was never
            # examined and the `header_key == secret` sabotage sailed past.)
            if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                for operand in [node.left, *node.comparators]:
                    if _node_refs_secret(operand):
                        raise AssertionError(
                            f"app/console/auth.py require_operator line "
                            f"{node.lineno} compares the secret with a bare "
                            f"==/!= — the secret must be compared ONLY through "
                            f"_constant_time_equal (H11/F3)"
                        )
        # A direct hmac.compare_digest bypass of the helper choke point.
        if _is_compare_digest_call(node):
            raise AssertionError(
                f"app/console/auth.py require_operator line {node.lineno} calls "
                f"hmac.compare_digest directly — the secret must be compared "
                f"only through _constant_time_equal, the single choke point "
                f"(H11/F3)"
            )


# F3 COVERAGE BOUNDARY — what the constant-time walk does and does not catch.
#
# CAUGHT:
#   - _constant_time_equal's body must contain a call to `X.compare_digest`
#     (any base; in this module the only one is hmac).  Swapping the helper's
#     body for `left == right` fails Half 1.
#   - require_operator must not compare the secret (the local `secret`
#     variable, or a direct console_auth_secret() call) with a bare ==/!=, and
#     must not call compare_digest directly — both fail Half 2.
#   - Comparisons of NON-secret values (request.url.path, scheme, username)
#     are untouched: none reference the secret, so the walk never trips.
#
# NOT caught (fail-open directions, recorded):
#   - `from hmac import compare_digest as cd; cd(...)` in require_operator —
#     the call's func is a Name other than compare_digest, invisible to the
#     attribute walk.  Still constant-time, so this is not a security hole,
#     but it bypasses the "single choke point" structure.
#   - The guard proves the CALL is compare_digest; it cannot prove the
#     arguments are the actual secret (that is behavioural, and covered by
#     the 401/503 tests).  A non-constant-time comparison hiding inside a
#     helper that still calls compare_digest elsewhere would evade Half 1.
#   - Timing itself is not measured — the guard pins the call, not the wall
#     clock, because timing is not deterministically testable in CI.
