# tests/test_static_snapshot_links.py — the "never dead-end into the
# credential wall" guarantee (found live, 2026-08-31: /test-run's target
# links redirected to /targets/{id}, which 401s with zero credentials).
#
# scripts/freeze_public_snapshot.py renders four templates
# (base.html/demo.html/targets.html/target_detail.html — run.html was
# already covered by its own static_steps_url mechanism) with
# static_mode=True so their body links and shared nav point at the frozen
# /test-run/* pages instead of the live, operator-only /targets/{id},
# /review/{id}, and /run/{id} routes. This file proves, by rendering each
# template directly the same way the freeze script does (no HTTP, no
# database — templates.env.get_template(...).render(**context), the same
# pattern test_console.py's demo.html tests already use), that:
#   1. static_mode substitutes every link that HAS a frozen equivalent;
#   2. a link with NO frozen equivalent (an operator-only screen, or a
#      run_id/target outside the frozen set) is OMITTED, never left
#      pointing at a live URL that will 401 a credential-less judge;
#   3. live mode (static_mode unset, exactly what every real console
#      route renders with today) is completely unchanged — this is a
#      regression guard as much as a bug-fix proof.
from app.console.app import templates

# ── Fixtures: minimal, hand-built contexts — enough for the link logic
# these templates gate behind {% if %}, not a full seeded database. Every
# other field these templates reference is either guarded by its own
# {% if %} (company/contact/signals) or unconditionally required
# (target/steps) and given a plausible minimal value below.

_TARGET = {
    "target_id": "tgt_test1", "state": "scored", "score": 80,
    "final_recommendation": "strong_fit", "offer_slug": "acme",
    "source": "csv", "created_at": "2026-08-31 00:00:00",
    "updated_at": "2026-08-31 00:00:00", "last_signal_refresh_at": None,
}

_FROZEN_RUN_ID = "run_frozen"
_OTHER_RUN_ID = "run_other"
_STEPS = [
    {
        "step_id": "step_1", "tool_name": "research", "status": "success",
        "agent_id": "system", "model_call_hash": None,
        "created_at": "2026-08-31 00:00:00", "input_json": "{}", "output_json": "{}",
        "run_id": _FROZEN_RUN_ID,
    },
    {
        "step_id": "step_2", "tool_name": "draft", "status": "success",
        "agent_id": "system", "model_call_hash": None,
        "created_at": "2026-08-31 00:01:00", "input_json": "{}", "output_json": "{}",
        "run_id": _OTHER_RUN_ID,
    },
]


def _render(name: str, **context) -> str:
    return templates.env.get_template(name).render(**context)


# ── targets.html — the exact bug this file exists because of ────────────────

def test_targets_html_static_mode_links_to_test_run_slug():
    """The reported bug: a showcase target's row must link to its frozen
    /test-run/<slug> page, never the live /targets/{id}."""
    html = _render(
        "targets.html",
        targets=[{**_TARGET, "company_name": "Acme"}],
        static_mode=True,
        static_target_slugs={"tgt_test1": "acme-inc"},
    )
    assert 'href="/test-run/acme-inc"' in html
    assert "/targets/tgt_test1" not in html


def test_targets_html_live_mode_unchanged():
    """Regression guard: the live console's own targets list (static_mode
    never passed there) must keep linking to /targets/{id} exactly as
    before this fix."""
    html = _render("targets.html", targets=[{**_TARGET, "company_name": "Acme"}])
    assert 'href="/targets/tgt_test1"' in html


# ── demo.html — three separate live links on one page ────────────────────────

def _demo_context(**overrides):
    base = {
        "showcase": [{
            "company_name": "Acme", "why": "test", "target_id": "tgt_test1",
            "state": "scored",
        }],
        "meeting": {"target_id": "tgt_test1", "company_name": "Acme",
                    "contact_name": None, "scheduled_at": "t", "duration_minutes": 30,
                    "reasoning": None},
        "meeting_draft": {"subject": "s", "revision_number": 1,
                           "created_at": "t", "body": "b", "footer": "f"},
        "static_mode": True,
        "static_target_slugs": {"tgt_test1": "acme-inc"},
    }
    base.update(overrides)
    return base


def test_demo_html_showcase_row_links_to_test_run_when_mapped():
    html = _render("demo.html", **_demo_context())
    assert 'href="/test-run/acme-inc"' in html
    assert "/targets/tgt_test1" not in html
    assert "/review/tgt_test1" not in html


def test_demo_html_showcase_row_prefers_audit_trail_even_when_awaiting_review():
    """A showcase target sitting at awaiting_review still gets the frozen
    audit-trail link in static mode -- there is no static review screen,
    but the audit trail is real and available, so it is strictly better
    than a dead end."""
    ctx = _demo_context()
    ctx["showcase"][0]["state"] = "awaiting_review"
    html = _render("demo.html", **ctx)
    assert 'href="/test-run/acme-inc"' in html
    assert "/review/tgt_test1" not in html


def test_demo_html_meeting_link_omitted_when_target_outside_showcase_set():
    """The 'most recent meeting' can belong to a target that isn't one of
    the four frozen showcase companies (a later batch booked a newer one)
    -- static mode must OMIT the link, never fall back to the live,
    credential-walled /review/{id}."""
    ctx = _demo_context()
    ctx["meeting"]["target_id"] = "tgt_not_in_showcase"
    ctx["meeting_draft"] = {**ctx["meeting_draft"]}
    html = _render("demo.html", **ctx)
    assert "/review/tgt_not_in_showcase" not in html
    assert "open the review screen" not in html


def test_demo_html_live_mode_unchanged():
    """Regression guard: the live /demo route (before it became a frozen
    file) rendered this exact template with static_mode unset -- kept here
    so the fallback path stays provably correct even though no live route
    calls it today."""
    ctx = _demo_context(static_mode=False, static_target_slugs={})
    html = _render("demo.html", **ctx)
    assert 'href="/targets/tgt_test1"' in html
    assert 'href="/review/tgt_test1"' in html  # the meeting section's live fallback


# ── target_detail.html — the trace-log run link ──────────────────────────────

def test_target_detail_static_mode_links_only_the_frozen_run():
    html = _render(
        "target_detail.html",
        target=_TARGET, company=None, contact=None, signals=None, steps=_STEPS,
        static_mode=True, static_target_slugs={}, static_frozen_run_id=_FROZEN_RUN_ID,
    )
    assert 'href="/test-run/run"' in html
    assert html.count("Watch this run live") == 1  # the OTHER run_id's link is omitted, not just its href changed
    assert f"/run/{_FROZEN_RUN_ID}" not in html
    assert f"/run/{_OTHER_RUN_ID}" not in html


def test_target_detail_live_mode_links_every_run():
    """Regression guard: the live /targets/{id} route (static_mode unset)
    must keep linking every distinct run_id, unchanged."""
    html = _render(
        "target_detail.html",
        target=_TARGET, company=None, contact=None, signals=None, steps=_STEPS,
    )
    assert f'href="/run/{_FROZEN_RUN_ID}"' in html
    assert f'href="/run/{_OTHER_RUN_ID}"' in html
    assert html.count("Watch this run live") == 2


# ── base.html nav (exercised through any child template) ────────────────────

def test_nav_static_mode_hides_review_queue_and_repoints_logo():
    html = _render("targets.html", targets=[], static_mode=True, static_target_slugs={})
    assert 'href="/test-run">Outbound Agency Console' in html
    assert "/review/queue" not in html
    assert 'fetch("/api/review/queue/count")' not in html


def test_nav_live_mode_unchanged():
    html = _render("targets.html", targets=[])
    assert 'href="/">Outbound Agency Console' in html
    assert 'href="/review/queue"' in html
    assert 'fetch("/api/review/queue/count")' in html
