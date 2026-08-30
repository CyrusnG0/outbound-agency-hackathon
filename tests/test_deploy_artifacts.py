"""
Tests for the A5b deploy artifacts (ticket A5b) — pure file-content checks.

These run in CI without Docker, so they assert on the FILES the deploy
depends on, not on a built image:

- .dockerignore             — the only thing standing between `COPY . .` and
                              the repo's real API keys ending up in a
                              registry layer
- Dockerfile                — the image contract (no .env copy, no packaging
                              install, Cloud Run port handling)
- requirements-console.txt  — the console-only dependency closure
- scripts/deploy_console.sh — the operator-facing deploy script

Each test's docstring names the failure it prevents, so the day one of
these regresses, the assertion message is the explanation.
"""

import ast
import os
import re
from pathlib import Path

# The artifacts live at the repo root (and in scripts/), never under tests/.
ROOT = Path(__file__).resolve().parent.parent
DOCKERIGNORE = ROOT / ".dockerignore"
DOCKERFILE = ROOT / "Dockerfile"
REQUIREMENTS = ROOT / "requirements-console.txt"
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy_console.sh"
# The console app whose FastAPI features drive what requirements-console.txt
# must declare (ticket H10) — the completeness guard's trigger file.
CONSOLE_APP = ROOT / "app" / "console" / "app.py"


def _read(path: Path) -> str:
    """Read an artifact, failing with a name-the-file message if missing."""
    assert path.is_file(), f"{path.name} missing — was it deleted or renamed?"
    return path.read_text(encoding="utf-8")


def _dockerignore_patterns() -> list[str]:
    # Parse .dockerignore the way Docker does, approximately: one pattern
    # per non-blank, non-comment line. Tests assert on THESE patterns, not
    # on substrings of the whole file — a comment mentioning ".env" (e.g.
    # "# do not copy .env") must not satisfy an exclusion test.
    return [
        line.strip()
        for line in _read(DOCKERIGNORE).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _dockerfile_instructions() -> list[str]:
    # Only Dockerfile INSTRUCTION lines (not comments): the Dockerfile's own
    # header comment documents the forbidden spellings (`pip install .`,
    # `${PORT:-8080}`) by name, so substring checks must ignore comments —
    # the same comment-trap the requirements test avoids.
    return [
        line
        for line in _read(DOCKERFILE).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_dockerignore_excludes_env_git_and_data():
    # Failure prevented: a Dockerfile with `COPY . .` baking the repo's real
    # .env (GOOGLE_API_KEY, DEEPSEEK_API_KEY) into an image layer — anyone
    # who can pull the image can extract the layer and read the keys.
    # .git (full history in the image) and data/ (a live database must never
    # ship inside an image) are the same class of mistake.
    patterns = _dockerignore_patterns()
    assert ".env" in patterns
    assert ".git" in patterns
    assert "data/" in patterns


def test_dockerignore_keeps_env_example():
    # Failure prevented: a blanket ".env*" pattern silently excluding
    # .env.example — which is committed deliberately and holds documented
    # placeholder names only, no values. Dropping it would deny a fresh
    # operator the one file that teaches which env vars exist, and would
    # falsely signal to a reader that it holds secrets. The broad ".env.*"
    # pattern therefore REQUIRES the explicit "!.env.example" negation;
    # assert both halves of that contract.
    patterns = _dockerignore_patterns()
    assert ".env.*" in patterns
    assert ".env.example" not in patterns
    assert "!.env.example" in patterns


def test_dockerfile_never_copies_env():
    # Failure prevented: a COPY instruction that reaches .env (or a whole-
    # context `COPY . .`) re-introducing the exact key leak .dockerignore
    # guards against. .dockerignore is one line of defense; the Dockerfile
    # must not have a copy path that can reach .env at all, even if a
    # .dockerignore entry were ever removed.
    dockerfile = _read(DOCKERFILE)
    assert not re.search(r"^\s*COPY\b.*\.env", dockerfile, re.MULTILINE)
    assert "COPY . " not in dockerfile


def test_dockerfile_installs_from_requirements_file_only():
    # Failure prevented: `pip install .` / `pip install -e .` — this repo's
    # pyproject.toml deliberately has NO [build-system] section
    # (docs/current_status.md: a documented non-fix, not an oversight), so
    # a packaging install fails at image build time. The only install that
    # can work is `-r requirements-console.txt`. Instruction lines only —
    # the Dockerfile's own comment block mentions the forbidden spellings.
    instructions = _dockerfile_instructions()
    for line in instructions:
        assert "pip install ." not in line
        assert "pip install -e" not in line
    assert any("-r requirements-console.txt" in line for line in instructions)


def test_dockerfile_cmd_uses_injected_port_and_binds_all_interfaces():
    # Failure prevented: two Cloud Run health-check killers in the CMD —
    # (a) a hardcoded port that ignores Cloud Run's injected $PORT, so the
    # revision fails its health check whenever the injected port differs
    # from the hardcode; (b) binding uvicorn's default 127.0.0.1, which
    # Cloud Run's router cannot reach, so the revision never serves
    # traffic. The CMD must reference ${PORT:-8080} AND --host 0.0.0.0.
    # The check is for the BRACED form ${PORT:-8080}: that is the only
    # valid shell spelling of a defaulted expansion (`$PORT:-8080` unbraced
    # is a syntax error), so a bare substring search for "$PORT" would
    # miss the correct spelling and only match broken ones.
    instructions = _dockerfile_instructions()
    cmd_lines = [
        line for line in instructions if line.upper().startswith("CMD")
    ]
    assert cmd_lines, "no CMD instruction found in Dockerfile"
    cmd = "\n".join(cmd_lines)
    assert "${PORT:-8080}" in cmd
    assert "0.0.0.0" in cmd


def test_requirements_console_excludes_pipeline_dependencies():
    # Failure prevented: the read-only console image silently growing the
    # whole pipeline — playwright alone adds hundreds of MB of browser
    # binaries to a container that never renders a page, and the LLM SDKs
    # drag in the entire model stack for a module that cannot call a model.
    # Only non-comment lines are checked: the file's header comment is
    # REQUIRED to name these exclusions (ticket A5b), so a naive whole-file
    # substring match would fail the test on the very comment that
    # documents the exclusions.
    req_lines = [
        line.strip()
        for line in _read(REQUIREMENTS).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert req_lines, "requirements-console.txt has no requirements at all"
    for banned in ("playwright", "google-adk", "anthropic", "google-genai"):
        for line in req_lines:
            assert banned not in line, (
                f"requirements-console.txt requires {banned!r} — the "
                "read-only console must not drag the pipeline into the image"
            )


def test_deploy_script_is_executable():
    # Failure prevented: the operator following the runbook and getting
    # "permission denied" on scripts/deploy_console.sh. The executable bit
    # is part of the artifact contract (and git tracks it).
    assert DEPLOY_SCRIPT.is_file(), "scripts/deploy_console.sh missing"
    assert os.access(DEPLOY_SCRIPT, os.X_OK), (
        "scripts/deploy_console.sh is not executable (needs chmod +x)"
    )


def test_deploy_script_uses_set_secrets_not_literal_password():
    # Failure prevented: the DB password appearing as a literal in the repo
    # (shell assignment, --set-env-vars, or inline) — from there it leaks
    # into git history, shell history and deploy logs. The only legal
    # spelling is --set-secrets OUTBOUND_DB_PASSWORD=<secret-name>:latest,
    # which makes Cloud Run inject the VALUE at container start so it never
    # appears in the image, the repo or any log.
    script = _read(DEPLOY_SCRIPT)
    assert "--set-secrets" in script
    # The exact secret reference, matching docs/gcp-setup.md §8's secret
    # name and the :latest version convention.
    assert "OUTBOUND_DB_PASSWORD=outbound-db-password:latest" in script
    # No line may be a literal assignment (with or without export/readonly)
    # of OUTBOUND_DB_PASSWORD to a value. Comment lines start with "#" and
    # cannot match; the --set-secrets flag line starts with "--", so it
    # cannot match either — only a real assignment line would.
    assignment = re.compile(
        r"^\s*(?:export\s+|readonly\s+)*OUTBOUND_DB_PASSWORD="
    )
    for line in script.splitlines():
        assert not assignment.search(line), (
            "literal OUTBOUND_DB_PASSWORD assignment in deploy script: "
            f"{line.strip()!r}"
        )
    # And the password must not be smuggled in via --set-env-vars either.
    assert not re.search(
        r"--set-env-vars[^\n]*OUTBOUND_DB_PASSWORD", script
    )


# ── H10: a FastAPI feature with an extra dependency must be DECLARED ──────────
#
# The first Cloud Run deploy of the console died at route-build time with
#   RuntimeError: Form data requires "python-multipart" to be installed.
# (fastapi/dependencies/utils.py, ensure_multipart_is_installed — raised from
# app/console/app.py's Form(...) parameters while create_app() builds the app
# object, so the container cannot boot at all, not a runtime edge case.)
# requirements-console.txt did not declare python-multipart, and NOTHING
# asserted the file was COMPLETE: the tests above check that the console
# EXCLUDES pipeline deps and that the artifacts' properties hold, never that
# the app BOOTS with the declared closure. This guard pins the completeness
# direction: if app/console/app.py references a FastAPI feature that pulls in
# an extra package, that package must be declared in requirements-console.txt.
#
# The feature → required-extra map comes from reading the installed FastAPI's
# own ensure_*_is_installed helpers (fastapi 0.136.3): there is EXACTLY ONE
# such helper, ensure_multipart_is_installed, and every form-data feature
# routes through it — Form(...) (the check fires on isinstance(field_info,
# params.Form)), File(...) (File subclasses Form), and the UploadFile
# annotation (FastAPI auto-wraps it in File(...), a Form). All three therefore
# require python-multipart, the package FastAPI's helper imports.
_FORM_FEATURES = frozenset({"Form", "File", "UploadFile"})
_FORM_FEATURE_REQUIRES = "python-multipart"


def _form_feature_uses(path: Path) -> list[tuple[str, int]]:
    """Return sorted ``(feature, lineno)`` for every form-data feature
    referenced in one Python file, deduplicated.

    Form(...) and File(...) are always CALLS (they construct a FieldInfo), so
    they are matched as Call nodes whose func is a bare Name — a method call
    like ``obj.Form(...)`` (an Attribute, not a Name) is not a FastAPI feature
    and is deliberately not matched. UploadFile is used as a TYPE ANNOTATION,
    never a call, so it is matched as a Name node in a Load (read/annotation)
    position; a Store/Del position would be a local shadowing the name, not a
    feature use.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))  # The console app's AST — the trigger source.
    found: set[tuple[str, int]] = set()  # A set dedups repeated uses (7 Form calls -> one entry per line).
    for node in ast.walk(tree):  # Walk EVERY node, so a use nested in any expression is seen.
        if (
            isinstance(node, ast.Call)  # Form(...) / File(...) — a FieldInfo constructor.
            and isinstance(node.func, ast.Name)  # Bare name, not a method attribute.
            and node.func.id in _FORM_FEATURES
        ):
            found.add((node.func.id, node.lineno))  # Record the feature name + the source line for the message.
        elif (
            isinstance(node, ast.Name)  # UploadFile appears as an annotation, not a call.
            and node.id == "UploadFile"
            and isinstance(node.ctx, ast.Load)  # A read/annotation use, not an assignment target.
        ):
            found.add((node.id, node.lineno))  # Same record shape as the call branch.
    # Dedup already happened via the set; sort by line so the failure message
    # reads top-to-bottom like the source.
    return sorted(found, key=lambda item: item[1])


def _requirement_names() -> list[str]:
    """The bare package name of every non-comment requirement line.

    Cut each line at the first character that ends a package name: an extras
    marker '[', a version specifier '<>=!~', an environment marker ';', or
    whitespace.  "uvicorn[standard]>=0.30,<1" -> "uvicorn";
    "python-multipart>=0.0.20,<1" -> "python-multipart".
    """
    names = []  # The accumulator: one bare package name per requirement line.
    for line in _read(REQUIREMENTS).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue  # Same comment-trap rule as the exclusion test: only real requirements.
        names.append(re.split(r"[\[<>=!~;\s]", stripped, maxsplit=1)[0])  # Cut the name at the first extras/specifier char.
    return names


def test_requirements_console_declares_every_fastapi_extra_dependency():
    """If the console's app.py uses a FastAPI feature that needs an extra
    package, requirements-console.txt must declare it — the completeness guard
    (ticket H10).  The conditional shape is deliberate: if the app stops using
    form data entirely, python-multipart is genuinely no longer needed and this
    guard has nothing to enforce, so it stays silent.

    COVERAGE BOUNDARY — what this walk does and does not catch, so the next
    reader trusts it only as far as it goes.  Caught: Form(...)/File(...)
    calls and UploadFile annotations under their LITERAL names, mapped to
    python-multipart — the complete set of ensure_*_is_installed extras in
    fastapi 0.136.3, verified by reading the installed source.  NOT caught: a
    future FastAPI that adds a new ensure_*_is_installed pair (the _FORM_*
    table above must be extended by hand — it is static, not introspected);
    an imported alias (`from fastapi import Form as F; F(...)` is invisible to
    a walk that matches only the literal names); and a non-FastAPI direct
    import (e.g. `import multipart`), which is the console's import-allowlist
    test's concern (tests/test_console.py), not this one.  A false positive is
    the safe direction: a local variable named Form/File/UploadFile would trip
    this and force a deliberate review.
    """
    uses = _form_feature_uses(CONSOLE_APP)
    if not uses:
        # No form-data feature in the app -> the extra is not required; the
        # guard is silent by design (documented above).
        return
    declared = set(_requirement_names())
    assert _FORM_FEATURE_REQUIRES in declared, (
        f"app/console/app.py uses FastAPI form-data features "
        f"{', '.join(f'{name}@{line}' for name, line in uses)}, which require "
        f"{_FORM_FEATURE_REQUIRES!r} (fastapi's ensure_multipart_is_installed "
        f"raises 'Form data requires python-multipart' at route-build time "
        f"without it), but requirements-console.txt does not declare it"
    )

def test_deploy_script_passes_set_secrets_exactly_once() -> None:
    """`--set-secrets` is a LIST flag, not a repeatable one — passing it twice drops the first.

    gcloud documents it as `--set-secrets=[KEY=VALUE,...]` with "All existing
    secrets will be removed first", so a second occurrence overwrites the
    first rather than adding to it.  The console needs BOTH secrets:
    OUTBOUND_DB_PASSWORD (or every query fails) and OUTBOUND_CONSOLE_API_KEY
    (or auth fails closed and every route 503s).  Written as two flags the
    deploy silently ships a console with no database credential — a defect
    only a real deploy would surface, exactly the H10 failure mode.  Both
    pairs must therefore live in ONE comma-separated flag.

    COVERAGE BOUNDARY: this counts occurrences and checks both names are
    present; it does not validate the secret names resolve in Secret Manager
    (the script's own Prerequisite 3 does that at deploy time).
    """
    script = DEPLOY_SCRIPT.read_text()
    # Count only the flag as passed to gcloud (leading whitespace + flag),
    # so the explanatory comments further down the file are not counted.
    occurrences = re.findall(r"^\s*--set-secrets ", script, flags=re.MULTILINE)
    assert len(occurrences) == 1, (
        f"deploy_console.sh passes --set-secrets {len(occurrences)} times; gcloud keeps "
        "only the LAST one, so every earlier secret is silently dropped. Put every "
        "KEY=secret:version pair in ONE comma-separated --set-secrets flag."
    )
    # Both secrets the console cannot run without must be in that one flag.
    for name in ("OUTBOUND_DB_PASSWORD", "OUTBOUND_CONSOLE_API_KEY"):
        assert name in script, f"deploy_console.sh no longer injects {name}"


# ── H13: the deploy script can ship a stale image and call it success ────────
#
# The 2026-08-27 incident: the operator ran scripts/deploy_console.sh, it
# printed a clean success and reported a new revision — and deployed the
# pre-H11 image, putting an unauthenticated console back on a public URL.
# Three independent causes, all in the script:
#   1. IMAGE defaulted to a hardcoded, permanently stale tag (ticket A5b's
#      tag), so every deploy shipped A5b's code unless the operator remembered
#      to override IMAGE.
#   2. The build/push commands were PRINTED as a comment block, never run — a
#      printed prerequisite is not a prerequisite.
#   3. Nothing verified the deployed service afterwards — the script's last
#      word was "Deployed."
#
# COVERAGE BOUNDARY: these are TEXT assertions, the same style as the rest of
# this module. They prove the guard is PRESENT in the script, not that it
# behaves correctly at runtime — a wrong variable name, a broken curl, or a
# logic error in the retry loop would all pass these tests. The runtime
# behaviour is the lead's to verify with a real deploy.


def test_deploy_script_default_image_tag_derives_from_git() -> None:
    """The IMAGE default must derive from the current git commit, never a hardcoded tag.

    Failure prevented: the H13 incident's first cause — IMAGE defaulted to a
    hardcoded tag (ticket A5b's), so every deploy shipped A5b's code unless the
    operator remembered to override IMAGE. A tag derived from the current git
    commit is unique per commit, so reusing a stale tag is impossible by
    construction. IMAGE stays overridable for a deliberate rollback, but the
    DEFAULT must track the code in the working tree.
    """
    script = DEPLOY_SCRIPT.read_text()
    # The known stale tag must be gone entirely — it WAS the default, and a
    # comment mentioning it could be mistaken for the real thing.
    assert "console:a5b" not in script, (
        "deploy_console.sh still hardcodes ticket A5b's tag (console:a5b) — the "
        "H13 incident. The IMAGE default must derive from the current git commit."
    )
    # No literal console:<tag> may appear at all: the only console: reference
    # in the script must be the git-derived ${TAG}. Catches any future
    # hardcoded tag, not just the known a5b.
    for line in script.splitlines():
        if "console:" in line and "console:${TAG}" not in line:
            raise AssertionError(
                f"deploy_console.sh has a literal image tag: {line.strip()!r} — "
                "the IMAGE default must derive from git, and an explicit image "
                "must come from the operator's IMAGE env override at runtime, "
                "never a literal in the script"
            )
    # The default derives from git: the tag is the commit's short hash.
    assert "git rev-parse --short HEAD" in script, (
        "deploy_console.sh no longer derives the default image tag from git — "
        "without this the default can silently point at stale code (H13)."
    )
    # ...and that derived tag is wired into the image name itself — the whole
    # point of deriving it. A tag computed but never used would not help.
    assert "console:${TAG}" in script, (
        "deploy_console.sh computes a git-derived TAG but never uses it in the "
        "IMAGE name — the default still cannot track the working tree."
    )


def test_deploy_script_has_post_deploy_smoke_check() -> None:
    """The script must verify the deployed console refuses anonymous write access.

    Failure prevented: the H13 incident's third cause — the old script's last
    word was "Deployed." and nothing checked that the thing just put on the
    internet actually enforced auth. This test pins the smoke check's two
    load-bearing references: the /kill-switch write route that must not be
    anonymously reachable, and the 422 status — the specific FastAPI
    *validation* error that means the request REACHED the handler, which can
    only happen when there is NO auth layer. If either reference is deleted,
    the smoke check can no longer detect the incident it exists to catch.
    """
    script = DEPLOY_SCRIPT.read_text()
    # The section header — anchors the block for the exit-1 test below.
    assert "POST-DEPLOY SMOKE CHECK" in script, (
        "deploy_console.sh lost its post-deploy smoke check — nothing verifies "
        "the deployed console actually enforces auth."
    )
    # 422 is the non-obvious tell this incident produced (see the block comment
    # in the script): an auth layer rejects with 401 before FastAPI reaches body
    # validation, so 422 on an anonymous POST is proof of no auth layer.
    assert "422" in script, (
        "deploy_console.sh's smoke check no longer looks for 422 — the status "
        "that means 'reached the handler with no auth layer'. Without it the "
        "check cannot tell 'shaped wrong' from 'open'."
    )
    # The write route that must not be anonymously reachable.
    assert "/kill-switch" in script, (
        "deploy_console.sh's smoke check no longer probes /kill-switch — the "
        "write route whose anonymous reachability is the H11 defect. Without it "
        "the check can pass while the console is publicly writable."
    )


def test_deploy_script_builds_or_verifies_image_before_deploy() -> None:
    """The image must be built+push or verified to exist before deploying.

    Failure prevented: the H13 incident's second cause — the old script PRINTED
    the docker build/push commands as a comment block and then deployed
    regardless of whether anyone ran them, so a forgotten build shipped whatever
    was last pushed (A5b's image). The script must either build and push the
    image itself (the preferred path) or verify the exact image exists in
    Artifact Registry and exit loudly if it does not.
    """
    script = DEPLOY_SCRIPT.read_text()
    assert (
        "docker push" in script
        or "gcloud artifacts docker images describe" in script
    ), (
        "deploy_console.sh neither builds/pushes the image nor verifies the "
        "pinned image exists before deploying — a forgotten build silently "
        "ships whatever was last pushed (the H13 incident)."
    )


def test_deploy_script_exits_nonzero_on_smoke_check_failure() -> None:
    """The smoke-check failure path must exit non-zero — never report success.

    Failure prevented: a smoke check that merely PRINTS a warning and lets the
    script fall through to "Deployed." would still call a broken deploy success —
    the exact failure mode the incident was. The failure must abort with exit 1
    so the operator's run ends in failure, not a clean success.
    """
    script = DEPLOY_SCRIPT.read_text()
    smoke_marker = "POST-DEPLOY SMOKE CHECK"
    assert smoke_marker in script, (
        "smoke check header missing — see "
        "test_deploy_script_has_post_deploy_smoke_check"
    )
    smoke_block = script.split(smoke_marker, 1)[1]
    assert "exit 1" in smoke_block, (
        "deploy_console.sh's smoke check has no 'exit 1' in its failure path — "
        "a failed smoke check must abort with a non-zero status, not fall "
        "through to 'Deployed.'"
    )

def test_deploy_script_captures_image_pinned_flag_before_defaulting_it() -> None:
    """`IMAGE_WAS_SET` must be captured BEFORE `IMAGE` is given its default.

    `${IMAGE+x}` expands to "x" whenever IMAGE is set to anything at all.  So
    once `IMAGE="${IMAGE:-...}"` has run, IMAGE is always set and the capture
    can never be empty.  Written in the wrong order the script takes the
    pinned/rollback branch on EVERY run: the build-and-push path becomes dead
    code that never executes, and deploying freshly written code fails with
    "the pinned image does not exist in Artifact Registry".

    That is not hypothetical -- it is what happened on the very first run
    after H13 landed, and it defeated H13's headline fix (build from the
    working tree) while every text-level guard still passed.  Ordering is the
    whole contract here, so the test asserts ordering, not presence.

    COVERAGE BOUNDARY: this asserts the order of two lines in the script
    text.  It does not execute the script or prove the branch logic is right;
    the branch behaviour was verified by running both paths by hand.
    """
    script = DEPLOY_SCRIPT.read_text().splitlines()
    # Locate the capture and the default assignment by their leading token, so
    # comments that merely mention either name are never matched.
    capture_line = next(
        (i for i, line in enumerate(script) if line.startswith("IMAGE_WAS_SET=")),
        None,
    )
    default_line = next(
        (i for i, line in enumerate(script) if line.startswith("IMAGE=")),
        None,
    )
    assert capture_line is not None, "deploy_console.sh no longer captures IMAGE_WAS_SET"
    assert default_line is not None, "deploy_console.sh no longer defaults IMAGE"
    assert capture_line < default_line, (
        f"IMAGE_WAS_SET is captured on line {capture_line + 1}, AFTER IMAGE is defaulted on line "
        f"{default_line + 1}. ${{IMAGE+x}} is then always 'x', so the script always takes the "
        "pinned/rollback branch and never builds from the working tree. Capture the flag first."
    )

def test_deploy_script_builds_from_repo_root_not_cwd() -> None:
    """`docker build` must use an absolute repo root as its context, never ".".

    The script lives in scripts/, which is the natural place to run it from --
    and a bare "." context then makes Docker look for a Dockerfile inside
    scripts/, where there is none. The first post-H13 build failed exactly
    that way ("failed to read dockerfile: open Dockerfile: no such file or
    directory") after the operator ran ./deploy_console.sh from scripts/.

    git rev-parse --show-toplevel resolves the repo root from any
    subdirectory, so the build works regardless of the operator's cwd. This
    test pins that the context is that variable and not a relative path.

    COVERAGE BOUNDARY: a text assertion on the build line. It proves the
    context is not cwd-relative; it does not run docker. Building from
    scripts/ was verified by hand.
    """
    script = DEPLOY_SCRIPT.read_text()
    build_lines = [
        line.strip()
        for line in script.splitlines()
        # Only the executed command, never the comments that discuss it.
        if line.strip().startswith("docker build")
    ]
    assert build_lines, "deploy_console.sh no longer runs docker build"
    for line in build_lines:
        assert not line.endswith(" ."), (
            f"docker build uses a cwd-relative '.' context: {line!r}. Run from scripts/ and "
            "Docker looks for a Dockerfile there and fails. Use the git-resolved repo root."
        )
        assert "${REPO_ROOT}" in line, (
            f"docker build context is not ${{REPO_ROOT}}: {line!r}. The context must be resolved "
            "from git so the script works from any directory."
        )
    # The variable the build line depends on must actually be defined from git.
    assert 'REPO_ROOT="$(git rev-parse --show-toplevel)"' in script, (
        "REPO_ROOT is used as the build context but is not resolved from git rev-parse "
        "--show-toplevel, so it may be empty or wrong depending on the operator's cwd."
    )


# ── H14: the build line's platform flags and the deploy/binding ORDER ─────────
#
# The behaviour tests in tests/test_deploy_script_behaviour.py prove the script
# DOES the right thing at runtime. These two guards stay text-level because they
# pin things a runtime test cannot see: a flag silently DROPPED from the build
# line (the text is the only record of what was requested), and the FILE ORDER
# of the deploy and binding commands (the binding must be written after the
# deploy so a failed deploy can never be what opens the service).


def test_deploy_script_build_line_pins_amd64_and_no_provenance() -> None:
    """The executed `docker build` must force linux/amd64 and no provenance.

    Failure prevented (H14): a NATIVE `docker build` on Apple Silicon produces
    an arm64 image that Cloud Run rejects at deploy time, and buildx's default
    OCI provenance/attestation manifest turns the pushed artifact into an image
    INDEX (application/vnd.oci.image.index.v1+json) that Cloud Run also
    rejects. Both flags on the build line make the pushed artifact a plain amd64
    manifest regardless of the host machine.

    COVERAGE BOUNDARY: text assertion on the build line only. It proves the
    flags are PRESENT in the command, not that a build actually produces a
    runnable image — the platform is verified at runtime against the REGISTRY
    image by tests/test_deploy_script_behaviour.py.
    """
    script = DEPLOY_SCRIPT.read_text()
    build_lines = [
        line.strip()
        for line in script.splitlines()
        # Only the executed command, never the comments that discuss it.
        if line.strip().startswith("docker build")
    ]
    assert build_lines, "deploy_console.sh no longer runs docker build"
    for line in build_lines:
        assert "--platform linux/amd64" in line, (
            f"docker build lost --platform linux/amd64: {line!r}. A native build "
            "on Apple Silicon produces an arm64 image Cloud Run rejects (H14)."
        )
        assert "--provenance=false" in line, (
            f"docker build lost --provenance=false: {line!r}. The provenance "
            "manifest turns the push into an OCI image index Cloud Run rejects (H14)."
        )


def test_deploy_script_adds_public_binding_after_deploy_call() -> None:
    """The add-iam-policy-binding call must be written AFTER the deploy call.

    Failure prevented (H14): gcloud run deploy applies its IAM flag BEFORE the
    revision succeeds, so the deploy must be CLOSED and the public binding must
    be a SEPARATE step later in the file — never before or on the deploy line.
    This asserts the FILE ORDER of the two executed commands (comments are
    ignored), the same ordering contract IMAGE_WAS_SET established.

    COVERAGE BOUNDARY: text-level ordering only. It proves the binding step is
    written after the deploy; the runtime behaviour (a failed deploy never adds
    the binding, the binding is added only after smoke A sees 403) is covered by
    tests/test_deploy_script_behaviour.py.
    """
    script = DEPLOY_SCRIPT.read_text().splitlines()
    deploy_idx = next(
        (
            i
            for i, line in enumerate(script)
            if line.strip().startswith("gcloud run deploy")
        ),
        None,
    )
    add_idx = next(
        (
            i
            for i, line in enumerate(script)
            if line.strip().startswith("gcloud run services add-iam-policy-binding")
        ),
        None,
    )
    assert deploy_idx is not None, "deploy_console.sh no longer calls gcloud run deploy"
    assert add_idx is not None, (
        "deploy_console.sh no longer adds the public binding as a separate step"
    )
    assert deploy_idx < add_idx, (
        f"gcloud run deploy is on line {deploy_idx + 1}, AFTER "
        f"add-iam-policy-binding on line {add_idx + 1}. The public binding must "
        "come after the deploy so a failed deploy can never be what opens the "
        "service (H14)."
    )
