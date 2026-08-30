#!/usr/bin/env bash
#
# deploy_console.sh — deploy the read-only console to Cloud Run (tickets A5b,
# H11, H13, H14).
#
# What this script IS: the operator's one command to build, push and deploy
# the console to Cloud Run, wired to Cloud SQL and Secret Manager, for the
# hackathon demo window. It builds a linux/amd64 image (Cloud Run's required
# platform — H14), verifies the PUSHED image actually advertises amd64 BEFORE
# any cloud mutation (H14), deploys the service CLOSED (--no-allow-
# unauthenticated, so a failed deploy can never open the door — H14), and
# only then, if ALLOW_UNAUTH=true, adds the public allUsers binding as a
# SEPARATE, echoed step after confirming the service is closed. After the
# deploy it PROBES the live URL and refuses to report success unless
# anonymous requests are refused (the H13 smoke check — the gate that would
# have caught the stale-image/no-auth incident — split into a closed-at-edge
# check before the public binding and an app-auth check after it).
#
# What this script is NOT:
#   - it is INERT until the operator runs it; it echoes every command it
#     executes (set -x), so every mutation is visible (CLAUDE.md §3);
#   - the ONLY IAM it ever mutates is the public allUsers binding it adds
#     ITSELF, seconds after verifying the service is closed — and, when the
#     post-binding smoke check fails, the removal of that same binding. It
#     never touches pre-existing IAM state (the deliberate distinction from
#     H13's "never mutate IAM": undoing its OWN just-made mutation is strictly
#     safer than leaving a known-open service up).
#   - it does NOT create secrets (Prerequisite 3 refuses to deploy if the
#     console API-key secret is missing).
#
# Do not run this until docs/gcp-setup.md §0–§8 are done.

set -euo pipefail
# -e: any failing command aborts the script — a half-finished deploy must
#     not report success (CLAUDE.md §3: failures surface clearly).
# -u: referencing an unset variable is a bug, not a silent default.
# -o pipefail: a failing command inside a pipeline fails the pipeline.

# ── Configuration (overridable via env; defaults from docs/gcp-setup.md) ─────
PROJECT="${PROJECT:-outbound-agency-devpost}"   # gcp-setup.md §2
REGION="${REGION:-us-central1}"                 # cheapest region, §6
SERVICE="${SERVICE:-outbound-console}"          # Cloud Run service name
# The Cloud SQL instance CONNECTION NAME (project:region:instance), as shown
# by `gcloud sql instances describe outbound-db` (gcp-setup.md §6).
INSTANCE="${INSTANCE:-outbound-agency-devpost:us-central1:outbound-db}"
# The repo's database-location convention for the cloudsql:// dialect
# (app/db.py connect(), docs/gcp-setup.md §6). Built from INSTANCE so the
# two can never drift apart.
DB_TARGET="cloudsql://${INSTANCE}/outbound"
# Where the image lives in Artifact Registry. The DEFAULT tag is derived from
# the current git commit so the deployed image always corresponds to the code
# in the working tree — unique per commit, so reusing a stale tag (the H13
# incident: the old default was hardcoded to ticket A5b's tag, so every deploy
# since shipped A5b's code) is impossible by construction. Override IMAGE for a
# deliberate rollback to a previously built image; that pinned path is verified
# to exist and never rebuilt (a rebuild would overwrite the tag you chose).
# The repo root, resolved from git so every path below is independent of the
# directory the operator happened to run this script from.
REPO_ROOT="$(git rev-parse --show-toplevel)"
TAG="${TAG:-$(git rev-parse --short HEAD)}"
# Did the operator pin IMAGE explicitly (rollback), or are we on the
# commit-derived default? Only the default path builds from the working tree.
#
# THIS CAPTURE MUST COME BEFORE THE DEFAULT ASSIGNMENT BELOW.  ${IMAGE+x}
# expands to "x" whenever IMAGE is set to anything at all, including an
# empty string -- so once the `${IMAGE:-...}` default has run, IMAGE is
# always set and this test can never be false.  Written in the other order
# (as it was when H13 first landed) the script takes the pinned/rollback
# branch on EVERY run, the build path becomes dead code that never executes,
# and a deploy of freshly written code fails with "the pinned image does not
# exist" -- which is exactly what happened on the first run after H13.
IMAGE_WAS_SET="${IMAGE+x}"
IMAGE="${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT}/outbound-console/console:${TAG}}"
if [[ -n "${IMAGE_WAS_SET}" ]]; then
    BUILD_FROM_TREE="false"
else
    BUILD_FROM_TREE="true"
fi
# ALLOW_UNAUTH=true (the default) means a PUBLIC URL in front of real pipeline
# data. This is NOT implemented as a flag on the deploy call — see the deploy
# comment for the H14 reason — but as a separate add-iam-policy-binding step
# that runs only AFTER smoke check A has confirmed the service is closed.
ALLOW_UNAUTH="${ALLOW_UNAUTH:-true}"

# ── Working-tree check ───────────────────────────────────────────────────────
# A commit-derived tag is a LIE about what code is inside the image when there
# are uncommitted changes, so refuse to build from a dirty tree unless the
# operator explicitly opts in (ALLOW_DIRTY=true). Only enforced on the
# build-from-tree path: a pinned rollback image is deliberately NOT the working
# tree's code, so the tree's state is irrelevant there.
if [[ "${BUILD_FROM_TREE}" == "true" ]]; then
    # git diff --quiet exits 1 when there ARE differences; the leading ! turns
    # that into "true" for the test, and being inside an `if` keeps set -e from
    # aborting on the expected non-zero status.
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Working tree has uncommitted changes:"
        git status --short
        if [[ "${ALLOW_DIRTY:-}" != "true" ]]; then
            echo "ERROR: refusing to build from a dirty tree — the commit-derived tag ${TAG}" >&2
            echo "would mislabel the image (H13: the deployed image must match the code)." >&2
            echo "Commit your changes and re-run, or re-run with ALLOW_DIRTY=true to deploy anyway." >&2
            exit 1
        fi
        echo "WARNING: deploying from a dirty tree anyway (ALLOW_DIRTY=true)." >&2
    fi
fi

cat <<'EOF'
[!] WARNING — running this script creates BILLABLE Google Cloud resources:

    0. It BUILDS the console image and PUSHES it to Artifact Registry — a
       stored image bills for its storage (a fresh copy on every deploy).
    1. A Cloud Run service (free tier covers it, but it is a live resource).
    2. By default (ALLOW_UNAUTH=true) a PUBLICLY REACHABLE URL. Since
       ticket H11 the app itself requires a credential on every route (the
       OUTBOUND_CONSOLE_API_KEY secret, via X-Internal-API-Key or HTTP
       Basic) — so "unauthenticated" at the Google edge does NOT mean "open".
       Cloud Run IAM is a second, independent layer: with ALLOW_UNAUTH=true
       the URL is reachable, and the console's own auth is what keeps real
       pipeline data (targets, scores, ICP verdicts, policy decisions, the
       full steps trace) behind the key.
       Since H14 the public binding is added as a SEPARATE step, only after
       the deploy has been confirmed closed — a failed deploy can never open
       the service.

    The public URL exists only so hackathon judges can open the demo — they
    will need the key (docs/runbook.md §12).  Public is correct ONLY for the
    demo window; tear it down afterwards (docs/gcp-setup.md §9) — the
    teardown commands are printed at the end of this script.

    Nothing has been created yet. Ctrl-C now to abort.
EOF

# ── Prerequisite 1: the image must exist in Artifact Registry ────────────────
# H13: this step is EXECUTED, not printed. The old script printed the docker
# build/push commands as a comment block and deployed regardless of whether
# anyone ran them — so a forgotten build shipped whatever was last pushed
# (ticket A5b's image, the incident). Two paths:
#   * default (commit-derived tag): build + push from the working tree.
#   * IMAGE=... pinned (rollback): verify the exact image exists in Artifact
#     Registry and exit loudly if it does not — never rebuild a pinned image
#     (a rebuild would overwrite the tag the operator chose to roll back to).
#
# The one-time Artifact Registry docker REPO creation stays a printed
# prerequisite: it genuinely is one-time and needs operator judgement
# (gcp-setup.md §5 enabled the API, not the repo).
cat <<EOF

Prerequisite 1 — one-time: create the Artifact Registry docker repo (once per
PROJECT, by hand; gcp-setup.md §5 already enabled the artifactregistry API):

    gcloud artifacts repositories create outbound-console \\
        --repository-format=docker --location=${REGION} --project=${PROJECT}
EOF

if [[ "${BUILD_FROM_TREE}" == "true" ]]; then
    # Build and push the image tagged with the current commit's short hash, so
    # the deployed artifact is exactly the working tree's code. Every command
    # is echoed by set -x before it runs (no hidden side effects, CLAUDE.md §3);
    # a build or push failure aborts via set -e — the deploy never runs with an
    # image that was not just built and pushed.
    echo "Building and pushing ${IMAGE} from the working tree (tag = commit ${TAG})."
    set -x
    # Build context is the REPO ROOT, resolved from git -- never a bare "."
    # and never a path relative to this file.  The operator runs this script
    # from wherever they happen to be (scripts/ is the obvious place, since
    # that is where the script lives), and "." would then make Docker look
    # for a Dockerfile in scripts/, which does not exist.  That is exactly
    # how the first post-H13 build failed: "failed to read dockerfile: open
    # Dockerfile: no such file or directory".  git rev-parse --show-toplevel
    # is correct from any subdirectory of the repo.
    # --platform linux/amd64: Cloud Run only runs linux/amd64 containers, and
    #   the operator's machine is Apple Silicon, so a NATIVE `docker build`
    #   produces an arm64 image that Cloud Run rejects at deploy time (H14:
    #   "Container manifest type ... must support amd64/linux"). Forcing the
    #   platform makes buildx build (via emulation, or natively if the host is
    #   amd64) the amd64 target regardless of the host. This is INVISIBLE on
    #   an amd64 machine — the previously working :a5b image was amd64 only by
    #   accident of where it was built, which is why the defect went unnoticed
    #   until the script finally built for real on Apple Silicon.
    # --provenance=false: buildx attaches an OCI provenance/attestation
    #   manifest by default. That extra manifest is what turns the pushed
    #   artifact into an OCI image INDEX (media type
    #   application/vnd.oci.image.index.v1+json) instead of a plain single
    #   manifest — and the H14 error was Cloud Run rejecting exactly that
    #   index form. Disabling provenance makes the push a plain manifest that
    #   Cloud Run accepts. (The platform check AFTER the push is the result
    #   assertion that catches either misconfiguration even if this one is
    #   ever "simplified" away.)
    docker build --platform linux/amd64 --provenance=false -t "${IMAGE}" "${REPO_ROOT}"
    docker push "${IMAGE}"
    set +x
else
    # Operator pinned IMAGE for a deliberate rollback: verify the exact image
    # exists in Artifact Registry. Same fail-closed shape as Prerequisite 3 —
    # non-interactive, output suppressed, and the check FAILS loudly if the
    # image is not there (gcloud run deploy would fail anyway, but here the
    # error names the problem).
    echo "IMAGE pinned explicitly (${IMAGE}) — verifying it exists in Artifact Registry..."
    if ! gcloud artifacts docker images describe "${IMAGE}" \
        --project "${PROJECT}" </dev/null >/dev/null 2>&1; then
        echo "ERROR: the pinned image '${IMAGE}' does not exist in Artifact Registry." >&2
        echo "It cannot be deployed. Push it first, or drop IMAGE to build from the working tree." >&2
        exit 1
    fi
    echo "Pinned image verified."
fi

# ── Image platform verification — do not hand a bad image to gcloud ───────────
# H14: the arm64 OCI index above was only rejected by gcloud AT DEPLOY TIME —
# AFTER gcloud had already applied the IAM flag and opened the service. So the
# manifest that actually landed in the registry must be verified BEFORE any
# cloud mutation. This is a RESULT assertion, not a precondition: it inspects
# what was pushed (or what the pinned rollback image IS), never what was
# requested.
# docker buildx imagetools inspect reads the manifest from the REGISTRY (not
# the local build cache), so it proves what gcloud will actually pull. If it
# fails (bad tag, wrong registry, no credentials) set -e aborts here — before
# any gcloud run deploy or add-iam-policy-binding can run.
# `--format '{{json .Image}}'` is used rather than the default human output
# because the default prints NO platform at all for a single-platform
# manifest, which is exactly what `--provenance=false` produces above. The
# first version of this check grepped the default output for "linux/amd64",
# and therefore rejected every correctly-built amd64 image -- a false
# positive that blocked a good deploy. It failed CLOSED, which is why it cost
# a re-run rather than an exposure, but the predicate was still wrong.
#
# The JSON form reports BOTH manifest shapes, and they differ:
#   * single manifest  -> one object with "architecture": "amd64", "os": "linux"
#   * multi-platform   -> a MAP keyed by platform, e.g. "linux/amd64": {...}
# so the check below accepts either. Verified against three real images: this
# repo's amd64 build (single, passes), the earlier arm64 build (fails), and
# python:3.13-slim (true multi-arch index, passes).
set -x
PUSHED_IMAGE_INFO="$(docker buildx imagetools inspect "${IMAGE}" --format '{{json .Image}}')"
set +x
# Pass if EITHER shape advertises linux/amd64. An arm64-only image -- what an
# Apple Silicon machine builds without --platform -- matches neither and is
# rejected before any cloud mutation happens.
if ! { grep -q '"linux/amd64"' <<<"${PUSHED_IMAGE_INFO}" \
       || { grep -q '"architecture": *"amd64"' <<<"${PUSHED_IMAGE_INFO}" \
            && grep -q '"os": *"linux"' <<<"${PUSHED_IMAGE_INFO}"; }; }; then
    echo "ERROR: image '${IMAGE}' does NOT advertise linux/amd64 — Cloud Run cannot run it." >&2
    echo "This is the H14 failure: an arm64 (or arm64-only-index) image built on Apple Silicon." >&2
    echo "What the registry reports for this image:" >&2
    echo "${PUSHED_IMAGE_INFO}" >&2
    echo "Rebuild with --platform linux/amd64 --provenance=false and re-run this script." >&2
    exit 1
fi
echo "Image platform verified: '${IMAGE}' advertises linux/amd64."

# ── Prerequisite 2: IAM on the dedicated runtime service account (S1) ────────
# Cloud Run runs the container as a DEDICATED, least-privilege runtime service
# account, NOT the project's default COMPUTE service account. The default
# compute SA carries project-wide roles/editor by GCP default — a blast radius
# the read-only console must not inherit (S1) — so the dedicated SA is created
# here and granted EXACTLY two roles:
#   roles/cloudsql.client              — reach Cloud SQL via the connector
#   roles/secretmanager.secretAccessor — let Cloud Run inject the password
#                                         secret (--set-secrets below)
# These are printed, NOT executed: they are one-time, project-level changes,
# and a script that silently granted IAM would be a hidden side effect
# (CLAUDE.md §3). Run them once, by hand, before the first deploy.
# The \$(...) and \${...} below are escaped so this printout executes
# nothing — the lines are meant to be copy-pasted.
cat <<EOF

Prerequisite 2 — create the dedicated runtime service account and grant it
its two roles (one-time, by hand):

    gcloud iam service-accounts create outbound-console-runtime \\
        --project=${PROJECT} \\
        --display-name="Outbound console Cloud Run runtime"
    gcloud projects add-iam-policy-binding ${PROJECT} \\
        --member="serviceAccount:outbound-console-runtime@${PROJECT}.iam.gserviceaccount.com" \\
        --role="roles/cloudsql.client"
    gcloud projects add-iam-policy-binding ${PROJECT} \\
        --member="serviceAccount:outbound-console-runtime@${PROJECT}.iam.gserviceaccount.com" \\
        --role="roles/secretmanager.secretAccessor"
EOF

# ── Prerequisite 2b (informational): strip roles/editor from the default SA ───
# S1: the PREVIOUS runtime identity — the project's default COMPUTE service
# account — carries project-wide roles/editor by GCP default. The console no
# longer runs as it, so that broad grant is now pure blast radius. This block
# is printed, NOT executed, and it is NOT a prerequisite the script enforces:
#   * the default compute SA is a project-wide credential that OTHER
#     services/jobs in this project (outside this repo's control) may depend
#     on — the operator MUST verify nothing else relies on its roles/editor
#     before running the removal by hand;
#   * a script silently removing a role from a shared credential would be a
#     hidden side effect (CLAUDE.md §3).
# Run it once, by hand, only after that verification. The \$(...) and \${...}
# are escaped exactly like Prerequisite 2 so the printout executes nothing.
cat <<EOF

Prerequisite 2b — remove project-wide roles/editor from the default COMPUTE
service account (one-time, BY HAND, ONLY after you verify nothing else in the
project depends on that account's roles/editor):

    PROJECT_NUMBER="\$(gcloud projects describe ${PROJECT} --format='value(projectNumber)')"
    gcloud projects remove-iam-policy-binding ${PROJECT} \\
        --member="serviceAccount:\${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \\
        --role="roles/editor"
EOF

# ── Prerequisite 3: the console API-key secret must exist ────────────────────
# The console FAILS CLOSED (app/console/auth.py, ticket H11): with
# OUTBOUND_CONSOLE_API_KEY unset, every route returns 503 and the deployed
# console is useless — and an EMPTY secret would authenticate nobody.  So this
# script refuses to deploy unless the secret Cloud Run will inject actually
# exists.  It never CREATES the secret itself: creating a secret is a hidden
# side effect (CLAUDE.md §3), and a script-created key would never be seen by
# the operator.  One-time, by hand (see docs/runbook.md §12):
#
#   printf '%s' "$(openssl rand -hex 32)" | gcloud secrets create outbound-console-api-key \
#       --project "${PROJECT}" --replication-policy automatic --data-file=-
#
# `printf '%s'` emits NO trailing newline.  The `<<<` herestring form this
# script used to print appended one, so the stored secret carried a trailing
# "\n", Cloud Run injected it verbatim, and the clean key never matched —
# every login 401'd (H15).  The app ALSO strips surrounding whitespace from
# the configured secret (app/console/auth.py::console_auth_secret), so an
# existing newline-suffixed secret keeps working; this form is for NEW
# secrets.
#
# The --quiet and </dev/null keep this non-interactive: if the Secret Manager
# API is not enabled or gcloud has no credentials, it must FAIL the check,
# not sit waiting for a y/N prompt mid-deploy.
if ! gcloud --quiet secrets describe "outbound-console-api-key" \
    --project "${PROJECT}" </dev/null >/dev/null 2>&1; then
    echo "ERROR: secret 'outbound-console-api-key' does not exist (or is unreachable)." >&2
    echo "The console fails closed without it (H11) — refusing to deploy an unauthenticated console." >&2
    echo "Create it once, by hand:" >&2
    echo "  printf '%s' \"\$(openssl rand -hex 32)\" | gcloud secrets create outbound-console-api-key \\" >&2
    echo "      --project ${PROJECT} --replication-policy automatic --data-file=-" >&2
    exit 1
fi

# ── The deploy — ALWAYS closed ───────────────────────────────────────────────
# Every command below is echoed by `set -x` before it runs, so the operator
# sees exactly what will happen — no hidden side effects (CLAUDE.md §3).
set -x

# The deploy is ALWAYS `--no-allow-unauthenticated`, regardless of ALLOW_UNAUTH.
# The public binding (if ALLOW_UNAUTH=true) is added LATER as a separate,
# explicit step, never as a flag on this call.
#
# WHY — the entire H14 lesson, stated once so the next reader does not
# "simplify" it back into a single flag:
#   `gcloud run deploy` is NOT atomic. It applies the IAM flag BEFORE the
#   revision is created. With `--allow-unauthenticated` on the deploy line, a
#   revision that then FAILS leaves the service publicly reachable while still
#   serving the OLD revision — the H14 incident: gcloud printed "Deployment
#   failed" while the allUsers binding had already been applied and the old
#   pre-H11 image was live at the URL. A deploy must never be the thing that
#   opens the service. Deploying closed, and opening as a separate step only
#   after the closure is verified, means a failure at ANY point can only ever
#   leave the service MORE closed than it was before the run.
gcloud run deploy "${SERVICE}" \
    --project "${PROJECT}" \
    --region "${REGION}" \
    --image "${IMAGE}" \
    --service-account "outbound-console-runtime@${PROJECT}.iam.gserviceaccount.com" \
    --port 8080 \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 2 \
    --add-cloudsql-instances "${INSTANCE}" \
    --set-env-vars "OUTBOUND_DB_TARGET=${DB_TARGET}" \
    --set-secrets "OUTBOUND_DB_PASSWORD=outbound-db-password:latest,OUTBOUND_CONSOLE_API_KEY=outbound-console-api-key:latest" \
    --no-allow-unauthenticated

set +x

# Flag-by-flag, why each is here:
#   --service-account  outbound-console-runtime@<project>.iam.gserviceaccount.com
#                  The dedicated runtime service account (S1): the console runs
#                  as THIS identity, not the default COMPUTE service account.
#                  Granted EXACTLY two roles (Prerequisite 2):
#                  roles/cloudsql.client and roles/secretmanager.secretAccessor.
#                  The old default SA carries project-wide roles/editor by GCP
#                  default — that would be the container's blast radius if the
#                  console were ever compromised. Least privilege keeps it to
#                  "read two secrets and open a DB connection", nothing more.
#   --port 8080    matches the container's CMD (${PORT:-8080} in the
#                  Dockerfile, which also defaults to 8080). Cloud Run
#                  routes external traffic to this port; a mismatch means a
#                  revision that builds but never serves.
#   --memory 512Mi the console is a read-only page server with no
#                  in-process pipeline; 512Mi is plenty.
#   --cpu 1        one vCPU is plenty for the same reason.
#   --max-instances 2  CAPPED DELIBERATELY: this is a single-operator demo
#                  tool, and with ALLOW_UNAUTH=true it sits on a public URL.
#                  An uncapped autoscale turns any traffic spike (or a
#                  judge's scraper loop) into unbounded billing; two
#                  instances is enough for the demo and bounds the worst
#                  case.
#   --add-cloudsql-instances  attaches the Cloud SQL instance so the Cloud
#                  SQL Python Connector inside the container (app/db.py) can
#                  reach it over its private path, authenticating with the
#                  dedicated runtime service account (--service-account above)
#                  — no IP allowlisting, no password in the container.
#   --set-env-vars OUTBOUND_DB_TARGET  the repo-wide convention for where
#                  the database lives (app/db.py, docs/gcp-setup.md §6); it
#                  contains no secret, so an env var is right.
#   --set-secrets OUTBOUND_DB_PASSWORD=outbound-db-password:latest
#                  WHY THIS FORM instead of a Secret Manager client library
#                  in the image: Cloud Run injects the secret's VALUE as an
#                  env var at container start, so the password never appears
#                  in the image, in the repo, in shell history or in deploy
#                  logs — and it needs no code change and no extra
#                  dependency (no google-cloud-secret-manager in
#                  requirements-console.txt). Rotating the secret is a
#                  redeploy with the new :latest value, not a rebuild. The
#                  secret name matches docs/gcp-setup.md §8.
#   --set-secrets OUTBOUND_CONSOLE_API_KEY=outbound-console-api-key:latest
#                  The console's auth secret (ticket H11).  Same injection
#                  mechanics as the DB password: Cloud Run puts the value
#                  into the OUTBOUND_CONSOLE_API_KEY env var at container
#                  start, so the key never appears in the image, the repo,
#                  shell history or deploy logs.  The console FAILS CLOSED
#                  without it (every route 503), so Prerequisite 3 above
#                  refuses to deploy if the secret does not exist.  Rotate by
#                  adding a new secret version and redeploying (runbook §12).
#   --no-allow-unauthenticated
#                  The service is deployed CLOSED, always. The public binding
#                  (if ANY) is a separate step later. See the WHY comment
#                  above the deploy for the H14 reason this is not a flag
#                  chosen by ALLOW_UNAUTH.

# ── POST-DEPLOY SMOKE CHECK A — the service must be CLOSED at the edge ────────
# H14 restructure: before we even CONSIDER adding a public binding (or, when
# ALLOW_UNAUTH=false, before we report success), prove the deploy left the
# service CLOSED. With --no-allow-unauthenticated, anonymous requests are
# refused at the Cloud Run edge with 403 — the app's own auth (H11) is never
# reached, so the pass condition is anonymous GET / -> 403.
#
# Why 200 is NOT an instant failure in THIS phase (unlike H13's check): the
# IAM change that closes the service propagates in up to ~88s, and while it is
# propagating the OLD revision — possibly an open one — is still what serves
# traffic, so 200 (or 401) here is EXPECTED transiently. It only becomes a
# failure if it persists to the settle deadline: then the deploy did NOT close
# the service, and we must fail loudly — and must NOT add a public binding on
# top of an open service.

# Capture the deployed service's URL from gcloud, not a hardcoded hostname.
# This is the URL the deploy actually produced, and the only one guaranteed to
# route to this revision.
set -x
SERVICE_URL="$(gcloud run services describe "${SERVICE}" --project "${PROJECT}" --region "${REGION}" --format='value(status.url)')"
set +x
if [[ -z "${SERVICE_URL}" ]]; then
    echo "ERROR: could not read the deployed service's URL from gcloud." >&2
    echo "The service may be publicly reachable right now — investigate and close it by hand if so." >&2
    exit 1
fi
echo "Smoke check A — probing ${SERVICE_URL} for closure at the edge..."

# Bounded settle window for Cloud Run IAM propagation / revision startup.
# SMOKE_WINDOW_SECONDS is the hard cap; SMOKE_INTERVAL_SECONDS is the pause
# between probes. Both overridable, both bounded by construction (CLAUDE.md §7:
# retries must be bounded).
# Default 180s, evidence-based: on 2026-08-27 an IAM change took 88s to
# propagate (an earlier one took 32s). H13's 90s default was too tight — it
# left a 2s margin over the worst measured propagation, so a single slow poll
# would fail a healthy deploy. 180s is ~2x headroom over the worst observation.
SMOKE_WINDOW_SECONDS="${SMOKE_WINDOW_SECONDS:-180}"
SMOKE_INTERVAL_SECONDS="${SMOKE_INTERVAL_SECONDS:-5}"
smoke_a_deadline=$(( $(date +%s) + SMOKE_WINDOW_SECONDS ))

while :; do
    # Anonymous probe: no X-Internal-API-Key, no Basic auth. The `|| echo "000"`
    # maps a curl failure (DNS, connection refused, timeout) to a sentinel so
    # the loop RETRIES instead of dying on a transient edge state. --max-time
    # 15 bounds each individual request.
    root_status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "${SERVICE_URL}/" || echo "000")"
    echo "smoke A: anonymous GET / -> ${root_status}  (want 403 — closed at the edge)"

    # PASS: closed at the edge. Proceed to the public-binding step (or, when
    # ALLOW_UNAUTH=false, straight to the teardown banner).
    if [[ "${root_status}" == "403" ]]; then
        echo "smoke A: PASS — the service is closed at the edge (403)."
        break
    fi

    # Not settled yet (old revision still serving, IAM still propagating, or
    # curl could not connect). If the hard-capped window elapsed without ever
    # seeing 403, the deploy did NOT close the service — fail loudly and DO
    # NOT add a public binding.
    if [[ "$(date +%s)" -ge "${smoke_a_deadline}" ]]; then
        echo "ERROR: could not confirm the service is CLOSED within ${SMOKE_WINDOW_SECONDS}s." >&2
        echo "  anonymous GET / -> ${root_status}  (wanted 403 at the edge)" >&2
        echo "The deploy was supposed to close this service and did not. NOT adding any public binding." >&2
        echo "If the service is reachable, close it by hand:" >&2
        echo "  gcloud run services remove-iam-policy-binding ${SERVICE} \\" >&2
        echo "      --project ${PROJECT} --region ${REGION} --member=allUsers --role=roles/run.invoker" >&2
        exit 1
    fi

    echo "smoke A: not settled yet — retrying in ${SMOKE_INTERVAL_SECONDS}s..."
    sleep "${SMOKE_INTERVAL_SECONDS}"
done

# ── The public binding — ONLY if the operator asked for it, ONLY after A ─────
# ALLOW_UNAUTH=false: the service stays closed; nothing more to do, fall
# through to the teardown banner. The binding step below is the ONLY place a
# public URL can come from, and it cannot run until smoke check A has seen 403.
if [[ "${ALLOW_UNAUTH}" == "true" ]]; then
    # The operator explicitly chose a PUBLIC URL (ALLOW_UNAUTH=true). Add the
    # allUsers / roles.run.invoker binding as its OWN, explicit, echoed action
    # — AFTER smoke check A proved the service was closed. Making this a
    # SEPARATE step from the deploy is the core of the H14 fix (see the WHY
    # comment above the deploy): the deploy is always closed, so a failed
    # deploy can never be what opened the service; opening is its own act,
    # executed only after verification. Echoed by set -x so the operator SEES
    # the exposure happen as its own action.
    echo "ALLOW_UNAUTH=true — adding the public allUsers binding as a separate step (smoke check A passed)."
    set -x
    gcloud run services add-iam-policy-binding "${SERVICE}" \
        --project "${PROJECT}" --region "${REGION}" \
        --member=allUsers --role=roles/run.invoker \
        --quiet
    set +x

    # ── POST-DEPLOY SMOKE CHECK B — open at the edge, closed at the app ────────
    # Now that the binding is live, anonymous requests REACH the app. The pass
    # contract (H13, carried forward):
    #   * GET /             must be 401 (the app's own auth, ticket H11) —
    #                       never 200. 200 = the page served = no auth layer.
    #   * POST /kill-switch must be 401 — never 422. 422 is the SPECIFIC tell
    #                       this incident produced: it is FastAPI's
    #                       *validation* error, which means the request REACHED
    #                       the handler and was refused only for payload shape
    #                       — a well-formed body would have been accepted. With
    #                       an auth layer (H11's global dependency) the request
    #                       is rejected 401 BEFORE body validation is reached,
    #                       so 422 can only mean there is no auth layer at all.
    #   * GET /_health      observed, not gated: it should be 200 (the app's
    #                       deliberate health carve-out). A wrong /_health is
    #                       a health-check regression, not an auth exposure.
    #                       The path is /_health, not /healthz (ticket H16):
    #                       Google's Cloud Run frontend intercepts the exact
    #                       path /healthz, so probing /healthz here would
    #                       never reach the app.
    #
    # During the settle window the binding may not have propagated yet, so the
    # edge can still answer 403 (closed) — that is transient, keep retrying.
    # 200 on GET / or 422 on POST is DEFINITIVE — a property of the deployed
    # code, not a transient edge state — fail immediately.
    echo "Smoke check B — probing ${SERVICE_URL} for app-level auth..."
    smoke_b_deadline=$(( $(date +%s) + SMOKE_WINDOW_SECONDS ))
    # smoke_b_ok records which way the loop broke: true = PASS, false = the
    # shared failure handler below must undo the binding and exit 1.
    smoke_b_ok="false"

    while :; do
        # Anonymous probes: no credential, EMPTY body on the POST. The
        # `|| echo "000"` sentinel keeps transient curl failures in the retry
        # loop instead of aborting. --max-time 15 bounds each request.
        root_status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "${SERVICE_URL}/" || echo "000")"
        kill_status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 -X POST "${SERVICE_URL}/kill-switch" || echo "000")"
        health_status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "${SERVICE_URL}/_health" || echo "000")"

        echo "smoke B: anonymous GET / -> ${root_status}  POST /kill-switch -> ${kill_status}  GET /_health -> ${health_status}"

        # DEFINITIVE VIOLATION: the console is exposed RIGHT NOW — the page
        # served without a credential (200) or the request reached a handler
        # with no auth layer (422). Neither is transient; break to the failure
        # handler, which removes the binding this script added seconds ago.
        if [[ "${root_status}" == "200" || "${kill_status}" == "422" ]]; then
            echo "ERROR: the deployed console is NOT protected." >&2
            echo "  anonymous GET /             -> ${root_status}  (must NOT be 200)" >&2
            echo "  anonymous POST /kill-switch -> ${kill_status}  (must NOT be 422 — 422 means the request" >&2
            echo "                                 reached the handler with NO auth layer to stop it)" >&2
            echo "Removing the allUsers binding this script added, then failing." >&2
            break
        fi

        # PASS: both protected endpoints answer 401 — the app's own auth (H11).
        # /_health is observed above and is expected to be 200.
        if [[ "${root_status}" == "401" && "${kill_status}" == "401" ]]; then
            echo "smoke B: PASS — anonymous access is refused by the app's auth (401)."
            smoke_b_ok="true"
            break
        fi

        # Transient edge state (binding still propagating -> 403, revision still
        # starting, curl could not connect). Retry until the hard-capped window
        # elapses, then fail with what we observed.
        if [[ "$(date +%s)" -ge "${smoke_b_deadline}" ]]; then
            echo "ERROR: could not confirm app-level auth within ${SMOKE_WINDOW_SECONDS}s." >&2
            echo "  anonymous GET /             -> ${root_status}  (wanted 401)" >&2
            echo "  anonymous POST /kill-switch -> ${kill_status}  (wanted 401)" >&2
            echo "Removing the allUsers binding this script added, then failing." >&2
            break
        fi

        echo "smoke B: not settled yet — retrying in ${SMOKE_INTERVAL_SECONDS}s..."
        sleep "${SMOKE_INTERVAL_SECONDS}"
    done

    # ── FAILURE: undo the script's OWN binding, then exit non-zero ────────────
    # The deliberate exception to H13's rule "never mutate IAM automatically":
    # this removes the binding the SCRIPT ITSELF added seconds ago — the
    # script's own mutation, not pre-existing state. Undoing it is strictly
    # safer than leaving a known-open service up. If the automatic removal
    # itself fails, print the manual command AND still exit 1 — a removal
    # failure must be impossible to mistake for success.
    if [[ "${smoke_b_ok}" != "true" ]]; then
        echo "Removing the allUsers binding this script just added..." >&2
        set -x
        if ! gcloud --quiet run services remove-iam-policy-binding "${SERVICE}" \
            --project "${PROJECT}" --region "${REGION}" \
            --member=allUsers --role=roles/run.invoker </dev/null; then
            set +x
            echo "WARNING: automatic removal of the binding FAILED — remove it BY HAND NOW:" >&2
            echo "  gcloud run services remove-iam-policy-binding ${SERVICE} \\" >&2
            echo "      --project ${PROJECT} --region ${REGION} --member=allUsers --role=roles/run.invoker" >&2
        else
            set +x
            echo "Binding removed — the service should now be closed at the edge again." >&2
        fi
        echo "Deploy FAILED the smoke check. This is a FAILURE, not a success." >&2
        exit 1
    fi
fi

# ── Teardown — printed, never executed ───────────────────────────────────────
# The demo URL must not outlive the demo (see the warning at the top).
cat <<EOF

Deployed and smoke-checked. Mode: ALLOW_UNAUTH=${ALLOW_UNAUTH}.
  * true:  the URL is public at Google's edge; the console's own auth refuses
           anonymous access (401 on protected routes, /_health 200).
  * false: the URL is closed at the edge (403 for anonymous requests).

TEARDOWN — run these when the demo window is over (docs/gcp-setup.md §9):

    # Delete the Cloud Run service and its public URL:
    gcloud run services delete ${SERVICE} --project ${PROJECT} --region ${REGION} --quiet

    # Stop Cloud SQL. THIS is the only resource that bills whether or not
    # you use it (db-f1-micro ~\$8-10/month while RUNNABLE; storage-only,
    # ~\$1.70/month, while stopped):
    gcloud sql instances patch ${INSTANCE##*:} --project ${PROJECT} --activation-policy=NEVER

    # Full nuke, if you want zero ongoing spend (recoverable for 30 days):
    # gcloud projects delete ${PROJECT}
EOF
