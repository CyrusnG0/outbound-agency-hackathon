"""
Thin reusable wrapper around the Anthropic and google-genai SDKs for
structured (Pydantic-validated) LLM output.

Every LLM-backed node in the pipeline (summarize_company, detect_signals, later
drafting/classification nodes) calls ``call_structured()`` rather than each
hand-rolling its own provider client call.  This module also enforces the
model-pinning discipline: application code never hardcodes a model string;
instead, ``_resolve_model()`` reads a stable role alias from
``config/models.yaml`` and resolves it to the actual model via an environment
variable.  The system refuses to boot if that env var is unset — no silent
fallback to an SDK default.

Three providers are supported behind one interface: "anthropic" (the real
Anthropic API), "deepseek" (its Anthropic-compatible endpoint), and "gemini"
(the google-genai SDK, in either API-key or Vertex AI mode).  Callers never
see the difference — they get a validated Pydantic instance or one of the
three documented exception types.
"""

import hashlib
import os

import anthropic
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
import httpx
from pydantic import BaseModel, ValidationError

# Load .env into the process environment before anything else in this module
# needs an API key or model pin. override=False (the default) means a real,
# already-set environment variable always wins over .env — .env is a
# convenience for local dev, never an override of the real environment.
load_dotenv()


# ── Exception types ───────────────────────────────────────────────────────────
# Three distinct exception types rather than one generic "LLM call failed" so
# callers like summarize_company can log the specific failure mode, and so future
# callers can potentially react differently to "model said nothing" (maybe retry
# with a different prompt) vs. "model said something that didn't match the
# schema" (maybe the schema itself is too strict for this input) vs. "the HTTP
# transport itself failed" (maybe retry after a pause, maybe fail immediately —
# see LLMTransportError.retryable).

class LLMSchemaValidationError(Exception):
    """Raised when the model returned structured output but it failed Pydantic
    validation against the expected response_schema.

    On the anthropic/deepseek path this means the tool_use block's ``input``
    dict didn't validate; on the gemini path it means ``response.parsed``
    (handed back as a raw dict) didn't validate.  Either way, this is one of
    the three documented failure modes of ``call_structured``.  It means the
    model *tried* to produce structured output but got it wrong — the shape or
    values didn't satisfy the schema's constraints."""
    pass


class LLMEmptyResponseError(Exception):
    """Raised when the model returned no structured output at all.

    This is the second of the three documented failure modes of
    ``call_structured``.  On the anthropic/deepseek path it means the model
    didn't call the ``emit_result`` tool — possibly because it refused,
    misread the instructions, or produced only prose.  On the gemini path it
    means ``response.parsed`` is None.  This is distinct from a schema-mismatch
    because the fix is usually different (e.g. re-prompt rather than relax the
    schema)."""
    pass


class LLMTransportError(Exception):
    """Raised when the provider's HTTP transport failed before any structured
    output could be produced.

    This is the third documented failure mode of ``call_structured``, and it is
    categorically different from the two above: the model never got a chance to
    answer (or its answer never arrived) — the failure happened at the HTTP
    layer (rate limit, provider overload, dropped connection, bad API key,
    malformed request, ...).  ``retryable`` tells the caller whether an
    identical second attempt could plausibly succeed, so the bounded retry loop
    in the caller can skip doomed attempts instead of burning them.
    ``status_code`` is the HTTP status the provider sent, or None when no HTTP
    response was ever received (connection reset / timeout)."""

    def __init__(self, message: str, *, provider: str, status_code: int | None, retryable: bool):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


# Fixed pause a caller takes before re-attempting a retryable transport error.
# Deliberately NOT exponential backoff: the retry count itself stays bounded at
# the callers' existing two attempts (Golden Rule: retries must be bounded), and
# a single fixed pause is enough to make the common case (a transient 429 or
# 503) likely to succeed on the second try.  A 429 retried with zero delay is
# almost certain to 429 again — the provider's rate-limit window simply hasn't
# elapsed yet.  Public (no underscore) because the two caller modules import it
# so their sleep and the tests' assertions can't drift apart from this value.
TRANSPORT_RETRY_SLEEP_SECONDS = 2.0


def _classify_transport_error(provider: str, exc: Exception) -> LLMTransportError:
    """Convert a provider SDK transport exception into an ``LLMTransportError``.

    Pure function — no I/O, no client calls, no retries; it only maps an SDK
    exception onto the three facts a caller needs (provider, status_code,
    retryable).  The mapping below is measured fact against the pinned SDKs
    (anthropic==0.120.2, google-genai), not guesswork — the two SDK families
    put the HTTP status in different places, and one of the anthropic classes
    has no status at all.
    """
    if provider == "gemini":
        # Two exception families can arrive here (see the widened except tuple
        # in _call_gemini): genai_errors.APIError — google-genai's own
        # status-carrying family — and raw httpx.RequestError, which the SDK
        # lets escape unwrapped on connection-level failures.  Branch on the
        # type with an explicit isinstance check rather than
        # getattr(exc, "code", None): it must be obvious which SDK family
        # carries a status, because httpx.RequestError has no .code attribute
        # at all and reading it unconditionally turned a clean connection
        # failure into an AttributeError crash.
        if isinstance(exc, genai_errors.APIError):
            # google-genai puts the HTTP status on APIError.code, and both
            # ClientError (4xx) and ServerError (5xx) subclass APIError — so
            # this one attribute covers every status-carrying failure the
            # gemini SDK raises out of generate_content.
            status_code = exc.code
        else:
            # A raw httpx.RequestError (ConnectError, ReadTimeout, ...): the
            # request never received an HTTP response, so there IS no status
            # code to report.  None also feeds the retryable predicate below,
            # which classifies it retryable=True — correct, because a dropped
            # connection is worth exactly one more try.
            status_code = None
    elif isinstance(exc, anthropic.APIConnectionError):
        # Connection reset / timeout (APIConnectionError and its
        # APITimeoutError subclass): the request never received an HTTP
        # response, so there IS no status code — and the class has no
        # status_code attribute at all.  This branch is checked FIRST, before
        # the APIStatusError branch below, precisely because touching
        # exc.status_code on a connection error would raise AttributeError.
        status_code = None
    else:
        # Everything else the anthropic-family try block lets through is an
        # APIStatusError (or a subclass: RateLimitError, InternalServerError,
        # AuthenticationError, ...) — those all carry the HTTP status the
        # server actually sent.
        status_code = exc.status_code

    # Retryable predicate — identical for all providers, factored out here so
    # "should we bother trying again" has one definition, not three.
    retryable = (
        status_code is None               # connection/timeout — worth one more try
        or status_code in (408, 409, 429)  # timeout / conflict / rate limit
        or status_code >= 500             # provider-side fault (503 overload, etc.)
    )
    # Everything else — 400 malformed request, 401 bad key, 403 denied, 404
    # wrong model — is NOT retryable: a second identical call cannot fix a
    # bad request or a bad key, it just costs another round trip.  Fail fast
    # instead so the target lands in "failed" without burning the attempt.

    # Name the provider and the status in the message itself so a steps row is
    # self-explanatory without opening the SDK (or the raw response body).
    status_part = f"status {status_code}" if status_code is not None else "no HTTP response"
    return LLMTransportError(
        f"{provider} transport error ({status_part}): {exc}",
        provider=provider,
        status_code=status_code,
        retryable=retryable,
    )


# ── Model resolution ──────────────────────────────────────────────────────────

def _resolve_model(model_alias: str, config_path: str = "config/models.yaml") -> tuple[str, str]:
    """Resolve a stable role alias (e.g. "research_model") to (provider, model)
    via a two-step lookup:

    1. Read config/models.yaml to get the "{provider}.{role}" alias (e.g.
       "deepseek.v4_flash").
    2. Derive the env var name from that alias (uppercased, dots→underscores,
       + "_MODEL") and read it.

    Returns the provider prefix (e.g. "deepseek") alongside the resolved model
    string so call_structured can pick the right API client configuration —
    different providers need different base_url/api_key wiring even though
    they share the same alias-resolution mechanism.

    Raises RuntimeError if the env var is unset — the system must refuse to
    boot on an unpinned model rather than guessing which model to call.
    """
    with open(config_path) as f:
        aliases = yaml.safe_load(f)
    # The raw alias value, e.g. "deepseek.v4_flash" — split once on "." to
    # get the provider prefix. This is what call_structured uses to decide
    # which API client to construct.
    alias_value = aliases[model_alias]
    provider = alias_value.split(".", 1)[0]
    # Derive the env var name: uppercase the alias, replace dots with
    # underscores, append _MODEL. E.g. "deepseek.v4_flash" becomes
    # "DEEPSEEK_V4_FLASH_MODEL". Convention copied from docs/data-flow.md §9.
    env_var = alias_value.upper().replace(".", "_") + "_MODEL"
    model = os.environ.get(env_var)
    if not model:
        raise RuntimeError(f"{env_var} is not set — refusing to boot on an unpinned model")
    return provider, model


# DeepSeek's Anthropic-compatible endpoint — confirmed against DeepSeek's own
# API docs (https://api-docs.deepseek.com/guides/anthropic_api) and a live
# test call by the lead before this ticket was written. Accepts the standard
# Anthropic Python SDK's request shape, including forced tool_choice, which
# is exactly what call_structured's emit_result pattern needs. Not a secret —
# hardcoded here the same way the Anthropic SDK bakes in its own default
# base_url.
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"

# Gemini's output-token budget. Deliberately 8x the anthropic path's
# max_tokens=1024, because the two numbers do NOT mean the same thing:
# Gemini 3.x bills its internal "thinking" tokens against max_output_tokens,
# while Anthropic's max_tokens covers only the visible reply. Measured live
# on 2026-08-19 against gemini-3.5-flash with a realistic research blob:
# thinking alone consumed 979 of 1024 tokens, leaving 26 tokens of JSON,
# finish_reason=MAX_TOKENS and response.parsed=None — i.e. every LLM node in
# the pipeline would have failed. Observed worst case at a healthy budget is
# ~713 thinking + ~177 output, so 8192 is generous headroom. This is a cap,
# not a spend: thinking tokens are billed whether or not the cap is raised.
_GEMINI_MAX_OUTPUT_TOKENS = 8192

# ── Request-level transport timeouts (ticket B1g) ────────────────────────────
# Every client this module constructs gets an explicit request timeout, so a
# hung socket raises instead of parking the pipeline forever (the measured
# 9h48m hang had ESTABLISHED-but-idle connections and no timeout anywhere).
# WHY 300s per request: healthy structured calls (summarize/detect) complete
# in seconds on the measured runs; 300s is two orders of magnitude of
# headroom for a slow provider, while still bounding the worst case — the
# callers' A4c retry loop (2 attempts + one 2s pause) then costs at most
# ~602s per node, and the per-target ceiling in app/agents/phase1.py (600s,
# PHASE1_TARGET_TIMEOUT_SECONDS) cuts off anything beyond that anyway.
# A timeout that fires here raises the SDK's timeout exception, which
# _classify_transport_error maps to retryable=True (status_code=None), so
# A4c's bounded retry treats it as a transient transport failure.
_LLM_REQUEST_TIMEOUT_SECONDS = 300.0
# The SAME deadline for the gemini client, in the unit google-genai actually
# uses: HttpOptions.timeout is MILLISECONDS (verified against the installed
# types.py docstring and _api_client.get_timeout_in_seconds, which divides by
# 1000).  Passing seconds here would silently set a 300ms timeout and fail
# every real call — the off-by-1000 trap, kept as one derived constant so the
# two units can never drift apart.
_GEMINI_HTTP_TIMEOUT_MS = int(_LLM_REQUEST_TIMEOUT_SECONDS * 1000)


def _build_client(provider: str) -> anthropic.Anthropic | genai.Client:
    """Construct the right client object for the resolved provider.

    "anthropic" and "deepseek" both return anthropic.Anthropic instances
    (deepseek speaks the Anthropic wire protocol at its own base_url);
    "gemini" returns a google-genai Client in one of two auth modes — Vertex
    AI (Cloud Run deployment, ADC credentials, no API key) or the Gemini API
    key (local dev).  Any other provider prefix in config/models.yaml raises
    immediately here rather than silently constructing a client that would
    fail later at the API boundary with a confusing error."""
    if provider == "anthropic":
        # Default client — the SDK reads ANTHROPIC_API_KEY from the
        # environment itself.  timeout= is the anthropic SDK's request
        # timeout in SECONDS (a float — verified against the installed
        # signature: ``timeout: float | Timeout | None``); when it fires the
        # SDK raises APITimeoutError, which the caller's except tuple catches
        # and _classify_transport_error marks retryable (B1g).
        return anthropic.Anthropic(timeout=_LLM_REQUEST_TIMEOUT_SECONDS)
    if provider == "deepseek":
        # DeepSeek needs its own key, read directly (not via the Anthropic
        # SDK's default env var, which would collide with a real Anthropic
        # key if both providers are configured at once).  Same timeout
        # wiring as the anthropic branch — deepseek speaks the Anthropic wire
        # protocol at its own base_url, so the SDK applies the timeout
        # identically.
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set — refusing to call deepseek without a key")
        return anthropic.Anthropic(
            api_key=api_key, base_url=_DEEPSEEK_BASE_URL,
            timeout=_LLM_REQUEST_TIMEOUT_SECONDS,
        )
    if provider == "gemini":
        # Vertex mode is chosen by an explicit env toggle — never inferred
        # from which vars happen to be present — so an environment cannot
        # silently flip between ADC auth and API-key auth.  Both gemini
        # branches pass http_options= carrying the request timeout — the
        # google-genai client knob (per-request config http_options are
        # patched over the client's at call time), in MILLISECONDS
        # (_GEMINI_HTTP_TIMEOUT_MS — see the constant's comment for the
        # verified unit).
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
            # Vertex AI needs a project and location to know where the model
            # runs; it authenticates via ADC (Cloud Run service account), so
            # no API key exists in this mode.
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set — vertexai mode requires a project")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION")
            if not location:
                raise RuntimeError("GOOGLE_CLOUD_LOCATION is not set — vertexai mode requires a location")
            return genai.Client(
                vertexai=True, project=project, location=location,
                http_options=types.HttpOptions(timeout=_GEMINI_HTTP_TIMEOUT_MS),
            )
        # Gemini API key mode (local dev). Refuse loudly without a key —
        # same discipline as the deepseek branch: never let the SDK fall
        # back to some default credential it might find elsewhere.
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set — refusing to call gemini without a key")
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=_GEMINI_HTTP_TIMEOUT_MS),
        )
    # Unknown provider — fail loudly rather than guessing a client config
    # that will break at the API boundary in a way that's harder to debug.
    raise RuntimeError(
        f"call_structured has no client configuration for provider '{provider}' — "
        f"only 'anthropic', 'deepseek', and 'gemini' are supported today"
    )


def _extra_request_kwargs(provider: str) -> dict:
    """Provider-specific extra kwargs merged into every messages.create() call.

    DeepSeek's deepseek-v4-flash defaults to an extended-thinking mode that is
    incompatible with forced tool_choice — confirmed via a live API call
    during this task's design: without this, the API returns
    '400 Bad Request: Thinking mode does not support this tool_choice'.
    Anthropic's own API has no such default-on behavior that would break
    forced tool_choice, so this is scoped to deepseek only — never applied to
    the anthropic path, to avoid changing already-working behavior.  It is
    also never consulted on the gemini path, whose SDK must not receive
    anthropic-only params at all."""
    if provider == "deepseek":
        return {"thinking": {"type": "disabled"}}
    return {}


# ── Structured LLM call ───────────────────────────────────────────────────────

def call_structured(
    *, model_alias: str, system_prompt: str, user_content: str, response_schema: type[BaseModel]
) -> BaseModel:
    """Call the LLM and return a validated Pydantic model instance.

    Supports three providers — "anthropic" (the real API), "deepseek" (via
    its Anthropic-compatible endpoint), and "gemini" (google-genai SDK) —
    chosen per-call based on the resolved model_alias's provider prefix in
    config/models.yaml.

    The anthropic/deepseek path uses the Anthropic SDK's tool-use pattern with
    a single forced tool call named ``emit_result`` whose ``input_schema`` is
    the Pydantic model's JSON schema; the gemini path uses the google-genai
    SDK's structured-output mode (``response_mime_type="application/json"``
    plus ``response_schema``).  Both paths end in the same guarantee: this
    function returns a schema-validated Pydantic instance or raises one of the
    three documented exception types.

    Raises:
        LLMEmptyResponseError: The model returned no structured output (no
            tool_use block, or gemini ``response.parsed`` is None).
        LLMSchemaValidationError: The structured output failed Pydantic
            validation against ``response_schema``.
        LLMTransportError: The provider's HTTP transport failed (rate limit,
            provider overload, dropped connection, bad key, ...).  Its
            ``retryable`` flag tells the caller whether an identical retry
            could plausibly succeed; ``status_code`` is None when no HTTP
            response was ever received.
        RuntimeError: The resolved provider has no supported client config,
            or its required API key / project / location env var is unset.
    """
    # Resolve the stable role alias to (provider, model) via the two-step
    # lookup described in _resolve_model's docstring. Called on every
    # invocation so env var / config changes take effect without a restart.
    provider, model = _resolve_model(model_alias)
    # Construct the right client for this provider — an anthropic.Anthropic
    # for anthropic/deepseek (different base_url/api_key each), a
    # google-genai Client for gemini.
    client = _build_client(provider)
    if provider == "gemini":
        # Gemini has its own SDK request shape (contents + config, structured
        # output mode) — dispatch to the gemini helper. The anthropic-path
        # extra kwargs below are deliberately NOT computed for gemini: its
        # SDK must never receive anthropic-only params like
        # thinking={"type": "disabled"}.
        return _call_gemini(
            client,
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            response_schema=response_schema,
        )
    # Any provider-specific extra request kwargs for the anthropic wire
    # protocol (currently: disabling thinking mode for deepseek only).
    extra_kwargs = _extra_request_kwargs(provider)
    return _call_anthropic_compatible(
        client,
        # The helper serves BOTH "anthropic" and "deepseek", so it needs the
        # resolved provider name to classify transport errors correctly (the
        # exception classes are the same, but the error message must say which
        # provider actually failed).
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        response_schema=response_schema,
        extra_kwargs=extra_kwargs,
    )


def _call_anthropic_compatible(
    client: anthropic.Anthropic,
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str,
    response_schema: type[BaseModel],
    extra_kwargs: dict,
) -> BaseModel:
    """Run the Anthropic-SDK request path and return the validated Pydantic
    instance.

    Used by both the "anthropic" and "deepseek" providers — deepseek speaks
    the Anthropic wire protocol at its own base_url, so the request shape is
    identical and only the client wiring differs.  This is the pre-gemini
    body of ``call_structured`` moved verbatim: forced ``emit_result``
    tool_choice, tool_use block extraction, and the two output-failure
    exception types are unchanged; the only addition is the transport-error
    wrapping around the network call below.
    """
    # Call the Messages API with a single forced tool named "emit_result".
    # The tool's input_schema is the Pydantic model's JSON schema — the model
    # must fill in the fields of that schema. This is the reliable way to get
    # structured output: the model is constrained to produce valid JSON
    # matching a known shape, rather than arbitrary text that may or may not
    # parse.
    #
    # The try block wraps ONLY client.messages.create — the network call.
    # Deliberately NOT widened to cover the response parsing, the
    # response.parsed inspection, the finish_reason branch, or the
    # model_validate calls below: those already raise their own documented
    # exception types (LLMEmptyResponseError / LLMSchemaValidationError), and
    # swallowing them into LLMTransportError would destroy the failure
    # taxonomy and mislead callers into retrying model-output problems as if
    # they were provider hiccups.
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,  # Company profiles and signals fit well within 1K tokens.
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            tools=[
                {
                    "name": "emit_result",
                    "description": "Emit the structured result.",
                    "input_schema": response_schema.model_json_schema(),
                }
            ],
            # Force the model to call exactly "emit_result" — it cannot choose to
            # reply in prose or call a different tool, so every response is either
            # a valid tool_use or nothing (which we catch below).
            tool_choice={"type": "tool", "name": "emit_result"},
            **extra_kwargs,
        )
    except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
        # The two anthropic-SDK families that mean "the transport failed":
        # connection/timeout failures (APIConnectionError, no status code) and
        # HTTP-status failures (APIStatusError and its subclasses).  Convert
        # them to the single documented LLMTransportError so callers branch on
        # one type; `from exc` keeps the original SDK traceback chained for
        # debugging.  Anything the SDK raises that is NOT in this tuple (e.g.
        # a bug in our own request construction) propagates untouched.
        raise _classify_transport_error(provider, exc) from exc

    # Extract all tool_use blocks from the response content. In theory there
    # should be exactly one (since tool_choice forces emit_result), but we
    # filter rather than assuming — if the SDK ever returns multiple blocks
    # we take the first rather than crashing.
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    if not tool_use_blocks:
        raise LLMEmptyResponseError("model returned no tool_use block")

    raw = tool_use_blocks[0].input
    try:
        return response_schema.model_validate(raw)
    except ValidationError as exc:
        raise LLMSchemaValidationError(str(exc)) from exc


def _call_gemini(
    client: genai.Client,
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    response_schema: type[BaseModel],
) -> BaseModel:
    """Run the google-genai request path and return the validated Pydantic
    instance.

    Gemini has no tool-use trick; instead the SDK's structured-output mode
    (``response_mime_type="application/json"`` + ``response_schema``) makes
    the API itself constrain the output to the schema's JSON shape.  The same
    three failure modes as the anthropic path are surfaced with the same three
    exception types, so callers (summarize_company, detect_signals) branch on
    them identically regardless of provider.
    """
    # Single generate_content call: the system prompt goes in the config (not
    # the contents list), the Pydantic model class is passed directly as
    # response_schema (the SDK converts it to JSON schema internally), and
    # max_output_tokens mirrors the anthropic path's max_tokens=1024 cap.
    #
    # The try block wraps ONLY client.models.generate_content — the network
    # call.  Same narrowness rule as the anthropic path: the parsed-None
    # inspection and the model_validate calls below raise
    # LLMEmptyResponseError / LLMSchemaValidationError and must never be
    # reclassified as transport errors by a widened try block.
    #
    # The provider name is hardcoded to "gemini" here rather than passed in:
    # unlike _call_anthropic_compatible (which serves both "anthropic" and
    # "deepseek"), this helper is gemini-only by construction, so there is no
    # provider to disambiguate.
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_schema,
                max_output_tokens=_GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )
    except (genai_errors.APIError, httpx.RequestError) as exc:
        # Two exception families share this handler, and the asymmetry with
        # the anthropic path is the whole reason both are here:
        #  - genai_errors.APIError (with ClientError / ServerError subclasses
        #    for 4xx / 5xx) is what google-genai raises when the transport
        #    fails WITH an HTTP response — the same "routine provider failure"
        #    family the anthropic path wraps.
        #  - httpx.RequestError is the raw transport exception google-genai
        #    lets escape UNWRAPPED when no HTTP response was ever received
        #    (DNS failure, connection reset, socket timeout).  The anthropic
        #    SDK wraps those same failures into anthropic.APIConnectionError,
        #    so its except tuple never needed this type — only this gemini
        #    branch does.  Do not "tidy" this tuple back to APIError alone.
        #    Deliberately NOT httpx.HTTPError: its HTTPStatusError subclass
        #    represents failures WITH a response, which already arrive as
        #    genai_errors.APIError and must keep taking that branch so the
        #    status code survives into _classify_transport_error.
        # Convert the failure to the single documented LLMTransportError the
        # same way as the anthropic path, with the SDK traceback chained via
        # `from exc`.
        raise _classify_transport_error("gemini", exc) from exc
    parsed = response.parsed
    if parsed is None:
        # The gemini analogue of "model returned no tool_use block". Two very
        # different causes land here, and telling them apart is the difference
        # between "raise the token cap" and "fix the prompt" — so inspect
        # finish_reason rather than reporting a bare empty response. A silent
        # truncation is exactly the bug that broke the first live run of this
        # code, and it cost a debugging session to identify; the message below
        # is what makes the next occurrence self-explanatory.
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        # thoughts_token_count is how many tokens the model spent reasoning
        # before emitting anything — the number that actually explains a
        # truncation, so it belongs in the error text.
        thoughts = getattr(response.usage_metadata, "thoughts_token_count", None)
        if finish_reason is not None and "MAX_TOKENS" in str(finish_reason):
            raise LLMEmptyResponseError(
                f"gemini hit max_output_tokens={_GEMINI_MAX_OUTPUT_TOKENS} and truncated its "
                f"JSON before it could be parsed (thinking tokens: {thoughts}). "
                f"Raise _GEMINI_MAX_OUTPUT_TOKENS or shorten the input."
            )
        # Genuinely empty for some other reason (safety block, refusal, empty
        # candidate list). Surface finish_reason so the cause isn't guesswork.
        raise LLMEmptyResponseError(
            f"gemini returned no parsed structured output (finish_reason={finish_reason})"
        )
    if isinstance(parsed, response_schema):
        # The SDK already validated the payload into the schema (it does this
        # itself when response_schema is a Pydantic class) — return it
        # directly rather than re-validating.
        return parsed
    try:
        # The SDK handed back a raw dict (e.g. when it could not do the
        # validation itself) — run the same Pydantic validation the
        # anthropic path runs on the tool_use input.
        return response_schema.model_validate(parsed)
    except ValidationError as exc:
        # Same exception type as the anthropic path: the model produced JSON
        # that does not satisfy the schema — callers branch on this the same
        # way for every provider.
        raise LLMSchemaValidationError(str(exc)) from exc


# ── Call fingerprinting ───────────────────────────────────────────────────────

def hash_call(system_prompt: str, user_content: str) -> str:
    """Produce a short, deterministic fingerprint of a specific prompt+input pair.

    The result is logged in ``steps.model_call_hash`` so an operator can
    correlate trace-log rows without storing the full prompt text redundantly
    on every row.  Two rows with the same hash were made with the same prompt
    and input — useful for spotting "this is the same call that succeeded
    earlier" or "this identical call failed on retry."

    The hash is truncated to 16 hex chars: short enough to be a compact log
    field, long enough (64 bits) to avoid practical collisions at this
    single-operator system's scale.
    """
    # Concatenate the system prompt and user content, hash with SHA-256, and
    # take the first 16 hex characters.  The "sha256:" prefix makes it
    # self-identifying in logs — a reader can tell at a glance what algorithm
    # produced the hash.
    return "sha256:" + hashlib.sha256((system_prompt + user_content).encode()).hexdigest()[:16]
