# tests/test_llm.py
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types  # build the exact HttpOptions the B1g client wiring must pass

from app.llm import (
    _GEMINI_HTTP_TIMEOUT_MS,  # B1g: the gemini client's request timeout, in the SDK's millisecond unit
    _GEMINI_MAX_OUTPUT_TOKENS,
    _LLM_REQUEST_TIMEOUT_SECONDS,  # B1g: the anthropic/deepseek client's request timeout, in seconds
    LLMEmptyResponseError,
    LLMSchemaValidationError,
    LLMTransportError,
    _build_client,
    _resolve_model,
    call_structured,
)
from app.schemas import CompanyProfile


def _write_config(tmp_path, content: str) -> str:
    path = tmp_path / "models.yaml"
    path.write_text(content)
    return str(path)


def test_resolve_model_returns_provider_and_model_for_deepseek_alias(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, "research_model: deepseek.v4_flash\n")
    monkeypatch.setenv("DEEPSEEK_V4_FLASH_MODEL", "deepseek-v4-flash")
    provider, model = _resolve_model("research_model", config_path=config_path)
    assert provider == "deepseek"
    assert model == "deepseek-v4-flash"


def test_resolve_model_returns_provider_and_model_for_anthropic_alias(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, "research_model: anthropic.claude_primary\n")
    monkeypatch.setenv("ANTHROPIC_CLAUDE_PRIMARY_MODEL", "claude-sonnet-stable")
    provider, model = _resolve_model("research_model", config_path=config_path)
    assert provider == "anthropic"
    assert model == "claude-sonnet-stable"


def test_resolve_model_raises_when_env_var_unset(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, "research_model: deepseek.v4_flash\n")
    monkeypatch.delenv("DEEPSEEK_V4_FLASH_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        _resolve_model("research_model", config_path=config_path)


def _fake_anthropic_response():
    """Build a fake anthropic SDK response with one tool_use block matching
    CompanyProfile's schema, so call_structured's post-processing (extracting
    the block, validating against the Pydantic schema) runs for real without
    a live API call."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"one_line_summary": "Acme is a logistics company.", "confidence": 0.8}
    response = MagicMock()
    response.content = [block]
    return response


def test_call_structured_uses_deepseek_base_url_and_key_and_disables_thinking(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("app.llm._resolve_model", return_value=("deepseek", "deepseek-v4-flash")), \
         patch("app.llm.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_anthropic_response()
        mock_anthropic_cls.return_value = mock_client

        result = call_structured(
            model_alias="research_model", system_prompt="sys", user_content="text",
            response_schema=CompanyProfile,
        )

    # The client must be constructed with DeepSeek's Anthropic-compatible
    # endpoint and the DeepSeek-specific API key — never the real Anthropic
    # endpoint for this provider.
    mock_anthropic_cls.assert_called_once_with(
        api_key="test-deepseek-key", base_url="https://api.deepseek.com/anthropic",
        timeout=_LLM_REQUEST_TIMEOUT_SECONDS,  # B1g: the request timeout must reach every client this module builds
    )
    # thinking must be disabled for deepseek — omitting this makes forced
    # tool_choice fail against the real API (confirmed live by the lead).
    _, call_kwargs = mock_client.messages.create.call_args
    assert call_kwargs.get("thinking") == {"type": "disabled"}
    assert isinstance(result, CompanyProfile)
    assert result.one_line_summary == "Acme is a logistics company."


def test_call_structured_uses_default_anthropic_client_with_no_thinking_param(monkeypatch):
    with patch("app.llm._resolve_model", return_value=("anthropic", "claude-sonnet-stable")), \
         patch("app.llm.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_anthropic_response()
        mock_anthropic_cls.return_value = mock_client

        call_structured(
            model_alias="research_model", system_prompt="sys", user_content="text",
            response_schema=CompanyProfile,
        )

    # The real Anthropic path keeps default client construction (SDK reads
    # ANTHROPIC_API_KEY itself) with no explicit base_url/api_key and no
    # thinking param — that workaround is deepseek-specific.  B1g adds only
    # the request timeout.
    mock_anthropic_cls.assert_called_once_with(timeout=_LLM_REQUEST_TIMEOUT_SECONDS)
    _, call_kwargs = mock_client.messages.create.call_args
    assert "thinking" not in call_kwargs


def test_call_structured_raises_for_unsupported_provider(monkeypatch):
    with patch("app.llm._resolve_model", return_value=("openai", "gpt-5-mini-stable")), \
         patch("app.llm.anthropic.Anthropic") as mock_anthropic_cls:
        with pytest.raises(RuntimeError):
            call_structured(
                model_alias="reply_classifier_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )
    # Must fail before ever constructing a client for a provider it can't
    # actually talk to — no partial/incorrect client construction.
    mock_anthropic_cls.assert_not_called()


# ── Gemini provider tests ────────────────────────────────────────────────────
# All mocked — no live API calls, no network. The google-genai SDK client is
# patched at the class level the same way the anthropic SDK client is patched
# in the tests above.

def test_resolve_model_returns_provider_and_model_for_gemini_alias(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, "research_model: gemini.flash\n")
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash")
    provider, model = _resolve_model("research_model", config_path=config_path)
    assert provider == "gemini"
    assert model == "gemini-3.5-flash"


def test_build_client_gemini_vertexai_mode(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    with patch("app.llm.genai.Client") as mock_genai_cls:
        client = _build_client("gemini")

    # Vertex mode is selected by the env toggle and must pass project and
    # location — the SDK needs both to know where the model runs. No api_key
    # param: this mode authenticates via ADC, never an API key.  B1g adds
    # http_options carrying the request timeout, in MILLISECONDS (the unit
    # the installed SDK uses — the off-by-1000 trap, pinned here by
    # asserting the exact derived constant).
    mock_genai_cls.assert_called_once_with(
        vertexai=True, project="my-gcp-project", location="us-central1",
        http_options=types.HttpOptions(timeout=_GEMINI_HTTP_TIMEOUT_MS),
    )
    assert client is mock_genai_cls.return_value


def test_build_client_gemini_api_key_mode(monkeypatch):
    # Deterministic: the vertex toggle must be absent, not merely "false",
    # so this test can never accidentally take the vertex branch.
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm.genai.Client") as mock_genai_cls:
        client = _build_client("gemini")

    # API-key mode (local dev): the key is passed explicitly to the client
    # constructor — no vertexai/project/location params in this mode.  The
    # B1g timeout rides in http_options here too (same ms constant).
    mock_genai_cls.assert_called_once_with(
        api_key="test-gemini-key",
        http_options=types.HttpOptions(timeout=_GEMINI_HTTP_TIMEOUT_MS),
    )
    assert client is mock_genai_cls.return_value


def test_build_client_gemini_raises_without_key_and_without_vertex(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Missing key and no vertex mode = refuse immediately, naming the exact
    # env var — never let the SDK fall back to a default credential.
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        _build_client("gemini")


def test_build_client_gemini_vertexai_raises_without_project(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Vertex mode without a project can't know where the model runs — fail
    # loudly naming the missing var rather than constructing a half-wired
    # client that would break at the API boundary.
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        _build_client("gemini")


def _fake_gemini_response(parsed):
    """Build a fake google-genai response carrying a ``parsed`` attribute, so
    call_structured's gemini post-processing (None check / instance check /
    dict validation) runs for real without a live API call."""
    response = MagicMock()
    response.parsed = parsed
    return response


def test_call_structured_gemini_happy_path_returns_validated_instance(monkeypatch):
    # Deterministic API-key mode, same setup discipline as the build_client
    # tests above.
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")
    expected = CompanyProfile(one_line_summary="Acme is a logistics company.", confidence=0.8)

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # The real SDK hands back an already-validated Pydantic instance when
        # response_schema is a Pydantic class — mirror that here.
        mock_client.models.generate_content.return_value = _fake_gemini_response(expected)
        mock_genai_cls.return_value = mock_client

        result = call_structured(
            model_alias="research_model", system_prompt="sys", user_content="text",
            response_schema=CompanyProfile,
        )

    # The client must be constructed in API-key mode with the env key, plus
    # the B1g request timeout in http_options.  Asserted as the EXACT
    # HttpOptions value — not ANY — because timeout is in MILLISECONDS and a
    # future edit that silently passes seconds (a 300ms timeout) is exactly
    # the off-by-1000 trap this assertion exists to catch.
    mock_genai_cls.assert_called_once_with(
        api_key="test-gemini-key",
        http_options=types.HttpOptions(timeout=_GEMINI_HTTP_TIMEOUT_MS),
    )
    # The gemini request shape: contents + GenerateContentConfig carrying the
    # system prompt, JSON mime type, the Pydantic schema itself, and gemini's
    # OWN output cap — deliberately not the anthropic path's 1024, because
    # gemini bills thinking tokens against this budget (see llm.py's constant).
    _, call_kwargs = mock_client.models.generate_content.call_args
    assert call_kwargs["model"] == "gemini-3.5-flash"
    assert call_kwargs["contents"] == "text"
    config = call_kwargs["config"]
    assert config.system_instruction == "sys"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is CompanyProfile
    assert config.max_output_tokens == _GEMINI_MAX_OUTPUT_TOKENS
    # The SDK-validated instance passes straight through.
    assert result is expected


def test_call_structured_gemini_validates_dict_into_schema(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # Defensive path: the SDK returns a raw dict instead of a validated
        # instance — call_structured must run the Pydantic validation itself.
        mock_client.models.generate_content.return_value = _fake_gemini_response(
            {"one_line_summary": "Acme is a logistics company.", "confidence": 0.8}
        )
        mock_genai_cls.return_value = mock_client

        result = call_structured(
            model_alias="research_model", system_prompt="sys", user_content="text",
            response_schema=CompanyProfile,
        )

    assert isinstance(result, CompanyProfile)
    assert result.one_line_summary == "Acme is a logistics company."


def test_call_structured_gemini_raises_empty_response_when_parsed_is_none(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _fake_gemini_response(None)
        mock_genai_cls.return_value = mock_client

        # parsed=None is the gemini analogue of "no tool_use block" — same
        # exception type so callers branch on it identically.
        with pytest.raises(LLMEmptyResponseError):
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )


def test_call_structured_gemini_raises_schema_error_when_dict_fails_validation(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # confidence=2.0 violates CompanyProfile's ge=0.0/le=1.0 clamp — the
        # model produced JSON, but it doesn't satisfy the schema.
        mock_client.models.generate_content.return_value = _fake_gemini_response(
            {"one_line_summary": "Acme is a logistics company.", "confidence": 2.0}
        )
        mock_genai_cls.return_value = mock_client

        # Same exception type as the anthropic path's schema-mismatch, so
        # callers branch on it identically for every provider.
        with pytest.raises(LLMSchemaValidationError):
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )


def test_call_structured_gemini_truncation_names_the_token_budget(monkeypatch):
    """A MAX_TOKENS finish must produce a truncation-specific message.

    Regression test for the bug found on the first live Gemini call: gemini
    3.x bills its thinking tokens against max_output_tokens, so a too-small
    budget yields truncated JSON and response.parsed is None. The generic
    "no parsed structured output" message pointed at the wrong cause and cost
    a debugging session; this locks in an error that names the cap and the
    thinking spend so the next occurrence is self-explanatory.
    """
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    # A truncated response: nothing parsed, a MAX_TOKENS finish reason, and a
    # thinking spend large enough to explain where the budget went.
    response = _fake_gemini_response(None)
    response.candidates = [MagicMock(finish_reason="FinishReason.MAX_TOKENS")]
    response.usage_metadata.thoughts_token_count = 979

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        mock_genai_cls.return_value = mock_client

        with pytest.raises(LLMEmptyResponseError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    message = str(excinfo.value)
    # The operator must be able to act on this without re-deriving the cause.
    assert "max_output_tokens" in message
    assert "979" in message


# ── Transport error classification tests ──────────────────────────────────────
# The provider SDKs raise a whole family of HTTP transport exceptions (rate
# limit, overload, connection drop, bad key) that call_structured must convert
# into LLMTransportError so callers can distinguish "retryable provider hiccup"
# from "my request is broken".  The SDK exception construction recipes below
# are verified against the installed SDK versions — the classes are fiddly to
# build by hand (they need httpx request/response objects), so do not
# "simplify" them.

def _anthropic_transport_request():
    """Build the httpx.Request every anthropic SDK transport error needs — a
    dummy request is enough because the SDK only reads it for its error
    message, never sends it."""
    return httpx.Request("POST", "https://example.invalid/v1")


def test_anthropic_rate_limit_error_raises_retryable_transport_error(monkeypatch):
    with patch("app.llm._resolve_model", return_value=("anthropic", "claude-sonnet-stable")), \
         patch("app.llm.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        # 429 = rate limit: transient, a retry after a pause can succeed.
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            "rate limited", response=httpx.Response(429, request=_anthropic_transport_request()), body=None,
        )
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(LLMTransportError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    err = excinfo.value
    assert err.status_code == 429
    assert err.retryable is True
    assert err.provider == "anthropic"
    # The message must be self-explanatory from a steps row: provider + status.
    assert "anthropic transport error" in str(err)
    assert "status 429" in str(err)


def test_anthropic_authentication_error_raises_non_retryable_transport_error(monkeypatch):
    with patch("app.llm._resolve_model", return_value=("anthropic", "claude-sonnet-stable")), \
         patch("app.llm.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        # 401 = bad key: a second identical call with the same key cannot fix it.
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            "bad key", response=httpx.Response(401, request=_anthropic_transport_request()), body=None,
        )
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(LLMTransportError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    err = excinfo.value
    assert err.status_code == 401
    assert err.retryable is False


def test_anthropic_connection_error_raises_retryable_transport_error_without_status(monkeypatch):
    with patch("app.llm._resolve_model", return_value=("anthropic", "claude-sonnet-stable")), \
         patch("app.llm.anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        # Connection reset: no HTTP response was ever received, so there is no
        # status code — the SDK class has no status_code attribute at all, and
        # the classifier must not touch it.
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            message="connection reset", request=_anthropic_transport_request(),
        )
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(LLMTransportError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    err = excinfo.value
    assert err.status_code is None
    assert err.retryable is True


def test_gemini_client_error_429_raises_retryable_transport_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # 429 quota exhausted: transient, retryable — the common gemini failure.
        mock_client.models.generate_content.side_effect = genai_errors.ClientError(
            429, {"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
        )
        mock_genai_cls.return_value = mock_client

        with pytest.raises(LLMTransportError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    err = excinfo.value
    assert err.status_code == 429
    assert err.retryable is True
    assert err.provider == "gemini"
    assert "gemini transport error" in str(err)
    assert "status 429" in str(err)


def test_gemini_server_error_503_is_retryable(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # 503 overload: provider-side fault, a retry after a pause can succeed.
        mock_client.models.generate_content.side_effect = genai_errors.ServerError(
            503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}},
        )
        mock_genai_cls.return_value = mock_client

        with pytest.raises(LLMTransportError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    assert excinfo.value.status_code == 503
    assert excinfo.value.retryable is True


def test_gemini_client_error_400_is_not_retryable(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # 400 malformed request: the request itself is wrong, so an identical
        # second call is guaranteed to fail the same way — not retryable.
        mock_client.models.generate_content.side_effect = genai_errors.ClientError(
            400, {"error": {"message": "bad request", "status": "INVALID_ARGUMENT"}},
        )
        mock_genai_cls.return_value = mock_client

        with pytest.raises(LLMTransportError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    assert excinfo.value.status_code == 400
    assert excinfo.value.retryable is False


def test_gemini_raw_httpx_connection_error_becomes_retryable_transport_error(monkeypatch):
    """A raw httpx.ConnectError escaping the gemini SDK must still become a
    retryable LLMTransportError.

    Regression test for the hole A4c left open on the gemini path: unlike the
    anthropic SDK (which wraps httpx transport failures into
    APIConnectionError), google-genai lets them escape generate_content
    unwrapped — measured live, a connection failure raises httpx.ConnectError
    and NOT google.genai.errors.APIError.  _call_gemini therefore catches
    httpx.RequestError by name; this test locks that in so a dropped
    connection can never crash the pipeline node as a raw SDK escape.
    """
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # DNS failure / connection refused: no HTTP response was ever
        # received, and google-genai does not wrap this — it escapes raw and
        # must be caught by the httpx.RequestError entry in _call_gemini's
        # except tuple.
        mock_client.models.generate_content.side_effect = httpx.ConnectError("connection refused")
        mock_genai_cls.return_value = mock_client

        with pytest.raises(LLMTransportError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    err = excinfo.value
    assert err.provider == "gemini"
    # No HTTP response was ever received, so there is no status code — and
    # the classifier must have taken the isinstance branch rather than
    # crashing on the missing .code attribute.
    assert err.status_code is None
    assert err.retryable is True
    assert "no HTTP response" in str(err)


def test_gemini_raw_httpx_read_timeout_becomes_retryable_transport_error(monkeypatch):
    """ReadTimeout must be caught too, via the httpx.RequestError BASE class.

    This proves the widened except tuple covers the whole transport-error
    family (ConnectError, ConnectTimeout, ReadTimeout, ReadError,
    RemoteProtocolError, ...) and not just the one class the first test
    happens to use — if the tuple named ConnectError specifically, this test
    would crash the pipeline node with a raw ReadTimeout escape.
    """
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # Socket read timeout mid-response: also no HTTP response received,
        # also let through unwrapped by google-genai.
        mock_client.models.generate_content.side_effect = httpx.ReadTimeout("timed out")
        mock_genai_cls.return_value = mock_client

        with pytest.raises(LLMTransportError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    err = excinfo.value
    assert err.provider == "gemini"
    assert err.status_code is None
    assert err.retryable is True


def test_schema_failure_is_not_reclassified_as_transport_error(monkeypatch):
    """A schema-validation failure must STILL raise LLMSchemaValidationError.

    This is the taxonomy regression test for the narrow try blocks in
    _call_anthropic_compatible / _call_gemini: if someone later widens the
    try block that wraps the network call to also cover the response parsing
    (the .parsed inspection / model_validate calls), a model-output problem
    like this one would be reclassified as LLMTransportError — and callers
    would then "retry" a broken schema instead of fixing it.  The model
    returned valid JSON here; only the Pydantic validation failed, so the
    failure is NOT a transport failure.
    """
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-gemini-key")

    with patch("app.llm._resolve_model", return_value=("gemini", "gemini-3.5-flash")), \
         patch("app.llm.genai.Client") as mock_genai_cls:
        mock_client = MagicMock()
        # confidence=2.0 violates CompanyProfile's 0.0-1.0 clamp — the model
        # produced JSON, but it doesn't satisfy the schema.
        mock_client.models.generate_content.return_value = _fake_gemini_response(
            {"one_line_summary": "Acme is a logistics company.", "confidence": 2.0}
        )
        mock_genai_cls.return_value = mock_client

        # pytest.raises(LLMSchemaValidationError) already proves the error was
        # NOT reclassified as LLMTransportError (a subclass mismatch would
        # fail the raises check) — asserted explicitly for readability.
        with pytest.raises(LLMSchemaValidationError) as excinfo:
            call_structured(
                model_alias="research_model", system_prompt="sys", user_content="text",
                response_schema=CompanyProfile,
            )

    assert not isinstance(excinfo.value, LLMTransportError)
