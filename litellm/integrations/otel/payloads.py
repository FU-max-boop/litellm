from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    cast,
)
from urllib.parse import urlsplit

from litellm.integrations.otel.semconv import (
    DEFAULT_BAGGAGE_METADATA_KEYS,
    GenAI,
    GenAIOperation,
    LiteLLM,
    resolve_operation,
    resolve_provider,
)

if TYPE_CHECKING:
    from litellm.types.services import ServiceLoggerPayload
    from litellm.types.utils import StandardLoggingPayload


# --- small typed coercion helpers (localize reading of heterogeneous dicts) -- #


def _as_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _as_str_tuple(value: object) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return None


# --- typed sub-structures ---------------------------------------------------- #


@dataclass(frozen=True)
class LLMRequestParams:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[Tuple[str, ...]] = None
    seed: Optional[int] = None

    @classmethod
    def from_model_parameters(cls, params: Mapping[str, object]) -> "LLMRequestParams":
        max_tokens = _as_int(params.get("max_tokens"))
        if max_tokens is None:
            max_tokens = _as_int(params.get("max_completion_tokens"))
        return cls(
            temperature=_as_float(params.get("temperature")),
            top_p=_as_float(params.get("top_p")),
            top_k=_as_int(params.get("top_k")),
            max_tokens=max_tokens,
            frequency_penalty=_as_float(params.get("frequency_penalty")),
            presence_penalty=_as_float(params.get("presence_penalty")),
            stop_sequences=_as_str_tuple(params.get("stop")),
            seed=_as_int(params.get("seed")),
        )


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass(frozen=True)
class SpanError:
    error_type: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class ServerInfo:
    address: Optional[str] = None
    port: Optional[int] = None

    @classmethod
    def from_api_base(cls, api_base: Optional[str]) -> Optional["ServerInfo"]:
        if not api_base:
            return None
        parsed = urlsplit(api_base if "://" in api_base else f"//{api_base}")
        if not parsed.hostname:
            return None
        return cls(address=parsed.hostname, port=parsed.port)


@dataclass(frozen=True)
class RequestIdentity:
    call_id: Optional[str] = None
    team_id: Optional[str] = None
    team_alias: Optional[str] = None
    key_hash: Optional[str] = None
    end_user: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: "StandardLoggingPayload") -> "RequestIdentity":
        raw_meta = cast(Mapping[str, object], payload.get("metadata") or {})
        metadata: Dict[str, str] = {}
        for key, value in raw_meta.items():
            if isinstance(value, (str, bool, int, float)):
                metadata[key] = str(value)
        return cls(
            call_id=_as_str(payload.get("litellm_call_id"))
            or _as_str(payload.get("id")),
            team_id=_as_str(raw_meta.get("team_id")),
            team_alias=_as_str(raw_meta.get("team_alias")),
            key_hash=_as_str(raw_meta.get("user_api_key_hash")),
            end_user=_as_str(payload.get("end_user")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class GuardrailSpanData:
    guardrail_name: str
    mode: Optional[str] = None
    status: Optional[str] = None
    masked_entity_count: Optional[int] = None


@dataclass(frozen=True)
class ServiceSpanData:
    service_name: str
    call_type: Optional[str] = None
    error: Optional[SpanError] = None
    # Arbitrary caller-supplied attributes to stamp on the service span (V1
    # contract: pass-through from ``async_service_*_hook(event_metadata=...)``).
    # The mapper owns how these are namespaced (canonical: ``litellm.metadata.*``;
    # legacy: bare keys to match V1).
    event_metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: "ServiceLoggerPayload",
        event_metadata: Optional[Mapping[str, object]] = None,
    ) -> "ServiceSpanData":
        # ``payload.service`` is a ``ServiceTypes(str, Enum)`` and ``error`` is
        # ``Optional[str]`` on the Pydantic model — no defensive reads needed.
        coerced: Dict[str, str] = {}
        for key, value in (event_metadata or {}).items():
            if value is None:
                coerced[key] = "None"
            elif isinstance(value, (str, bool, int, float)):
                coerced[key] = str(value)
            else:
                coerced[key] = str(value)
        return cls(
            service_name=payload.service.value,
            call_type=payload.call_type,
            error=SpanError(message=payload.error) if payload.error else None,
            event_metadata=coerced,
        )


@dataclass(frozen=True)
class ProxyRequestSpanData:
    http_method: str
    route: str
    url_path: Optional[str] = None
    status_code: Optional[int] = None
    identity: Optional[RequestIdentity] = None


@dataclass(frozen=True)
class ManagementSpanData:
    route: str
    error: Optional[SpanError] = None


# --- the primary LLM-call model ---------------------------------------------- #


@dataclass(frozen=True)
class ToolDefinition:
    """A single function/tool declared on a chat-completion request."""

    name: str
    description: Optional[str] = None
    parameters_json: Optional[str] = (
        None  # JSON-serialized schema (str so it's an AttrValue)
    )


@dataclass(frozen=True)
class LLMCallSpanData:
    operation: GenAIOperation
    provider: str
    request_model: str
    response_model: Optional[str]
    response_id: Optional[str]
    request_params: LLMRequestParams
    usage: LLMUsage
    finish_reasons: Tuple[str, ...]
    error: Optional[SpanError]
    response_cost: Optional[float]
    server: Optional[ServerInfo]
    identity: RequestIdentity
    is_streaming: Optional[bool] = None
    tools: Tuple[ToolDefinition, ...] = ()
    # Raw messages + response — needed by vendor mappers (OpenInference,
    # Langfuse, Weave) that stamp message-level attributes. ``messages_in`` is
    # the request payload, ``choices_out`` mirrors ``response.choices`` from
    # the StandardLoggingPayload. Both are typed as immutable mappings so the
    # dataclass stays hashable / frozen.
    messages_in: Tuple[Mapping[str, object], ...] = ()
    choices_out: Tuple[Mapping[str, object], ...] = ()
    system_fingerprint: Optional[str] = None

    @classmethod
    def from_standard_logging_payload(
        cls, payload: "StandardLoggingPayload"
    ) -> "LLMCallSpanData":
        model_parameters = cast(
            Mapping[str, object], payload.get("model_parameters") or {}
        )
        response = payload.get("response")
        response_id: Optional[str] = None
        response_model: Optional[str] = None
        finish_reasons: List[str] = []
        if isinstance(response, dict):
            response_id = _as_str(response.get("id"))
            response_model = _as_str(response.get("model"))
            choices = response.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        reason = _as_str(choice.get("finish_reason"))
                        if reason:
                            finish_reasons.append(reason)

        error: Optional[SpanError] = None
        if payload.get("status") == "failure":
            error_info = cast(
                Mapping[str, object], payload.get("error_information") or {}
            )
            error = SpanError(
                error_type=_as_str(error_info.get("error_class"))
                or _as_str(error_info.get("error_code")),
                message=_as_str(error_info.get("error_message"))
                or _as_str(payload.get("error_str")),
            )

        hidden_params = cast(Mapping[str, object], payload.get("hidden_params") or {})
        messages_in: Tuple[Mapping[str, object], ...] = ()
        raw_messages = payload.get("messages")
        if isinstance(raw_messages, list):
            messages_in = tuple(m for m in raw_messages if isinstance(m, dict))
        choices_out: Tuple[Mapping[str, object], ...] = ()
        system_fingerprint: Optional[str] = None
        if isinstance(response, dict):
            raw_choices = response.get("choices")
            if isinstance(raw_choices, list):
                choices_out = tuple(c for c in raw_choices if isinstance(c, dict))
            system_fingerprint = _as_str(response.get("system_fingerprint"))
        return cls(
            operation=resolve_operation(_as_str(payload.get("call_type"))),
            provider=resolve_provider(_as_str(payload.get("custom_llm_provider"))),
            request_model=_as_str(payload.get("model")) or "",
            response_model=response_model,
            response_id=response_id,
            request_params=LLMRequestParams.from_model_parameters(model_parameters),
            usage=LLMUsage(
                input_tokens=_as_int(payload.get("prompt_tokens")),
                output_tokens=_as_int(payload.get("completion_tokens")),
                total_tokens=_as_int(payload.get("total_tokens")),
            ),
            finish_reasons=tuple(finish_reasons),
            error=error,
            response_cost=_as_float(payload.get("response_cost")),
            server=ServerInfo.from_api_base(
                _as_str(payload.get("api_base"))
                or _as_str(hidden_params.get("api_base"))
            ),
            identity=RequestIdentity.from_payload(payload),
            is_streaming=_as_bool(payload.get("stream")),
            tools=_extract_tools(model_parameters),
            messages_in=messages_in,
            choices_out=choices_out,
            system_fingerprint=system_fingerprint,
        )


def _extract_tools(
    model_parameters: Mapping[str, object],
) -> Tuple[ToolDefinition, ...]:
    """Pull ``tools`` (the OpenAI / Anthropic function-calling shape) from request params.

    Accepts the chat-completion ``tools=[{"type":"function", "function": {...}}, ...]``
    shape; tolerates older ``functions=[...]`` shape too. Empty tuple if absent.
    """
    import json as _json

    raw_tools = model_parameters.get("tools")
    if not isinstance(raw_tools, list):
        raw_tools = model_parameters.get("functions")  # legacy OpenAI shape
    if not isinstance(raw_tools, list):
        return ()
    out: List[ToolDefinition] = []
    for entry in raw_tools:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") if "function" in entry else entry
        if not isinstance(fn, dict):
            continue
        name = _as_str(fn.get("name"))
        if not name:
            continue
        params = fn.get("parameters")
        params_json: Optional[str] = None
        if params is not None:
            try:
                params_json = _json.dumps(params, default=str)
            except Exception:
                params_json = None
        out.append(
            ToolDefinition(
                name=name,
                description=_as_str(fn.get("description")),
                parameters_json=params_json,
            )
        )
    return tuple(out)


def promoted_baggage(
    identity: RequestIdentity,
    request_model: Optional[str],
    promoted_keys: Tuple[str, ...],
    metadata_keys: Tuple[str, ...] = DEFAULT_BAGGAGE_METADATA_KEYS,
) -> Dict[str, str]:
    """Assemble the bounded set of values to write into Baggage for promotion."""
    candidate: Dict[str, Optional[str]] = {
        LiteLLM.TEAM_ID: identity.team_id,
        LiteLLM.TEAM_ALIAS: identity.team_alias,
        LiteLLM.KEY_HASH: identity.key_hash,
        LiteLLM.END_USER: identity.end_user,
        GenAI.REQUEST_MODEL: request_model,
    }
    out: Dict[str, str] = {
        key: value for key, value in candidate.items() if key in promoted_keys and value
    }
    for meta_key in metadata_keys:
        value = identity.metadata.get(meta_key)
        if value:
            out[f"{LiteLLM.METADATA_PREFIX}{meta_key}"] = value
    return out
