"""``CustomLogger`` adapter on the V2 OpenTelemetry engine.

One class. One ``TracerProvider``. Multi-backend by listing ``ExporterSpec``s
on the config. Multi-vocabulary by composing mappers. No subclassing.

Customers configure an :class:`OpenTelemetryV2` instance the same way they'd
configure any OTel SDK: pick exporters, pick attribute conventions, plug it
into the proxy. Each "integration" (Arize, Phoenix, Langfuse, Weave, AgentOps,
Levo, Langtrace) reduces to a *preset function* under
``litellm.integrations.otel.presets`` that returns the right
:class:`OpenTelemetryV2Config` — not a subclass.

Out of scope for V2 (opt-in release): per-request multi-tenant credential
routing. V1 builds a TracerProvider per credential set, caches them, and
hands out a different ``Tracer`` per request. That belongs at the OTel
Collector layer (a routing processor splits the span stream by tenant). The
factory steers customers who need it back to V1.
"""

from datetime import datetime
from typing import Any, List, Mapping, Optional, Union, cast

from opentelemetry import trace
from opentelemetry.context import Context, get_current
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer
from opentelemetry.trace.status import Status, StatusCode

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel.config import (
    CaptureMessageContent,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel.context import (
    context_from_span,
    extract_traceparent,
    set_request_baggage,
)
from litellm.integrations.otel.emitter import SpanEmitter
from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    ManagementSpanData,
    ServiceSpanData,
    SpanError,
    promoted_baggage,
)
from litellm.integrations.otel.providers import build_tracer_provider, get_tracer
from litellm.integrations.otel.semconv import Error as ErrorSemconv
from litellm.integrations.otel.semconv import HTTP, LiteLLM
from litellm.integrations.otel.spans import SpanRole

LITELLM_PROXY_REQUEST_SPAN_NAME = "Received Proxy Server Request"
LITELLM_TRACER_NAME = "litellm"

# Any callback whose class belongs to one of these modules is "the OTel
# callback" for proxy-global-registration purposes.
_OTEL_MODULES = (
    "litellm.integrations.otel",
    "litellm.integrations.opentelemetry",
)


# --------------------------------------------------------------------------- #
#  Small typed helpers
# --------------------------------------------------------------------------- #


def _to_ns(value: Optional[Union[datetime, float, int]]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp() * 1e9)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(float(value) * 1e9)
    return None


def _to_seconds(
    value: Optional[Union[datetime, float, int, str]],
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
    return None


def _is_recordable_span(obj: object) -> bool:
    if obj is None or not isinstance(obj, trace.Span):
        return False
    try:
        ctx = obj.get_span_context()
    except Exception:
        return False
    return ctx is not None and ctx.is_valid


# --------------------------------------------------------------------------- #
#  The V2 logger
# --------------------------------------------------------------------------- #


class OpenTelemetryV2(CustomLogger):
    """The V2 ``CustomLogger`` for OpenTelemetry.

    Constructor mirrors V1's positional shape so existing factory call sites
    don't care which version they're holding.
    """

    def __init__(
        self,
        config: Optional[OpenTelemetryV2Config] = None,
        callback_name: Optional[str] = None,
        tracer_provider: Optional[TracerProvider] = None,
        logger_provider: Optional[Any] = None,  # reserved for OTel logs
        meter_provider: Optional[Any] = None,  # reserved for metrics
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config: OpenTelemetryV2Config = config or OpenTelemetryV2Config()
        self.callback_name = callback_name
        self._tracer_provider: TracerProvider = (
            tracer_provider
            if tracer_provider is not None
            else build_tracer_provider(self.config)
        )
        self.tracer: Tracer = get_tracer(self._tracer_provider, LITELLM_TRACER_NAME)
        # Mapper chain — resolved once from the config's name list. The chain
        # is the *only* source of attribute keys; the engine never inlines them.
        self._mappers = resolve_mappers(self.config.mapper_names)
        self._emitter = SpanEmitter(self.tracer, self.config, mappers=self._mappers)
        self._init_otel_logger_on_litellm_proxy()

    # ====================================================================== #
    #  Proxy global registration
    # ====================================================================== #

    def _init_otel_logger_on_litellm_proxy(self) -> None:
        """Claim ``proxy_server.open_telemetry_logger`` if no one else has."""
        try:
            from litellm.proxy import proxy_server
        except Exception:
            return
        try:
            existing = getattr(litellm, "service_callback", None) or []
            already_otel = any(
                cb.__class__.__module__.startswith(_OTEL_MODULES)
                for cb in existing
                if hasattr(cb, "__class__")
            )
            if not already_otel:
                existing.append(self)
        except Exception:
            pass
        if getattr(proxy_server, "open_telemetry_logger", None) is None:
            setattr(proxy_server, "open_telemetry_logger", self)

    # ====================================================================== #
    #  Capture mode
    # ====================================================================== #

    def _resolve_capture_mode(self) -> str:
        if getattr(litellm, "turn_off_message_logging", False):
            return CaptureMessageContent.NO_CONTENT
        return self.config.capture_message_content

    def _capture_in_span(self) -> bool:
        return self._resolve_capture_mode() in (
            CaptureMessageContent.SPAN_ONLY,
            CaptureMessageContent.SPAN_AND_EVENT,
        )

    def _capture_in_event(self) -> bool:
        return self._resolve_capture_mode() in (
            CaptureMessageContent.EVENT_ONLY,
            CaptureMessageContent.SPAN_AND_EVENT,
        )

    # ====================================================================== #
    #  LLM-call callbacks
    # ====================================================================== #

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    def _emit_llm_call(
        self,
        kwargs: Mapping[str, Any],
        start_time: Optional[Union[datetime, float]],
        end_time: Optional[Union[datetime, float]],
    ) -> Optional[Span]:
        payload = kwargs.get("standard_logging_object")
        if not payload:
            return None
        data = LLMCallSpanData.from_standard_logging_payload(cast("Any", payload))
        parent_ctx = self._resolve_parent_context(kwargs)
        # Identity → Baggage so child spans (guardrails, services) inherit it.
        bag = promoted_baggage(
            data.identity,
            data.request_model,
            promoted_keys=tuple(self.config.baggage_promoted_keys),
            metadata_keys=tuple(self.config.baggage_metadata_keys),
        )
        if bag:
            parent_ctx = set_request_baggage(bag, context=parent_ctx)
        return self._emitter.emit(
            SpanRole.LLM_CALL,
            data,
            parent_context=parent_ctx,
            start_time_ns=_to_ns(start_time),
            end_time_ns=_to_ns(end_time),
        )

    # ====================================================================== #
    #  Parent context resolution
    # ====================================================================== #

    def _resolve_parent_context(self, kwargs: Mapping[str, Any]) -> Optional[Context]:
        """4-priority resolution: explicit parent span → traceparent header →
        current global context → no parent.
        """
        litellm_params = kwargs.get("litellm_params") or {}
        if isinstance(litellm_params, dict):
            metadata = litellm_params.get("metadata") or {}
            parent_span = (
                metadata.get("litellm_parent_otel_span")
                if isinstance(metadata, dict)
                else None
            )
            if _is_recordable_span(parent_span):
                return context_from_span(cast(Span, parent_span))
            proxy_request = litellm_params.get("proxy_server_request") or {}
            if isinstance(proxy_request, dict):
                headers = proxy_request.get("headers")
                if (
                    isinstance(headers, Mapping)
                    and not self.config.ignore_context_propagation
                ):
                    ctx = extract_traceparent(headers)
                    if ctx is not None:
                        return ctx
        return get_current()

    # ====================================================================== #
    #  Service hooks
    # ====================================================================== #

    async def async_service_success_hook(
        self,
        payload: Any,
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=None,
        )

    async def async_service_failure_hook(
        self,
        payload: Any,
        error: Optional[str] = "",
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=error or "error",
        )

    def _emit_service(
        self,
        payload: Any,
        *,
        parent_otel_span: Optional[Span],
        start_time: Optional[Union[datetime, float]],
        end_time: Optional[Union[datetime, float]],
        event_metadata: Optional[dict],
        error_override: Optional[str],
    ) -> Optional[Span]:
        if not _is_recordable_span(parent_otel_span):
            return None
        data = ServiceSpanData.from_payload(payload, event_metadata=event_metadata)
        if error_override is not None and data.error is None:
            data = ServiceSpanData(
                service_name=data.service_name,
                call_type=data.call_type,
                error=SpanError(message=error_override),
                event_metadata=data.event_metadata,
            )
        return self._emitter.emit(
            SpanRole.SERVICE,
            data,
            parent_context=context_from_span(cast(Span, parent_otel_span)),
            start_time_ns=_to_ns(start_time),
            end_time_ns=_to_ns(end_time),
        )

    # ====================================================================== #
    #  async_post_call_* hooks (success: stamp duration + guardrails;
    #                           failure: ERROR status + guardrails)
    # ====================================================================== #

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        parent_span = self._resolve_proxy_parent_span(data)
        self._emit_guardrail_spans(data, parent_span)
        if parent_span is not None:
            self.set_preprocessing_duration_attribute(parent_span, dict(data))
        return response

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, Any],
        original_exception: Optional[BaseException],
        user_api_key_dict: Any,
        traceback_str: Optional[str] = None,
    ) -> None:
        parent_span = self._resolve_proxy_parent_span(request_data)
        if parent_span is not None:
            err_type = (
                original_exception.__class__.__name__
                if original_exception is not None
                else "error"
            )
            parent_span.set_attribute(ErrorSemconv.TYPE, err_type)
            parent_span.set_status(
                Status(StatusCode.ERROR, str(original_exception or "error"))
            )
            if original_exception is not None:
                try:
                    parent_span.record_exception(original_exception)
                except Exception:
                    pass
        self._emit_guardrail_spans(request_data, parent_span)

    def _emit_guardrail_spans(
        self,
        request_data: Mapping[str, Any],
        parent_span: Optional[Span],
    ) -> None:
        metadata = request_data.get("metadata")
        guardrails: List[Any] = []
        if isinstance(metadata, dict):
            info = metadata.get("standard_logging_guardrail_information")
            if isinstance(info, list):
                guardrails = info
            elif isinstance(info, dict):
                guardrails = [info]
        if not guardrails:
            return
        parent_ctx = (
            context_from_span(cast(Span, parent_span))
            if _is_recordable_span(parent_span)
            else None
        )
        for entry in guardrails:
            if not isinstance(entry, dict):
                continue
            name = entry.get("guardrail_name") or entry.get("name") or "guardrail"
            data = GuardrailSpanData(
                guardrail_name=str(name),
                mode=entry.get("guardrail_mode") or entry.get("mode"),
                status=entry.get("guardrail_status") or entry.get("status"),
                masked_entity_count=entry.get("masked_entity_count"),
            )
            self._emitter.emit(SpanRole.GUARDRAIL, data, parent_context=parent_ctx)

    # ====================================================================== #
    #  Management endpoint hooks
    # ====================================================================== #

    async def async_management_endpoint_success_hook(
        self,
        logging_payload: Any,
        parent_otel_span: Optional[Span] = None,
    ) -> None:
        self._emit_management(logging_payload, parent_otel_span, success=True)

    async def async_management_endpoint_failure_hook(
        self,
        logging_payload: Any,
        parent_otel_span: Optional[Span] = None,
    ) -> None:
        self._emit_management(logging_payload, parent_otel_span, success=False)

    def _emit_management(
        self,
        logging_payload: Any,
        parent_otel_span: Optional[Span],
        *,
        success: bool,
    ) -> None:
        route = getattr(logging_payload, "route", None) or "management"
        exc = getattr(logging_payload, "exception", None)
        error = (
            None
            if success
            else SpanError(
                error_type=(
                    exc.__class__.__name__
                    if isinstance(exc, BaseException)
                    else "error"
                ),
                message=str(exc) if exc is not None else "error",
            )
        )
        ManagementSpanData(route=str(route), error=error)  # validated shape
        parent_ctx = (
            context_from_span(cast(Span, parent_otel_span))
            if _is_recordable_span(parent_otel_span)
            else None
        )
        start_time = getattr(logging_payload, "start_time", None)
        end_time = getattr(logging_payload, "end_time", None)
        span = self._emitter._start(
            SpanRole.MANAGEMENT,
            str(route),
            parent_context=parent_ctx,
            start_time_ns=_to_ns(start_time),
        )
        request_data = getattr(logging_payload, "request_data", None)
        if isinstance(request_data, dict):
            for key, value in request_data.items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(f"request.{key}", value)
        if success:
            response = getattr(logging_payload, "response", None)
            if isinstance(response, dict):
                for key, value in response.items():
                    if isinstance(value, (str, bool, int, float)):
                        span.set_attribute(f"response.{key}", value)
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_attribute(
                ErrorSemconv.TYPE,
                (error.error_type if error is not None else None) or "error",
            )
            span.set_status(Status(StatusCode.ERROR, str(exc or "error")))
            if isinstance(exc, BaseException):
                try:
                    span.record_exception(exc)
                except Exception:
                    pass
        span.end(end_time=_to_ns(end_time))

    # ====================================================================== #
    #  Proxy SERVER-span lifecycle
    # ====================================================================== #

    def create_litellm_proxy_request_started_span(
        self, start_time: datetime, headers: Optional[Mapping[str, str]]
    ) -> Optional[Span]:
        parent_ctx: Optional[Context] = None
        if (
            headers is not None
            and isinstance(headers, Mapping)
            and not self.config.ignore_context_propagation
        ):
            parent_ctx = extract_traceparent(headers)
        return self._emitter._start(
            SpanRole.PROXY_REQUEST,
            LITELLM_PROXY_REQUEST_SPAN_NAME,
            parent_context=parent_ctx,
            start_time_ns=_to_ns(start_time),
        )

    @staticmethod
    def set_proxy_request_route_attributes(
        span: Optional[Span],
        *,
        url_path: Optional[str] = None,
        http_route: Optional[str] = None,
    ) -> None:
        if span is None:
            return
        if url_path is not None:
            span.set_attribute(HTTP.URL_PATH, url_path)
        if http_route is not None:
            span.set_attribute(HTTP.ROUTE, http_route)

    @staticmethod
    def set_response_status_code_attribute(
        span: Optional[Span], status_code: Optional[int]
    ) -> None:
        if span is None or status_code is None:
            return
        span.set_attribute(HTTP.RESPONSE_STATUS_CODE, int(status_code))

    @staticmethod
    def set_preprocessing_duration_attribute(
        span: Optional[Span], container: Any
    ) -> None:
        if span is None or not isinstance(container, dict):
            return
        first_handoff = container.get("first_api_call_start_time")
        received_at = None
        litellm_params = container.get("litellm_params")
        candidates = (
            (
                litellm_params.get("metadata")
                if isinstance(litellm_params, dict)
                else None
            ),
            container.get("metadata"),
            container.get("litellm_metadata"),
        )
        for source in candidates:
            if isinstance(source, dict):
                received_at = received_at or source.get("litellm_received_at")
        if received_at is None or first_handoff is None:
            return
        start_ts = _to_seconds(received_at)
        end_ts = _to_seconds(first_handoff)
        if start_ts is None or end_ts is None:
            return
        duration_ms = (end_ts - start_ts) * 1000.0
        if duration_ms < 0:
            return
        span.set_attribute(LiteLLM.PREPROCESSING_MS, duration_ms)

    # ====================================================================== #
    #  Helpers
    # ====================================================================== #

    @staticmethod
    def _resolve_proxy_parent_span(
        container: Mapping[str, Any],
    ) -> Optional[Span]:
        metadata = container.get("metadata") if isinstance(container, dict) else None
        if isinstance(metadata, dict):
            parent = metadata.get("litellm_parent_otel_span")
            if _is_recordable_span(parent):
                return cast(Span, parent)
        litellm_params = (
            container.get("litellm_params") if isinstance(container, dict) else None
        )
        if isinstance(litellm_params, dict):
            inner_meta = litellm_params.get("metadata")
            if isinstance(inner_meta, dict):
                parent = inner_meta.get("litellm_parent_otel_span")
                if _is_recordable_span(parent):
                    return cast(Span, parent)
        return None
