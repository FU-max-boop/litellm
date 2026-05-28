"""Provider / exporter factory + the Baggage span processor.

One ``TracerProvider`` per process; *N* ``SpanProcessor``s attached, one per
:class:`~litellm.integrations.otel.config.ExporterSpec`. That replaces V1's
"one private TracerProvider per integration" hack: instead of building parallel
provider singletons, the same trace flows through every configured exporter.

OTLP exporter imports are lazy inside :func:`build_span_exporter` so importing
this module never pulls in ``grpc``.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from opentelemetry import baggage
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import Span, SpanKind, Tracer

from litellm.integrations.otel.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.semconv import LiteLLM
from litellm.integrations.otel.spans import LiteLLMSpanKind

_SPAN_KIND_BY_ROLE_KIND: Dict[LiteLLMSpanKind, SpanKind] = {
    LiteLLMSpanKind.SERVER: SpanKind.SERVER,
    LiteLLMSpanKind.CLIENT: SpanKind.CLIENT,
    LiteLLMSpanKind.INTERNAL: SpanKind.INTERNAL,
    LiteLLMSpanKind.PRODUCER: SpanKind.PRODUCER,
    LiteLLMSpanKind.CONSUMER: SpanKind.CONSUMER,
}


def to_otel_span_kind(kind: LiteLLMSpanKind) -> SpanKind:
    return _SPAN_KIND_BY_ROLE_KIND[kind]


class LiteLLMBaggageSpanProcessor(SpanProcessor):
    """Stamps an allowlisted set of Baggage entries onto every span at start."""

    def __init__(
        self,
        allowed_keys: Iterable[str],
        allowed_prefixes: Tuple[str, ...] = (LiteLLM.METADATA_PREFIX,),
    ) -> None:
        self._allowed_keys = frozenset(allowed_keys)
        self._allowed_prefixes = tuple(allowed_prefixes)

    def _is_allowed(self, key: str) -> bool:
        return key in self._allowed_keys or any(
            key.startswith(prefix) for prefix in self._allowed_prefixes
        )

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        for key, value in baggage.get_all(parent_context).items():
            if self._is_allowed(key) and isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:  # noqa: D401 - no-op
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def parse_headers(raw: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if not raw:
        return headers
    for pair in raw.split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            headers[key.strip()] = value.strip()
    return headers


def _exporter_from_spec(spec: ExporterSpec) -> SpanExporter:
    kind = (spec.kind or "console").lower()
    if kind in ("in_memory", "inmemory", "memory"):
        return InMemorySpanExporter()
    if kind in ("otlp_http", "http", "http/protobuf", "http/json"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )

        return HTTPExporter(endpoint=spec.endpoint, headers=parse_headers(spec.headers))
    if kind in ("otlp_grpc", "grpc"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GRPCExporter,
        )

        return GRPCExporter(endpoint=spec.endpoint, headers=parse_headers(spec.headers))
    return ConsoleSpanExporter()


def _processor_for(exporter: SpanExporter, use_simple: Optional[bool]) -> SpanProcessor:
    """Pick Simple vs Batch. Default: Simple for console/in-memory (synchronous
    test ergonomics), Batch otherwise (production export semantics).
    """
    if use_simple is None:
        use_simple = isinstance(exporter, (ConsoleSpanExporter, InMemorySpanExporter))
    return SimpleSpanProcessor(exporter) if use_simple else BatchSpanProcessor(exporter)


def build_span_exporter(config: OpenTelemetryV2Config) -> SpanExporter:
    """Single-exporter convenience used by older call sites + tests.

    Reads the legacy ``exporter`` / ``endpoint`` / ``headers`` triple. New code
    should configure ``config.exporters`` directly.
    """
    return _exporter_from_spec(
        ExporterSpec(
            kind=config.exporter, endpoint=config.endpoint, headers=config.headers
        )
    )


def build_resource(config: OpenTelemetryV2Config) -> Resource:
    attributes: Dict[str, str] = {"service.name": config.service_name}
    if config.deployment_environment:
        attributes["deployment.environment"] = config.deployment_environment
    attributes.update(config.resource_attributes)
    return Resource.create(attributes)


def build_tracer_provider(
    config: OpenTelemetryV2Config,
    exporter: Optional[SpanExporter] = None,
    baggage_processor: Optional[SpanProcessor] = None,
    use_simple_processor: Optional[bool] = None,
) -> TracerProvider:
    """Build the shared :class:`TracerProvider`.

    Attaches the Baggage processor first (so identity attrs land before any
    export decision), then one ``SpanProcessor`` per ``config.exporters``
    entry — that's how multi-backend works. ``exporter`` and
    ``use_simple_processor`` are back-compat overrides for tests.
    """
    provider = TracerProvider(resource=build_resource(config))
    if baggage_processor is None:
        baggage_processor = LiteLLMBaggageSpanProcessor(
            allowed_keys=config.baggage_promoted_keys
        )
    provider.add_span_processor(baggage_processor)

    if exporter is not None:
        provider.add_span_processor(_processor_for(exporter, use_simple_processor))
        return provider

    specs: List[ExporterSpec] = (
        list(config.exporters)
        if config.exporters
        else [
            ExporterSpec(
                kind=config.exporter, endpoint=config.endpoint, headers=config.headers
            )
        ]
    )
    for spec in specs:
        exp = _exporter_from_spec(spec)
        provider.add_span_processor(
            _processor_for(
                exp,
                (
                    spec.use_simple_processor
                    if spec.use_simple_processor is not None
                    else use_simple_processor
                ),
            )
        )
    return provider


def get_tracer(provider: TracerProvider, name: str = "litellm") -> Tracer:
    return provider.get_tracer(name)


def in_memory_provider(
    config: Optional[OpenTelemetryV2Config] = None,
) -> Tuple[TracerProvider, InMemorySpanExporter]:
    """Convenience for tests: a provider exporting to an in-memory buffer."""
    cfg = config or OpenTelemetryV2Config(exporter="in_memory")
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider(cfg, exporter=exporter)
    return provider, exporter
