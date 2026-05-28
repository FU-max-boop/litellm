"""Typed configuration for the V2 OpenTelemetry instrumentation.

Two new shapes versus V1's flat ``OpenTelemetryConfig``:

- :class:`ExporterSpec` — *one* destination. Each spec attaches a
  ``SpanProcessor`` to the shared ``TracerProvider``, so a single trace
  fans out to every configured backend without per-tenant providers.
- :class:`OpenTelemetryV2Config.mapper_names` — ordered list of attribute
  vocabularies to compose. ``"genai"`` is the canonical one; vendor entries
  (``"openinference"``, ``"langfuse"``, ``"weave"``, ``"langtrace"``) layer
  on top so the same span carries multiple naming schemes.

That replaces the V1 + Phase 2-port patterns of (a) one ``TracerProvider``
per integration, (b) per-credential ``TracerProvider`` caches for
multi-tenancy, (c) ``set_attributes`` subclass overrides per vendor.
Multi-tenancy is intentionally out of scope; route at the collector layer.
"""

from typing import Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.integrations.otel.semconv import (
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
)

#: Master feature-flag env var. The new logger is inert until this is truthy.
OTEL_V2_ENV = "LITELLM_OTEL_V2"


class CaptureMessageContent(str):
    NO_CONTENT = "no_content"
    SPAN_ONLY = "span_only"
    EVENT_ONLY = "event_only"
    SPAN_AND_EVENT = "span_and_event"


class _OTelV2Flag(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, validation_alias=AliasChoices(OTEL_V2_ENV))


def is_otel_v2_enabled() -> bool:
    return _OTelV2Flag().enabled


class ExporterSpec(BaseModel):
    """One span-export destination.

    Attaching ``N`` ``ExporterSpec``s to a logger gives the same trace ``N``
    destinations (Arize + Phoenix + Langfuse + your own Honeycomb, etc.) — no
    extra ``TracerProvider``s, no fan-out daemons.
    """

    model_config = {"extra": "forbid"}

    kind: str = Field(
        default="console",
        description="console | in_memory | otlp_http | otlp_grpc",
    )
    endpoint: Optional[str] = None
    headers: Optional[str] = None
    use_simple_processor: Optional[bool] = Field(
        default=None,
        description=(
            "Force SimpleSpanProcessor regardless of exporter kind. Default: "
            "auto (Simple for console/in_memory, Batch otherwise)."
        ),
    )


class OpenTelemetryV2Config(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    # ----- the basic env-derived shape (kept so customers' OTEL_* envs work) - #
    exporter: str = Field(
        default="console",
        validation_alias=AliasChoices("OTEL_EXPORTER", "OTEL_EXPORTER_OTLP_PROTOCOL"),
        description=(
            "Legacy single-exporter knob. Folded into ``exporters`` by the "
            "model validator; new code should use ``exporters`` directly."
        ),
    )
    endpoint: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"),
    )
    headers: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_HEADERS", "OTEL_EXPORTER_OTLP_HEADERS"),
    )
    service_name: str = Field(
        default="litellm", validation_alias=AliasChoices("OTEL_SERVICE_NAME")
    )
    deployment_environment: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("OTEL_ENVIRONMENT_NAME")
    )

    enable_metrics: bool = Field(
        default=False,
        validation_alias=AliasChoices("LITELLM_OTEL_INTEGRATION_ENABLE_METRICS"),
    )
    enable_events: bool = Field(
        default=False,
        validation_alias=AliasChoices("LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS"),
    )
    capture_message_content: str = Field(
        default=CaptureMessageContent.NO_CONTENT,
        validation_alias=AliasChoices(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        ),
    )
    ignore_context_propagation: bool = Field(
        default=False,
        validation_alias=AliasChoices("OTEL_IGNORE_CONTEXT_PROPAGATION"),
    )

    legacy_compat: bool = Field(
        default=True, validation_alias=AliasChoices("LITELLM_OTEL_LEGACY_COMPAT")
    )

    # ----- the V2-native shape ---------------------------------------------- #

    exporters: List[ExporterSpec] = Field(
        default_factory=list,
        description=(
            "One destination per spec. The shared TracerProvider attaches a "
            "SpanProcessor per entry. Empty means: fall back to the single "
            "legacy ``exporter`` / ``endpoint`` / ``headers`` triple."
        ),
    )

    mapper_names: List[str] = Field(
        default_factory=lambda: ["genai"],
        description=(
            "Ordered attribute vocabularies. ``genai`` is canonical (always "
            "first). Vendor names: ``openinference`` (Arize+Phoenix), "
            "``langfuse``, ``weave``, ``langtrace``."
        ),
    )

    resource_attributes: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra Resource attributes beyond ``service.name`` and "
            "``deployment.environment`` (e.g. integration-specific markers)."
        ),
    )

    baggage_promoted_keys: List[str] = Field(
        default_factory=lambda: list(BAGGAGE_PROMOTED_KEYS)
    )
    baggage_metadata_keys: List[str] = Field(
        default_factory=lambda: list(DEFAULT_BAGGAGE_METADATA_KEYS)
    )

    @model_validator(mode="after")
    def _normalize(self) -> "OpenTelemetryV2Config":
        # An endpoint with no explicit exporter implies OTLP/HTTP (matches V1).
        if self.endpoint and self.exporter == "console":
            self.exporter = "otlp_http"
        # If exporters is empty, fold the legacy triple into one spec so the
        # provider always has at least one destination.
        if not self.exporters:
            self.exporters = [
                ExporterSpec(
                    kind=self.exporter,
                    endpoint=self.endpoint,
                    headers=self.headers,
                )
            ]
        # Ensure ``genai`` is always present and first.
        names = list(self.mapper_names)
        if "genai" in names:
            names = ["genai"] + [n for n in names if n != "genai"]
        else:
            names = ["genai"] + names
        # ``legacy`` dual-emit follows from the toggle. Append at the tail so
        # canonical keys win on conflict.
        if self.legacy_compat and "legacy" not in names:
            names.append("legacy")
        self.mapper_names = names
        return self

    @classmethod
    def from_env(cls) -> "OpenTelemetryV2Config":
        return cls()
