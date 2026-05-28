"""Arize preset — OTLP exporter to Arize + OpenInference vocabulary."""

import os
from typing import Optional

from litellm.integrations.arize.arize import ArizeLogger as _V1ArizeLogger
from litellm.integrations.otel.config import ExporterSpec, OpenTelemetryV2Config


def arize_preset(
    *,
    config_overrides: Optional[OpenTelemetryV2Config] = None,
) -> OpenTelemetryV2Config:
    """Read ``ARIZE_*`` env vars and produce a V2 config.

    The exporter is added as one of (potentially many) ``exporters`` entries,
    not a private TracerProvider — so spans flow to Arize *and* whatever else
    the customer has wired (Honeycomb, Langfuse, ...) without per-integration
    provider singletons.
    """
    arize_cfg = _V1ArizeLogger.get_arize_config()
    headers = _arize_headers(arize_cfg)
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=arize_cfg.protocol or "otlp_grpc",
                    endpoint=arize_cfg.endpoint or "https://otlp.arize.com/v1",
                    headers=headers,
                ),
            ],
            "mapper_names": _ensure_mapper(base.mapper_names, "openinference"),
            "resource_attributes": {
                **base.resource_attributes,
                **(
                    {"model_id": arize_cfg.project_name}
                    if arize_cfg.project_name
                    else {}
                ),
            },
        }
    )


def _arize_headers(arize_cfg) -> Optional[str]:
    pieces = []
    if arize_cfg.space_id or arize_cfg.space_key:
        pieces.append(f"space_id={arize_cfg.space_id or arize_cfg.space_key}")
    if arize_cfg.api_key:
        pieces.append(f"api_key={arize_cfg.api_key}")
    if not pieces:
        # Honour the V1 fallback of letting OTEL_EXPORTER_OTLP_TRACES_HEADERS
        # be set out of band; nothing for us to attach here.
        return os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
    return ",".join(pieces)


def _ensure_mapper(mapper_names, name: str):
    if name in mapper_names:
        return list(mapper_names)
    return [*mapper_names, name]
