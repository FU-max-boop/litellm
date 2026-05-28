"""Langfuse-OTEL preset."""

from typing import Optional

from litellm.integrations.langfuse.langfuse_otel import (
    LangfuseOtelLogger as _V1Langfuse,
)
from litellm.integrations.otel.config import ExporterSpec, OpenTelemetryV2Config


def langfuse_preset(
    *,
    config_overrides: Optional[OpenTelemetryV2Config] = None,
) -> OpenTelemetryV2Config:
    """Read ``LANGFUSE_*`` env vars and produce a V2 config."""
    cfg = _V1Langfuse.get_langfuse_otel_config()
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=cfg.exporter if hasattr(cfg, "exporter") else "otlp_http",
                    endpoint=cfg.endpoint,
                    headers=cfg.headers,
                ),
            ],
            "mapper_names": _ensure_mapper(base.mapper_names, "langfuse"),
        }
    )


def _ensure_mapper(mapper_names, name: str):
    if name in mapper_names:
        return list(mapper_names)
    return [*mapper_names, name]
