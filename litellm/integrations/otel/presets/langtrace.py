"""Langtrace preset — Langtrace consumes generic OTLP + a vendor mapper."""

from typing import Optional

from litellm.integrations.otel.config import OpenTelemetryV2Config


def langtrace_preset(
    *,
    config_overrides: Optional[OpenTelemetryV2Config] = None,
) -> OpenTelemetryV2Config:
    """Compose the Langtrace mapper on top of the customer's OTLP destination.

    Unlike Arize / Phoenix / Langfuse, Langtrace doesn't ship its own endpoint
    — customers point their existing OTLP collector at Langtrace and just
    need the vendor attribute schema applied to outgoing spans.
    """
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "mapper_names": _ensure(base.mapper_names, "langtrace"),
        }
    )


def _ensure(mapper_names, name: str):
    if name in mapper_names:
        return list(mapper_names)
    return [*mapper_names, name]
