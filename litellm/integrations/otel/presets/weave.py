"""Weave (W&B) preset."""

from typing import Optional

from litellm.integrations.otel.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.weave.weave_otel import get_weave_otel_config


def weave_preset(
    *,
    config_overrides: Optional[OpenTelemetryV2Config] = None,
) -> OpenTelemetryV2Config:
    """Read ``WANDB_*`` env vars and produce a V2 config."""
    weave_cfg = get_weave_otel_config()
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=weave_cfg.protocol or "otlp_http",
                    endpoint=weave_cfg.endpoint,
                    headers=weave_cfg.otlp_auth_headers,
                ),
            ],
            # Weave consumes OpenInference + a small Weave-specific overlay.
            "mapper_names": _ensure(
                _ensure(base.mapper_names, "openinference"), "weave"
            ),
        }
    )


def _ensure(mapper_names, name: str):
    if name in mapper_names:
        return list(mapper_names)
    return [*mapper_names, name]
