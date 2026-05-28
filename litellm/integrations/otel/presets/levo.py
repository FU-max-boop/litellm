"""Levo preset — OTLP/HTTP to a Levo collector with org+workspace headers."""

from typing import Optional

from litellm.integrations.levo.levo import LevoLogger as _V1Levo
from litellm.integrations.otel.config import ExporterSpec, OpenTelemetryV2Config


def levo_preset(
    *,
    config_overrides: Optional[OpenTelemetryV2Config] = None,
) -> OpenTelemetryV2Config:
    """Read ``LEVOAI_*`` env vars and produce a V2 config."""
    cfg = _V1Levo.get_levo_config()
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=cfg.endpoint,
                    headers=cfg.otlp_auth_headers,
                ),
            ],
        }
    )
