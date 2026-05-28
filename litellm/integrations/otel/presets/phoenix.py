"""Arize-Phoenix preset.

Phoenix consumes the OpenInference vocabulary just like Arize. The only
distinct knob is the project name, which becomes a Resource attribute
(``openinference.project.name``) so it flows on every span without a custom
per-project TracerProvider.
"""

import os
from typing import Optional

from litellm.integrations.arize.arize_phoenix import (
    ArizePhoenixLogger as _V1Phoenix,
)
from litellm.integrations.otel.config import ExporterSpec, OpenTelemetryV2Config


def phoenix_preset(
    *,
    config_overrides: Optional[OpenTelemetryV2Config] = None,
) -> OpenTelemetryV2Config:
    """Read ``PHOENIX_*`` env vars and produce a V2 config."""
    cfg = _V1Phoenix.get_arize_phoenix_config()
    headers = cfg.otlp_auth_headers if hasattr(cfg, "otlp_auth_headers") else None
    project_name = (
        os.environ.get("PHOENIX_PROJECT_NAME")
        or os.environ.get("PHOENIX_COLLECTOR_PROJECT_NAME")
        or "default"
    )
    base = config_overrides or OpenTelemetryV2Config()
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind=cfg.protocol if hasattr(cfg, "protocol") else "otlp_http",
                    endpoint=cfg.endpoint,
                    headers=headers,
                ),
            ],
            "mapper_names": _ensure_mapper(base.mapper_names, "openinference"),
            "resource_attributes": {
                **base.resource_attributes,
                "openinference.project.name": project_name,
            },
        }
    )


def _ensure_mapper(mapper_names, name: str):
    if name in mapper_names:
        return list(mapper_names)
    return [*mapper_names, name]
