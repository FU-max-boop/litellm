"""AgentOps preset — OTLP/HTTP to AgentOps' endpoint with a fetched JWT."""

import os
from typing import Any, Dict, Optional

from litellm.integrations.otel.config import ExporterSpec, OpenTelemetryV2Config

_AGENTOPS_ENDPOINT = "https://otlp.agentops.cloud/v1/traces"
_AGENTOPS_AUTH_ENDPOINT = "https://api.agentops.ai/v3/auth/token"


def agentops_preset(
    *,
    config_overrides: Optional[OpenTelemetryV2Config] = None,
    auth_override: Optional[Dict[str, Any]] = None,
) -> OpenTelemetryV2Config:
    """Pre-fetch AgentOps' JWT, then produce a V2 config.

    ``auth_override`` is only for tests; in production the JWT comes from
    AgentOps' auth endpoint keyed off ``AGENTOPS_API_KEY``.
    """
    api_key = os.environ.get("AGENTOPS_API_KEY")
    headers: Optional[str] = None
    auth = auth_override
    if auth is None and api_key:
        try:
            auth = _fetch_agentops_jwt(api_key)
        except Exception:
            auth = None
    if auth and auth.get("token"):
        headers = f"Authorization=Bearer {auth['token']}"
    base = config_overrides or OpenTelemetryV2Config()
    resource_extra: Dict[str, str] = {
        "service.name": os.environ.get("AGENTOPS_SERVICE_NAME", "agentops"),
        "telemetry.sdk.name": "agentops",
    }
    project_id = (auth or {}).get("project_id")
    if project_id:
        resource_extra["project.id"] = str(project_id)
    if os.environ.get("AGENTOPS_ENVIRONMENT"):
        resource_extra["deployment.environment"] = os.environ["AGENTOPS_ENVIRONMENT"]
    return base.model_copy(
        update={
            "exporters": [
                *base.exporters,
                ExporterSpec(
                    kind="otlp_http",
                    endpoint=_AGENTOPS_ENDPOINT,
                    headers=headers,
                ),
            ],
            "resource_attributes": {**base.resource_attributes, **resource_extra},
        }
    )


def _fetch_agentops_jwt(api_key: str) -> Dict[str, Any]:
    from litellm.llms.custom_httpx.http_handler import _get_httpx_client

    client = _get_httpx_client()
    try:
        response = client.post(
            url=_AGENTOPS_AUTH_ENDPOINT,
            headers={"Content-Type": "application/json", "Connection": "keep-alive"},
            json={"api_key": api_key},
            timeout=10,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch AgentOps token: {response.text}")
        return response.json()
    finally:
        client.close()
