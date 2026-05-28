"""Integration presets — each one returns an :class:`OpenTelemetryV2Config`.

A preset is a small function that reads the integration's env vars and produces
the right exporter list + mapper-name list. The factory in ``litellm_logging``
resolves a callback name (``"arize"``, ``"langfuse_otel"``, ...) to a preset
and passes the result to a single ``OpenTelemetryV2`` instance.

No subclasses. No private TracerProviders. No callback_name string dispatch.
"""

from litellm.integrations.otel.presets.agentops import agentops_preset
from litellm.integrations.otel.presets.arize import arize_preset
from litellm.integrations.otel.presets.langfuse import langfuse_preset
from litellm.integrations.otel.presets.langtrace import langtrace_preset
from litellm.integrations.otel.presets.levo import levo_preset
from litellm.integrations.otel.presets.phoenix import phoenix_preset
from litellm.integrations.otel.presets.weave import weave_preset

#: Callback name → preset function.
PRESET_BY_CALLBACK = {
    "agentops": agentops_preset,
    "arize": arize_preset,
    "arize_phoenix": phoenix_preset,
    "langfuse_otel": langfuse_preset,
    "langtrace": langtrace_preset,
    "levo": levo_preset,
    "weave_otel": weave_preset,
}

__all__ = [
    "PRESET_BY_CALLBACK",
    "agentops_preset",
    "arize_preset",
    "langfuse_preset",
    "langtrace_preset",
    "levo_preset",
    "phoenix_preset",
    "weave_preset",
]
