"""Weave (W&B) attribute mapper.

Weave consumes OpenInference + a small set of Weave-specific keys (display
name, thread id, output value). This mapper layers the latter on top of
OpenInference's vocabulary — compose ``["genai", "openinference", "weave"]``
to feed a Weave backend.
"""

import json

from litellm.integrations.otel.mappers.base import AttributeMap, SpanData
from litellm.integrations.otel.payloads import LLMCallSpanData
from litellm.integrations.otel.spans import SpanRole


class WeaveMapper:
    """Maps ``LLMCallSpanData`` to Weave's vendor attributes."""

    def map(self, role: SpanRole, data: SpanData) -> AttributeMap:
        if not isinstance(data, LLMCallSpanData):
            return {}
        attrs: AttributeMap = {}
        # ``display_name`` follows V1's ``{operation} {model}`` form, but the
        # canonical span name already covers that — Weave reads the attr too.
        if data.request_model:
            attrs["weave.display_name"] = f"{data.operation.value} {data.request_model}"
        if data.identity.call_id:
            attrs["weave.call_id"] = data.identity.call_id
        # Weave treats the response choices as the "output" payload.
        if data.choices_out:
            try:
                attrs["weave.output"] = json.dumps(list(data.choices_out), default=str)
            except Exception:
                pass
        return attrs
