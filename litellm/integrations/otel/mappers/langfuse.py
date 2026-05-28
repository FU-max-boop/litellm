"""Langfuse OTLP attribute mapper.

Langfuse ingests OTLP spans and reads from its own vendor namespace
(``langfuse.observation.*``, ``langfuse.trace.*``). Compose this mapper after
``GenAIMapper`` to send canonical + Langfuse-flavored spans simultaneously.
"""

import json
from typing import Optional

from litellm.integrations.otel.mappers.base import AttributeMap, SpanData
from litellm.integrations.otel.payloads import LLMCallSpanData
from litellm.integrations.otel.spans import SpanRole


def _stringify_message(message: object) -> Optional[str]:
    if not isinstance(message, dict):
        return None
    try:
        return json.dumps(message, default=str)
    except Exception:
        return None


class LangfuseMapper:
    """Maps ``LLMCallSpanData`` to Langfuse's observation-level attributes."""

    def map(self, role: SpanRole, data: SpanData) -> AttributeMap:
        if not isinstance(data, LLMCallSpanData):
            return {}
        attrs: AttributeMap = {"langfuse.observation.type": "generation"}
        if data.request_model:
            attrs["langfuse.observation.model.name"] = data.request_model
        if data.provider:
            attrs["langfuse.observation.metadata.provider"] = data.provider
        if data.identity.call_id:
            attrs["langfuse.observation.id"] = data.identity.call_id

        rp = data.request_params
        model_params: dict = {}
        for key, value in (
            ("temperature", rp.temperature),
            ("top_p", rp.top_p),
            ("max_tokens", rp.max_tokens),
            ("frequency_penalty", rp.frequency_penalty),
            ("presence_penalty", rp.presence_penalty),
            ("seed", rp.seed),
        ):
            if value is not None:
                model_params[key] = value
        if model_params:
            attrs["langfuse.observation.model.parameters"] = json.dumps(model_params)

        if data.messages_in:
            input_serialized = [
                json.loads(s)
                for s in (_stringify_message(m) for m in data.messages_in)
                if s is not None
            ]
            if input_serialized:
                attrs["langfuse.observation.input"] = json.dumps(input_serialized)

        if data.choices_out:
            output_messages = [
                choice.get("message")
                for choice in data.choices_out
                if isinstance(choice, dict)
            ]
            output_serialized = [
                json.loads(s)
                for s in (_stringify_message(m) for m in output_messages)
                if s is not None
            ]
            if output_serialized:
                attrs["langfuse.observation.output"] = json.dumps(output_serialized)

        usage_payload: dict = {}
        if data.usage.input_tokens is not None:
            usage_payload["input"] = data.usage.input_tokens
        if data.usage.output_tokens is not None:
            usage_payload["output"] = data.usage.output_tokens
        if data.usage.total_tokens is not None:
            usage_payload["total"] = data.usage.total_tokens
        if usage_payload:
            attrs["langfuse.observation.usage_details"] = json.dumps(usage_payload)
        if data.response_cost is not None:
            attrs["langfuse.observation.cost_details"] = json.dumps(
                {"total": data.response_cost}
            )
        if data.identity.team_id:
            attrs["langfuse.trace.metadata.team_id"] = data.identity.team_id
        if data.identity.team_alias:
            attrs["langfuse.trace.metadata.team_alias"] = data.identity.team_alias
        return attrs
