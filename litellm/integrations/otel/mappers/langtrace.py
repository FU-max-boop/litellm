"""Langtrace attribute mapper.

V1 reached Langtrace via ``callback_name == "langtrace"`` string dispatch
inside ``set_attributes``. V2 makes it an explicit mapper that customers
compose like any other vocabulary.
"""

import json

from litellm.integrations.otel.mappers.base import AttributeMap, SpanData
from litellm.integrations.otel.payloads import LLMCallSpanData
from litellm.integrations.otel.spans import SpanRole


class LangtraceMapper:
    """Maps ``LLMCallSpanData`` to Langtrace's vendor attributes."""

    def map(self, role: SpanRole, data: SpanData) -> AttributeMap:
        if not isinstance(data, LLMCallSpanData):
            return {}
        attrs: AttributeMap = {"gen_ai.operation.name": "chat"}
        if data.provider:
            attrs["langtrace.service.name"] = data.provider
        if data.request_model:
            attrs["llm.model"] = data.request_model
        if data.response_model:
            attrs["gen_ai.response.model"] = data.response_model
        if data.response_id:
            attrs["gen_ai.response_id"] = data.response_id
        if data.system_fingerprint:
            attrs["gen_ai.system_fingerprint"] = data.system_fingerprint
        rp = data.request_params
        if rp.temperature is not None:
            attrs["llm.temperature"] = rp.temperature
        if rp.top_p is not None:
            attrs["llm.top_p"] = rp.top_p
        if rp.top_k is not None:
            attrs["llm.top_k"] = rp.top_k
        if rp.max_tokens is not None:
            attrs["llm.max_tokens"] = rp.max_tokens
        if rp.frequency_penalty is not None:
            attrs["llm.frequency_penalty"] = rp.frequency_penalty
        if rp.presence_penalty is not None:
            attrs["llm.presence_penalty"] = rp.presence_penalty
        if data.is_streaming is not None:
            attrs["llm.stream"] = data.is_streaming
        if data.usage.input_tokens is not None:
            attrs["llm.token.counts.prompt"] = data.usage.input_tokens
        if data.usage.output_tokens is not None:
            attrs["llm.token.counts.completion"] = data.usage.output_tokens
        if data.usage.total_tokens is not None:
            attrs["llm.token.counts.total"] = data.usage.total_tokens
        # Prompts + completions serialized into Langtrace's blob shape.
        if data.messages_in:
            try:
                attrs["llm.prompts"] = json.dumps(list(data.messages_in), default=str)
            except Exception:
                pass
        if data.choices_out:
            completions = [
                choice.get("message")
                for choice in data.choices_out
                if isinstance(choice, dict)
            ]
            try:
                attrs["llm.completions"] = json.dumps(completions, default=str)
            except Exception:
                pass
        return attrs
