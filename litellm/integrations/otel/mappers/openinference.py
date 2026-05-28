"""OpenInference attribute mapper (Arize + Arize-Phoenix shared vocabulary).

Spec: https://github.com/Arize-ai/openinference/tree/main/spec — the standard
both Arize and Phoenix consume. Composing this mapper after ``GenAIMapper``
gives the same span both vocabularies, so a single trace lights up Arize +
Phoenix + any other OpenInference-aware backend simultaneously.
"""

import json
from typing import Optional

from litellm.integrations.otel.mappers.base import AttributeMap, SpanData
from litellm.integrations.otel.payloads import LLMCallSpanData
from litellm.integrations.otel.spans import SpanRole


def _message_content(message: object) -> Optional[str]:
    """Extract the textual ``content`` from a chat message dict."""
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # multimodal: concatenate text parts only
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(p for p in parts if isinstance(p, str)) or None
    return None


class OpenInferenceMapper:
    """Emits OpenInference attributes for LLM_CALL spans.

    Key families (per the OpenInference spec):
    - ``openinference.span.kind`` — discriminator (``"LLM"`` here)
    - ``llm.model_name`` / ``llm.provider`` / ``llm.invocation_parameters``
    - ``llm.input_messages.{i}.message.role`` / ``...content``
    - ``llm.output_messages.{i}.message.role`` / ``...content``
    - ``llm.token_count.prompt`` / ``...completion`` / ``...total``
    - ``input.value`` / ``output.value`` — JSON-serialized request / response
    """

    def map(self, role: SpanRole, data: SpanData) -> AttributeMap:
        if not isinstance(data, LLMCallSpanData):
            return {}
        attrs: AttributeMap = {"openinference.span.kind": "LLM"}
        if data.request_model:
            attrs["llm.model_name"] = data.request_model
        if data.provider:
            attrs["llm.provider"] = data.provider
        rp = data.request_params
        invocation: dict = {}
        for key, value in (
            ("temperature", rp.temperature),
            ("top_p", rp.top_p),
            ("top_k", rp.top_k),
            ("max_tokens", rp.max_tokens),
            ("frequency_penalty", rp.frequency_penalty),
            ("presence_penalty", rp.presence_penalty),
            ("seed", rp.seed),
        ):
            if value is not None:
                invocation[key] = value
        if invocation:
            attrs["llm.invocation_parameters"] = json.dumps(invocation)

        # Input messages — flat numbered keys per OpenInference.
        input_blob: list = []
        for idx, msg in enumerate(data.messages_in):
            role_val = msg.get("role") if isinstance(msg, dict) else None
            if isinstance(role_val, str):
                attrs[f"llm.input_messages.{idx}.message.role"] = role_val
            content = _message_content(msg)
            if content is not None:
                attrs[f"llm.input_messages.{idx}.message.content"] = content
            input_blob.append({"role": role_val, "content": content})
        if input_blob:
            attrs["input.value"] = json.dumps(input_blob)

        # Output messages — only the chosen completion(s).
        output_blob: list = []
        for idx, choice in enumerate(data.choices_out):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            role_val = message.get("role") if isinstance(message, dict) else None
            if isinstance(role_val, str):
                attrs[f"llm.output_messages.{idx}.message.role"] = role_val
            content = _message_content(message)
            if content is not None:
                attrs[f"llm.output_messages.{idx}.message.content"] = content
            output_blob.append({"role": role_val, "content": content})
        if output_blob:
            attrs["output.value"] = json.dumps(output_blob)

        # Tools (declared on the request).
        for idx, tool in enumerate(data.tools):
            attrs[f"llm.tools.{idx}.tool.name"] = tool.name
            if tool.description:
                attrs[f"llm.tools.{idx}.tool.description"] = tool.description
            if tool.parameters_json:
                attrs[f"llm.tools.{idx}.tool.json_schema"] = tool.parameters_json

        if data.usage.input_tokens is not None:
            attrs["llm.token_count.prompt"] = data.usage.input_tokens
        if data.usage.output_tokens is not None:
            attrs["llm.token_count.completion"] = data.usage.output_tokens
        if data.usage.total_tokens is not None:
            attrs["llm.token_count.total"] = data.usage.total_tokens
        return attrs
