"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

import copy
import json
import os
import re
from typing import (
    Any,
    Coroutine,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

from packaging.version import InvalidVersion, Version

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _get_image_mime_type_from_url,
)
from litellm.litellm_core_utils.prompt_templates.factory import _parse_mime_type
from litellm.litellm_core_utils.reasoning_effort_utils import (
    reasoning_effort_from_thinking_budget,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantToolCall,
    ChatCompletionFileObject,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionVideoObject,
    ChatCompletionVideoUrlObject,
)
from litellm.utils import _remove_additional_properties, _remove_strict_from_schema

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

# Pre-0.24.0 vLLM rejects tools[].function.strict (pydantic extra_forbidden);
# see BerriAI/litellm#6088 and vllm-project/vllm#15526. v0.24.0+ accepts the
# field (and uses it for strict tool calling).
HOSTED_VLLM_TOOL_SCHEMA_MIN_VERSION = Version("0.24.0")

# Successful GET {api_root}/version lookups, keyed by normalized api_base.
_vllm_version_cache: Dict[str, str] = {}


def _parse_vllm_version(raw: Optional[str]) -> Optional[Version]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.startswith(("v", "V")):
        s = s[1:]
    try:
        return Version(s)
    except InvalidVersion:
        match = re.match(r"(\d+\.\d+\.\d+)", s)
        if not match:
            return None
        try:
            return Version(match.group(1))
        except InvalidVersion:
            return None


def _should_strip_legacy_tool_schema_fields(vllm_version: Optional[str]) -> bool:
    """Return True only when we know the server is older than v0.24.0."""
    parsed = _parse_vllm_version(vllm_version)
    if parsed is None:
        return False
    return parsed < HOSTED_VLLM_TOOL_SCHEMA_MIN_VERSION


def _hosted_vllm_version_url(api_base: str) -> str:
    """Map chat api_base (.../v1) to the instrumentator /version URL."""
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base.rstrip('/')}/version"


def _fetch_vllm_version(api_base: str, timeout: float = 1.0) -> Optional[str]:
    if api_base in _vllm_version_cache:
        return _vllm_version_cache[api_base]
    try:
        import httpx

        response = httpx.get(_hosted_vllm_version_url(api_base), timeout=timeout)
        if response.status_code != 200:
            return None
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        version = payload.get("version")
        if not isinstance(version, str) or not version.strip():
            return None
        _vllm_version_cache[api_base] = version
        return version
    except Exception:
        return None


def _resolve_vllm_version(
    *,
    explicit_version: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[str]:
    if explicit_version:
        return explicit_version
    env_version = os.getenv("HOSTED_VLLM_VERSION")
    if env_version:
        return env_version
    resolved_api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")
    if resolved_api_base:
        return _fetch_vllm_version(resolved_api_base)
    return None


def _strip_legacy_tool_schema_fields(tools: Any) -> Any:
    tools_copy = copy.deepcopy(tools)
    tools_copy = _remove_additional_properties(tools_copy)
    tools_copy = _remove_strict_from_schema(tools_copy)
    return tools_copy


class HostedVLLMChatConfig(OpenAIGPTConfig):
    def _convert_custom_tools_to_function_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        vLLM chat completions currently accepts only OpenAI function tools.
        Convert custom tools into function tools so request validation does not fail.
        """
        converted_tools: List[Dict[str, Any]] = []
        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                converted_tools.append(tool)
                continue

            if tool.get("type") != "custom":
                converted_tools.append(tool)
                continue

            custom_tool = tool.get("custom", {})
            if not isinstance(custom_tool, dict):
                custom_tool = {}

            tool_name = custom_tool.get("name") or tool.get("name") or f"custom_tool_{idx}"
            tool_description = custom_tool.get("description") or tool.get("description")
            tool_parameters = custom_tool.get("input_schema") or tool.get("input_schema")

            if not isinstance(tool_parameters, dict):
                tool_parameters = {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Raw tool input payload.",
                        }
                    },
                    "required": ["input"],
                }

            function_tool: Dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": str(tool_name),
                    "parameters": tool_parameters,
                },
            }
            if isinstance(tool_description, str):
                function_tool["function"]["description"] = tool_description

            converted_tools.append(function_tool)

        return converted_tools

    def get_supported_openai_params(self, model: str) -> List[str]:
        params = super().get_supported_openai_params(model)
        params.extend(["reasoning_effort", "thinking"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        explicit_version = non_default_params.pop("vllm_version", None)
        if explicit_version is None:
            explicit_version = optional_params.pop("vllm_version", None)

        _tools = non_default_params.pop("tools", None)
        if _tools is not None:
            # Preserve tools[].function.strict and additionalProperties on
            # modern vLLM (v0.24.0+). Strip only when we know the server is
            # older — BerriAI/litellm#6088 / vllm-project/vllm#15526.
            if isinstance(_tools, list):
                _tools = self._convert_custom_tools_to_function_tools(_tools)
            vllm_version = _resolve_vllm_version(
                explicit_version=explicit_version if isinstance(explicit_version, str) else None,
                api_base=optional_params.get("api_base"),
            )
            if _should_strip_legacy_tool_schema_fields(vllm_version):
                _tools = _strip_legacy_tool_schema_fields(_tools)
        if _tools is not None:
            non_default_params["tools"] = _tools

        thinking = non_default_params.pop("thinking", None)
        if thinking is not None and isinstance(thinking, dict):
            if thinking.get("type") == "enabled":
                if "reasoning_effort" not in non_default_params:
                    non_default_params["reasoning_effort"] = reasoning_effort_from_thinking_budget(
                        thinking.get("budget_tokens", 0)
                    )

        return super().map_openai_params(non_default_params, optional_params, model, drop_params)

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        tools = optional_params.get("tools")
        if tools is not None:
            explicit_version = litellm_params.get("vllm_version")
            if not isinstance(explicit_version, str):
                explicit_version = None
            api_base = litellm_params.get("api_base")
            if not isinstance(api_base, str):
                api_base = None
            vllm_version = _resolve_vllm_version(
                explicit_version=explicit_version,
                api_base=api_base,
            )
            if _should_strip_legacy_tool_schema_fields(vllm_version):
                optional_params = {
                    **optional_params,
                    "tools": _strip_legacy_tool_schema_fields(tools),
                }

        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")
        dynamic_api_key = api_key or get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"
        return api_base, dynamic_api_key

    def _is_video_file(self, content_item: ChatCompletionFileObject) -> bool:
        file = content_item.get("file", {})
        format = file.get("format")
        file_data = file.get("file_data")
        file_id = file.get("file_id")
        if content_item.get("type") != "file":
            return False
        if format and format.startswith("video/"):
            return True
        elif file_data:
            mime_type = _parse_mime_type(file_data)
            if mime_type and mime_type.startswith("video/"):
                return True
        elif file_id:
            mime_type = _get_image_mime_type_from_url(file_id)
            if mime_type and mime_type.startswith("video/"):
                return True
        return False

    def _convert_file_to_video_url(self, content_item: ChatCompletionFileObject) -> ChatCompletionVideoObject:
        file = content_item.get("file", {})
        file_id = file.get("file_id")
        file_data = file.get("file_data")

        if file_id:
            return ChatCompletionVideoObject(type="video_url", video_url=ChatCompletionVideoUrlObject(url=file_id))
        elif file_data:
            return ChatCompletionVideoObject(type="video_url", video_url=ChatCompletionVideoUrlObject(url=file_data))
        raise ValueError("file_id or file_data is required")

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]: ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Support translating:
        - video files from file_id or file_data to video_url
        - thinking_blocks on assistant messages are removed, and content lists
          are converted to strings for vLLM compatibility
        """
        for message in messages:
            if message["role"] == "assistant":
                message.pop("thinking_blocks", None)
                existing_content = message.get("content")
                if isinstance(existing_content, list):
                    text_parts = []
                    tool_calls: list[ChatCompletionAssistantToolCall] = []
                    content_blocks: list[object] = []
                    has_structured_content = False
                    for c in existing_content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                            content_blocks.append(c)
                        elif isinstance(c, dict) and c.get("type") == "tool_use":
                            tool_input = c.get("input", {})
                            tool_calls.append(
                                ChatCompletionAssistantToolCall(
                                    id=c.get("id"),
                                    type="function",
                                    function=ChatCompletionToolCallFunctionChunk(
                                        name=c.get("name"),
                                        arguments=(
                                            tool_input
                                            if isinstance(
                                                tool_input,
                                                str,
                                            )
                                            else json.dumps(tool_input)
                                        ),
                                    ),
                                )
                            )
                        else:
                            content_blocks.append(c)
                            has_structured_content = True
                    if tool_calls:
                        existing_tool_calls = message.get("tool_calls")
                        if isinstance(existing_tool_calls, list):
                            existing_tool_call_ids = {
                                tool_call.get("id")
                                for tool_call in existing_tool_calls
                                if isinstance(tool_call, dict) and tool_call.get("id") is not None
                            }
                            new_tool_calls = [
                                tool_call
                                for tool_call in tool_calls
                                if tool_call.get("id") not in existing_tool_call_ids
                            ]
                            if new_tool_calls:
                                message["tool_calls"] = existing_tool_calls + new_tool_calls
                        else:
                            message["tool_calls"] = tool_calls
                    content_str = "\n".join(text_parts)
                    new_content = content_blocks if has_structured_content else content_str
                    message["content"] = new_content  # type: ignore[typeddict-item]
            elif message["role"] == "user":
                message_content = message.get("content")
                if message_content and isinstance(message_content, list):
                    replaced_content_items: List[Tuple[int, ChatCompletionFileObject]] = []
                    for idx, content_item in enumerate(message_content):
                        if content_item.get("type") == "file":
                            content_item = cast(ChatCompletionFileObject, content_item)
                            if self._is_video_file(content_item):
                                replaced_content_items.append((idx, content_item))
                    for idx, content_item in replaced_content_items:
                        message_content[idx] = self._convert_file_to_video_url(content_item)

        if is_async:
            return super()._transform_messages(messages, model, is_async=cast(Literal[True], True))
        else:
            return super()._transform_messages(messages, model, is_async=cast(Literal[False], False))
