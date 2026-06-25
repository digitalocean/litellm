# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
from collections import deque
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Literal,
    Optional,
    Tuple,
)

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.anthropic import UsageDelta
from litellm.types.utils import AdapterCompletionStreamWrapper

if TYPE_CHECKING:
    from litellm.types.llms.anthropic import ContentBlockContentBlockDict
    from litellm.types.utils import ModelResponseStream


class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - first chunk return 'message_start'
    - content block must be started and stopped
    - finish_reason must map exactly to anthropic reason, else anthropic client won't be able to parse it.
    """

    from litellm.types.llms.anthropic import (
        ContentBlockContentBlockDict,
        ContentBlockStart,
        ContentBlockStartText,
        TextBlock,
    )

    sent_first_chunk: bool = False
    sent_content_block_start: bool = False
    sent_content_block_finish: bool = False
    current_content_block_type: Literal["text", "tool_use", "thinking"] = "text"
    sent_last_message: bool = False
    holding_chunk: Optional[Any] = None
    holding_stop_reason_chunk: Optional[Any] = None
    queued_usage_chunk: bool = False
    current_content_block_index: int = 0
    current_content_block_start: ContentBlockContentBlockDict = TextBlock(
        type="text",
        text="",
    )
    chunk_queue: deque = deque()  # Queue for buffering multiple chunks
    _peek_buffer: deque = deque()  # Chunks consumed during initial type peek
    _tool_use_header_seen: bool = False

    def __init__(
        self,
        completion_stream: Any,
        model: str,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(completion_stream)
        self.model = model
        # Mapping of truncated tool names to original names (for OpenAI's 64-char limit)
        self.tool_name_mapping = tool_name_mapping or {}
        # Initialize queues per-instance so concurrent streams don't share state.
        self.chunk_queue = deque()
        self._peek_buffer = deque()
        self._tool_use_header_seen = False

    def _create_initial_usage_delta(self) -> UsageDelta:
        """
        Create the initial UsageDelta for the message_start event.

        Initializes cache token fields (cache_creation_input_tokens, cache_read_input_tokens)
        to 0 to indicate to clients (like Claude Code) that prompt caching is supported.

        The actual cache token values will be provided in the message_delta event at the
        end of the stream, since Bedrock Converse API only returns usage data in the final
        response chunk.

        Returns:
            UsageDelta with all token counts initialized to 0.
        """
        return UsageDelta(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def __next__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                buffered, block_type, content_block_start = (
                    self._collect_initial_peek_chunks_sync(
                        LiteLLMAnthropicMessagesAdapter()
                    )
                )
                self.current_content_block_type = block_type
                self.current_content_block_start = content_block_start
                if buffered:
                    self._peek_buffer = deque(buffered)
                    self._tool_use_header_seen = False
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self._build_initial_content_block(
                                block_type, content_block_start
                            ),
                        }
                    )
                else:
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                    )
                return self.chunk_queue.popleft()

            for chunk in self._iter_stream_chunks():
                if chunk == "None" or chunk is None:
                    raise Exception

                should_start_new_block = self._should_start_new_content_block(chunk)
                if should_start_new_block:
                    self._increment_content_block_index()
                elif self._chunk_is_tool_use_header(chunk):
                    self._tool_use_header_seen = True

                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )

                if should_start_new_block and not self.sent_content_block_finish:
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": max(self.current_content_block_index - 1, 0),
                        }
                    )

                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )
                    self.chunk_queue.append(processed_chunk)
                    if self._chunk_is_tool_use_header(chunk):
                        self._tool_use_header_seen = True
                    self.sent_content_block_finish = False
                    return self.chunk_queue.popleft()

                if (
                    processed_chunk["type"] == "message_delta"
                    and self.sent_content_block_finish is False
                ):
                    # Queue both the content_block_stop and the message_delta
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": self.current_content_block_index,
                        }
                    )
                    self.sent_content_block_finish = True
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()
                elif self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.chunk_queue.append(processed_chunk)
                    self.holding_chunk = None
                    return self.chunk_queue.popleft()
                else:
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends
            if self.holding_chunk is not None:
                self.chunk_queue.append(self.holding_chunk)
                self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration
        except StopIteration:
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            if self.sent_last_message is False:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopIteration
        except Exception as e:
            verbose_logger.error(
                "Anthropic Adapter - {}\n{}".format(e, traceback.format_exc())
            )
            raise StopAsyncIteration

    async def __anext__(self):  # noqa: PLR0915
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                buffered, block_type, content_block_start = (
                    await self._collect_initial_peek_chunks_async(
                        LiteLLMAnthropicMessagesAdapter()
                    )
                )
                self.current_content_block_type = block_type
                self.current_content_block_start = content_block_start
                if buffered:
                    self._peek_buffer = deque(buffered)
                    self._tool_use_header_seen = False
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self._build_initial_content_block(
                                block_type, content_block_start
                            ),
                        }
                    )
                else:
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                    )
                return self.chunk_queue.popleft()

            async for chunk in self._async_iter_stream_chunks():
                if chunk == "None" or chunk is None:
                    raise Exception

                # Check if we need to start a new content block
                should_start_new_block = self._should_start_new_content_block(chunk)
                if should_start_new_block:
                    self._increment_content_block_index()
                elif self._chunk_is_tool_use_header(chunk):
                    self._tool_use_header_seen = True

                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )

                # Check if this is a usage chunk and we have a held stop_reason chunk
                if (
                    self.holding_stop_reason_chunk is not None
                    and getattr(chunk, "usage", None) is not None
                ):
                    # Merge usage into the held stop_reason chunk
                    merged_chunk = self.holding_stop_reason_chunk.copy()
                    if "delta" not in merged_chunk:
                        merged_chunk["delta"] = {}

                    # Add usage to the held chunk
                    uncached_input_tokens = chunk.usage.prompt_tokens or 0
                    if (
                        hasattr(chunk.usage, "prompt_tokens_details")
                        and chunk.usage.prompt_tokens_details
                    ):
                        cached_tokens = (
                            getattr(
                                chunk.usage.prompt_tokens_details, "cached_tokens", 0
                            )
                            or 0
                        )
                        uncached_input_tokens -= cached_tokens

                    usage_dict: UsageDelta = {
                        "input_tokens": uncached_input_tokens,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
                    # Add cache tokens if available (for prompt caching support)
                    if (
                        hasattr(chunk.usage, "_cache_creation_input_tokens")
                        and chunk.usage._cache_creation_input_tokens > 0
                    ):
                        usage_dict["cache_creation_input_tokens"] = (
                            chunk.usage._cache_creation_input_tokens
                        )
                    if (
                        hasattr(chunk.usage, "_cache_read_input_tokens")
                        and chunk.usage._cache_read_input_tokens > 0
                    ):
                        usage_dict["cache_read_input_tokens"] = (
                            chunk.usage._cache_read_input_tokens
                        )
                    merged_chunk["usage"] = usage_dict

                    # Queue the merged chunk and reset
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()

                # Check if this processed chunk has a stop_reason - hold it for next chunk

                if not self.queued_usage_chunk:
                    if should_start_new_block and not self.sent_content_block_finish:
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": max(self.current_content_block_index - 1, 0),
                            }
                        )

                        self.chunk_queue.append(
                            {
                                "type": "content_block_start",
                                "index": self.current_content_block_index,
                                "content_block": self.current_content_block_start,
                            }
                        )
                        self.chunk_queue.append(processed_chunk)
                        if self._chunk_is_tool_use_header(chunk):
                            self._tool_use_header_seen = True
                        self.sent_content_block_finish = False

                        return self.chunk_queue.popleft()

                    if (
                        processed_chunk["type"] == "message_delta"
                        and self.sent_content_block_finish is False
                    ):
                        # Queue both the content_block_stop and the holding chunk
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": self.current_content_block_index,
                            }
                        )
                        self.sent_content_block_finish = True
                        if (
                            processed_chunk.get("delta", {}).get("stop_reason")
                            is not None
                        ):
                            self.holding_stop_reason_chunk = processed_chunk
                        else:
                            self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()
                    elif self.holding_chunk is not None:
                        # Queue both chunks
                        self.chunk_queue.append(self.holding_chunk)
                        self.chunk_queue.append(processed_chunk)
                        self.holding_chunk = None
                        return self.chunk_queue.popleft()
                    else:
                        # Queue the current chunk
                        self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends
            if not self.queued_usage_chunk:
                if self.holding_stop_reason_chunk is not None:
                    self.chunk_queue.append(self.holding_stop_reason_chunk)
                    self.holding_stop_reason_chunk = None

                if self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            # Return queued items if any
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration

        except StopIteration:
            # Handle any remaining queued chunks before stopping
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            # Handle any held stop_reason chunk
            if self.holding_stop_reason_chunk is not None:
                return self.holding_stop_reason_chunk
            if not self.sent_last_message:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopAsyncIteration

    def anthropic_sse_wrapper(self) -> Iterator[bytes]:
        """
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        Similar to the Bedrock bedrock_sse_wrapper implementation.

        This wrapper ensures dict chunks are SSE formatted with both event and data lines.
        """
        for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of anthropic_sse_wrapper.
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        """
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    def _iter_stream_chunks(self) -> Iterator[Any]:
        while self._peek_buffer:
            yield self._peek_buffer.popleft()
        yield from self.completion_stream

    async def _async_iter_stream_chunks(self) -> AsyncIterator[Any]:
        while self._peek_buffer:
            yield self._peek_buffer.popleft()
        async for chunk in self.completion_stream:
            yield chunk

    def _chunk_has_any_content(self, chunk: "ModelResponseStream") -> bool:
        if not chunk.choices:
            return False
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            return True
        if getattr(delta, "reasoning_content", None):
            return True
        thinking_blocks = getattr(delta, "thinking_blocks", None)
        if thinking_blocks:
            for thinking_block in thinking_blocks:
                if thinking_block.get("thinking") or thinking_block.get("signature"):
                    return True
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                if function and (function.name or function.arguments):
                    return True
                if getattr(tool_call, "id", None):
                    return True
        return False

    def _is_definitive_block_type_chunk(
        self, chunk: "ModelResponseStream", block_type: str
    ) -> bool:
        if block_type != "text":
            return True
        if self._chunk_has_any_content(chunk):
            return True
        if chunk.choices and chunk.choices[0].finish_reason is not None:
            return True
        return False

    def _build_initial_content_block(
        self,
        block_type: Literal["text", "tool_use", "thinking"],
        content_block_start: "ContentBlockContentBlockDict",
    ) -> dict:
        if block_type == "thinking":
            return {"type": "thinking", "thinking": ""}
        if block_type == "tool_use":
            return dict(content_block_start)
        return {"type": "text", "text": ""}

    def _collect_initial_peek_chunks_sync(
        self, adapter: Any
    ) -> Tuple[list[Any], Literal["text", "tool_use", "thinking"], Any]:
        buffered: list[Any] = []
        block_type: Literal["text", "tool_use", "thinking"] = "text"
        content_block_start = self.current_content_block_start
        determining_index = -1

        for chunk in self.completion_stream:
            if chunk == "None" or chunk is None:
                continue
            buffered.append(chunk)
            (
                block_type,
                content_block_start,
            ) = adapter._translate_streaming_openai_chunk_to_anthropic_content_block(
                choices=chunk.choices
            )
            if self._is_definitive_block_type_chunk(chunk, block_type):
                determining_index = len(buffered) - 1
                break

        if determining_index < 0 and buffered:
            determining_index = len(buffered) - 1

        replay_buffer = buffered[determining_index:] if determining_index >= 0 else []
        return replay_buffer, block_type, content_block_start

    async def _collect_initial_peek_chunks_async(
        self, adapter: Any
    ) -> Tuple[list[Any], Literal["text", "tool_use", "thinking"], Any]:
        buffered: list[Any] = []
        block_type: Literal["text", "tool_use", "thinking"] = "text"
        content_block_start = self.current_content_block_start
        determining_index = -1

        async for chunk in self.completion_stream:
            if chunk == "None" or chunk is None:
                continue
            buffered.append(chunk)
            (
                block_type,
                content_block_start,
            ) = adapter._translate_streaming_openai_chunk_to_anthropic_content_block(
                choices=chunk.choices
            )
            if self._is_definitive_block_type_chunk(chunk, block_type):
                determining_index = len(buffered) - 1
                break

        if determining_index < 0 and buffered:
            determining_index = len(buffered) - 1

        replay_buffer = buffered[determining_index:] if determining_index >= 0 else []
        return replay_buffer, block_type, content_block_start

    def _increment_content_block_index(self):
        self.current_content_block_index += 1

    def _should_start_new_content_block(self, chunk: "ModelResponseStream") -> bool:
        """
        Determine if we should start a new content block based on the processed chunk.
        Override this method with your specific logic for detecting new content blocks.

        Examples of when you might want to start a new content block:
        - Switching from text to tool calls
        - Different content types in the response
        - Specific markers in the content
        """
        from .transformation import LiteLLMAnthropicMessagesAdapter

        # Example logic - customize based on your needs:
        # If chunk indicates a tool call
        if chunk.choices[0].finish_reason is not None:
            return False

        (
            block_type,
            content_block_start,
        ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=chunk.choices  # type: ignore
        )

        # Restore original tool name if it was truncated for OpenAI's 64-char limit
        if block_type == "tool_use":
            # Type narrowing: content_block_start is ToolUseBlock when block_type is "tool_use"
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)

            if tool_block.get("name"):
                truncated_name = tool_block["name"]
                original_name = self.tool_name_mapping.get(
                    truncated_name, truncated_name
                )
                tool_block["name"] = original_name

        if block_type != self.current_content_block_type:
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            self._tool_use_header_seen = False
            return True

        # For parallel tool calls, a new content block starts when a different
        # tool id/name arrives, or when a repeated header arrives after the
        # current tool's opening header was already consumed.
        if block_type == "tool_use":
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)
            tool_name = tool_block.get("name")
            if tool_name:
                current_tool = cast(ToolUseBlock, self.current_content_block_start)
                current_name = current_tool.get("name")
                current_id = current_tool.get("id")
                tool_id = tool_block.get("id")
                same_header = (
                    self.current_content_block_type == "tool_use"
                    and tool_name == current_name
                    and tool_id == current_id
                )
                if same_header and not self._tool_use_header_seen:
                    return False
                if not same_header or self._tool_use_header_seen:
                    self.current_content_block_type = block_type
                    self.current_content_block_start = content_block_start
                    self._tool_use_header_seen = False
                    return True

        return False

    def _chunk_is_tool_use_header(self, chunk: "ModelResponseStream") -> bool:
        if not chunk.choices:
            return False
        tool_calls = getattr(chunk.choices[0].delta, "tool_calls", None)
        if not tool_calls:
            return False
        function = getattr(tool_calls[0], "function", None)
        return bool(function and function.name)
