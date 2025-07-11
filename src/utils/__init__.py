import logging
import uuid
import json
import traceback
from typing import Any, Dict, Union, AsyncGenerator

from models import MessagesResponse, Usage, MessagesRequest

logger = logging.getLogger(__name__)
from .conversion_utils import convert_anthropic_to_litellm
from .parser_utils import parse_tool_result_content
from .schema_utils import clean_gemini_schema


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest
) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""

    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")

        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = (
                message.content if message and hasattr(message, "content") else ""
            )
            tool_calls = (
                message.tool_calls
                if message and hasattr(message, "tool_calls")
                else None
            )
            finish_reason = (
                choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            )
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = (
                    litellm_response
                    if isinstance(litellm_response, dict)
                    else litellm_response.dict()
                )
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__
                try:
                    response_dict = (
                        litellm_response.model_dump()
                        if hasattr(litellm_response, "model_dump")
                        else litellm_response.__dict__
                    )
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, "id", f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, "choices", [{}]),
                        "usage": getattr(litellm_response, "usage", {}),
                    }

            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = (
                choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            )
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = (
                choices[0].get("finish_reason", "stop")
                if choices and len(choices) > 0
                else "stop"
            )
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        # Create content list for Anthropic format
        content = []

        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        # Add tool calls if present (tool_use in Anthropic format) - only for Claude models
        if tool_calls and is_claude_model:
            logger.debug(f"Processing tool calls: {tool_calls}")

            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = (function.get("arguments", "{}"),)
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = (
                        getattr(function, "arguments", "{}") if function else "{}"
                    )

                logger.debug(
                    f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}"
                )

                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": arguments,
                    }
                )
        elif tool_calls and not is_claude_model:
            # For non-Claude models, convert tool calls to text format
            logger.debug(
                f"Converting tool calls to text for non-Claude model: {clean_model}"
            )

            # We'll append tool info to the text content
            tool_text = "\n\nTool usage:\n"

            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = (function.get("arguments", "{}"),)
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = (
                        getattr(function, "arguments", "{}") if function else "{}"
                    )

                try:
                    args_dict = json.loads(arguments)
                    arguments_str = json.dumps(args_dict, indent=2)
                except Exception:
                    arguments_str = str(arguments)

                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"

            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})

        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})

        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

        return anthropic_response

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error converting response: {e}\n{error_traceback}")
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {e}"}],
            stop_reason=None,
            usage=Usage(input_tokens=0, output_tokens=0),
        )


async def handle_streaming(
    response_generator, original_request: MessagesRequest
) -> AsyncGenerator[str, None]:
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    # ... rest of implementation unchanged ...
