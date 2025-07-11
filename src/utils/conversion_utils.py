import json
import logging
import traceback
import uuid
from typing import Any, Dict, Union

from models import MessagesRequest

logger = logging.getLogger(__name__)


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    messages = []
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, "type") and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # flatten tool_result blocks into user content
            text_content = ""
            for block in content:
                if hasattr(block, "type") and block.type == "text":
                    text_content += block.text + "\n"
                elif hasattr(block, "type") and block.type == "tool_result":
                    result = block.content if hasattr(block, "content") else ""
                    text_content += f"Tool result: {result}\n"
            messages.append({"role": msg.role, "content": text_content.strip()})
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith(
        "openai/"
    ) or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(
            f"Capping max_tokens to 16384 (original: {anthropic_request.max_tokens})"
        )
    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    return litellm_request
