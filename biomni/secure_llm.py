"""
Custom LangChain wrapper for Stanford Healthcare's secure LLM API.

This module provides a custom chat model implementation that works with
the Stanford Healthcare secure API gateway, supporting multiple model types
including GPT, Claude, and other models accessible through the secure endpoint.
"""

import json
from typing import Any, Iterator, List, Optional, Union

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


class SecureChatModel(BaseChatModel):
    """Custom chat model for Stanford Healthcare's secure API gateway.

    This model supports multiple underlying model types (GPT, Claude, Llama, etc.)
    accessed through a secure API gateway with subscription key authentication.

    Args:
        model_id: The model identifier (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
        api_url: The API endpoint URL
        api_key: The subscription key for API authentication
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 8192)

    Example:
        >>> from biomni.secure_llm import SecureChatModel
        >>> llm = SecureChatModel(
        ...     model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        ...     api_url="https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa",
        ...     api_key=os.environ["OPENAI_API_KEY"]
        ... )
        >>> response = llm.invoke([HumanMessage(content="What is DNA?")])
    """

    model_id: str
    api_url: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 8192
    bound_tools: Optional[List[dict]] = None

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "secure-chat"

    def bind_tools(
        self,
        tools: List[Union[dict, type, BaseTool]],
        **kwargs: Any,
    ) -> "SecureChatModel":
        """Bind tools to the model.

        Args:
            tools: List of tools to bind (can be dicts, types, or BaseTool objects)
            **kwargs: Additional keyword arguments

        Returns:
            A new instance of SecureChatModel with bound tools
        """
        # Convert tools to OpenAI format
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            else:
                formatted_tools.append(convert_to_openai_tool(tool))

        # Create a new instance with bound tools
        return self.__class__(
            model_id=self.model_id,
            api_url=self.api_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            bound_tools=formatted_tools,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the secure API.

        Args:
            messages: List of messages to send to the model
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional keyword arguments

        Returns:
            ChatResult containing the model's response
        """
        # Check if we're using tools (requires OpenAI-style format)
        if self.bound_tools:
            # Use OpenAI-style message format for tool calling
            openai_messages = self._messages_to_openai_format(messages)
            payload = self._build_payload_with_tools(openai_messages)
        else:
            # Convert messages to the appropriate format
            prompt = self._messages_to_prompt(messages)
            # Build the request payload based on model type
            payload = self._build_payload(prompt)

        # Set up headers
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json"
        }

        # Debug logging
        print(f"\n=== DEBUG: API Request ===")
        print(f"URL: {self.api_url}")
        print(f"Model ID: {self.model_id}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print("=" * 50)

        # Make the API request
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))

        # Debug logging for response
        print(f"\n=== DEBUG: API Response ===")
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Response Text: {response.text}")
        print("=" * 50)

        response.raise_for_status()

        # Extract the response content and tool calls
        response_json = response.json()

        if self.bound_tools:
            content, tool_calls = self._extract_content_and_tool_calls(response_json)
            message = AIMessage(content=content, tool_calls=tool_calls)
        else:
            content = self._extract_content(response_json)
            message = AIMessage(content=content)

        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to a prompt string.

        Args:
            messages: List of LangChain messages

        Returns:
            Formatted prompt string
        """
        # For models that expect a simple prompt string (like Claude on Bedrock)
        # we concatenate all messages
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")
            else:
                prompt_parts.append(str(msg.content))

        return "\n\n".join(prompt_parts)

    def _messages_to_openai_format(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to OpenAI chat format.

        Args:
            messages: List of LangChain messages

        Returns:
            List of message dictionaries in OpenAI format
        """
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                message_dict = {"role": "assistant", "content": msg.content or ""}
                # Include tool calls if present
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    message_dict["tool_calls"] = []
                    for tool_call in msg.tool_calls:
                        message_dict["tool_calls"].append({
                            "id": tool_call.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"])
                            }
                        })
                openai_messages.append(message_dict)
            elif isinstance(msg, ToolMessage):
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
            else:
                openai_messages.append({"role": "user", "content": str(msg.content)})

        return openai_messages

    def _build_payload(self, prompt: str) -> dict:
        """Build the API request payload based on model type.

        Args:
            prompt: The formatted prompt string

        Returns:
            Dictionary payload for the API request
        """
        # Detect model type and build appropriate payload
        if self.model_id in ["gpt-4o", "gpt-4.1"]:
            # OpenAI-style models
            return {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}]
            }
        elif self.model_id.startswith("anthropic.claude") or self.model_id.startswith("arn:aws:bedrock"):
            # Claude models on Bedrock
            return {
                "model_id": self.model_id,
                "prompt_text": prompt
            }
        elif self.model_id in ["Llama-3.3-70B-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-4-Scout-17B-16E-Instruct"]:
            # Llama models
            return {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens
            }
        elif self.model_id == "deepseek-chat":
            # DeepSeek models
            return {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": 1,
                "stream": False
            }
        elif self.model_id == "gemini-1-5":
            # Gemini models
            return {
                "contents": {"role": "user", "parts": {"text": prompt}},
                "safety_settings": {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_LOW_AND_ABOVE"},
                "generation_config": {"temperature": self.temperature, "topP": 0.8, "topK": 40}
            }
        else:
            # Default to OpenAI-style format
            return {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

    def _extract_content(self, response_json: dict) -> str:
        """Extract the content from the API response.

        Args:
            response_json: The JSON response from the API

        Returns:
            The text content from the response
        """
        # Unwrap Bedrock response if needed
        if "body" in response_json and isinstance(response_json["body"], dict):
            response_json = response_json["body"]

        # Try different response formats based on model type
        if "choices" in response_json:
            # OpenAI-style response (GPT, Llama, DeepSeek)
            return response_json["choices"][0]["message"]["content"]
        elif "content" in response_json:
            # Anthropic-style response (Claude)
            if isinstance(response_json["content"], list):
                return response_json["content"][0]["text"]
            else:
                return response_json["content"]
        else:
            # Fallback - try to find any text content
            raise ValueError(f"Unable to extract content from response: {response_json}")

    def _build_payload_with_tools(self, messages: List[dict]) -> dict:
        """Build the API request payload with tools included.

        Args:
            messages: List of message dictionaries in OpenAI format

        Returns:
            Dictionary payload for the API request with tools
        """
        # Build base payload for models that support tool calling
        if self.model_id in ["gpt-4o", "gpt-4.1"] or self.model_id.startswith("gpt-"):
            # OpenAI-style models
            payload = {
                "model": self.model_id,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "tools": self.bound_tools,
            }
        elif self.model_id.startswith("anthropic.claude") or self.model_id.startswith("arn:aws:bedrock"):
            # Claude models on Bedrock - convert to Anthropic Messages API format with tools
            anthropic_messages = []
            system_content = None

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    system_content = content
                elif role == "user":
                    anthropic_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    message_dict = {"role": "assistant", "content": content or ""}
                    # Include tool calls if present
                    if "tool_calls" in msg and msg["tool_calls"]:
                        # Convert OpenAI format tool calls to Anthropic format
                        content_blocks = []
                        if content:
                            content_blocks.append({"type": "text", "text": content})
                        for tool_call in msg["tool_calls"]:
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tool_call.get("id", ""),
                                "name": tool_call["function"]["name"],
                                "input": json.loads(tool_call["function"]["arguments"])
                            })
                        message_dict["content"] = content_blocks
                    anthropic_messages.append(message_dict)
                elif role == "tool":
                    # Tool results in Anthropic format
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content
                        }]
                    })

            # Build payload with Anthropic Messages API format
            # The Stanford API gateway wraps Bedrock calls
            inner_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # Add system message if present
            if system_content:
                inner_payload["system"] = system_content

            # Add tools if present (convert to Anthropic format)
            if self.bound_tools:
                anthropic_tools = []
                for tool in self.bound_tools:
                    func = tool.get("function", {})
                    anthropic_tools.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {})
                    })
                inner_payload["tools"] = anthropic_tools

            # For Stanford's Bedrock proxy, send the Anthropic format directly
            # The API expects the Anthropic Messages API format, not wrapped
            payload = inner_payload
            payload["model_id"] = self.model_id
        elif self.model_id in ["Llama-3.3-70B-Instruct", "Llama-4-Maverick-17B-128E-Instruct-FP8", "Llama-4-Scout-17B-16E-Instruct"]:
            # Llama models may support tools
            payload = {
                "model": self.model_id,
                "messages": messages,
                "tools": self.bound_tools,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        elif self.model_id == "deepseek-chat":
            # DeepSeek models with tools
            payload = {
                "model": self.model_id,
                "messages": messages,
                "tools": self.bound_tools,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": 1,
                "stream": False
            }
        else:
            # Default to OpenAI-style format with tools
            payload = {
                "model": self.model_id,
                "messages": messages,
                "tools": self.bound_tools,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

        return payload

    def _extract_content_and_tool_calls(self, response_json: dict) -> tuple:
        """Extract content and tool calls from the API response.

        Args:
            response_json: The JSON response from the API

        Returns:
            Tuple of (content, tool_calls) where tool_calls is a list of dicts
        """
        tool_calls = []
        content = ""

        # Unwrap Bedrock response if needed
        if "body" in response_json and isinstance(response_json["body"], dict):
            response_json = response_json["body"]

        # Try different response formats based on model type
        if "choices" in response_json:
            # OpenAI-style response (GPT, Llama, DeepSeek)
            message = response_json["choices"][0]["message"]
            content = message.get("content", "")

            # Extract tool calls if present
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    tool_calls.append({
                        "name": tool_call["function"]["name"],
                        "args": json.loads(tool_call["function"]["arguments"]),
                        "id": tool_call.get("id", ""),
                    })

        elif "content" in response_json:
            # Anthropic-style response (Claude)
            content_list = response_json["content"]
            if isinstance(content_list, list):
                for item in content_list:
                    if item.get("type") == "text":
                        content += item.get("text", "")
                    elif item.get("type") == "tool_use":
                        tool_calls.append({
                            "name": item.get("name", ""),
                            "args": item.get("input", {}),
                            "id": item.get("id", ""),
                        })
            else:
                content = content_list

        return content, tool_calls

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "api_url": self.api_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
