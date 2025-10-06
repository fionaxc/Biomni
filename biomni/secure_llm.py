"""
Custom LangChain wrapper for Stanford Healthcare's secure LLM API.

This module provides a custom chat model implementation that works with
the Stanford Healthcare secure API gateway, supporting multiple model types
including GPT, Claude, and other models accessible through the secure endpoint.
"""

import json
from typing import Any, Iterator, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


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

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "secure-chat"

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
        # Convert messages to the appropriate format
        prompt = self._messages_to_prompt(messages)

        # Build the request payload based on model type
        payload = self._build_payload(prompt)

        # Set up headers
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/json"
        }

        # Make the API request
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        # Extract the response content based on model type
        content = self._extract_content(response.json())

        # Create the chat generation
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
                openai_messages.append({"role": "assistant", "content": msg.content})
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

    @property
    def _identifying_params(self) -> dict:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "api_url": self.api_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
