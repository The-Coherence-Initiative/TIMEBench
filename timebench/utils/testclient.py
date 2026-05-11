"""Lightweight OpenAI-compatible chat-completion client (sync + async).

Provides :class:`ChatCompletionClient` for synchronous usage and
:class:`AsyncChatCompletionClient` for async/streaming workflows.
Both return typed wrapper objects so callers can use attribute access
(``response.choices[0].message.content``) in addition to dict-style access.
"""

import json
from typing import AsyncGenerator, Dict, Generator, List, Optional, Union

import aiohttp
import requests


# ---------------------------------------------------------------------------
# Response wrapper types
# ---------------------------------------------------------------------------


class ChatMessage:
    """Thin wrapper around a message object enabling attribute access."""

    def __init__(self, data: Dict) -> None:
        self._data = data

    @property
    def content(self) -> Optional[str]:
        return self._data.get("content")

    @property
    def role(self) -> Optional[str]:
        return self._data.get("role")

    @property
    def function_call(self):
        return self._data.get("function_call")

    @property
    def tool_calls(self):
        return self._data.get("tool_calls")

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


class ChatChoice:
    """Thin wrapper around a single choice object."""

    def __init__(self, data: Dict) -> None:
        self._data = data

    @property
    def message(self) -> ChatMessage:
        return ChatMessage(self._data.get("message", {}))

    @property
    def delta(self) -> ChatMessage:
        """Populated for streaming responses."""
        return ChatMessage(self._data.get("delta", {}))

    @property
    def index(self) -> Optional[int]:
        return self._data.get("index")

    @property
    def finish_reason(self) -> Optional[str]:
        return self._data.get("finish_reason")

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


class ChatUsage:
    """Thin wrapper around token-usage information."""

    def __init__(self, data: Dict) -> None:
        self._data = data

    @property
    def prompt_tokens(self) -> int:
        return self._data.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self._data.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self._data.get("total_tokens", 0)

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


class ChatCompletionResponse:
    """Thin wrapper around a chat-completion response enabling attribute access."""

    def __init__(self, data: Dict) -> None:
        self._data = data

    @property
    def choices(self) -> List[ChatChoice]:
        return [ChatChoice(c) for c in self._data.get("choices", [])]

    @property
    def usage(self) -> ChatUsage:
        return ChatUsage(self._data.get("usage", {}))

    @property
    def id(self) -> Optional[str]:
        return self._data.get("id")

    @property
    def object(self) -> Optional[str]:
        return self._data.get("object")

    @property
    def created(self) -> Optional[int]:
        return self._data.get("created")

    @property
    def model(self) -> Optional[str]:
        return self._data.get("model")

    @property
    def system_fingerprint(self) -> Optional[str]:
        """Backend configuration fingerprint returned by the server."""
        return self._data.get("system_fingerprint")

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


# ---------------------------------------------------------------------------
# Synchronous client
# ---------------------------------------------------------------------------


class ChatCompletionClient:
    """Synchronous OpenAI-compatible chat-completion client."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletionResponse, Generator[ChatCompletionResponse, None, None]]:
        """Send a chat-completion request.

        Parameters
        ----------
        messages:
            Conversation history as a list of ``{"role": ..., "content": ...}`` dicts.
        model:
            Model identifier to use.
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum tokens to generate.  Mapped to ``max_tokens`` in the
            payload unless ``max_completion_tokens`` is passed via *kwargs*,
            in which case that takes precedence.
        stream:
            If ``True``, returns a generator of partial :class:`ChatCompletionResponse`
            objects instead of a single complete response.
        **kwargs:
            Any additional parameters forwarded verbatim to the API (e.g.
            ``seed``, ``top_p``, ``top_k``).
        """
        url = f"{self.base_url}/chat/completions"

        payload: Dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        if "max_completion_tokens" in kwargs:
            payload["max_completion_tokens"] = kwargs.pop("max_completion_tokens")
        elif max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if stream:
            return self._stream_completion(url, payload)

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return ChatCompletionResponse(response.json())

    def _stream_completion(
        self,
        url: str,
        payload: Dict,
    ) -> Generator[ChatCompletionResponse, None, None]:
        """Yield partial :class:`ChatCompletionResponse` objects for a streaming request."""
        with requests.post(url, headers=self.headers, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]  # strip "data: " prefix
                if data == "[DONE]":
                    break
                try:
                    yield ChatCompletionResponse(json.loads(data))
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Asynchronous client
# ---------------------------------------------------------------------------


class AsyncChatCompletionClient:
    """Async OpenAI-compatible chat-completion client."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionResponse, None]]:
        """Async version of :meth:`ChatCompletionClient.create_chat_completion`."""
        url = f"{self.base_url}/chat/completions"

        payload: Dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        if "max_completion_tokens" in kwargs:
            payload["max_completion_tokens"] = kwargs.pop("max_completion_tokens")
        elif max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if stream:
            return self._stream_completion_async(url, payload)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return ChatCompletionResponse(data)

    async def _stream_completion_async(
        self,
        url: str,
        payload: Dict,
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Yield partial responses for an async streaming request."""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield ChatCompletionResponse(json.loads(data))
                    except json.JSONDecodeError:
                        continue
