import logging

import requests

logger = logging.getLogger(__name__)

_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """Send chat-completion requests to OpenRouter with automatic retries.

    Args:
        api_key: OpenRouter API key.
        retries: Number of *extra* attempts after the first (default 1 → 2 total).
    """

    def __init__(self, api_key: str, retries: int = 1):
        self.api_key = api_key
        self.retries = retries

    def chat(
        self,
        model: str,
        messages: list[dict],
        response_format: dict | None = None,
        timeout: int = 60,
    ) -> str:
        """Send a chat request and return the assistant message content.

        Retries up to ``self.retries`` additional times on any exception.

        Args:
            model: OpenRouter model identifier (e.g. ``"openai/gpt-4o-mini"``).
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            response_format: Optional ``{"type": "json_object"}`` or similar.
            timeout: Per-request timeout in seconds.

        Returns:
            The stripped content string from the first choice.

        Raises:
            The last exception encountered when all attempts fail.
        """
        payload: dict = {"model": model, "messages": messages}
        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    _URL, headers=headers, json=payload, timeout=timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_exc = e
                if attempt < self.retries:
                    logger.warning(
                        "OpenRouter attempt %d/%d failed: %s — retrying…",
                        attempt + 1,
                        self.retries + 1,
                        e,
                    )

        raise last_exc  # type: ignore[misc]
