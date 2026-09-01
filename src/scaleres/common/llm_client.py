"""Shared OpenAI-compatible async chat-completion helpers with retry/backoff.

Used by scaleres.generation.generate_topics, generate_synthetic_answers, and
translate_answers, which otherwise each reimplemented the same
exponential-backoff-with-jitter retry loop around
``client.chat.completions.create``.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Dict, Optional, Sequence

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_JITTER_MAX = 0.5


def build_async_client(
    *,
    api_base: str,
    api_key: str,
    timeout: Optional[float] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> AsyncOpenAI:
    """Construct an AsyncOpenAI client pointed at an OpenAI-compatible endpoint.

    Calls ``load_dotenv()`` so callers can rely on a ``.env``-provided API key.
    """

    load_dotenv()
    kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": api_base}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if default_headers:
        kwargs["default_headers"] = default_headers
    return AsyncOpenAI(**kwargs)


async def retrying_chat_completion(
    client: AsyncOpenAI,
    messages: Sequence[ChatCompletionMessageParam],
    request_semaphore: asyncio.Semaphore,
    *,
    max_retries: int,
    request_delay: float = 0.0,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    jitter_max: float = DEFAULT_JITTER_MAX,
    **create_kwargs: Any,
) -> ChatCompletion:
    """Call ``chat.completions.create`` with exponential backoff + jitter retry.

    ``create_kwargs`` is forwarded verbatim (model, temperature, top_p,
    max_tokens, reasoning_effort, extra_body, frequency_penalty,
    response_format, ...) alongside the semaphore-gated ``messages``.
    """

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            async with request_semaphore:
                response = await client.chat.completions.create(
                    messages=list(messages), **create_kwargs
                )
            if request_delay:
                await asyncio.sleep(request_delay)
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                raise
            sleep_for = backoff_base ** (attempt - 1)
            await asyncio.sleep(sleep_for + random.uniform(0, jitter_max))

    raise RuntimeError("Unreachable state in retrying_chat_completion") from last_error


def parse_extra_body_json(
    raw: Optional[str], default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Parse an optional ``--extra-body`` JSON string into a dict.

    Returns a copy of ``default`` (or ``{}``) when ``raw`` is falsy.
    """

    if not raw:
        return dict(default) if default else {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise SystemExit(f"Invalid JSON for --extra-body: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--extra-body must be a JSON object")
    return payload
