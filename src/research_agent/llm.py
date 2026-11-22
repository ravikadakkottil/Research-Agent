"""
LLM access in one place.

Every structured LLM call goes through ask_json(): it sends the prompt in
JSON mode, then VALIDATES the reply against a Pydantic model. Swapping
providers later (Phase 2) means changing this file only.

Async (AsyncGroq): while one LLM call waits on the network, other work
(e.g. OpenAlex requests or other LLM calls) can run.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypeVar

from groq import AsyncGroq, BadRequestError
from pydantic import BaseModel, ValidationError

from research_agent.cache import ResponseCache
from research_agent.config import Settings, get_settings
from research_agent.tracing import observe

# T = "whichever Pydantic model you pass in"; ask_json returns that same type.
T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """The LLM replied, but not in the shape we asked for."""


@lru_cache
def get_client() -> AsyncGroq:
    """
    One shared async Groq client (reuses the HTTPS connection).

    max_retries: the SDK itself retries rate limits (429) and server errors (5xx)
    with exponential backoff — the same idea as our OpenAlex retries.
    """
    settings = get_settings()
    return AsyncGroq(
        api_key=settings.groq_api_key.get_secret_value(),
        max_retries=settings.llm_max_retries,
    )


def create_llm_cache(settings: Settings) -> ResponseCache | None:
    """
    Cache for LLM answers (same file format as the OpenAlex cache).

    Why: LLMs are NOT deterministic — the same question gives slightly different
    search queries each run, so the OpenAlex cache never gets a hit. Caching the
    LLM output makes repeat runs identical, instant and free.
    """
    if not settings.cache_enabled:
        return None
    return ResponseCache(
        settings.cache_dir / "llm.sqlite", ttl_seconds=settings.cache_ttl_hours * 3600
    )


async def ask_json(
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    *,
    cache: ResponseCache | None = None,
    model: str | None = None,
) -> T:
    """
    One LLM call that must return JSON matching `schema`.

        prompt  ->  Groq (JSON mode)  ->  JSON string  ->  validated Pydantic object

    JSON mode guarantees valid JSON *syntax*; Pydantic checks the *shape*
    (required keys, types). Both are needed.

    With `cache`: identical (model, temperature, prompts, schema) -> stored answer.
    `model` overrides settings.llm_model (e.g. a stronger model for eval judging).
    """
    settings = get_settings()
    model = model or settings.llm_model

    key = None
    if cache:
        key = ResponseCache.make_key(
            "llm",
            {
                "model": model,
                "temperature": settings.llm_temperature,
                "system": system_prompt,
                "user": user_prompt,
                "schema": schema.__name__,
            },
        )
        if (hit := cache.get(key)) is not None:
            # Still traced, so the trace shows WHAT the plan/summary was.
            with observe(
                f"llm: {schema.__name__} (cached)",
                as_type="generation",
                model=model,
                input=_messages(system_prompt, user_prompt),
                output=hit,
                metadata={"cache_hit": True},
            ):
                return schema.model_validate(hit)

    # Models occasionally produce broken/empty JSON (Groq then answers 400
    # "json_validate_failed"), or valid JSON of the wrong shape. That's random,
    # so asking again usually works. Network errors are retried by the SDK.
    attempts = 1 + settings.llm_json_retries
    for attempt in range(1, attempts + 1):
        try:
            result = await _complete_json(system_prompt, user_prompt, schema, model, settings)
            break
        except LLMOutputError:
            if attempt == attempts:
                raise

    if key:  # only VALID answers are cached
        cache.set(key, result.model_dump(mode="json"))
    return result


async def _complete_json(
    system_prompt: str, user_prompt: str, schema: type[T], model: str, settings: Settings
) -> T:
    """One attempt. Bad output -> LLMOutputError (worth retrying); other errors as-is."""
    messages = _messages(system_prompt, user_prompt)
    # A "generation" observation: Langfuse shows prompt, reply, model, tokens, latency.
    # A failed attempt is marked ERROR, so retries are visible in the trace.
    with observe(
        f"llm: {schema.__name__}",
        as_type="generation",
        model=model,
        model_parameters={"temperature": settings.llm_temperature},
        input=messages,
    ) as gen:
        try:
            completion = await get_client().chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=settings.llm_temperature,
            )
        except BadRequestError as exc:
            if "json_validate_failed" in str(exc):
                raise LLMOutputError(f"LLM produced invalid JSON: {exc}") from exc
            raise
        raw = completion.choices[0].message.content or ""
        gen.update(output=raw, usage_details=usage_details(completion))

        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMOutputError(
                f"LLM output did not match {schema.__name__}:\n{exc}\n\nRaw output:\n{raw[:1000]}"
            ) from exc


def _messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def usage_details(completion: Any) -> dict[str, int] | None:
    """Token counts from a Groq/OpenAI-style completion, in Langfuse's format."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    return {
        "input": usage.prompt_tokens or 0,
        "output": usage.completion_tokens or 0,
        "total": usage.total_tokens or 0,
    }
