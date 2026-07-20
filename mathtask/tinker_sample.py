"""Async Tinker sampling helpers for generation / validation prompts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from mathtask.ask_arithmetic import (
    FINAL_CHANNEL_PREFILL,
    MAX_TOKENS,
    NUM_IN_FLIGHT,
    RENDERER_BY_EFFORT,
    load_hf_token,
    load_tinker_api_key,
)

from core.defaults import DEFAULT_MODEL

MAX_SAMPLE_RETRIES = 3


@dataclass(frozen=True)
class SampleRequest:
    """One chat-style user prompt to sample."""

    key: str
    user_content: str


@dataclass(frozen=True)
class SampleResult:
    key: str
    text: str


async def sample_many(
    requests: Sequence[SampleRequest],
    *,
    reasoning_effort: str = "high",
    model: str = DEFAULT_MODEL,
    model_path: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS,
    num_in_flight: int = NUM_IN_FLIGHT,
    max_retries: int = MAX_SAMPLE_RETRIES,
) -> list[SampleResult]:
    """Sample free-form responses for many prompts (high reasoning by default).

    Each request is retried up to ``max_retries`` times with backoff. Requests
    that still fail yield empty text (callers treat them as unusable) instead
    of aborting the whole batch; a systemic failure where **every** request
    fails raises RuntimeError so the loop stops loudly rather than training
    on silently empty data.
    """
    if not requests:
        return []

    load_tinker_api_key()
    load_hf_token()

    import tinker
    from tinker_cookbook.renderers import get_renderer, get_text_content

    if reasoning_effort not in RENDERER_BY_EFFORT:
        raise ValueError(
            f"Unknown reasoning effort {reasoning_effort!r}; "
            f"expected one of {sorted(RENDERER_BY_EFFORT)}"
        )
    renderer_name = RENDERER_BY_EFFORT[reasoning_effort]
    generation_prefill = (
        FINAL_CHANNEL_PREFILL if reasoning_effort == "instant" else None
    )

    service_client = tinker.ServiceClient()
    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model)

    renderer = get_renderer(
        renderer_name,
        sampling_client.get_tokenizer(),
        model_name=model,
    )
    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )

    in_flight = asyncio.Semaphore(num_in_flight)
    renderer_lock = asyncio.Lock()
    results: dict[str, str] = {}
    failed: dict[str, Exception] = {}
    attempts = max(1, max_retries)

    async def one(req: SampleRequest) -> None:
        async with in_flight:
            for attempt in range(attempts):
                try:
                    async with renderer_lock:
                        model_input = renderer.build_generation_prompt(
                            [{"role": "user", "content": req.user_content}],
                            prefill=generation_prefill,
                        )
                    sample = await sampling_client.sample_async(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    async with renderer_lock:
                        message, _ = renderer.parse_response(
                            sample.sequences[0].tokens
                        )
                        results[req.key] = get_text_content(message)
                    return
                except Exception as exc:  # transient API / network / parse errors
                    if attempt + 1 >= attempts:
                        failed[req.key] = exc
                        print(
                            f"[tinker_sample] {req.key}: failed after "
                            f"{attempts} attempts: {exc}",
                            flush=True,
                        )
                    else:
                        await asyncio.sleep(min(2.0**attempt, 8.0))

    await asyncio.gather(*(one(req) for req in requests))
    if len(failed) == len(requests):
        raise RuntimeError(
            f"All {len(requests)} sampling requests failed; "
            f"last error: {next(iter(failed.values()))}"
        )
    if failed:
        print(
            f"[tinker_sample] WARNING: {len(failed)}/{len(requests)} requests "
            "failed; continuing with partial results",
            flush=True,
        )
    return [SampleResult(key=req.key, text=results.get(req.key, "")) for req in requests]


def sample_many_sync(
    requests: Sequence[SampleRequest],
    **kwargs,
) -> list[SampleResult]:
    """Sync wrapper around :func:`sample_many`."""
    return asyncio.run(sample_many(requests, **kwargs))
