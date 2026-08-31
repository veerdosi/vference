from __future__ import annotations

import statistics
import time
from pathlib import Path

import mlx.core as mx

from .model import load_streaming_qwen


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
        return ordered[index]

    return {
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(values),
    }


def generate_greedy(
    artifact: Path,
    prompt: str,
    *,
    max_tokens: int,
    cache_capacity: int,
    chat_template: bool,
    enable_thinking: bool,
) -> dict[str, object]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    load_started = time.perf_counter()
    model, tokenizer, store = load_streaming_qwen(
        artifact, cache_capacity=cache_capacity
    )
    load_seconds = time.perf_counter() - load_started
    try:
        if chat_template:
            prompt_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=enable_thinking,
            )
        else:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_tokens:
            raise ValueError("prompt encoded to zero tokens")

        cache = model.make_cache()
        prefill_started = time.perf_counter()
        logits = None
        for token_id in prompt_tokens:
            logits = model(mx.array([[token_id]]), cache=cache)
            mx.eval(logits)
        prefill_seconds = time.perf_counter() - prefill_started

        output_tokens: list[int] = []
        decode_latencies: list[float] = []
        eos_ids = set(tokenizer.eos_token_ids)
        for step in range(max_tokens):
            assert logits is not None
            token_id = int(mx.argmax(logits[0, -1]).item())
            output_tokens.append(token_id)
            if token_id in eos_ids or step + 1 == max_tokens:
                break
            started = time.perf_counter()
            logits = model(mx.array([[token_id]]), cache=cache)
            mx.eval(logits)
            decode_latencies.append(time.perf_counter() - started)

        return {
            "prompt": prompt,
            "chat_template": chat_template,
            "enable_thinking": enable_thinking,
            "prompt_tokens": len(prompt_tokens),
            "output_tokens": output_tokens,
            "output_text": tokenizer.decode(output_tokens),
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "decode_model_calls": len(decode_latencies),
            "decode_tokens_per_second": (
                len(decode_latencies) / sum(decode_latencies)
                if decode_latencies
                else None
            ),
            "decode_latency_seconds": (
                _latency_summary(decode_latencies) if decode_latencies else None
            ),
            "expert_store": store.stats(),
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
        }
    finally:
        store.close()
