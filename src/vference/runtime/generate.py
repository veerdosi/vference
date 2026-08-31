from __future__ import annotations

import statistics
import time
import json
import resource
from pathlib import Path

import mlx.core as mx
import psutil

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
    nocache: bool = False,
    trace_output: Path | None = None,
    store_kind: str = "python",
    prefill_chunk_size: int = 1,
    repeat_raw_prompt_to_tokens: int | None = None,
    cache_policy: str = "global",
    decode_cache_policy: str | None = None,
) -> dict[str, object]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if prefill_chunk_size < 1:
        raise ValueError("prefill chunk size must be positive")
    if repeat_raw_prompt_to_tokens is not None and repeat_raw_prompt_to_tokens < 1:
        raise ValueError("repeated raw prompt token count must be positive")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    swap_before = psutil.swap_memory().used
    load_started = time.perf_counter()
    model, tokenizer, store = load_streaming_qwen(
        artifact,
        cache_capacity=cache_capacity,
        nocache=nocache,
        trace_routes=trace_output is not None,
        store_kind=store_kind,
        cache_policy=cache_policy,
    )
    load_seconds = time.perf_counter() - load_started
    try:
        prompt_mode = "chat" if chat_template else "raw"
        if repeat_raw_prompt_to_tokens is not None:
            base_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if not base_tokens:
                raise ValueError("prompt encoded to zero tokens")
            repetitions = (
                repeat_raw_prompt_to_tokens + len(base_tokens) - 1
            ) // len(base_tokens)
            prompt_tokens = (base_tokens * repetitions)[:repeat_raw_prompt_to_tokens]
            prompt_mode = "synthetic_repeated_raw_tokens"
        elif chat_template:
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
        for start in range(0, len(prompt_tokens), prefill_chunk_size):
            chunk = prompt_tokens[start : start + prefill_chunk_size]
            logits = model(mx.array([chunk]), cache=cache)
            mx.eval(logits)
        prefill_seconds = time.perf_counter() - prefill_started

        if decode_cache_policy is not None and decode_cache_policy != cache_policy:
            if store_kind != "stable":
                raise ValueError("phase-specific cache policy requires the stable store")
            store.set_cache_policy(decode_cache_policy)

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

        swap_after = psutil.swap_memory().used
        result = {
            "prompt": prompt,
            "chat_template": chat_template,
            "prompt_mode": prompt_mode,
            "enable_thinking": enable_thinking,
            "store_kind": store_kind,
            "cache_policy": cache_policy,
            "decode_cache_policy": decode_cache_policy or cache_policy,
            "prompt_tokens": len(prompt_tokens),
            "prefill_chunk_size": prefill_chunk_size,
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
            "process_rss_bytes": {
                "before_load": rss_before,
                "after_generation": process.memory_info().rss,
                "high_water": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "system_swap_bytes": {
                "before": swap_before,
                "after": swap_after,
                "delta": swap_after - swap_before,
            },
        }
        if trace_output is not None:
            trace_output.parent.mkdir(parents=True, exist_ok=True)
            trace_output.write_text(
                json.dumps(
                    {
                        "format": "vference.route-trace.v1",
                        "artifact": str(artifact.resolve()),
                        "prompt_tokens": len(prompt_tokens),
                        "prompt_mode": prompt_mode,
                        "output_tokens": output_tokens,
                        "records": store.trace(),
                    },
                    separators=(",", ":"),
                )
            )
            result["trace_output"] = str(trace_output.resolve())
        return result
    finally:
        store.close()
