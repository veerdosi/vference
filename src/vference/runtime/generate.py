from __future__ import annotations

import statistics
import time
import json
import resource
from pathlib import Path

import mlx.core as mx
import psutil

from .admission import estimate_qwen35_admission
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


def _state_bytes(cache: list[object]) -> int:
    return sum(int(getattr(item, "nbytes", 0)) for item in cache)


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
    clear_cache_between_prefill_chunks: bool = False,
    needle: str | None = None,
    needle_context_tokens: int | None = None,
    max_mlx_memory_bytes: int | None = None,
    prefetch_policy: str = "none",
    prefetch_budget: int = 1,
) -> dict[str, object]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if prefill_chunk_size < 1:
        raise ValueError("prefill chunk size must be positive")
    if repeat_raw_prompt_to_tokens is not None and repeat_raw_prompt_to_tokens < 1:
        raise ValueError("repeated raw prompt token count must be positive")
    if (needle is None) != (needle_context_tokens is None):
        raise ValueError("needle and needle context token count must be set together")
    if needle_context_tokens is not None and needle_context_tokens < 1:
        raise ValueError("needle context token count must be positive")
    if needle is not None and repeat_raw_prompt_to_tokens is not None:
        raise ValueError("needle context and repeated-token stress mode are exclusive")
    effective_decode_cache_policy = decode_cache_policy
    if effective_decode_cache_policy is None:
        effective_decode_cache_policy = "layer" if store_kind == "stable" else cache_policy
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
        prefetch_policy=prefetch_policy,
        prefetch_budget=prefetch_budget,
    )
    load_seconds = time.perf_counter() - load_started
    try:
        prompt_mode = "chat" if chat_template else "raw"
        if needle is not None and needle_context_tokens is not None:
            filler_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            needle_tokens = tokenizer.encode(
                f"\nIMPORTANT FACT: The verification code is {needle}.\n",
                add_special_tokens=False,
            )
            query_tokens = tokenizer.encode(
                "\nQuestion: What is the verification code? Respond with only the code.\nAnswer:",
                add_special_tokens=False,
            )
            filler_count = needle_context_tokens - len(needle_tokens) - len(query_tokens)
            if not filler_tokens or filler_count < 1:
                raise ValueError("needle context target is too small for the prompt")
            repeated = (
                filler_tokens
                * ((filler_count + len(filler_tokens) - 1) // len(filler_tokens))
            )[:filler_count]
            insertion = filler_count // 4
            prompt_tokens = [
                *repeated[:insertion],
                *needle_tokens,
                *repeated[insertion:],
                *query_tokens,
            ]
            prompt_mode = "synthetic_needle_retrieval"
        elif repeat_raw_prompt_to_tokens is not None:
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

        config = json.loads((artifact / "config.json").read_text())
        if max_mlx_memory_bytes is None:
            max_mlx_memory_bytes = int(psutil.virtual_memory().total * 0.375)
        admission = estimate_qwen35_admission(
            config,
            resident_bytes=mx.get_active_memory(),
            total_tokens=len(prompt_tokens) + max_tokens,
            prefill_chunk_size=prefill_chunk_size,
            budget_bytes=max_mlx_memory_bytes,
            runtime_reserve_bytes=(
                (16 + 2 * max(0, prefetch_budget - 1)) * 1024**2
                if prefetch_policy != "none"
                else 0
            ),
        )
        if not admission.admitted:
            raise MemoryError(
                "context rejected before prefill: estimated MLX peak "
                f"{admission.estimated_peak_bytes / 1024**3:.2f} GiB exceeds "
                f"the {admission.budget_bytes / 1024**3:.2f} GiB budget"
            )

        cache = model.make_cache()
        prefill_started = time.perf_counter()
        logits = None
        prefill_chunks: list[dict[str, int]] = []
        for start in range(0, len(prompt_tokens), prefill_chunk_size):
            chunk = prompt_tokens[start : start + prefill_chunk_size]
            logits = model(mx.array([chunk]), cache=cache)
            mx.eval(logits)
            chunk_memory = {
                "start_token": start,
                "token_count": len(chunk),
                "active_bytes": mx.get_active_memory(),
                "cache_bytes_before_clear": mx.get_cache_memory(),
            }
            if clear_cache_between_prefill_chunks:
                mx.clear_cache()
            chunk_memory["cache_bytes_after_clear"] = mx.get_cache_memory()
            prefill_chunks.append(chunk_memory)
        prefill_seconds = time.perf_counter() - prefill_started

        if effective_decode_cache_policy != cache_policy:
            if store_kind != "stable":
                raise ValueError("phase-specific cache policy requires the stable store")
            store.set_cache_policy(effective_decode_cache_policy)

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
            "needle": needle,
            "admission": admission.as_dict(),
            "enable_thinking": enable_thinking,
            "store_kind": store_kind,
            "cache_policy": cache_policy,
            "decode_cache_policy": effective_decode_cache_policy,
            "prefetch_policy": prefetch_policy,
            "prefetch_budget": prefetch_budget,
            "prompt_tokens": len(prompt_tokens),
            "prefill_chunk_size": prefill_chunk_size,
            "clear_cache_between_prefill_chunks": clear_cache_between_prefill_chunks,
            "prefill_chunk_memory": prefill_chunks,
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
            "mlx_cache_bytes": mx.get_cache_memory(),
            "model_state_bytes": _state_bytes(cache),
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
