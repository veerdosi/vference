from __future__ import annotations

import gc
import statistics
import time
import json
import resource
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil

from .admission import estimate_qwen35_admission
from .model import load_streaming_qwen
from .sampling import make_token_sampler, select_token

QUALIFIED_QWEN35_PREFILL_CHUNK_SIZE = 512
DECODE_TRANSIENT_RESERVE_BYTES = 448 * 1024**2


def _detach_last_logits(logits: mx.array) -> mx.array:
    value = logits[:, -1:]
    if value.dtype == mx.bfloat16:
        host = np.asarray(value.view(mx.uint16)).copy()
        detached = mx.array(host).view(mx.bfloat16)
    else:
        detached = mx.array(np.asarray(value).copy())
    mx.eval(detached)
    return detached


def _validate_prefill_chunk_size(
    prompt_tokens: int, chunk_size: int, allow_unqualified: bool
) -> None:
    if (
        prompt_tokens > chunk_size
        and chunk_size != QUALIFIED_QWEN35_PREFILL_CHUNK_SIZE
        and not allow_unqualified
    ):
        raise ValueError(
            f"multi-chunk Qwen3.5 prefill size {chunk_size} is not output-qualified; "
            f"use {QUALIFIED_QWEN35_PREFILL_CHUNK_SIZE} or pass the explicit "
            "experimental override"
        )


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


def _memory_snapshot() -> dict[str, int | float]:
    value = psutil.virtual_memory()
    return {
        "available": value.available,
        "used": value.used,
        "free": value.free,
        "active": value.active,
        "inactive": value.inactive,
        "wired": getattr(value, "wired", 0),
        "percent": value.percent,
    }


def _store_stats_delta(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    def counters(
        left: dict[str, object], right: dict[str, object], fields: tuple[str, ...]
    ) -> dict[str, int | float]:
        return {
            field: right.get(field, 0) - left.get(field, 0)  # type: ignore[operator]
            for field in fields
        }

    result: dict[str, object] = counters(before, after, ("hits", "misses", "bytes_read"))
    before_timing = before.get("timing_seconds", {})
    after_timing = after.get("timing_seconds", {})
    assert isinstance(before_timing, dict) and isinstance(after_timing, dict)
    result["timing_seconds"] = counters(
        before_timing,
        after_timing,
        ("pread", "materialize", "router_and_graph_wait", "execute_total"),
    )

    before_prefetch = before.get("prefetch")
    after_prefetch = after.get("prefetch")
    if isinstance(before_prefetch, dict) and isinstance(after_prefetch, dict):
        result["prefetch"] = counters(
            before_prefetch,
            after_prefetch,
            (
                "submitted",
                "bytes_read",
                "useful",
                "late",
                "wrong",
                "unused_completed",
                "cancelled",
                "failed",
                "skipped_busy",
                "already_resident",
                "wait_seconds",
                "copy_seconds",
                "total_physical_bytes",
            ),
        )

    before_layers = before.get("per_layer", [])
    after_layers = after.get("per_layer", [])
    assert isinstance(before_layers, list) and isinstance(after_layers, list)
    result["per_layer"] = [
        counters(
            left,
            right,
            ("hits", "misses", "read_seconds", "materialize_seconds"),
        )
        for left, right in zip(before_layers, after_layers)
        if isinstance(left, dict) and isinstance(right, dict)
    ]
    return result


def _decode_resize_admission(
    *,
    resident_bytes: int,
    model_state_bytes: int,
    record_size: int,
    prefill_capacity: int,
    decode_capacity: int,
    runtime_reserve_bytes: int,
    budget_bytes: int,
) -> dict[str, int | bool]:
    extra_pool_bytes = (decode_capacity - prefill_capacity) * record_size
    estimated_peak_bytes = (
        resident_bytes
        + model_state_bytes
        + extra_pool_bytes
        + runtime_reserve_bytes
        + DECODE_TRANSIENT_RESERVE_BYTES
    )
    return {
        "admitted": estimated_peak_bytes <= budget_bytes,
        "budget_bytes": budget_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "prefill_capacity": prefill_capacity,
        "decode_capacity": decode_capacity,
        "extra_pool_bytes": extra_pool_bytes,
        "decode_transient_reserve_bytes": DECODE_TRANSIENT_RESERVE_BYTES,
        "runtime_reserve_bytes": runtime_reserve_bytes,
    }


def generate_greedy(
    artifact: Path,
    prompt: str,
    *,
    max_tokens: int,
    cache_capacity: int,
    chat_template: bool,
    enable_thinking: bool,
    decode_cache_capacity: int | None = None,
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
    prefetch_min_observations: int = 8,
    allow_unqualified_prefill_chunk_size: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
) -> dict[str, object]:
    request_started = time.perf_counter()
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if prefill_chunk_size < 1:
        raise ValueError("prefill chunk size must be positive")
    if decode_cache_capacity is not None and decode_cache_capacity < 1:
        raise ValueError("decode expert cache capacity must be positive")
    sampler = make_token_sampler(temperature=temperature, top_p=top_p, top_k=top_k)
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
    effective_decode_cache_capacity = decode_cache_capacity or cache_capacity
    if effective_decode_cache_capacity != cache_capacity and store_kind != "stable":
        raise ValueError("phase-specific cache capacity requires the stable store")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    memory_before = _memory_snapshot()
    swap_before = psutil.swap_memory()
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
        prefetch_min_observations=prefetch_min_observations,
    )
    load_seconds = time.perf_counter() - load_started
    model_ready_at = time.perf_counter()
    memory_after_load = _memory_snapshot()
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
                filler_tokens * ((filler_count + len(filler_tokens) - 1) // len(filler_tokens))
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
            repetitions = (repeat_raw_prompt_to_tokens + len(base_tokens) - 1) // len(base_tokens)
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
        _validate_prefill_chunk_size(
            len(prompt_tokens),
            prefill_chunk_size,
            allow_unqualified_prefill_chunk_size,
        )

        config = json.loads((artifact / "config.json").read_text())
        if max_mlx_memory_bytes is None:
            max_mlx_memory_bytes = int(psutil.virtual_memory().total * 0.375)
        runtime_reserve_bytes = (
            (16 + 2 * max(0, prefetch_budget - 1)) * 1024**2 if prefetch_policy != "none" else 0
        )
        admission = estimate_qwen35_admission(
            config,
            resident_bytes=mx.get_active_memory(),
            total_tokens=len(prompt_tokens) + max_tokens,
            prefill_chunk_size=prefill_chunk_size,
            budget_bytes=max_mlx_memory_bytes,
            runtime_reserve_bytes=runtime_reserve_bytes,
        )
        if not admission.admitted:
            raise MemoryError(
                "context rejected before prefill: estimated MLX peak "
                f"{admission.estimated_peak_bytes / 1024**3:.2f} GiB exceeds "
                f"the {admission.budget_bytes / 1024**3:.2f} GiB budget"
            )
        decode_resize_admission = _decode_resize_admission(
            resident_bytes=admission.resident_bytes,
            model_state_bytes=admission.model_state_bytes,
            record_size=int(store.record_size),
            prefill_capacity=cache_capacity,
            decode_capacity=effective_decode_cache_capacity,
            runtime_reserve_bytes=runtime_reserve_bytes,
            budget_bytes=max_mlx_memory_bytes,
        )
        if (
            effective_decode_cache_capacity != cache_capacity
            and not decode_resize_admission["admitted"]
        ):
            raise MemoryError(
                "decode cache resize rejected before prefill: estimated MLX peak "
                f"{decode_resize_admission['estimated_peak_bytes'] / 1024**3:.2f} GiB "
                f"exceeds the {max_mlx_memory_bytes / 1024**3:.2f} GiB budget"
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
        memory_after_prefill = _memory_snapshot()
        prefill_store_stats = store.stats()
        prefill_mlx_peak_bytes = mx.get_peak_memory()

        transition_started = time.perf_counter()
        if effective_decode_cache_capacity != cache_capacity:
            logits = _detach_last_logits(logits)
            gc.collect()
            mx.clear_cache()
            store.resize_capacity(effective_decode_cache_capacity, effective_decode_cache_policy)
        elif effective_decode_cache_policy != cache_policy:
            if store_kind != "stable":
                raise ValueError("phase-specific cache policy requires the stable store")
            store.set_cache_policy(effective_decode_cache_policy)
        phase_transition_seconds = time.perf_counter() - transition_started
        memory_after_phase_transition = _memory_snapshot()
        phase_transition_mlx_active_bytes = mx.get_active_memory()
        phase_transition_mlx_cache_bytes = mx.get_cache_memory()
        mx.reset_peak_memory()
        if sampler is not None:
            mx.random.seed(seed)

        output_tokens: list[int] = []
        decode_latencies: list[float] = []
        first_token_at: float | None = None
        eos_ids = set(tokenizer.eos_token_ids)
        for step in range(max_tokens):
            assert logits is not None
            token_id = select_token(logits, sampler)
            output_tokens.append(token_id)
            if first_token_at is None:
                first_token_at = time.perf_counter()
            if token_id in eos_ids or step + 1 == max_tokens:
                break
            started = time.perf_counter()
            logits = model(mx.array([[token_id]]), cache=cache)
            mx.eval(logits)
            decode_latencies.append(time.perf_counter() - started)

        swap_after = psutil.swap_memory()
        memory_after_generation = _memory_snapshot()
        final_store_stats = store.stats()
        decode_mlx_peak_bytes = mx.get_peak_memory()
        result = {
            "prompt": prompt,
            "chat_template": chat_template,
            "prompt_mode": prompt_mode,
            "needle": needle,
            "admission": admission.as_dict(),
            "decode_resize_admission": decode_resize_admission,
            "enable_thinking": enable_thinking,
            "store_kind": store_kind,
            "cache_policy": cache_policy,
            "decode_cache_policy": effective_decode_cache_policy,
            "prefill_cache_capacity": cache_capacity,
            "decode_cache_capacity": effective_decode_cache_capacity,
            "prefetch_policy": prefetch_policy,
            "prefetch_budget": prefetch_budget,
            "prefetch_min_observations": prefetch_min_observations,
            "sampler": {
                "kind": "greedy" if sampler is None else "categorical",
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
            },
            "prompt_tokens": len(prompt_tokens),
            "prefill_chunk_size": prefill_chunk_size,
            "clear_cache_between_prefill_chunks": clear_cache_between_prefill_chunks,
            "prefill_chunk_memory": prefill_chunks,
            "output_tokens": output_tokens,
            "output_text": tokenizer.decode(output_tokens),
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "phase_transition_seconds": phase_transition_seconds,
            "phase_transition_mlx_active_bytes": phase_transition_mlx_active_bytes,
            "phase_transition_mlx_cache_bytes": phase_transition_mlx_cache_bytes,
            "time_to_first_token_seconds": first_token_at - model_ready_at,
            "cold_start_time_to_first_token_seconds": first_token_at - request_started,
            "decode_model_calls": len(decode_latencies),
            "decode_tokens_per_second": (
                len(decode_latencies) / sum(decode_latencies) if decode_latencies else None
            ),
            "decode_latency_seconds": (
                _latency_summary(decode_latencies) if decode_latencies else None
            ),
            "expert_store": final_store_stats,
            "expert_store_phases": {
                "prefill": prefill_store_stats,
                "decode_delta": _store_stats_delta(prefill_store_stats, final_store_stats),
            },
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_bytes": max(prefill_mlx_peak_bytes, decode_mlx_peak_bytes),
            "prefill_mlx_peak_bytes": prefill_mlx_peak_bytes,
            "decode_mlx_peak_bytes": decode_mlx_peak_bytes,
            "mlx_cache_bytes": mx.get_cache_memory(),
            "model_state_bytes": _state_bytes(cache),
            "process_rss_bytes": {
                "before_load": rss_before,
                "after_generation": process.memory_info().rss,
                "high_water": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "system_memory_bytes": {
                "before_load": memory_before,
                "after_load": memory_after_load,
                "after_prefill": memory_after_prefill,
                "after_phase_transition": memory_after_phase_transition,
                "after_generation": memory_after_generation,
            },
            "system_swap_bytes": {
                "before": swap_before.used,
                "after": swap_after.used,
                "delta": swap_after.used - swap_before.used,
                "swap_in_delta": swap_after.sin - swap_before.sin,
                "swap_out_delta": swap_after.sout - swap_before.sout,
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
