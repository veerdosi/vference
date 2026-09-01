from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vference.artifacts.safetensors import scan_model

from .expert_store import (
    StableSlotExpertStore,
    StreamingSwitchGLU,
    SynchronousExpertStore,
)
from .model import load_streaming_qwen
from .sampling import make_token_sampler, select_token


def _source_experts(
    source: Path, layer_id: int, expert_ids: tuple[int, ...]
) -> dict[str, mx.array]:
    prefix = f"language_model.model.layers.{layer_id}.mlp.switch_mlp."
    entries = scan_model(source)
    selected = {name: entry for name, entry in entries.items() if name.startswith(prefix)}
    if len(selected) != 9:
        raise ValueError(f"expected 9 expert tensors for layer {layer_id}, got {len(selected)}")
    result: dict[str, mx.array] = {}
    for name, entry in selected.items():
        bytes_per_expert = entry.nbytes // entry.shape[0]
        slices = []
        fd = os.open(entry.shard, os.O_RDONLY)
        try:
            for expert_id in expert_ids:
                data = os.pread(
                    fd,
                    bytes_per_expert,
                    entry.file_offset + expert_id * bytes_per_expert,
                )
                if len(data) != bytes_per_expert:
                    raise OSError(f"short source read for {name}[{expert_id}]")
                slices.append(data)
        finally:
            os.close(fd)
        raw = b"".join(slices)
        if entry.dtype == "U32":
            value = mx.array(np.frombuffer(raw, dtype="<u4").copy())
        elif entry.dtype == "BF16":
            value = mx.array(np.frombuffer(raw, dtype="<u2").copy()).view(mx.bfloat16)
        else:
            raise ValueError(f"unsupported source dtype: {entry.dtype}")
        result[name.removeprefix(prefix)] = value.reshape(len(expert_ids), *entry.shape[1:])
    return result


def _gather_qmm(x: mx.array, arrays: dict[str, mx.array], prefix: str, indices: mx.array):
    return mx.gather_qmm(
        x,
        arrays[f"{prefix}.weight"],
        arrays[f"{prefix}.scales"],
        arrays[f"{prefix}.biases"],
        rhs_indices=indices,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
    )


def verify_real_layer_math(
    source: Path,
    artifact: Path,
    *,
    layer_id: int,
    expert_ids: tuple[int, ...],
    seed: int = 20260901,
    store_kind: str = "python",
) -> dict[str, object]:
    """Compare the upstream stacked-expert operation with streamed real records."""
    arrays = _source_experts(source.resolve(), layer_id, expert_ids)
    input_dims = arrays["gate_proj.scales"].shape[-1] * 64
    mx.random.seed(seed)
    x = mx.random.normal((1, 1, input_dims)).astype(mx.bfloat16)
    indices = mx.array([[list(expert_ids)]], dtype=mx.int32)
    local_indices = mx.array([[list(range(len(expert_ids)))]], dtype=mx.int32)
    expanded = mx.expand_dims(x, (-2, -3))
    up = _gather_qmm(expanded, arrays, "up_proj", local_indices)
    gate = _gather_qmm(expanded, arrays, "gate_proj", local_indices)
    expected = _gather_qmm(nn.silu(gate) * up, arrays, "down_proj", local_indices).squeeze(-2)

    store_types = {
        "python": SynchronousExpertStore,
        "stable": StableSlotExpertStore,
    }
    try:
        store_type = store_types[store_kind]
    except KeyError as error:
        raise ValueError(f"unknown expert store: {store_kind}") from error
    with store_type(artifact, capacity=len(expert_ids)) as store:
        actual = store.execute(layer_id, x, indices)
        mx.eval(expected, actual)
        expected_bits = np.asarray(expected.view(mx.uint16))
        actual_bits = np.asarray(actual.view(mx.uint16))
        exact = np.array_equal(expected_bits, actual_bits)
        delta = np.abs(
            np.asarray(expected.astype(mx.float32)) - np.asarray(actual.astype(mx.float32))
        )
        stats = store.stats()
    return {
        "source": str(source.resolve()),
        "artifact": str(artifact.resolve()),
        "layer_id": layer_id,
        "expert_ids": list(expert_ids),
        "seed": seed,
        "store_kind": store_kind,
        "bit_exact": exact,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "elements_compared": int(delta.size),
        "store": stats,
    }


def verify_multi_turn_state(
    artifact: Path,
    *,
    first: str,
    second: str,
    continuation_tokens: int = 8,
    cache_capacity: int = 320,
) -> dict[str, object]:
    """Compare stable and Python stores under identical split-prefix updates."""
    if continuation_tokens < 1:
        raise ValueError("continuation token count must be positive")
    model, tokenizer, stable_store = load_streaming_qwen(
        artifact,
        cache_capacity=cache_capacity,
        store_kind="stable",
        cache_policy="global",
    )
    reference_store = None
    try:
        first_tokens = tokenizer.encode(first, add_special_tokens=False)
        second_tokens = tokenizer.encode(second, add_special_tokens=False)
        if not first_tokens or not second_tokens:
            raise ValueError("both multi-turn segments must encode to tokens")

        def run_split() -> tuple[np.ndarray, list[int], int]:
            cache = model.make_cache()
            logits = model(mx.array([first_tokens]), cache=cache)
            mx.eval(logits)
            logits = model(mx.array([second_tokens]), cache=cache)[:, -1:]
            mx.eval(logits)
            initial = np.asarray(logits.astype(mx.float32)).copy()
            output: list[int] = []
            for _ in range(continuation_tokens):
                token = int(mx.argmax(logits[0, -1]).item())
                output.append(token)
                logits = model(mx.array([[token]]), cache=cache)
                mx.eval(logits)
            return initial, output, _cache_bytes(cache)

        stable_logits, stable_output, stable_state_bytes = run_split()
        stable_stats = stable_store.stats()

        reference_store = SynchronousExpertStore(artifact, capacity=cache_capacity)
        for layer_id, layer in enumerate(model.language_model.layers):
            layer.mlp.switch_mlp = StreamingSwitchGLU(layer_id, reference_store)
        reference_logits, reference_output, reference_state_bytes = run_split()

        initial_delta = np.abs(stable_logits - reference_logits)
        initial_tokens_equal = int(stable_logits.argmax()) == int(reference_logits.argmax())

        return {
            "artifact": str(artifact.resolve()),
            "first_tokens": len(first_tokens),
            "second_tokens": len(second_tokens),
            "continuation_tokens": continuation_tokens,
            "initial_argmax_equal": initial_tokens_equal,
            "initial_logits_max_abs_error": float(initial_delta.max()),
            "initial_logits_mean_abs_error": float(initial_delta.mean()),
            "continuation_exact": stable_output == reference_output,
            "stable_output_tokens": stable_output,
            "reference_output_tokens": reference_output,
            "output_text": tokenizer.decode(stable_output),
            "stable_state_bytes": stable_state_bytes,
            "reference_state_bytes": reference_state_bytes,
            "stable_store": stable_stats,
            "reference_store": reference_store.stats(),
        }
    finally:
        stable_store.close()
        if reference_store is not None:
            reference_store.close()


def _cache_bytes(cache: list[object]) -> int:
    return sum(int(getattr(item, "nbytes", 0)) for item in cache)


def verify_prefill_chunk_invariance(
    artifact: Path,
    *,
    prompt: str,
    prompt_token_count: int,
    chunk_sizes: tuple[int, ...],
    continuation_tokens: int = 16,
    cache_capacity: int = 320,
    nocache: bool = False,
) -> dict[str, object]:
    """Compare logits and greedy tokens across prefill chunk boundaries."""
    if prompt_token_count < 1:
        raise ValueError("prompt token count must be positive")
    if continuation_tokens < 1:
        raise ValueError("continuation token count must be positive")
    if len(chunk_sizes) < 2 or any(size < 1 for size in chunk_sizes):
        raise ValueError("provide at least two positive prefill chunk sizes")

    model, tokenizer, store = load_streaming_qwen(
        artifact,
        cache_capacity=cache_capacity,
        nocache=nocache,
        trace_routes=True,
        store_kind="stable",
        cache_policy="global",
        prefetch_policy="none",
    )
    try:
        base_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if not base_tokens:
            raise ValueError("prompt encoded to zero tokens")
        repetitions = (prompt_token_count + len(base_tokens) - 1) // len(base_tokens)
        prompt_tokens = (base_tokens * repetitions)[:prompt_token_count]

        logits_by_chunk: list[np.ndarray] = []
        output_by_chunk: list[list[int]] = []
        routes_by_chunk: list[dict[int, list[list[int]]]] = []
        runs: list[dict[str, object]] = []
        for chunk_size in chunk_sizes:
            trace_start = len(store.trace())
            cache = model.make_cache()
            logits = None
            started = time.perf_counter()
            for start in range(0, len(prompt_tokens), chunk_size):
                chunk = prompt_tokens[start : start + chunk_size]
                logits = model(mx.array([chunk]), cache=cache)
                mx.eval(logits)
            prefill_seconds = time.perf_counter() - started
            assert logits is not None
            initial_logits = np.asarray(logits[0, -1].astype(mx.float32)).copy()
            output_tokens: list[int] = []
            for step in range(continuation_tokens):
                token_id = int(mx.argmax(logits[0, -1]).item())
                output_tokens.append(token_id)
                if step + 1 < continuation_tokens:
                    logits = model(mx.array([[token_id]]), cache=cache)
                    mx.eval(logits)
            token_encoding = json.dumps(output_tokens, separators=(",", ":")).encode()
            normalized_routes: dict[int, list[list[int]]] = {}
            for record in store.trace()[trace_start:]:
                layer_id = int(record["layer_id"])
                normalized_routes.setdefault(layer_id, []).extend(record["expert_ids"])
            route_encoding = json.dumps(
                normalized_routes, sort_keys=True, separators=(",", ":")
            ).encode()
            normalized_route_sets = {
                layer_id: [sorted(experts) for experts in rows]
                for layer_id, rows in normalized_routes.items()
            }
            route_set_encoding = json.dumps(
                normalized_route_sets, sort_keys=True, separators=(",", ":")
            ).encode()
            logits_by_chunk.append(initial_logits)
            output_by_chunk.append(output_tokens)
            routes_by_chunk.append(normalized_routes)
            runs.append(
                {
                    "chunk_size": chunk_size,
                    "prefill_seconds": prefill_seconds,
                    "initial_argmax": int(initial_logits.argmax()),
                    "output_tokens": output_tokens,
                    "output_token_sha256": hashlib.sha256(token_encoding).hexdigest(),
                    "route_sha256": hashlib.sha256(route_encoding).hexdigest(),
                    "route_set_sha256": hashlib.sha256(route_set_encoding).hexdigest(),
                }
            )
            del cache, logits
            mx.clear_cache()

        reference_logits = logits_by_chunk[0]
        reference_output = output_by_chunk[0]
        reference_routes = routes_by_chunk[0]
        comparisons = []
        for chunk_size, logits, output_tokens, routes in zip(
            chunk_sizes[1:],
            logits_by_chunk[1:],
            output_by_chunk[1:],
            routes_by_chunk[1:],
        ):
            delta = np.abs(reference_logits - logits)
            first_difference = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(reference_output, output_tokens))
                    if left != right
                ),
                None,
            )
            first_route_difference = None
            first_route_set_difference = None
            for layer_id in sorted(set(reference_routes) | set(routes)):
                reference_layer = reference_routes.get(layer_id, [])
                candidate_layer = routes.get(layer_id, [])
                for token_index, (left, right) in enumerate(zip(reference_layer, candidate_layer)):
                    if left != right:
                        if first_route_difference is None:
                            first_route_difference = {
                                "layer_id": layer_id,
                                "token_index": token_index,
                                "reference_experts": left,
                                "candidate_experts": right,
                            }
                        if sorted(left) != sorted(right):
                            first_route_set_difference = {
                                "layer_id": layer_id,
                                "token_index": token_index,
                                "reference_experts": left,
                                "candidate_experts": right,
                            }
                            break
                if first_route_set_difference is not None:
                    break
                if len(reference_layer) != len(candidate_layer):
                    length_difference = {
                        "layer_id": layer_id,
                        "reference_token_count": len(reference_layer),
                        "candidate_token_count": len(candidate_layer),
                    }
                    if first_route_difference is None:
                        first_route_difference = length_difference
                    first_route_set_difference = length_difference
                    break
            comparisons.append(
                {
                    "reference_chunk_size": chunk_sizes[0],
                    "candidate_chunk_size": chunk_size,
                    "initial_logits_exact": bool(np.array_equal(reference_logits, logits)),
                    "initial_logits_max_abs_error": float(delta.max()),
                    "initial_logits_mean_abs_error": float(delta.mean()),
                    "initial_argmax_equal": int(reference_logits.argmax()) == int(logits.argmax()),
                    "output_tokens_exact": reference_output == output_tokens,
                    "first_output_difference": first_difference,
                    "routes_exact": reference_routes == routes,
                    "first_route_difference": first_route_difference,
                    "route_expert_sets_exact": first_route_set_difference is None,
                    "first_route_set_difference": first_route_set_difference,
                }
            )

        return {
            "artifact": str(artifact.resolve()),
            "prompt_tokens": len(prompt_tokens),
            "continuation_tokens": continuation_tokens,
            "reference_chunk_size": chunk_sizes[0],
            "all_initial_logits_exact": all(item["initial_logits_exact"] for item in comparisons),
            "all_initial_argmax_equal": all(item["initial_argmax_equal"] for item in comparisons),
            "all_output_tokens_exact": all(item["output_tokens_exact"] for item in comparisons),
            "all_routes_exact": all(item["routes_exact"] for item in comparisons),
            "all_route_expert_sets_exact": all(
                item["route_expert_sets_exact"] for item in comparisons
            ),
            "runs": runs,
            "comparisons": comparisons,
            "store": store.stats(),
        }
    finally:
        store.close()


def verify_runtime_corpus(
    artifact: Path,
    corpus_path: Path,
    *,
    cache_capacity: int = 320,
    prefetch_policy: str = "none",
    prefetch_budget: int = 1,
    prefetch_min_observations: int = 8,
) -> dict[str, object]:
    """Compare deterministic chat cases between stable and Python stores."""
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("format") != "vference.runtime-corpus.v1":
        raise ValueError("unsupported runtime corpus format")
    cases = corpus.get("cases", [])
    if not cases:
        raise ValueError("runtime corpus has no cases")

    model, tokenizer, stable_store = load_streaming_qwen(
        artifact,
        cache_capacity=cache_capacity,
        store_kind="stable",
        cache_policy="global",
        prefetch_policy=prefetch_policy,
        prefetch_budget=prefetch_budget,
        prefetch_min_observations=prefetch_min_observations,
    )
    reference_store = None
    try:

        def run_case(case: dict[str, object]) -> dict[str, object]:
            prompt_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": str(case["prompt"])}],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )
            cache = model.make_cache()
            logits = model(mx.array([prompt_tokens]), cache=cache)[:, -1:]
            mx.eval(logits)
            initial = np.asarray(logits.astype(mx.float32)).copy()
            raw_sampler = case.get("sampler", {})
            if not isinstance(raw_sampler, dict):
                raise ValueError(f"case {case['id']} sampler must be an object")
            temperature = float(raw_sampler.get("temperature", 0.0))
            top_p = float(raw_sampler.get("top_p", 1.0))
            top_k = int(raw_sampler.get("top_k", 0))
            seed = int(raw_sampler.get("seed", 0))
            sampler = make_token_sampler(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            if sampler is not None:
                mx.random.seed(seed)
            output: list[int] = []
            step_logits_sha256: list[str] = []
            eos_ids = set(tokenizer.eos_token_ids)
            for _ in range(int(case["max_tokens"])):
                logit_bytes = np.asarray(logits[:, -1:].view(mx.uint16)).tobytes()
                step_logits_sha256.append(hashlib.sha256(logit_bytes).hexdigest())
                token = select_token(logits, sampler)
                output.append(token)
                if token in eos_ids:
                    break
                logits = model(mx.array([[token]]), cache=cache)
                mx.eval(logits)
            return {
                "initial_logits": initial,
                "output_tokens": output,
                "output_text": tokenizer.decode(output),
                "step_logits_sha256": step_logits_sha256,
                "sampler": {
                    "kind": "greedy" if sampler is None else "categorical",
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "seed": seed,
                },
            }

        stable_results = [run_case(case) for case in cases]
        stable_stats = stable_store.stats()

        reference_store = SynchronousExpertStore(artifact, capacity=cache_capacity)
        for layer_id, layer in enumerate(model.language_model.layers):
            layer.mlp.switch_mlp = StreamingSwitchGLU(layer_id, reference_store)
        reference_results = [run_case(case) for case in cases]

        results = []
        for case, stable, reference in zip(cases, stable_results, reference_results):
            delta = np.abs(stable["initial_logits"] - reference["initial_logits"])
            stable_step_logits = stable["step_logits_sha256"]
            reference_step_logits = reference["step_logits_sha256"]
            first_step_logits_difference = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(stable_step_logits, reference_step_logits)
                    )
                    if left != right
                ),
                None,
            )
            if first_step_logits_difference is None and len(stable_step_logits) != len(
                reference_step_logits
            ):
                first_step_logits_difference = min(
                    len(stable_step_logits), len(reference_step_logits)
                )
            results.append(
                {
                    "id": case["id"],
                    "initial_logits_max_abs_error": float(delta.max()),
                    "initial_logits_mean_abs_error": float(delta.mean()),
                    "step_logits_exact": first_step_logits_difference is None,
                    "first_step_logits_difference": first_step_logits_difference,
                    "tokens_exact": stable["output_tokens"] == reference["output_tokens"],
                    "stable_output_tokens": stable["output_tokens"],
                    "reference_output_tokens": reference["output_tokens"],
                    "output_text": stable["output_text"],
                    "sampler": stable["sampler"],
                }
            )
        return {
            "artifact": str(artifact.resolve()),
            "corpus": str(corpus_path.resolve()),
            "case_count": len(results),
            "prefetch_policy": prefetch_policy,
            "prefetch_budget": prefetch_budget,
            "prefetch_min_observations": prefetch_min_observations,
            "all_tokens_exact": all(result["tokens_exact"] for result in results),
            "all_step_logits_exact": all(result["step_logits_exact"] for result in results),
            "max_initial_logits_abs_error": max(
                result["initial_logits_max_abs_error"] for result in results
            ),
            "cases": results,
            "stable_store": stable_stats,
            "reference_store": reference_store.stats(),
        }
    finally:
        stable_store.close()
        if reference_store is not None:
            reference_store.close()
