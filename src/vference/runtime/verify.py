from __future__ import annotations

from pathlib import Path
import os
import json

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
    expected = _gather_qmm(
        nn.silu(gate) * up, arrays, "down_proj", local_indices
    ).squeeze(-2)

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

        reference_store = SynchronousExpertStore(
            artifact, capacity=cache_capacity
        )
        for layer_id, layer in enumerate(model.language_model.layers):
            layer.mlp.switch_mlp = StreamingSwitchGLU(layer_id, reference_store)
        reference_logits, reference_output, reference_state_bytes = run_split()

        initial_delta = np.abs(stable_logits - reference_logits)
        initial_tokens_equal = int(stable_logits.argmax()) == int(
            reference_logits.argmax()
        )

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


def verify_runtime_corpus(
    artifact: Path,
    corpus_path: Path,
    *,
    cache_capacity: int = 320,
    prefetch_policy: str = "none",
    prefetch_budget: int = 1,
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
    )
    reference_store = None
    try:
        def run_case(case: dict[str, object]) -> tuple[np.ndarray, list[int], str]:
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
            output: list[int] = []
            eos_ids = set(tokenizer.eos_token_ids)
            for _ in range(int(case["max_tokens"])):
                token = int(mx.argmax(logits[0, -1]).item())
                output.append(token)
                if token in eos_ids:
                    break
                logits = model(mx.array([[token]]), cache=cache)
                mx.eval(logits)
            return initial, output, tokenizer.decode(output)

        stable_results = [run_case(case) for case in cases]
        stable_stats = stable_store.stats()

        reference_store = SynchronousExpertStore(
            artifact, capacity=cache_capacity
        )
        for layer_id, layer in enumerate(model.language_model.layers):
            layer.mlp.switch_mlp = StreamingSwitchGLU(layer_id, reference_store)
        reference_results = [run_case(case) for case in cases]

        results = []
        for case, stable, reference in zip(cases, stable_results, reference_results):
            delta = np.abs(stable[0] - reference[0])
            results.append(
                {
                    "id": case["id"],
                    "initial_logits_max_abs_error": float(delta.max()),
                    "initial_logits_mean_abs_error": float(delta.mean()),
                    "tokens_exact": stable[1] == reference[1],
                    "stable_output_tokens": stable[1],
                    "reference_output_tokens": reference[1],
                    "output_text": stable[2],
                }
            )
        return {
            "artifact": str(artifact.resolve()),
            "corpus": str(corpus_path.resolve()),
            "case_count": len(results),
            "prefetch_policy": prefetch_policy,
            "prefetch_budget": prefetch_budget,
            "all_tokens_exact": all(result["tokens_exact"] for result in results),
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
