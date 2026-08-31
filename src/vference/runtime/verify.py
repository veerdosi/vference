from __future__ import annotations

from pathlib import Path
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vference.artifacts.safetensors import scan_model

from .expert_store import SynchronousExpertStore


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

    with SynchronousExpertStore(artifact, capacity=len(expert_ids)) as store:
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
        "bit_exact": exact,
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "elements_compared": int(delta.size),
        "store": stats,
    }
