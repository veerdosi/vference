from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


_NUMPY_DTYPES = {
    "U32": np.dtype("<u4"),
    "BF16": np.dtype("<u2"),
}


@dataclass(frozen=True)
class ComponentLayout:
    suffix: str
    dtype: str
    shape: tuple[int, ...]
    bytes_per_expert: int
    record_offset: int


@dataclass(frozen=True)
class ExpertWeights:
    gate_weight: mx.array
    gate_scales: mx.array
    gate_biases: mx.array
    up_weight: mx.array
    up_scales: mx.array
    up_biases: mx.array
    down_weight: mx.array
    down_scales: mx.array
    down_biases: mx.array


class SynchronousExpertStore:
    """Bounded exact-expert cache used by the Stage 2 reference runtime.

    This deliberately performs no prediction or asynchronous work. A miss
    blocks on the requested record, making it the correctness and performance
    baseline for later native implementations.
    """

    def __init__(self, artifact: Path, *, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("expert cache capacity must be positive")
        self.artifact = artifact.resolve()
        index = json.loads((self.artifact / "experts.index.json").read_text())
        if index["format"] != "vference.expert-pack.v1":
            raise ValueError(f"unsupported expert pack format: {index['format']}")
        self.layer_count = int(index["layer_count"])
        self.expert_count = int(index["expert_count_per_layer"])
        self.record_size = int(index["record_size"])
        self.layer_stride = int(index["layer_stride"])
        self.layer_ids = tuple(int(value) for value in index["layer_ids"])
        self.components = tuple(
            ComponentLayout(
                suffix=item["suffix"],
                dtype=item["dtype"],
                shape=tuple(item["source_shape"])[1:],
                bytes_per_expert=int(item["bytes_per_expert"]),
                record_offset=int(item["record_offset"]),
            )
            for item in index["components"]
        )
        self._validate_layout()
        self.capacity = capacity
        self._fd = os.open(self.artifact / "experts.pack", os.O_RDONLY)
        self._cache: OrderedDict[tuple[int, int], ExpertWeights] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.bytes_read = 0

    def _validate_layout(self) -> None:
        expected = {
            "gate_proj.weight",
            "gate_proj.scales",
            "gate_proj.biases",
            "up_proj.weight",
            "up_proj.scales",
            "up_proj.biases",
            "down_proj.weight",
            "down_proj.scales",
            "down_proj.biases",
        }
        actual = {component.suffix for component in self.components}
        if actual != expected:
            raise ValueError(f"expert component mismatch: {sorted(actual ^ expected)}")
        spans = sorted(
            (component.record_offset, component.record_offset + component.bytes_per_expert)
            for component in self.components
        )
        if spans[0][0] != 0 or spans[-1][1] != self.record_size:
            raise ValueError("expert components do not span the complete record")
        if any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("expert components overlap or leave gaps")

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._cache.clear()

    def __enter__(self) -> SynchronousExpertStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _record_offset(self, layer_id: int, expert_id: int) -> int:
        try:
            layer_ordinal = self.layer_ids.index(layer_id)
        except ValueError as error:
            raise IndexError(f"unknown expert layer {layer_id}") from error
        if not 0 <= expert_id < self.expert_count:
            raise IndexError(f"expert {expert_id} out of range")
        return layer_ordinal * self.layer_stride + expert_id * self.record_size

    @staticmethod
    def _array(data: memoryview, component: ComponentLayout) -> mx.array:
        try:
            dtype = _NUMPY_DTYPES[component.dtype]
        except KeyError as error:
            raise ValueError(f"unsupported packed dtype: {component.dtype}") from error
        array = mx.array(np.frombuffer(data, dtype=dtype).copy()).reshape(component.shape)
        if component.dtype == "BF16":
            array = array.view(mx.bfloat16)
        return array

    def _load(self, layer_id: int, expert_id: int) -> ExpertWeights:
        offset = self._record_offset(layer_id, expert_id)
        record = os.pread(self._fd, self.record_size, offset)
        if len(record) != self.record_size:
            raise OSError(
                f"short expert read for ({layer_id}, {expert_id}): "
                f"{len(record)} != {self.record_size}"
            )
        arrays = {
            component.suffix: self._array(
                memoryview(record)[
                    component.record_offset : component.record_offset
                    + component.bytes_per_expert
                ],
                component,
            )
            for component in self.components
        }
        mx.eval(*arrays.values())
        self.bytes_read += self.record_size
        return ExpertWeights(
            gate_weight=arrays["gate_proj.weight"],
            gate_scales=arrays["gate_proj.scales"],
            gate_biases=arrays["gate_proj.biases"],
            up_weight=arrays["up_proj.weight"],
            up_scales=arrays["up_proj.scales"],
            up_biases=arrays["up_proj.biases"],
            down_weight=arrays["down_proj.weight"],
            down_scales=arrays["down_proj.scales"],
            down_biases=arrays["down_proj.biases"],
        )

    def get(self, layer_id: int, expert_id: int) -> ExpertWeights:
        key = (layer_id, expert_id)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        weights = self._load(layer_id, expert_id)
        self._cache[key] = weights
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return weights

    @staticmethod
    def _qmm(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x,
            weight,
            scales,
            biases,
            transpose=True,
            group_size=64,
            bits=4,
            mode="affine",
        )

    def execute(self, layer_id: int, x: mx.array, indices: mx.array) -> mx.array:
        """Execute selected experts in the exact route order supplied by MLX."""
        mx.eval(indices)
        host_indices = np.asarray(indices, dtype=np.int64)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_indices = host_indices.reshape(-1, host_indices.shape[-1])
        token_outputs: list[mx.array] = []
        for token_id, token_experts in enumerate(flat_indices):
            route_outputs: list[mx.array] = []
            token_x = flat_x[token_id : token_id + 1]
            for expert_id in token_experts:
                weights = self.get(layer_id, int(expert_id))
                gate = self._qmm(
                    token_x, weights.gate_weight, weights.gate_scales, weights.gate_biases
                )
                up = self._qmm(
                    token_x, weights.up_weight, weights.up_scales, weights.up_biases
                )
                hidden = nn.silu(gate) * up
                down = self._qmm(
                    hidden,
                    weights.down_weight,
                    weights.down_scales,
                    weights.down_biases,
                )
                route_outputs.append(down.squeeze(0))
            token_outputs.append(mx.stack(route_outputs, axis=0))
        output = mx.stack(token_outputs, axis=0)
        return output.reshape(*indices.shape, x.shape[-1])

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "resident": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "bytes_read": self.bytes_read,
        }


class StreamingSwitchGLU(nn.Module):
    def __init__(self, layer_id: int, store: SynchronousExpertStore) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.store = store

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return self.store.execute(self.layer_id, x, indices)
