from __future__ import annotations

import gc
import json
import os
import fcntl
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vference.native import extension

from .integrity import file_identity


_NUMPY_DTYPES = {
    "U32": np.dtype("<u4"),
    "BF16": np.dtype("<u2"),
}
F_NOCACHE = 48


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

    def __init__(
        self,
        artifact: Path,
        *,
        capacity: int = 64,
        nocache: bool = False,
        trace_routes: bool = False,
    ) -> None:
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
        if nocache:
            fcntl.fcntl(self._fd, F_NOCACHE, 1)
        self.nocache = nocache
        self.trace_routes = trace_routes
        self._pack_identity = file_identity(self.artifact / "experts.pack")
        self.artifact_integrity: dict[str, object] | None = None
        self._cache: OrderedDict[tuple[int, int], ExpertWeights] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.bytes_read = 0
        self.read_ns = 0
        self.materialize_ns = 0
        self.router_wait_ns = 0
        self.execute_ns = 0
        self.layer_stats = [
            {"hits": 0, "misses": 0, "read_ns": 0, "materialize_ns": 0}
            for _ in range(self.layer_count)
        ]
        self.route_trace: list[dict[str, object]] = []

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

    def validate_source_unchanged(self) -> None:
        current = file_identity(self.artifact / "experts.pack")
        if current != self._pack_identity:
            raise OSError("experts.pack changed after runtime integrity admission")

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
        read_started = time.perf_counter_ns()
        record = os.pread(self._fd, self.record_size, offset)
        read_ns = time.perf_counter_ns() - read_started
        if len(record) != self.record_size:
            raise OSError(
                f"short expert read for ({layer_id}, {expert_id}): "
                f"{len(record)} != {self.record_size}"
            )
        materialize_started = time.perf_counter_ns()
        arrays = {
            component.suffix: self._array(
                memoryview(record)[
                    component.record_offset : component.record_offset + component.bytes_per_expert
                ],
                component,
            )
            for component in self.components
        }
        mx.eval(*arrays.values())
        materialize_ns = time.perf_counter_ns() - materialize_started
        self.bytes_read += self.record_size
        self.read_ns += read_ns
        self.materialize_ns += materialize_ns
        self.layer_stats[layer_id]["read_ns"] += read_ns
        self.layer_stats[layer_id]["materialize_ns"] += materialize_ns
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
            self.layer_stats[layer_id]["hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        self.layer_stats[layer_id]["misses"] += 1
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
        execute_started = time.perf_counter_ns()
        wait_started = time.perf_counter_ns()
        mx.eval(indices)
        self.router_wait_ns += time.perf_counter_ns() - wait_started
        host_indices = np.asarray(indices, dtype=np.int64)
        if self.trace_routes:
            self.route_trace.append(
                {
                    "layer_id": layer_id,
                    "expert_ids": host_indices.reshape(-1, host_indices.shape[-1]).tolist(),
                }
            )
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
                up = self._qmm(token_x, weights.up_weight, weights.up_scales, weights.up_biases)
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
        self.execute_ns += time.perf_counter_ns() - execute_started
        return output.reshape(*indices.shape, x.shape[-1])

    def stats(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "nocache": self.nocache,
            "resident": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "bytes_read": self.bytes_read,
            "artifact_integrity": self.artifact_integrity,
            "timing_seconds": {
                "pread": self.read_ns / 1_000_000_000,
                "materialize": self.materialize_ns / 1_000_000_000,
                "router_and_graph_wait": self.router_wait_ns / 1_000_000_000,
                "execute_total": self.execute_ns / 1_000_000_000,
            },
            "per_layer": [
                {
                    "hits": values["hits"],
                    "misses": values["misses"],
                    "read_seconds": values["read_ns"] / 1_000_000_000,
                    "materialize_seconds": values["materialize_ns"] / 1_000_000_000,
                }
                for values in self.layer_stats
            ],
        }

    def trace(self) -> list[dict[str, object]]:
        return list(self.route_trace)


class StableSlotExpertStore(SynchronousExpertStore):
    """Exact synchronous cache backed by fixed, directly filled MLX buffers.

    Unlike :class:`SynchronousExpertStore`, a miss does not create nine new MLX
    arrays. The native reader writes the packed record into one stable slot in
    each component pool, and the GPU gathers the selected slots directly.
    """

    def __init__(
        self,
        artifact: Path,
        *,
        capacity: int = 64,
        nocache: bool = False,
        trace_routes: bool = False,
        cache_policy: str = "global",
        prefetch_policy: str = "none",
        prefetch_budget: int = 1,
        prefetch_min_observations: int = 8,
    ) -> None:
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
        self._layer_ordinals = {
            layer_id: ordinal for ordinal, layer_id in enumerate(self.layer_ids)
        }
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
        self.nocache = nocache
        self.trace_routes = trace_routes
        self._pack_identity = file_identity(self.artifact / "experts.pack")
        self.artifact_integrity: dict[str, object] | None = None
        if prefetch_policy not in {"none", "adaptive_cross"}:
            raise ValueError(f"unknown prefetch policy: {prefetch_policy}")
        if not 1 <= prefetch_budget <= 8:
            raise ValueError("prefetch budget must be between one and eight records")
        if prefetch_min_observations < 1:
            raise ValueError("prefetch minimum observations must be positive")
        self.prefetch_policy = prefetch_policy
        self.prefetch_budget = prefetch_budget
        self.prefetch_min_observations = prefetch_min_observations
        if cache_policy not in {"global", "layer", "demand"}:
            raise ValueError(f"unknown stable cache policy: {cache_policy}")
        if cache_policy == "layer" and capacity < self.layer_count:
            raise ValueError("layer-partitioned cache needs at least one slot per layer")
        self.cache_policy = cache_policy

        native = extension()
        self._native = native
        self._reader = native.PackReader(str(self.artifact / "experts.pack"), nocache)
        dtype_names = {"U32": "uint32", "BF16": "bfloat16"}
        try:
            self._pools = {
                component.suffix: native.owned_zeros(
                    [capacity, *component.shape], dtype_names[component.dtype]
                )
                for component in self.components
            }
        except KeyError as error:
            raise ValueError(f"unsupported packed dtype: {error.args[0]}") from error
        self._ordered_pools = [self._pools[component.suffix] for component in self.components]
        self._segment_bytes = [component.bytes_per_expert for component in self.components]
        self._free_slots = list(range(capacity - 1, -1, -1))
        self._free_slots_by_layer: dict[int, list[int]] = {}
        self._layer_capacities: dict[int, int] = {}
        if cache_policy == "layer":
            base, extra = divmod(capacity, self.layer_count)
            start = 0
            for ordinal, layer_id in enumerate(self.layer_ids):
                count = base + (ordinal < extra)
                self._layer_capacities[layer_id] = count
                self._free_slots_by_layer[layer_id] = list(range(start + count - 1, start - 1, -1))
                start += count
        self._cache: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._slot_keys: list[tuple[int, int] | None] = [None] * capacity
        self.policy_transitions: list[dict[str, str]] = []
        self.capacity_transitions: list[dict[str, int]] = []

        self.hits = 0
        self.misses = 0
        self.bytes_read = 0
        self.read_ns = 0
        self.materialize_ns = 0
        self.router_wait_ns = 0
        self.execute_ns = 0
        self.layer_stats = [
            {"hits": 0, "misses": 0, "read_ns": 0, "materialize_ns": 0}
            for _ in range(self.layer_count)
        ]
        self.route_trace: list[dict[str, object]] = []
        self._transition_tables: dict[tuple[int, int], np.ndarray] = {}
        self._previous_layer_route: tuple[int, np.ndarray] | None = None
        self._prefetch_fd = -1
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._pending_prefetch: dict[tuple[int, int], Future[bytes]] = {}
        self.prefetch_submitted = 0
        self.prefetch_bytes_read = 0
        self.prefetch_useful = 0
        self.prefetch_late = 0
        self.prefetch_wrong = 0
        self.prefetch_unused = 0
        self.prefetch_cancelled = 0
        self.prefetch_failed = 0
        self.prefetch_skipped_busy = 0
        self.prefetch_already_resident = 0
        self.prefetch_wait_ns = 0
        self.prefetch_copy_ns = 0
        if prefetch_policy != "none":
            self._prefetch_fd = os.open(self.artifact / "experts.pack", os.O_RDONLY)
            if nocache:
                fcntl.fcntl(self._prefetch_fd, F_NOCACHE, 1)
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vference-prefetch"
            )

    def close(self) -> None:
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
            self._prefetch_executor = None
        for key in list(self._pending_prefetch):
            self._finish_prefetch(key)
        if self._prefetch_fd >= 0:
            os.close(self._prefetch_fd)
            self._prefetch_fd = -1
        mx.synchronize()
        self._cache.clear()
        self._slot_keys = []
        self._free_slots = []
        self._free_slots_by_layer = {}
        self._layer_capacities = {}
        self._ordered_pools = []
        self._pools = {}
        self._reader = None
        gc.collect()
        mx.clear_cache()

    def set_cache_policy(self, cache_policy: str) -> None:
        """Synchronously clear and repartition slots between inference phases."""
        if cache_policy not in {"global", "layer", "demand"}:
            raise ValueError(f"unknown stable cache policy: {cache_policy}")
        if cache_policy == self.cache_policy:
            return
        if cache_policy == "layer" and self.capacity < self.layer_count:
            raise ValueError("layer-partitioned cache needs at least one slot per layer")
        previous = self.cache_policy
        mx.synchronize()
        self._cache.clear()
        self._slot_keys = [None] * self.capacity
        self._free_slots = list(range(self.capacity - 1, -1, -1))
        self._free_slots_by_layer = {}
        self._layer_capacities = {}
        if cache_policy == "layer":
            base, extra = divmod(self.capacity, self.layer_count)
            start = 0
            for ordinal, layer_id in enumerate(self.layer_ids):
                count = base + (ordinal < extra)
                self._layer_capacities[layer_id] = count
                self._free_slots_by_layer[layer_id] = list(range(start + count - 1, start - 1, -1))
                start += count
        self.cache_policy = cache_policy
        self.policy_transitions.append({"from": previous, "to": cache_policy})

    def resize_capacity(self, capacity: int, cache_policy: str | None = None) -> None:
        """Replace stable pools at a synchronized inference-phase boundary."""
        if capacity < 1:
            raise ValueError("expert cache capacity must be positive")
        target_policy = cache_policy or self.cache_policy
        if target_policy not in {"global", "layer", "demand"}:
            raise ValueError(f"unknown stable cache policy: {target_policy}")
        if target_policy == "layer" and capacity < self.layer_count:
            raise ValueError("layer-partitioned cache needs at least one slot per layer")
        if capacity == self.capacity:
            self.set_cache_policy(target_policy)
            return

        mx.synchronize()
        for key, future in list(self._pending_prefetch.items()):
            self.prefetch_wrong += 1
            if future.cancel():
                self.prefetch_cancelled += 1
                del self._pending_prefetch[key]
            else:
                self._finish_prefetch(key)
                self.prefetch_unused += 1

        previous_capacity = self.capacity
        previous_policy = self.cache_policy
        self._cache.clear()
        self._slot_keys = []
        self._free_slots = []
        self._free_slots_by_layer = {}
        self._layer_capacities = {}
        self._ordered_pools = []
        self._pools = {}
        gc.collect()
        mx.clear_cache()

        dtype_names = {"U32": "uint32", "BF16": "bfloat16"}
        self._pools = {
            component.suffix: self._native.owned_zeros(
                [capacity, *component.shape], dtype_names[component.dtype]
            )
            for component in self.components
        }
        self._ordered_pools = [self._pools[component.suffix] for component in self.components]
        self.capacity = capacity
        self._slot_keys = [None] * capacity
        self._free_slots = list(range(capacity - 1, -1, -1))
        if target_policy == "layer":
            base, extra = divmod(capacity, self.layer_count)
            start = 0
            for ordinal, layer_id in enumerate(self.layer_ids):
                count = base + (ordinal < extra)
                self._layer_capacities[layer_id] = count
                self._free_slots_by_layer[layer_id] = list(range(start + count - 1, start - 1, -1))
                start += count
        self.cache_policy = target_policy
        self.capacity_transitions.append({"from": previous_capacity, "to": capacity})
        if previous_policy != target_policy:
            self.policy_transitions.append({"from": previous_policy, "to": target_policy})

    def _observe_transition(self, layer_id: int, host_indices: np.ndarray) -> None:
        if self.prefetch_policy == "none":
            return
        routes = host_indices.reshape(-1, host_indices.shape[-1]).astype(np.intp, copy=False)
        previous = self._previous_layer_route
        if layer_id == self.layer_ids[0]:
            previous = None
        if (
            previous is not None
            and self._layer_ordinals[previous[0]] + 1 == self._layer_ordinals[layer_id]
        ):
            left = previous[1]
            if len(left) == len(routes):
                table = self._transition_tables.setdefault(
                    (previous[0], layer_id),
                    np.zeros((self.expert_count, self.expert_count), dtype=np.uint32),
                )
                width = routes.shape[-1]
                sources = np.repeat(left[:, :, None], width, axis=2).reshape(-1)
                targets = np.repeat(routes[:, None, :], width, axis=1).reshape(-1)
                np.add.at(table, (sources, targets), 1)
        self._previous_layer_route = (layer_id, routes.copy())

    def _finish_prefetch(self, key: tuple[int, int]) -> bytes | None:
        future = self._pending_prefetch.pop(key, None)
        if future is None:
            return None
        try:
            record = future.result()
        except Exception:
            self.prefetch_failed += 1
            return None
        if len(record) != self.record_size:
            self.prefetch_failed += 1
            return None
        self.prefetch_bytes_read += len(record)
        return record

    def _retire_prefetch(self, layer_id: int, requested: set[tuple[int, int]]) -> None:
        for key, future in list(self._pending_prefetch.items()):
            if key[0] != layer_id or (key in requested and key not in self._cache):
                continue
            self.prefetch_wrong += 1
            if future.cancel():
                self.prefetch_cancelled += 1
                del self._pending_prefetch[key]
            elif future.done():
                self._finish_prefetch(key)
                self.prefetch_unused += 1

    def _take_prefetched(self, key: tuple[int, int]) -> bytes | None:
        future = self._pending_prefetch.get(key)
        if future is None:
            return None
        ready = future.done()
        wait_started = time.perf_counter_ns()
        record = self._finish_prefetch(key)
        waited = time.perf_counter_ns() - wait_started
        if record is None:
            return None
        self.prefetch_useful += 1
        if not ready:
            self.prefetch_late += 1
            self.prefetch_wait_ns += waited
        return record

    def _submit_prefetch(self, layer_id: int, expert_ids: tuple[int, ...]) -> None:
        if self._prefetch_executor is None:
            return
        for key, future in list(self._pending_prefetch.items()):
            if key[0] < layer_id and future.done():
                self._finish_prefetch(key)
                self.prefetch_unused += 1
        for expert_id in expert_ids:
            key = (layer_id, expert_id)
            if key in self._cache:
                self.prefetch_already_resident += 1
                continue
            if key in self._pending_prefetch:
                continue
            if len(self._pending_prefetch) >= self.prefetch_budget:
                self.prefetch_skipped_busy += 1
                break
            future = self._prefetch_executor.submit(
                os.pread,
                self._prefetch_fd,
                self.record_size,
                self._record_offset(layer_id, expert_id),
            )
            self._pending_prefetch[key] = future
            self.prefetch_submitted += 1

    def _predict_next(self, layer_id: int, host_indices: np.ndarray) -> None:
        if self.prefetch_policy != "adaptive_cross":
            return
        try:
            next_layer = self.layer_ids[self._layer_ordinals[layer_id] + 1]
        except IndexError:
            return
        routes = host_indices.reshape(-1, host_indices.shape[-1])
        if len(routes) != 1:
            return
        table = self._transition_tables.get((layer_id, next_layer))
        if table is None:
            return
        scores = table[routes[0]].sum(axis=0, dtype=np.uint64)
        ranked = sorted(
            range(self.expert_count), key=lambda expert_id: (-int(scores[expert_id]), expert_id)
        )
        selected = tuple(
            expert_id
            for expert_id in ranked[: self.prefetch_budget]
            if scores[expert_id] >= self.prefetch_min_observations * routes.shape[-1]
        )
        if not selected:
            return
        self._submit_prefetch(next_layer, selected)

    def _record_offset(self, layer_id: int, expert_id: int) -> int:
        try:
            layer_ordinal = self._layer_ordinals[layer_id]
        except KeyError as error:
            raise IndexError(f"unknown expert layer {layer_id}") from error
        if not 0 <= expert_id < self.expert_count:
            raise IndexError(f"expert {expert_id} out of range")
        return layer_ordinal * self.layer_stride + expert_id * self.record_size

    def _allocate_slot(self, key: tuple[int, int], protected: set[tuple[int, int]]) -> int:
        if self.cache_policy == "layer":
            free_slots = self._free_slots_by_layer[key[0]]
            if free_slots:
                return free_slots.pop()
            for victim, slot in self._cache.items():
                if victim[0] == key[0] and victim not in protected:
                    del self._cache[victim]
                    self._slot_keys[slot] = None
                    return slot
            raise RuntimeError(
                "layer cache partition is smaller than the simultaneously "
                "requested expert working set"
            )
        if self._free_slots:
            return self._free_slots.pop()
        for victim, slot in self._cache.items():
            if victim not in protected:
                del self._cache[victim]
                self._slot_keys[slot] = None
                return slot
        raise RuntimeError(
            "stable expert capacity is smaller than the simultaneously requested working set"
        )

    def _load_slot(
        self,
        layer_id: int,
        expert_id: int,
        protected: set[tuple[int, int]],
    ) -> int:
        key = (layer_id, expert_id)
        slot = self._allocate_slot(key, protected)
        record = self._take_prefetched(key)
        started = time.perf_counter_ns()
        if record is None:
            count = self._reader.read_into(
                self._ordered_pools,
                slot,
                self._record_offset(layer_id, expert_id),
                self._segment_bytes,
            )
        else:
            count = self._native.copy_record_into(
                self._ordered_pools,
                slot,
                self._segment_bytes,
                record,
            )
        read_ns = time.perf_counter_ns() - started
        if count != self.record_size:
            raise OSError(
                f"short expert read for ({layer_id}, {expert_id}): {count} != {self.record_size}"
            )
        layer_ordinal = self._layer_ordinals[layer_id]
        if record is None:
            self.bytes_read += count
            self.read_ns += read_ns
            self.layer_stats[layer_ordinal]["read_ns"] += read_ns
        else:
            self.prefetch_copy_ns += read_ns
        self._cache[key] = slot
        self._slot_keys[slot] = key
        return slot

    def _resolve_slots(self, layer_id: int, host_indices: np.ndarray) -> np.ndarray:
        requested = [(layer_id, int(expert_id)) for expert_id in host_indices.reshape(-1)]
        protected = set(requested)
        self._retire_prefetch(layer_id, protected)
        available = self.capacity
        if self.cache_policy == "layer":
            available = len(self._free_slots_by_layer[layer_id]) + sum(
                key[0] == layer_id for key in self._cache
            )
        if len(protected) > available:
            raise RuntimeError(
                "stable expert capacity is smaller than the simultaneously requested working set"
            )
        slots: dict[tuple[int, int], int] = {}
        layer_ordinal = self._layer_ordinals[layer_id]
        for key in requested:
            if key in slots:
                self.hits += 1
                self.layer_stats[layer_ordinal]["hits"] += 1
                continue
            if key in self._cache:
                self.hits += 1
                self.layer_stats[layer_ordinal]["hits"] += 1
                self._cache.move_to_end(key)
                slots[key] = self._cache[key]
            else:
                self.misses += 1
                self.layer_stats[layer_ordinal]["misses"] += 1
                slots[key] = self._load_slot(*key, protected)
        return np.asarray([slots[key] for key in requested], dtype=np.int32).reshape(
            host_indices.shape
        )

    def _execute_loaded(self, x: mx.array, slot_indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        def qmm(prefix: str, value: mx.array) -> mx.array:
            return mx.gather_qmm(
                value,
                self._pools[f"{prefix}.weight"],
                self._pools[f"{prefix}.scales"],
                self._pools[f"{prefix}.biases"],
                rhs_indices=slot_indices,
                transpose=True,
                group_size=64,
                bits=4,
                mode="affine",
            )

        up = qmm("up_proj", x)
        gate = qmm("gate_proj", x)
        output = qmm("down_proj", nn.silu(gate) * up)
        return output.squeeze(-2)

    def execute(self, layer_id: int, x: mx.array, indices: mx.array) -> mx.array:
        execute_started = time.perf_counter_ns()
        wait_started = time.perf_counter_ns()
        mx.eval(indices)
        self.router_wait_ns += time.perf_counter_ns() - wait_started
        host_indices = np.asarray(indices, dtype=np.int64)
        if self.cache_policy == "demand":
            self._cache.clear()
            self._slot_keys = [None] * self.capacity
            self._free_slots = list(range(self.capacity - 1, -1, -1))
        if self.trace_routes:
            self.route_trace.append(
                {
                    "layer_id": layer_id,
                    "expert_ids": host_indices.reshape(-1, host_indices.shape[-1]).tolist(),
                }
            )

        self._observe_transition(layer_id, host_indices)

        unique_count = len(set(int(value) for value in host_indices.reshape(-1)))
        working_capacity = self._layer_capacities.get(layer_id, self.capacity)
        if unique_count <= working_capacity:
            local = mx.array(self._resolve_slots(layer_id, host_indices))
            output = self._execute_loaded(x, local)
        else:
            flat_x = x.reshape(-1, x.shape[-1])
            flat_indices = host_indices.reshape(-1, host_indices.shape[-1])
            grouped_outputs = []
            start = 0
            while start < len(flat_indices):
                selected: set[int] = set()
                end = start
                while end < len(flat_indices):
                    candidate = selected | set(int(value) for value in flat_indices[end])
                    if len(candidate) > working_capacity:
                        break
                    selected = candidate
                    end += 1
                if end == start:
                    raise RuntimeError(
                        "cache partition cannot hold one token's exact routed experts"
                    )
                local_host = flat_indices[start:end].reshape(1, end - start, -1)
                local = mx.array(self._resolve_slots(layer_id, local_host))
                group_output = self._execute_loaded(
                    flat_x[start:end].reshape(1, end - start, -1), local
                )
                # The next group may overwrite these slots, so settle this graph first.
                mx.eval(group_output)
                grouped_outputs.append(group_output.reshape(-1, x.shape[-1]))
                start = end
            output = mx.concatenate(grouped_outputs, axis=0).reshape(*indices.shape, x.shape[-1])
        self.execute_ns += time.perf_counter_ns() - execute_started
        self._predict_next(layer_id, host_indices)
        return output

    def stats(self) -> dict[str, object]:
        values = super().stats()
        values["implementation"] = "stable_native_slots"
        values["cache_policy"] = self.cache_policy
        values["prefetch_policy"] = self.prefetch_policy
        values["prefetch_budget"] = self.prefetch_budget
        values["prefetch_min_observations"] = self.prefetch_min_observations
        values["policy_transitions"] = list(self.policy_transitions)
        values["capacity_transitions"] = list(self.capacity_transitions)
        values["prefetch"] = {
            "submitted": self.prefetch_submitted,
            "bytes_read": self.prefetch_bytes_read,
            "useful": self.prefetch_useful,
            "late": self.prefetch_late,
            "wrong": self.prefetch_wrong,
            "unused_completed": self.prefetch_unused,
            "cancelled": self.prefetch_cancelled,
            "failed": self.prefetch_failed,
            "skipped_busy": self.prefetch_skipped_busy,
            "already_resident": self.prefetch_already_resident,
            "wait_seconds": self.prefetch_wait_ns / 1_000_000_000,
            "copy_seconds": self.prefetch_copy_ns / 1_000_000_000,
            "total_physical_bytes": self.bytes_read + self.prefetch_bytes_read,
        }
        return values

    def pool_pointers(self) -> dict[str, int]:
        """Expose addresses for invariance tests and diagnostics."""
        return {suffix: self._native.data_pointer(pool) for suffix, pool in self._pools.items()}


class StreamingSwitchGLU(nn.Module):
    def __init__(self, layer_id: int, store: SynchronousExpertStore) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.store = store

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return self.store.execute(self.layer_id, x, indices)
