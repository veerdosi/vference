from __future__ import annotations

import json
from collections import Counter, OrderedDict
from pathlib import Path


def _requests(trace: dict) -> list[tuple[int, int]]:
    requests: list[tuple[int, int]] = []
    for record in trace["records"]:
        layer_id = int(record["layer_id"])
        for token_experts in record["expert_ids"]:
            requests.extend((layer_id, int(expert_id)) for expert_id in token_experts)
    return requests


def _global_lru(requests: list[tuple[int, int]], capacity: int) -> tuple[int, int]:
    cache: OrderedDict[tuple[int, int], None] = OrderedDict()
    hits = 0
    for key in requests:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits, len(requests) - hits


def _partitioned_lru(
    requests: list[tuple[int, int]], capacity: int, layer_ids: list[int]
) -> tuple[int, int]:
    base, remainder = divmod(capacity, len(layer_ids))
    limits = {
        layer_id: base + (ordinal < remainder)
        for ordinal, layer_id in enumerate(layer_ids)
    }
    caches = {layer_id: OrderedDict() for layer_id in layer_ids}
    hits = 0
    for layer_id, expert_id in requests:
        cache = caches[layer_id]
        if expert_id in cache:
            hits += 1
            cache.move_to_end(expert_id)
        elif limits[layer_id] > 0:
            cache[expert_id] = None
            if len(cache) > limits[layer_id]:
                cache.popitem(last=False)
    return hits, len(requests) - hits


def _result(policy: str, capacity: int, hits: int, misses: int, record_size: int) -> dict:
    total = hits + misses
    return {
        "policy": policy,
        "capacity_records": capacity,
        "capacity_bytes": capacity * record_size,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total,
        "logical_read_bytes": misses * record_size,
        "logical_read_bytes_per_request": misses * record_size / total,
    }


def replay_trace(trace_path: Path, capacities: tuple[int, ...], record_size: int) -> dict:
    trace = json.loads(trace_path.read_text())
    if trace.get("format") != "vference.route-trace.v1":
        raise ValueError(f"unsupported trace format: {trace.get('format')}")
    requests = _requests(trace)
    layer_ids = sorted({layer_id for layer_id, _ in requests})
    per_layer_unique = Counter()
    seen_by_layer: dict[int, set[int]] = {layer_id: set() for layer_id in layer_ids}
    for layer_id, expert_id in requests:
        seen_by_layer[layer_id].add(expert_id)
    per_layer_unique.update(
        {layer_id: len(experts) for layer_id, experts in seen_by_layer.items()}
    )

    results = []
    for capacity in capacities:
        if capacity < 1:
            raise ValueError("cache capacities must be positive")
        hits, misses = _global_lru(requests, capacity)
        results.append(_result("global_lru", capacity, hits, misses, record_size))
        if capacity >= len(layer_ids):
            hits, misses = _partitioned_lru(requests, capacity, layer_ids)
            results.append(
                _result("partitioned_lru", capacity, hits, misses, record_size)
            )

    contiguous_pairs = 0
    possible_pairs = 0
    for record in trace["records"]:
        for expert_ids in record["expert_ids"]:
            ordered = sorted(int(value) for value in expert_ids)
            contiguous_pairs += sum(
                right == left + 1 for left, right in zip(ordered, ordered[1:])
            )
            possible_pairs += max(0, len(ordered) - 1)

    return {
        "trace": str(trace_path.resolve()),
        "prompt_tokens": trace.get("prompt_tokens"),
        "output_tokens": len(trace.get("output_tokens", [])),
        "layers": len(layer_ids),
        "expert_requests": len(requests),
        "unique_experts": len(set(requests)),
        "per_layer_unique": dict(sorted(per_layer_unique.items())),
        "adjacent_id_pair_rate": contiguous_pairs / possible_pairs,
        "results": results,
    }
