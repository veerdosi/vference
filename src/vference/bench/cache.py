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
    limits = {layer_id: base + (ordinal < remainder) for ordinal, layer_id in enumerate(layer_ids)}
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
    per_layer_unique.update({layer_id: len(experts) for layer_id, experts in seen_by_layer.items()})

    results = []
    for capacity in capacities:
        if capacity < 1:
            raise ValueError("cache capacities must be positive")
        hits, misses = _global_lru(requests, capacity)
        results.append(_result("global_lru", capacity, hits, misses, record_size))
        if capacity >= len(layer_ids):
            hits, misses = _partitioned_lru(requests, capacity, layer_ids)
            results.append(_result("partitioned_lru", capacity, hits, misses, record_size))

    contiguous_pairs = 0
    possible_pairs = 0
    for record in trace["records"]:
        for expert_ids in record["expert_ids"]:
            ordered = sorted(int(value) for value in expert_ids)
            contiguous_pairs += sum(right == left + 1 for left, right in zip(ordered, ordered[1:]))
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


def _trace_calls(trace: dict) -> tuple[list[list[dict]], list[list[dict]]]:
    records = trace["records"]
    layer_ids = sorted({int(record["layer_id"]) for record in records})
    layer_count = len(layer_ids)
    if not layer_count or len(records) % layer_count:
        raise ValueError("route trace does not contain complete model calls")
    calls = [records[start : start + layer_count] for start in range(0, len(records), layer_count)]
    for call in calls:
        if [int(record["layer_id"]) for record in call] != layer_ids:
            raise ValueError("route trace layer order changes within a model call")
        row_counts = {len(record["expert_ids"]) for record in call}
        if len(row_counts) != 1:
            raise ValueError("route trace layers disagree on model-call row count")
    decode_count = max(0, len(trace.get("output_tokens", [])) - 1)
    if decode_count > len(calls):
        raise ValueError("route trace has fewer model calls than emitted tokens")
    split = len(calls) - decode_count
    prefill = calls[:split]
    decode = calls[split:]
    if any(len(call[0]["expert_ids"]) != 1 for call in decode):
        raise ValueError("route trace decode section contains a multi-row model call")
    return prefill, decode


def _training_counts(
    prefill: list[list[dict]], layer_ids: list[int]
) -> tuple[
    dict[int, Counter],
    dict[tuple[int, int], dict[int, Counter]],
    dict[int, tuple[int, ...]],
]:
    popularity = {layer_id: Counter() for layer_id in layer_ids}
    transitions: dict[tuple[int, int], dict[int, Counter]] = {}
    previous_by_layer: dict[int, tuple[int, ...]] = {}

    def update_transition(
        pair: tuple[int, int], sources: tuple[int, ...], targets: tuple[int, ...]
    ) -> None:
        table = transitions.setdefault(pair, {})
        for source in sources:
            table.setdefault(source, Counter()).update(targets)

    for call in prefill:
        for record in call:
            layer_id = int(record["layer_id"])
            for experts in record["expert_ids"]:
                selected = tuple(int(expert) for expert in experts)
                popularity[layer_id].update(selected)
                previous = previous_by_layer.get(layer_id)
                if previous is not None:
                    update_transition((layer_id, layer_id), previous, selected)
                previous_by_layer[layer_id] = selected
        for left, right in zip(call, call[1:]):
            pair = (int(left["layer_id"]), int(right["layer_id"]))
            for left_experts, right_experts in zip(left["expert_ids"], right["expert_ids"]):
                sources = tuple(int(expert) for expert in left_experts)
                targets = tuple(int(expert) for expert in right_experts)
                update_transition(pair, sources, targets)
    return popularity, transitions, previous_by_layer


def _top(counter: Counter, budget: int) -> tuple[int, ...]:
    return tuple(
        expert
        for expert, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:budget]
    )


def _simulate_prefetch(
    decode: list[list[dict]],
    layer_ids: list[int],
    capacity_per_layer: int,
    budget: int,
    predictor: str,
    popularity: dict[int, Counter],
    transitions: dict[tuple[int, int], dict[int, Counter]],
    initial_previous_by_layer: dict[int, tuple[int, ...]],
    min_prediction_observations: int = 1,
) -> dict:
    caches = {layer_id: OrderedDict() for layer_id in layer_ids}
    speculative: set[tuple[int, int]] = set()
    demand_hits = demand_misses = prefetch_reads = useful = prediction_existing = 0
    evicted_unused = 0
    adaptive_transitions = {
        pair: {source: Counter(counts) for source, counts in table.items()}
        for pair, table in transitions.items()
    }
    previous_by_layer = dict(initial_previous_by_layer)

    def insert(layer_id: int, expert_id: int, *, prefetch: bool) -> None:
        nonlocal prefetch_reads, prediction_existing, evicted_unused
        key = (layer_id, expert_id)
        cache = caches[layer_id]
        if expert_id in cache:
            cache.move_to_end(expert_id)
            if prefetch:
                prediction_existing += 1
            return
        cache[expert_id] = None
        if prefetch:
            prefetch_reads += 1
            speculative.add(key)
        if len(cache) > capacity_per_layer:
            victim, _ = cache.popitem(last=False)
            victim_key = (layer_id, victim)
            if victim_key in speculative:
                speculative.remove(victim_key)
                evicted_unused += 1

    def predict_static(layer_id: int) -> tuple[int, ...]:
        return _top(popularity[layer_id], budget)

    def predict_transition(
        left_layer: int, right_layer: int, selected: tuple[int, ...]
    ) -> tuple[int, ...]:
        scores = Counter()
        table = adaptive_transitions.get((left_layer, right_layer), {})
        for source in selected:
            scores.update(table.get(source, {}))
        return tuple(
            expert_id
            for expert_id in _top(scores, budget)
            if scores[expert_id] >= min_prediction_observations * max(1, len(selected))
        )

    for call in decode:
        for ordinal, record in enumerate(call):
            layer_id = int(record["layer_id"])
            if predictor == "static_popularity":
                for expert_id in predict_static(layer_id):
                    insert(layer_id, expert_id, prefetch=True)
            previous = previous_by_layer.get(layer_id)
            if predictor == "same_layer_transition" and previous is not None:
                for expert_id in predict_transition(layer_id, layer_id, previous):
                    insert(layer_id, expert_id, prefetch=True)
            selected = tuple(int(expert) for expert in record["expert_ids"][0])
            for expert_id in selected:
                key = (layer_id, expert_id)
                if expert_id in caches[layer_id]:
                    demand_hits += 1
                    caches[layer_id].move_to_end(expert_id)
                    if key in speculative:
                        speculative.remove(key)
                        useful += 1
                else:
                    demand_misses += 1
                    insert(layer_id, expert_id, prefetch=False)
            if predictor == "cross_layer_adaptive" and ordinal > 0:
                left_layer = int(call[ordinal - 1]["layer_id"])
                left_selected = tuple(int(expert) for expert in call[ordinal - 1]["expert_ids"][0])
                table = adaptive_transitions.setdefault((left_layer, layer_id), {})
                for source in left_selected:
                    table.setdefault(source, Counter()).update(selected)
            if predictor == "same_layer_transition" and previous is not None:
                table = adaptive_transitions.setdefault((layer_id, layer_id), {})
                for source in previous:
                    table.setdefault(source, Counter()).update(selected)
            previous_by_layer[layer_id] = selected
            if predictor in {
                "cross_layer_transition",
                "cross_layer_adaptive",
            } and ordinal + 1 < len(call):
                next_layer = int(call[ordinal + 1]["layer_id"])
                for expert_id in predict_transition(layer_id, next_layer, selected):
                    insert(next_layer, expert_id, prefetch=True)

    total_demands = demand_hits + demand_misses
    return {
        "predictor": predictor,
        "prefetch_budget_per_layer": budget,
        "min_prediction_observations": min_prediction_observations,
        "capacity_per_layer": capacity_per_layer,
        "demand_hits": demand_hits,
        "exposed_demand_misses": demand_misses,
        "demand_hit_rate": demand_hits / total_demands,
        "prefetch_reads": prefetch_reads,
        "predictions_already_resident": prediction_existing,
        "useful_prefetch_reads": useful,
        "useful_prefetch_rate": useful / prefetch_reads if prefetch_reads else 0.0,
        "unused_prefetch_evictions": evicted_unused,
        "unused_prefetches_at_end": len(speculative),
        "total_physical_reads": demand_misses + prefetch_reads,
    }


def replay_prefetch(
    trace_path: Path,
    *,
    capacity_per_layer: int,
    budgets: tuple[int, ...],
    record_size: int,
    adaptive_min_observations: tuple[int, ...] = (1,),
) -> dict:
    """Evaluate ideal-completion prefetch policies without changing runtime math."""
    if capacity_per_layer < 1:
        raise ValueError("per-layer cache capacity must be positive")
    if not adaptive_min_observations or any(value < 1 for value in adaptive_min_observations):
        raise ValueError("adaptive minimum observations must be positive")
    trace = json.loads(trace_path.read_text())
    if trace.get("format") != "vference.route-trace.v1":
        raise ValueError(f"unsupported trace format: {trace.get('format')}")
    prefill, decode = _trace_calls(trace)
    if not decode:
        raise ValueError("route trace contains no single-token decode calls")
    layer_ids = [int(record["layer_id"]) for record in decode[0]]
    popularity, transitions, previous_by_layer = _training_counts(prefill, layer_ids)
    baseline = _simulate_prefetch(
        decode,
        layer_ids,
        capacity_per_layer,
        0,
        "none",
        popularity,
        transitions,
        previous_by_layer,
    )
    baseline["demand_read_bytes"] = baseline["exposed_demand_misses"] * record_size
    baseline["total_physical_read_bytes"] = baseline["total_physical_reads"] * record_size
    results = []
    for budget in budgets:
        if budget < 1:
            raise ValueError("prefetch budgets must be positive")
        for predictor in (
            "static_popularity",
            "cross_layer_transition",
            "cross_layer_adaptive",
            "same_layer_transition",
        ):
            observations = (
                adaptive_min_observations if predictor == "cross_layer_adaptive" else (1,)
            )
            for min_observations in observations:
                result = _simulate_prefetch(
                    decode,
                    layer_ids,
                    capacity_per_layer,
                    budget,
                    predictor,
                    popularity,
                    transitions,
                    previous_by_layer,
                    min_prediction_observations=min_observations,
                )
                result["demand_misses_avoided"] = (
                    baseline["exposed_demand_misses"] - result["exposed_demand_misses"]
                )
                result["physical_read_amplification"] = (
                    result["total_physical_reads"] / baseline["total_physical_reads"]
                )
                result["demand_read_bytes"] = result["exposed_demand_misses"] * record_size
                result["prefetch_read_bytes"] = result["prefetch_reads"] * record_size
                result["total_physical_read_bytes"] = result["total_physical_reads"] * record_size
                results.append(result)
    return {
        "trace": str(trace_path.resolve()),
        "prompt_tokens": trace.get("prompt_tokens"),
        "decode_model_calls": len(decode),
        "layers": len(layer_ids),
        "record_size": record_size,
        "adaptive_min_observations": list(adaptive_min_observations),
        "baseline": baseline,
        "results": results,
    }
