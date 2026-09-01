import json
from pathlib import Path

from vference.bench.cache import replay_prefetch, replay_trace


def test_replay_global_and_partitioned_lru(tmp_path: Path) -> None:
    trace = {
        "format": "vference.route-trace.v1",
        "prompt_tokens": 1,
        "output_tokens": [1],
        "records": [
            {"layer_id": 0, "expert_ids": [[0, 1]]},
            {"layer_id": 1, "expert_ids": [[0, 1]]},
            {"layer_id": 0, "expert_ids": [[0, 2]]},
            {"layer_id": 1, "expert_ids": [[0, 2]]},
        ],
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    result = replay_trace(path, capacities=(2, 4), record_size=100)
    by_key = {(item["policy"], item["capacity_records"]): item for item in result["results"]}
    assert by_key[("global_lru", 2)]["hits"] == 0
    assert by_key[("partitioned_lru", 2)]["hits"] == 0
    assert by_key[("global_lru", 4)]["hits"] == 2
    assert by_key[("partitioned_lru", 4)]["hits"] == 2
    assert result["unique_experts"] == 6


def test_prefetch_replay_separates_exposed_and_physical_reads(tmp_path: Path) -> None:
    trace = {
        "format": "vference.route-trace.v1",
        "prompt_tokens": 2,
        "output_tokens": [1, 2, 3],
        "records": [
            {"layer_id": 0, "expert_ids": [[0], [0]]},
            {"layer_id": 1, "expert_ids": [[2], [2]]},
            {"layer_id": 0, "expert_ids": [[0]]},
            {"layer_id": 1, "expert_ids": [[2]]},
            {"layer_id": 0, "expert_ids": [[0]]},
            {"layer_id": 1, "expert_ids": [[2]]},
        ],
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    result = replay_prefetch(path, capacity_per_layer=1, budgets=(1,), record_size=100)
    assert result["baseline"]["exposed_demand_misses"] == 2
    by_predictor = {item["predictor"]: item for item in result["results"]}
    transition = by_predictor["cross_layer_transition"]
    assert transition["exposed_demand_misses"] == 1
    assert transition["useful_prefetch_reads"] == 1
    assert transition["total_physical_reads"] == 2
    static = by_predictor["static_popularity"]
    assert static["exposed_demand_misses"] == 0
    assert static["total_physical_reads"] == 2

    supported = replay_prefetch(
        path,
        capacity_per_layer=1,
        budgets=(1,),
        record_size=100,
        adaptive_min_observations=(3,),
    )
    adaptive = next(
        item for item in supported["results"] if item["predictor"] == "cross_layer_adaptive"
    )
    assert adaptive["min_prediction_observations"] == 3
    assert adaptive["prefetch_reads"] == 0
    assert adaptive["exposed_demand_misses"] == 2
