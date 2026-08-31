import json
from pathlib import Path

from vference.bench.cache import replay_trace


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
    by_key = {
        (item["policy"], item["capacity_records"]): item for item in result["results"]
    }
    assert by_key[("global_lru", 2)]["hits"] == 0
    assert by_key[("partitioned_lru", 2)]["hits"] == 0
    assert by_key[("global_lru", 4)]["hits"] == 2
    assert by_key[("partitioned_lru", 4)]["hits"] == 2
    assert result["unique_experts"] == 6
