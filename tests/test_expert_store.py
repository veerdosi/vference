import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vference.runtime.expert_store import SynchronousExpertStore


def _bytes(array: mx.array) -> bytes:
    mx.eval(array)
    if array.dtype == mx.bfloat16:
        array = array.view(mx.uint16)
    return np.asarray(array).tobytes()


def test_streamed_experts_match_the_same_quantized_arrays(tmp_path: Path) -> None:
    mx.random.seed(7)
    expert_count = 2
    input_dims = 64
    hidden_dims = 64
    projections = {
        "gate_proj": (hidden_dims, input_dims),
        "up_proj": (hidden_dims, input_dims),
        "down_proj": (input_dims, hidden_dims),
    }
    experts: list[dict[str, mx.array]] = []
    for _ in range(expert_count):
        values: dict[str, mx.array] = {}
        for name, shape in projections.items():
            dense = mx.random.normal(shape).astype(mx.bfloat16)
            weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
            values[f"{name}.weight"] = weight
            values[f"{name}.scales"] = scales
            values[f"{name}.biases"] = biases
        experts.append(values)

    suffixes = (
        "gate_proj.weight",
        "gate_proj.scales",
        "gate_proj.biases",
        "up_proj.weight",
        "up_proj.scales",
        "up_proj.biases",
        "down_proj.weight",
        "down_proj.scales",
        "down_proj.biases",
    )
    record_offset = 0
    components = []
    for suffix in suffixes:
        value = experts[0][suffix]
        data = _bytes(value)
        components.append(
            {
                "suffix": suffix,
                "dtype": "U32" if value.dtype == mx.uint32 else "BF16",
                "source_shape": [expert_count, *value.shape],
                "bytes_per_expert": len(data),
                "record_offset": record_offset,
            }
        )
        record_offset += len(data)
    index = {
        "format": "vference.expert-pack.v1",
        "layer_count": 1,
        "expert_count_per_layer": expert_count,
        "record_size": record_offset,
        "layer_stride": record_offset * expert_count,
        "layer_ids": [0],
        "components": components,
    }
    (tmp_path / "experts.index.json").write_text(json.dumps(index))
    with (tmp_path / "experts.pack").open("wb") as handle:
        for expert in experts:
            for suffix in suffixes:
                handle.write(_bytes(expert[suffix]))

    x = mx.random.normal((1, 1, input_dims)).astype(mx.bfloat16)
    indices = mx.array([[[1, 0]]], dtype=mx.int32)
    expected = []
    for expert_id in (1, 0):
        expert = experts[expert_id]

        def qmm(prefix: str, value: mx.array) -> mx.array:
            return mx.quantized_matmul(
                value,
                expert[f"{prefix}.weight"],
                expert[f"{prefix}.scales"],
                expert[f"{prefix}.biases"],
                group_size=64,
                bits=4,
            )

        gate = qmm("gate_proj", x.reshape(1, input_dims))
        up = qmm("up_proj", x.reshape(1, input_dims))
        expected.append(qmm("down_proj", nn.silu(gate) * up).squeeze(0))
    expected_array = mx.stack(expected).reshape(1, 1, 2, input_dims)

    with SynchronousExpertStore(tmp_path, capacity=1) as store:
        actual = store.execute(0, x, indices)
        mx.eval(actual, expected_array)
        assert np.array_equal(
            np.asarray(actual.view(mx.uint16)),
            np.asarray(expected_array.view(mx.uint16)),
        )
        stats = store.stats()
        assert {key: stats[key] for key in ("capacity", "resident", "hits", "misses", "bytes_read")} == {
            "capacity": 1,
            "resident": 1,
            "hits": 0,
            "misses": 2,
            "bytes_read": 2 * record_offset,
        }
