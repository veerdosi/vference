import json
from pathlib import Path

import pytest
import numpy as np
from safetensors.numpy import save_file

from vference.adapters import Qwen3MoeAdapter, artifact_adapter_for_config
from vference.artifacts.builder import build_artifact, verify_artifact
from vference.artifacts.safetensors import TensorEntry
from vference.runtime.model import Qwen35RuntimeAdapter, runtime_adapter_for_artifact


def _qwen3_entries(tmp_path: Path, layer_ids: tuple[int, ...]) -> dict[str, TensorEntry]:
    component_sizes = [256] * 8 + [2048]
    entries: dict[str, TensorEntry] = {}
    for layer_id in layer_ids:
        for suffix, bytes_per_expert in zip(
            Qwen3MoeAdapter._component_suffixes, component_sizes, strict=True
        ):
            name = f"model.layers.{layer_id}.mlp.switch_mlp.{suffix}"
            entries[name] = TensorEntry(
                name=name,
                shard=tmp_path / "synthetic.safetensors",
                dtype="U8",
                shape=(2, bytes_per_expert),
                data_start=0,
                data_end=2 * bytes_per_expert,
                header_size=8,
            )
    return entries


def test_second_architecture_maps_sparse_layers_into_generic_records(tmp_path: Path) -> None:
    adapter = artifact_adapter_for_config(
        {
            "model_type": "qwen3_moe",
            "num_hidden_layers": 4,
            "num_experts": 2,
            "decoder_sparse_step": 2,
            "mlp_only_layers": [],
        }
    )
    assert isinstance(adapter, Qwen3MoeAdapter)

    layers = adapter.expert_layers(_qwen3_entries(tmp_path, (1, 3)))

    assert [layer.layer_id for layer in layers] == [1, 3]
    assert all(layer.expert_count == 2 for layer in layers)
    assert all(layer.record_size == 4096 for layer in layers)
    assert layers[0].components[-1].record_offset == 2048


def test_second_architecture_uses_generic_builder_and_verifier(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 2,
        "num_experts": 2,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [],
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }
    (source / "config.json").write_text(json.dumps(config))
    arrays: dict[str, np.ndarray] = {"model.embed_tokens.weight": np.arange(16, dtype=np.float16)}
    metadata_sizes = [64, 64, 64, 64, 64, 192]
    metadata_index = 0
    for suffix in Qwen3MoeAdapter._component_suffixes:
        name = f"model.layers.1.mlp.switch_mlp.{suffix}"
        if suffix.endswith("weight"):
            arrays[name] = np.arange(2 * 256, dtype=np.uint32).reshape(2, 256)
        else:
            size = metadata_sizes[metadata_index]
            metadata_index += 1
            arrays[name] = np.arange(2 * size, dtype=np.float16).reshape(2, size)
    save_file(arrays, source / "model.safetensors")

    artifact = tmp_path / "runtime"
    result = build_artifact(source, artifact, min_free_bytes=0)
    index = json.loads((artifact / "experts.index.json").read_text())
    manifest = json.loads((artifact / "manifest.json").read_text())

    assert result.expert_record_bytes == 4096
    assert index["layer_ids"] == [1]
    assert manifest["architecture_adapter"] == "qwen3_moe"
    assert verify_artifact(source, artifact)["verified"]


def test_runtime_adapter_is_selected_by_artifact_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"architecture_adapter": "qwen3_5_moe"})
    )

    adapter = runtime_adapter_for_artifact(tmp_path)

    assert isinstance(adapter, Qwen35RuntimeAdapter)


def test_runtime_rejects_unknown_adapter_before_model_load(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"architecture_adapter": "unqualified_test_model"})
    )

    with pytest.raises(ValueError, match="unsupported architecture adapter"):
        runtime_adapter_for_artifact(tmp_path)
