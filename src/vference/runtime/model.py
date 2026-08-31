from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
from mlx_lm.utils import load_tokenizer

from .expert_store import StreamingSwitchGLU, SynchronousExpertStore


def load_streaming_qwen(
    artifact: Path, *, cache_capacity: int = 64
) -> tuple[Model, object, SynchronousExpertStore]:
    """Load the resident text core and attach exact synchronous expert streaming."""
    artifact = artifact.resolve()
    config = json.loads((artifact / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    store = SynchronousExpertStore(artifact, capacity=cache_capacity)
    for layer_id, layer in enumerate(model.language_model.layers):
        layer.mlp.switch_mlp = StreamingSwitchGLU(layer_id, store)
    gc.collect()

    weights = mx.load(artifact / "core.safetensors")
    weights = model.sanitize(weights)
    quantization = config["quantization"]

    def should_quantize(path: str, module: nn.Module) -> bool:
        return hasattr(module, "to_quantized") and f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization.get("mode", "affine"),
        class_predicate=should_quantize,
    )
    model.eval()
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(
        artifact,
        {"trust_remote_code": True},
        eos_token_ids=config.get("eos_token_id"),
    )
    return model, tokenizer, store
