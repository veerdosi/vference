from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
from mlx_lm.utils import load_tokenizer

from .expert_store import (
    StableSlotExpertStore,
    StreamingSwitchGLU,
    SynchronousExpertStore,
)


def load_streaming_qwen(
    artifact: Path,
    *,
    cache_capacity: int = 64,
    nocache: bool = False,
    trace_routes: bool = False,
    store_kind: str = "python",
    cache_policy: str = "global",
    prefetch_policy: str = "none",
) -> tuple[Model, object, SynchronousExpertStore]:
    """Load the resident text core and attach exact synchronous expert streaming."""
    artifact = artifact.resolve()
    config = json.loads((artifact / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    store_types = {
        "python": SynchronousExpertStore,
        "stable": StableSlotExpertStore,
    }
    try:
        store_type = store_types[store_kind]
    except KeyError as error:
        raise ValueError(f"unknown expert store: {store_kind}") from error
    store_kwargs = {
        "capacity": cache_capacity,
        "nocache": nocache,
        "trace_routes": trace_routes,
    }
    if store_kind == "stable":
        store_kwargs["cache_policy"] = cache_policy
        store_kwargs["prefetch_policy"] = prefetch_policy
    elif cache_policy != "global":
        raise ValueError("the Python reference store only supports global LRU")
    elif prefetch_policy != "none":
        raise ValueError("the Python reference store does not support prefetch")
    store = store_type(artifact, **store_kwargs)
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
