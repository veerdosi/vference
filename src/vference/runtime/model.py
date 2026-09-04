from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
from mlx_lm.utils import load_tokenizer

from .admission import AdmissionEstimate, estimate_qwen35_admission
from .expert_store import (
    StableSlotExpertStore,
    StreamingSwitchGLU,
    SynchronousExpertStore,
)
from .integrity import verify_artifact_integrity


class RuntimeArchitectureAdapter(Protocol):
    """Architecture-owned behavior required by the generic generation loop."""

    name: str

    def route_shape(self, config: dict[str, object]) -> tuple[int, int]: ...

    def validate_prefill_chunk(
        self, prompt_tokens: int, chunk_size: int, allow_unqualified: bool
    ) -> None: ...

    def estimate_admission(
        self,
        config: dict[str, object],
        *,
        resident_bytes: int,
        total_tokens: int,
        prefill_chunk_size: int,
        budget_bytes: int,
        runtime_reserve_bytes: int,
    ) -> AdmissionEstimate: ...

    def make_cache(self, model: object) -> list[object]: ...


@dataclass(frozen=True)
class Qwen35RuntimeAdapter:
    """Qualified runtime policy for the first Qwen3.5 MoE adapter."""

    name: str = "qwen3_5_moe"
    qualified_prefill_chunk_size: int = 512

    def route_shape(self, config: dict[str, object]) -> tuple[int, int]:
        text = config.get("text_config", config)
        assert isinstance(text, dict)
        return int(text["num_experts_per_tok"]), int(text["num_hidden_layers"])

    def validate_prefill_chunk(
        self, prompt_tokens: int, chunk_size: int, allow_unqualified: bool
    ) -> None:
        if (
            prompt_tokens > chunk_size
            and chunk_size != self.qualified_prefill_chunk_size
            and not allow_unqualified
        ):
            raise ValueError(
                f"multi-chunk Qwen3.5 prefill size {chunk_size} is not output-qualified; "
                f"use {self.qualified_prefill_chunk_size} or pass the explicit "
                "experimental override"
            )

    def estimate_admission(
        self,
        config: dict[str, object],
        *,
        resident_bytes: int,
        total_tokens: int,
        prefill_chunk_size: int,
        budget_bytes: int,
        runtime_reserve_bytes: int,
    ) -> AdmissionEstimate:
        return estimate_qwen35_admission(
            config,
            resident_bytes=resident_bytes,
            total_tokens=total_tokens,
            prefill_chunk_size=prefill_chunk_size,
            budget_bytes=budget_bytes,
            runtime_reserve_bytes=runtime_reserve_bytes,
        )

    def make_cache(self, model: object) -> list[object]:
        return model.make_cache()  # type: ignore[attr-defined, no-any-return]


_RUNTIME_ADAPTERS: dict[str, RuntimeArchitectureAdapter] = {
    "qwen3_5_moe": Qwen35RuntimeAdapter(),
}


@dataclass
class LoadedStreamingModel:
    """Resident model/store bundle reusable across independent requests."""

    artifact: Path
    model: Model
    tokenizer: object
    store: SynchronousExpertStore
    adapter: RuntimeArchitectureAdapter

    def close(self) -> None:
        self.store.close()


def runtime_adapter_for_artifact(artifact: Path) -> RuntimeArchitectureAdapter:
    manifest = json.loads((artifact / "manifest.json").read_text())
    name = manifest.get("architecture_adapter")
    try:
        return _RUNTIME_ADAPTERS[str(name)]
    except KeyError as error:
        raise ValueError(
            f"runtime artifact requests unsupported architecture adapter: {name!r}"
        ) from error


def load_streaming_qwen(
    artifact: Path,
    *,
    cache_capacity: int = 64,
    nocache: bool = False,
    trace_routes: bool = False,
    store_kind: str = "python",
    cache_policy: str = "global",
    prefetch_policy: str = "none",
    prefetch_budget: int = 1,
    prefetch_min_observations: int = 8,
    demand_workers: int = 1,
) -> tuple[Model, object, SynchronousExpertStore]:
    """Load the resident text core and attach exact synchronous expert streaming."""
    artifact = artifact.resolve()
    integrity = verify_artifact_integrity(artifact)
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
        store_kwargs["prefetch_budget"] = prefetch_budget
        store_kwargs["prefetch_min_observations"] = prefetch_min_observations
        store_kwargs["demand_workers"] = demand_workers
    elif cache_policy != "global":
        raise ValueError("the Python reference store only supports global LRU")
    elif prefetch_policy != "none":
        raise ValueError("the Python reference store does not support prefetch")
    elif demand_workers != 1:
        raise ValueError("the Python reference store does not support parallel demand")
    store = store_type(artifact, **store_kwargs)
    store.artifact_integrity = integrity
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


def load_streaming_model(
    artifact: Path,
    **kwargs: object,
) -> LoadedStreamingModel:
    """Resolve the artifact's adapter, then load its qualified execution path."""
    artifact = artifact.resolve()
    adapter = runtime_adapter_for_artifact(artifact)
    if adapter.name != "qwen3_5_moe":
        raise ValueError(f"no model loader is registered for adapter: {adapter.name}")
    model, tokenizer, store = load_streaming_qwen(artifact, **kwargs)
    return LoadedStreamingModel(artifact, model, tokenizer, store, adapter)
