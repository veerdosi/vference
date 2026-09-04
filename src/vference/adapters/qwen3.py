from __future__ import annotations

from vference.artifacts.safetensors import TensorEntry

from .base import ArtifactAdapter, ExpertComponent, ExpertLayer


class Qwen3MoeAdapter(ArtifactAdapter):
    """Artifact adapter for the earlier dense-attention Qwen3 MoE family."""

    name = "qwen3_moe"
    _component_suffixes = (
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

    def __init__(self, *, layer_ids: tuple[int, ...], experts: int) -> None:
        if not layer_ids:
            raise ValueError("Qwen3 MoE adapter needs at least one sparse layer")
        self.layer_ids = layer_ids
        self.experts = experts

    @classmethod
    def from_config(cls, config: dict[str, object]) -> Qwen3MoeAdapter:
        layers = int(config["num_hidden_layers"])
        experts = int(config["num_experts"])
        sparse_step = int(config.get("decoder_sparse_step", 1))
        mlp_only = {int(value) for value in config.get("mlp_only_layers", [])}  # type: ignore[arg-type]
        layer_ids = tuple(
            layer_id
            for layer_id in range(layers)
            if layer_id not in mlp_only
            and experts > 0
            and (layer_id + 1) % sparse_step == 0
        )
        return cls(layer_ids=layer_ids, experts=experts)

    def expert_layers(self, entries: dict[str, TensorEntry]) -> tuple[ExpertLayer, ...]:
        result: list[ExpertLayer] = []
        used: set[str] = set()
        for layer_id in self.layer_ids:
            prefix = f"model.layers.{layer_id}.mlp.switch_mlp."
            components: list[ExpertComponent] = []
            record_offset = 0
            for suffix in self._component_suffixes:
                name = prefix + suffix
                if name not in entries:
                    raise ValueError(f"missing Qwen3 expert tensor: {name}")
                entry = entries[name]
                if not entry.shape or entry.shape[0] != self.experts:
                    raise ValueError(
                        f"{name}: expected leading expert dimension {self.experts}, "
                        f"got {entry.shape}"
                    )
                if entry.nbytes % self.experts:
                    raise ValueError(f"{name}: payload cannot be divided by expert count")
                bytes_per_expert = entry.nbytes // self.experts
                components.append(
                    ExpertComponent(
                        name=name,
                        dtype=entry.dtype,
                        shape=entry.shape,
                        source=entry,
                        bytes_per_expert=bytes_per_expert,
                        record_offset=record_offset,
                    )
                )
                record_offset += bytes_per_expert
                used.add(name)
            if record_offset % 4096:
                raise ValueError(
                    f"layer {layer_id}: expert record size {record_offset} is not 4 KiB aligned"
                )
            result.append(
                ExpertLayer(
                    layer_id=layer_id,
                    expert_count=self.experts,
                    record_size=record_offset,
                    components=tuple(components),
                )
            )

        routed = {name for name in entries if ".mlp.switch_mlp." in name}
        if routed != used:
            unknown = sorted(routed - used)
            raise ValueError(f"unrecognized routed expert tensors: {unknown[:5]}")
        return tuple(result)
