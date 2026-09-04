"""Architecture adapters for converting source checkpoints."""

from .base import ArtifactAdapter
from .qwen3 import Qwen3MoeAdapter
from .qwen35 import Qwen35MoeAdapter


def artifact_adapter_for_config(config: dict[str, object]) -> ArtifactAdapter:
    model_type = config.get("model_type")
    if model_type == "qwen3_5_moe":
        text = config.get("text_config", config)
        assert isinstance(text, dict)
        return Qwen35MoeAdapter(
            layers=int(text["num_hidden_layers"]), experts=int(text["num_experts"])
        )
    if model_type == "qwen3_moe":
        return Qwen3MoeAdapter.from_config(config)
    raise ValueError(f"unsupported artifact architecture: {model_type!r}")


__all__ = [
    "ArtifactAdapter",
    "Qwen3MoeAdapter",
    "Qwen35MoeAdapter",
    "artifact_adapter_for_config",
]
