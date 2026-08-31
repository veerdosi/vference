from __future__ import annotations

from dataclasses import dataclass

from vference.artifacts.safetensors import TensorEntry


@dataclass(frozen=True)
class ExpertComponent:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source: TensorEntry
    bytes_per_expert: int
    record_offset: int


@dataclass(frozen=True)
class ExpertLayer:
    layer_id: int
    expert_count: int
    record_size: int
    components: tuple[ExpertComponent, ...]


class ArtifactAdapter:
    """Boundary between architecture-specific tensor names and generic storage."""

    name: str

    def expert_layers(self, entries: dict[str, TensorEntry]) -> tuple[ExpertLayer, ...]:
        raise NotImplementedError
