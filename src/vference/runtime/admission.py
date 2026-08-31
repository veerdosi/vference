from __future__ import annotations

import math
from dataclasses import asdict, dataclass


MIB = 1024**2


@dataclass(frozen=True)
class AdmissionEstimate:
    admitted: bool
    budget_bytes: int
    estimated_peak_bytes: int
    resident_bytes: int
    model_state_bytes: int
    prefill_transient_reserve_bytes: int
    total_tokens: int
    prefill_chunk_size: int

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def qwen35_state_bytes(config: dict[str, object], total_tokens: int) -> int:
    """Derive allocated DeltaNet and KV state bytes for batch-1 BF16 Qwen3.5."""
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    interval = int(text["full_attention_interval"])
    attention_layers = layers // interval
    linear_layers = layers - attention_layers

    key_heads = int(text["linear_num_key_heads"])
    value_heads = int(text["linear_num_value_heads"])
    key_dim = int(text["linear_key_head_dim"])
    value_dim = int(text["linear_value_head_dim"])
    conv_kernel = int(text["linear_conv_kernel_dim"])
    conv_dim = 2 * key_heads * key_dim + value_heads * value_dim
    linear_per_layer = (
        (conv_kernel - 1) * conv_dim * 2
        + value_heads * key_dim * value_dim * 4
    )

    kv_heads = int(text["num_key_value_heads"])
    head_dim = int(text["head_dim"])
    allocated_tokens = math.ceil(total_tokens / 256) * 256
    kv_per_layer = 2 * kv_heads * allocated_tokens * head_dim * 2
    return linear_layers * linear_per_layer + attention_layers * kv_per_layer


def estimate_qwen35_admission(
    config: dict[str, object],
    *,
    resident_bytes: int,
    total_tokens: int,
    prefill_chunk_size: int,
    budget_bytes: int,
) -> AdmissionEstimate:
    """Conservative measured-profile admission estimate for the first adapter.

    The transient reserve is fitted above the local 512-token passing peak and
    the rejected 2,048-token single-chunk peak. It is deliberately conservative
    and must be re-profiled when the model adapter or MLX revision changes.
    """
    state_bytes = qwen35_state_bytes(config, total_tokens)
    transient_bytes = 576 * MIB + math.ceil(prefill_chunk_size * 384 * MIB / 512)
    peak_bytes = resident_bytes + state_bytes + transient_bytes
    return AdmissionEstimate(
        admitted=peak_bytes <= budget_bytes,
        budget_bytes=budget_bytes,
        estimated_peak_bytes=peak_bytes,
        resident_bytes=resident_bytes,
        model_state_bytes=state_bytes,
        prefill_transient_reserve_bytes=transient_bytes,
        total_tokens=total_tokens,
        prefill_chunk_size=prefill_chunk_size,
    )
