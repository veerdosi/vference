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
    runtime_reserve_bytes: int
    prefill_transient_reserve_bytes: int
    total_tokens: int
    prefill_chunk_size: int

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertCapacityPlan:
    admitted: bool
    budget_bytes: int
    requested_capacity: int
    selected_capacity: int
    maximum_capacity: int
    minimum_capacity: int
    current_capacity: int
    record_size: int
    fixed_resident_bytes: int
    non_pool_reserve_bytes: int
    estimated_peak_bytes: int

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def fit_expert_capacity(
    *,
    budget_bytes: int,
    resident_bytes: int,
    current_capacity: int,
    requested_capacity: int,
    minimum_capacity: int,
    record_size: int,
    non_pool_reserve_bytes: int,
) -> ExpertCapacityPlan:
    """Fit a bounded expert pool after reserving model state and transients.

    ``resident_bytes`` includes the currently allocated pool. Subtracting its
    exact fixed-record payload makes the planner architecture-neutral; the
    adapter supplies its simultaneous exact working-set minimum.
    """
    if budget_bytes < 1 or record_size < 1:
        raise ValueError("budget and expert record size must be positive")
    if current_capacity < 1 or requested_capacity < 1 or minimum_capacity < 1:
        raise ValueError("expert capacities must be positive")
    if non_pool_reserve_bytes < 0:
        raise ValueError("non-pool reserve cannot be negative")
    current_pool_bytes = current_capacity * record_size
    fixed_resident_bytes = resident_bytes - current_pool_bytes
    if fixed_resident_bytes < 0:
        raise ValueError("resident bytes are smaller than the current expert pool")
    available_pool_bytes = budget_bytes - fixed_resident_bytes - non_pool_reserve_bytes
    maximum_capacity = max(0, available_pool_bytes // record_size)
    selected_capacity = min(requested_capacity, maximum_capacity)
    admitted = selected_capacity >= minimum_capacity
    if not admitted:
        selected_capacity = 0
    estimated_peak_bytes = (
        fixed_resident_bytes
        + selected_capacity * record_size
        + non_pool_reserve_bytes
    )
    return ExpertCapacityPlan(
        admitted=admitted,
        budget_bytes=budget_bytes,
        requested_capacity=requested_capacity,
        selected_capacity=selected_capacity,
        maximum_capacity=maximum_capacity,
        minimum_capacity=minimum_capacity,
        current_capacity=current_capacity,
        record_size=record_size,
        fixed_resident_bytes=fixed_resident_bytes,
        non_pool_reserve_bytes=non_pool_reserve_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
    )


def minimum_expert_capacity(*, route_width: int, layer_count: int, policy: str) -> int:
    """Return the slots required for one exact route under a cache partition."""
    if route_width < 1 or layer_count < 1:
        raise ValueError("route width and layer count must be positive")
    if policy in {"global", "demand"}:
        return route_width
    if policy == "layer":
        return route_width * layer_count
    raise ValueError(f"unknown expert cache policy: {policy}")


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
    runtime_reserve_bytes: int = 0,
) -> AdmissionEstimate:
    """Conservative measured-profile admission estimate for the first adapter.

    The transient reserve is fitted above the local 512-token passing peak and
    the rejected 2,048-token single-chunk peak. It is deliberately conservative
    and must be re-profiled when the model adapter or MLX revision changes.
    """
    state_bytes = qwen35_state_bytes(config, total_tokens)
    transient_bytes = 576 * MIB + math.ceil(prefill_chunk_size * 384 * MIB / 512)
    peak_bytes = resident_bytes + state_bytes + transient_bytes + runtime_reserve_bytes
    return AdmissionEstimate(
        admitted=peak_bytes <= budget_bytes,
        budget_bytes=budget_bytes,
        estimated_peak_bytes=peak_bytes,
        resident_bytes=resident_bytes,
        model_state_bytes=state_bytes,
        runtime_reserve_bytes=runtime_reserve_bytes,
        prefill_transient_reserve_bytes=transient_bytes,
        total_tokens=total_tokens,
        prefill_chunk_size=prefill_chunk_size,
    )
