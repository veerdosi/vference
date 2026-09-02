from vference.runtime.admission import (
    estimate_qwen35_admission,
    fit_expert_capacity,
    minimum_expert_capacity,
    qwen35_state_bytes,
)


CONFIG = {
    "text_config": {
        "num_hidden_layers": 40,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
    }
}


def test_qwen_state_estimate_matches_measured_cache_allocations() -> None:
    assert qwen35_state_bytes(CONFIG, 1) == 69_632_000
    assert qwen35_state_bytes(CONFIG, 513) == 80_117_760
    assert qwen35_state_bytes(CONFIG, 8192) == 232_161_280


def test_admission_rejects_the_measured_unsafe_cache_budget() -> None:
    budget = 3 * 1024**3
    safe = estimate_qwen35_admission(
        CONFIG,
        resident_bytes=1_945_100_000,
        total_tokens=8192,
        prefill_chunk_size=512,
        budget_bytes=budget,
    )
    unsafe = estimate_qwen35_admission(
        CONFIG,
        resident_bytes=2_511_300_000,
        total_tokens=8192,
        prefill_chunk_size=512,
        budget_bytes=budget,
    )
    assert safe.admitted
    assert not unsafe.admitted

    with_prefetch = estimate_qwen35_admission(
        CONFIG,
        resident_bytes=1_945_100_000,
        total_tokens=8192,
        prefill_chunk_size=512,
        budget_bytes=budget,
        runtime_reserve_bytes=18 * 1024**2,
    )
    oversized_runtime = estimate_qwen35_admission(
        CONFIG,
        resident_bytes=1_945_100_000,
        total_tokens=8192,
        prefill_chunk_size=512,
        budget_bytes=budget,
        runtime_reserve_bytes=64 * 1024**2,
    )
    assert with_prefetch.admitted
    assert not oversized_runtime.admitted


def test_capacity_planner_shrinks_only_the_expert_pool() -> None:
    record_size = 1_769_472
    current_capacity = 320
    fixed_resident = 1_378_869_384
    plan = fit_expert_capacity(
        budget_bytes=3 * 1024**3,
        resident_bytes=fixed_resident + current_capacity * record_size,
        current_capacity=current_capacity,
        requested_capacity=current_capacity,
        minimum_capacity=8,
        record_size=record_size,
        non_pool_reserve_bytes=399_933_440 + 960 * 1024**2 + 18 * 1024**2,
    )

    assert plan.admitted
    assert plan.selected_capacity == 235
    assert plan.fixed_resident_bytes == fixed_resident
    assert plan.estimated_peak_bytes <= plan.budget_bytes


def test_capacity_planner_rejects_less_than_exact_working_set() -> None:
    plan = fit_expert_capacity(
        budget_bytes=100,
        resident_bytes=80,
        current_capacity=8,
        requested_capacity=8,
        minimum_capacity=8,
        record_size=10,
        non_pool_reserve_bytes=21,
    )

    assert not plan.admitted
    assert plan.maximum_capacity == 7
    assert plan.selected_capacity == 0


def test_minimum_capacity_accounts_for_layer_partitions() -> None:
    assert minimum_expert_capacity(route_width=8, layer_count=40, policy="global") == 8
    assert minimum_expert_capacity(route_width=8, layer_count=40, policy="demand") == 8
    assert minimum_expert_capacity(route_width=8, layer_count=40, policy="layer") == 320
