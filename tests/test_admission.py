from vference.runtime.admission import estimate_qwen35_admission, qwen35_state_bytes


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
