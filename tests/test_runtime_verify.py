from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from vference.runtime.verify import verify_prefill_chunk_invariance
from vference.runtime.generate import (
    _detach_last_logits,
    _decode_resize_admission,
    _make_token_sampler,
    _select_token,
    _store_stats_delta,
    _validate_prefill_chunk_size,
)


def test_greedy_and_seeded_sampling_are_reproducible() -> None:
    logits = mx.array([[[0.0, 0.5, 1.0, 1.5]]])
    assert _select_token(logits, None) == 3

    sampler = _make_token_sampler(temperature=0.8, top_p=0.9, top_k=3)
    mx.random.seed(17)
    first = [_select_token(logits, sampler) for _ in range(16)]
    mx.random.seed(17)
    second = [_select_token(logits, sampler) for _ in range(16)]
    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": -0.1, "top_p": 1.0, "top_k": 0}, "temperature"),
        ({"temperature": 0.8, "top_p": 0.0, "top_k": 0}, "top-p"),
        ({"temperature": 0.8, "top_p": 1.1, "top_k": 0}, "top-p"),
        ({"temperature": 0.8, "top_p": 1.0, "top_k": -1}, "top-k"),
    ],
)
def test_sampler_validation(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _make_token_sampler(**kwargs)


def test_detach_last_logits_preserves_bfloat16_bits() -> None:
    logits = mx.arange(24).reshape(1, 3, 8).astype(mx.bfloat16)
    detached = _detach_last_logits(logits)
    assert detached.shape == (1, 1, 8)
    assert np.array_equal(
        np.asarray(detached.view(mx.uint16)),
        np.asarray(logits[:, -1:].view(mx.uint16)),
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "chunk_sizes", "continuation_tokens", "message"),
    [
        (0, (256, 512), 1, "prompt token count"),
        (1, (256, 512), 0, "continuation token count"),
        (1, (512,), 1, "at least two"),
        (1, (0, 512), 1, "positive prefill chunk sizes"),
    ],
)
def test_prefill_chunk_verify_rejects_invalid_workloads(
    prompt_tokens: int,
    chunk_sizes: tuple[int, ...],
    continuation_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_prefill_chunk_invariance(
            Path("unused"),
            prompt="x",
            prompt_token_count=prompt_tokens,
            chunk_sizes=chunk_sizes,
            continuation_tokens=continuation_tokens,
        )


def test_qwen_prefill_chunk_guard_allows_qualified_or_single_chunk() -> None:
    _validate_prefill_chunk_size(8192, 512, False)
    _validate_prefill_chunk_size(256, 256, False)


def test_qwen_prefill_chunk_guard_requires_explicit_override() -> None:
    with pytest.raises(ValueError, match="not output-qualified"):
        _validate_prefill_chunk_size(7936, 256, False)
    _validate_prefill_chunk_size(7936, 256, True)


def test_store_stats_delta_separates_phase_counters() -> None:
    before = {
        "hits": 10,
        "misses": 20,
        "bytes_read": 30,
        "timing_seconds": {
            "pread": 1.0,
            "materialize": 0.0,
            "router_and_graph_wait": 2.0,
            "execute_total": 3.0,
        },
        "prefetch": {
            "submitted": 4,
            "bytes_read": 5,
            "useful": 1,
            "total_physical_bytes": 35,
        },
        "per_layer": [{"hits": 2, "misses": 3, "read_seconds": 0.5, "materialize_seconds": 0.0}],
    }
    after = {
        "hits": 17,
        "misses": 29,
        "bytes_read": 41,
        "timing_seconds": {
            "pread": 1.4,
            "materialize": 0.0,
            "router_and_graph_wait": 2.7,
            "execute_total": 4.2,
        },
        "prefetch": {
            "submitted": 6,
            "bytes_read": 8,
            "useful": 2,
            "total_physical_bytes": 49,
        },
        "per_layer": [{"hits": 7, "misses": 7, "read_seconds": 0.8, "materialize_seconds": 0.0}],
    }
    delta = _store_stats_delta(before, after)
    assert delta["hits"] == 7
    assert delta["misses"] == 9
    assert delta["bytes_read"] == 11
    assert delta["timing_seconds"]["router_and_graph_wait"] == pytest.approx(0.7)
    assert delta["prefetch"]["submitted"] == 2
    assert delta["prefetch"]["total_physical_bytes"] == 14
    assert delta["per_layer"][0]["misses"] == 4


def test_decode_resize_admission_preserves_headroom() -> None:
    common = {
        "resident_bytes": 1_945_100_424,
        "model_state_bytes": 232_161_280,
        "record_size": 1_769_472,
        "prefill_capacity": 320,
        "runtime_reserve_bytes": 18 * 1024**2,
        "budget_bytes": 3 * 1024**3,
    }
    assert _decode_resize_admission(decode_capacity=560, **common)["admitted"]
    assert not _decode_resize_admission(decode_capacity=640, **common)["admitted"]
