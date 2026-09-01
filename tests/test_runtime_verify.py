from pathlib import Path

import pytest

from vference.runtime.verify import verify_prefill_chunk_invariance
from vference.runtime.generate import _store_stats_delta, _validate_prefill_chunk_size


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
