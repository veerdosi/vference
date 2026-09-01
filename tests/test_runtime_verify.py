from pathlib import Path

import pytest

from vference.runtime.verify import verify_prefill_chunk_invariance
from vference.runtime.generate import _validate_prefill_chunk_size


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
