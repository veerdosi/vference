from pathlib import Path

import pytest

from vference.runtime.verify import verify_prefill_chunk_invariance


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
