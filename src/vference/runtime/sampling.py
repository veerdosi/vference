from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

TokenSampler = Callable[[mx.array], mx.array] | None


def make_token_sampler(*, temperature: float, top_p: float, top_k: int) -> TokenSampler:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top-p must be greater than zero and at most one")
    if top_k < 0:
        raise ValueError("top-k must be non-negative")
    if temperature == 0:
        return None
    return make_sampler(temp=temperature, top_p=top_p, top_k=top_k)


def select_token(logits: mx.array, sampler: TokenSampler) -> int:
    last_logits = logits[:, -1, :]
    if sampler is None:
        return int(mx.argmax(last_logits, axis=-1).item())
    logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
    return int(sampler(logprobs).item())
