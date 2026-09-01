from __future__ import annotations

import gc

import mlx.core as mx

from vference.native import extension


def test_owned_pool_releases_allocator_buffer() -> None:
    mx.clear_cache()
    baseline = mx.get_active_memory()
    pool = extension().owned_zeros([1024, 1024], "uint32")
    assert mx.get_active_memory() == baseline + 4 * 1024**2

    del pool
    gc.collect()
    mx.clear_cache()
    assert mx.get_active_memory() == baseline
