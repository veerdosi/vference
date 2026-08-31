from __future__ import annotations

import fcntl
import mmap
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any


F_NOCACHE = 48
ALIGNMENT = 4096


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def probe_storage(
    path: Path,
    *,
    read_size: int,
    reads: int,
    pattern: str,
    nocache: bool,
    seed: int,
) -> dict[str, Any]:
    if read_size <= 0 or read_size % ALIGNMENT:
        raise ValueError(f"read_size must be a positive multiple of {ALIGNMENT}")
    file_size = path.stat().st_size
    max_block = file_size // read_size
    if max_block < 2:
        raise ValueError("file is too small for the requested read size")
    rng = random.Random(seed)
    if pattern == "random":
        blocks = [rng.randrange(max_block) for _ in range(reads)]
    elif pattern == "sequential":
        start = rng.randrange(max(1, max_block - reads))
        blocks = [(start + index) % max_block for index in range(reads)]
    else:
        raise ValueError(f"unsupported pattern: {pattern}")

    fd = os.open(path, os.O_RDONLY)
    buffer = mmap.mmap(-1, read_size)
    view = memoryview(buffer)
    latencies_ms: list[float] = []
    checksum = 0
    total_bytes = 0
    try:
        if nocache:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        started = time.perf_counter_ns()
        for block in blocks:
            offset = block * read_size
            before = time.perf_counter_ns()
            count = os.preadv(fd, [view], offset)
            after = time.perf_counter_ns()
            if count != read_size:
                raise OSError(f"short read at {offset}: {count} != {read_size}")
            checksum = (checksum + view[0] + view[count - 1]) & 0xFFFFFFFF
            total_bytes += count
            latencies_ms.append((after - before) / 1_000_000)
        elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000
    finally:
        view.release()
        buffer.close()
        os.close(fd)

    return {
        "file": str(path),
        "file_size_bytes": file_size,
        "mode": "nocache" if nocache else "buffered",
        "pattern": pattern,
        "read_size_bytes": read_size,
        "reads": reads,
        "bytes_read": total_bytes,
        "elapsed_s": elapsed_s,
        "throughput_bytes_s": total_bytes / elapsed_s,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "p50": _percentile(latencies_ms, 0.50),
            "p95": _percentile(latencies_ms, 0.95),
            "p99": _percentile(latencies_ms, 0.99),
            "max": max(latencies_ms),
        },
        "checksum": checksum,
        "seed": seed,
    }
