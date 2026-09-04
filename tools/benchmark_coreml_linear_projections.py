#!/usr/bin/env python3
"""Benchmark a compressed Core ML GatedDeltaNet projection probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

from build_coreml_linear_projections import PROJECTIONS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("--runner-source", type=Path, default=Path("tools/coreml_runner.swift"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _qmm(x: mx.array, weights: dict[str, mx.array], prefix: str) -> mx.array:
    return mx.quantized_matmul(
        x,
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.scales"],
        weights[f"{prefix}.biases"],
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
    )


def _mlx_benchmark(
    manifest: dict[str, object], input_values: np.ndarray, warmup: int, iterations: int
) -> dict[str, object]:
    artifact = Path(str(manifest["artifact"]))
    layer = int(manifest["layer"])
    weights = mx.load(artifact / "core.safetensors")
    prefix = f"language_model.model.layers.{layer}.linear_attn"

    def execute(x: mx.array) -> mx.array:
        return mx.concatenate(
            [_qmm(x, weights, f"{prefix}.{name}") for name in PROJECTIONS], axis=-1
        )

    result: dict[str, object] = {}
    for name, dtype in (("bfloat16", mx.bfloat16), ("float16", mx.float16)):
        x = mx.array(input_values).astype(dtype)
        for _ in range(warmup):
            mx.eval(execute(x))
        latencies: list[float] = []
        output = None
        for _ in range(iterations):
            started = time.perf_counter_ns()
            output = execute(x)
            mx.eval(output)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000_000)
        assert output is not None
        result[name] = {
            "latency_seconds": _summary(latencies),
            "output_fp16": np.asarray(output.astype(mx.float16)).reshape(-1),
        }
    return result


def _error(output: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    difference = np.abs(output.astype(np.float32) - reference)
    return {
        "exact_fp16": bool(np.array_equal(output, reference)),
        "max_absolute": float(difference.max()),
        "mean_absolute": float(difference.mean()),
        "cosine_similarity": float(
            np.dot(output.astype(np.float32), reference)
            / (np.linalg.norm(output.astype(np.float32)) * np.linalg.norm(reference))
        ),
    }


def main() -> None:
    args = _parse_args()
    probe_dir = args.probe_dir.resolve()
    manifest = json.loads((probe_dir / "probe.json").read_text())
    package = probe_dir / manifest["package"]
    input_path = probe_dir / manifest["input"]
    reference_path = probe_dir / manifest["reference_output"]
    shape = (1, int(manifest["sequence_length"]), int(manifest["hidden_size"]))
    input_values = np.fromfile(input_path, dtype=np.float16).reshape(shape)
    reference = np.fromfile(reference_path, dtype=np.float16).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="vference-coreml-projections-") as temporary:
        runner = Path(temporary) / "coreml_runner"
        subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-parse-as-library",
                "-O",
                "-framework",
                "CoreML",
                str(args.runner_source.resolve()),
                "-o",
                str(runner),
            ],
            check=True,
        )
        coreml: dict[str, object] = {}
        for units in ("cpu_only", "cpu_gpu", "cpu_ne", "all"):
            output_path = Path(temporary) / f"output-{units}.raw"
            completed = subprocess.run(
                [
                    str(runner),
                    str(package),
                    str(input_path),
                    str(output_path),
                    units,
                    str(manifest["sequence_length"]),
                    str(manifest["hidden_size"]),
                    str(args.warmup),
                    str(args.iterations),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            item = json.loads(completed.stdout)
            output = np.fromfile(output_path, dtype=np.float16).astype(np.float32)
            item["reference_error"] = _error(output, reference)
            item["reference_error"]["output_sha256"] = _sha256(output_path)
            coreml[units] = item

    mlx_raw = _mlx_benchmark(manifest, input_values, args.warmup, args.iterations)
    mlx = {
        name: {
            "latency_seconds": item["latency_seconds"],
            "reference_error": _error(item["output_fp16"].astype(np.float32), reference),
        }
        for name, item in mlx_raw.items()
    }
    artifact_manifest = json.loads((Path(manifest["artifact"]) / "manifest.json").read_text())
    result = {
        "experiment": "Compressed Core ML GatedDeltaNet input-projection probe",
        "recorded_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "artifact": {
            "repository": artifact_manifest["source"]["repository"],
            "model_revision": artifact_manifest["source"]["revision"],
            "expert_pack_sha256": artifact_manifest["output_sha256"]["experts.pack"],
            "manifest_sha256": manifest["artifact_manifest_sha256"],
        },
        "environment": {
            "hardware": "MacBook Air Mac14,2, Apple M2, 8 GiB unified memory",
            "operating_system": platform.platform(),
            "coremltools": manifest["coremltools_version"],
        },
        "workload": {
            "layer": manifest["layer"],
            "shape": list(shape),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "input_source": manifest["input_source"],
            "input_sha256": manifest["input_sha256"],
            "output_size": manifest["output_size"],
        },
        "coreml_artifact": {
            "representation": manifest["representation"],
            "package_bytes": manifest["package_bytes"],
        },
        "mlx": mlx,
        "coreml": coreml,
        "decision": {
            "status": "microbenchmark only; not integrated",
            "quality": "Any route or output change in downstream qualification rejects the candidate.",
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
