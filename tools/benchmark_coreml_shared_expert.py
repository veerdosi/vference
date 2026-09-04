#!/usr/bin/env python3
"""Benchmark a generated shared-expert Core ML probe against MLX."""

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
import mlx.nn as nn
import numpy as np


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


def _latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _quantized_linear(x: mx.array, weights: dict[str, mx.array], prefix: str) -> mx.array:
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
    prefix = f"language_model.model.layers.{layer}.mlp"

    def execute(x: mx.array) -> mx.array:
        gate = _quantized_linear(x, weights, f"{prefix}.shared_expert.gate_proj")
        up = _quantized_linear(x, weights, f"{prefix}.shared_expert.up_proj")
        hidden = nn.silu(gate) * up
        down = _quantized_linear(hidden, weights, f"{prefix}.shared_expert.down_proj")
        scale = mx.sigmoid(_quantized_linear(x, weights, f"{prefix}.shared_expert_gate"))
        return down * scale

    results: dict[str, object] = {}
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
        results[name] = {
            "latency_seconds": _latency_summary(latencies),
            "output_fp16": np.asarray(output.astype(mx.float16)),
        }
    return results


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    probe_dir = args.probe_dir.resolve()
    manifest = json.loads((probe_dir / "probe.json").read_text())
    package = probe_dir / manifest["package"]
    input_path = probe_dir / manifest["input"]
    reference_path = probe_dir / manifest["reference_output"]
    input_shape = (1, int(manifest["sequence_length"]), int(manifest["hidden_size"]))
    input_values = np.fromfile(input_path, dtype=np.float16).reshape(input_shape)
    reference = np.fromfile(reference_path, dtype=np.float16).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="vference-coreml-runner-") as temporary:
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
        coreml_results: dict[str, object] = {}
        for compute_units in ("cpu_only", "cpu_gpu", "cpu_ne", "all"):
            output_path = Path(temporary) / f"output-{compute_units}.raw"
            completed = subprocess.run(
                [
                    str(runner),
                    str(package),
                    str(input_path),
                    str(output_path),
                    compute_units,
                    str(manifest["sequence_length"]),
                    str(manifest["hidden_size"]),
                    str(args.warmup),
                    str(args.iterations),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            output = np.fromfile(output_path, dtype=np.float16).astype(np.float32)
            error = np.abs(output - reference)
            result["reference_error"] = {
                "exact_fp16": bool(np.array_equal(output, reference)),
                "max_absolute": float(error.max()),
                "mean_absolute": float(error.mean()),
                "cosine_similarity": float(
                    np.dot(output, reference) / (np.linalg.norm(output) * np.linalg.norm(reference))
                ),
                "output_sha256": _sha256(output_path),
            }
            coreml_results[compute_units] = result

    mlx_results = _mlx_benchmark(manifest, input_values, args.warmup, args.iterations)
    serializable_mlx = {
        name: {
            "latency_seconds": result["latency_seconds"],
            "reference_error": {
                "exact_fp16": bool(
                    np.array_equal(result["output_fp16"].astype(np.float32).reshape(-1), reference)
                ),
                "max_absolute": float(
                    np.abs(result["output_fp16"].astype(np.float32).reshape(-1) - reference).max()
                ),
                "mean_absolute": float(
                    np.abs(result["output_fp16"].astype(np.float32).reshape(-1) - reference).mean()
                ),
            },
        }
        for name, result in mlx_results.items()
    }
    artifact_manifest = json.loads((Path(manifest["artifact"]) / "manifest.json").read_text())
    benchmark = {
        "experiment": "Standalone Core ML shared-expert placement and latency probe",
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
            "swift": subprocess.run(
                ["xcrun", "swift", "--version"], check=True, capture_output=True, text=True
            ).stdout.splitlines()[0],
        },
        "workload": {
            "kind": (
                "synthetic deterministic standalone layer input"
                if manifest.get("input_source") == "synthetic deterministic normal"
                else "captured live Qwen hidden state"
            ),
            "input_source": manifest.get("input_source", "synthetic deterministic normal"),
            "input_sha256": manifest["input_sha256"],
            "layer": manifest["layer"],
            "shape": list(input_shape),
            "seed": manifest["seed"],
            "warmup": args.warmup,
            "iterations": args.iterations,
            "source_dtype": manifest["source_dtype"],
            "coreml_boundary_dtype": manifest["coreml_boundary_dtype"],
        },
        "coreml_artifact": {
            "format": "dense float16 ML Program",
            "package_bytes": manifest["package_bytes"],
            "package_sha256_by_file": {
                str(path.relative_to(package)): _sha256(path)
                for path in sorted(package.rglob("*"))
                if path.is_file()
            },
        },
        "mlx": serializable_mlx,
        "coreml": coreml_results,
        "decision": {
            "status": "placement and microbenchmark evidence only; not integrated",
            "quality": "Core ML output is not bit-identical to MLX BF16 quantized matmul, so it cannot become a default path without downstream route/logit/token qualification.",
            "next_action": "Measure asynchronous overlap with exact expert I/O and reject unless end-to-end behavior and correctness gates pass.",
        },
    }
    encoded = json.dumps(benchmark, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
