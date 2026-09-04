#!/usr/bin/env python3
"""Build a standalone Core ML shared-expert probe from a vference artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


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


def _dense_weight(weights: dict[str, mx.array], prefix: str) -> np.ndarray:
    dense = mx.dequantize(
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.scales"],
        weights[f"{prefix}.biases"],
        group_size=64,
        bits=4,
        mode="affine",
        dtype=mx.float16,
    )
    mx.eval(dense)
    return np.asarray(dense)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    if args.sequence_length <= 0:
        raise ValueError("sequence length must be positive")
    artifact = args.artifact.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    config = json.loads((artifact / "config.json").read_text())
    text_config = config["text_config"]
    hidden_size = int(text_config["hidden_size"])
    intermediate_size = int(text_config["shared_expert_intermediate_size"])
    layer_count = int(text_config["num_hidden_layers"])
    if not 0 <= args.layer < layer_count:
        raise ValueError(f"layer must be in [0, {layer_count})")

    weights = mx.load(artifact / "core.safetensors")
    prefix = f"language_model.model.layers.{args.layer}.mlp"
    gate_prefix = f"{prefix}.shared_expert.gate_proj"
    up_prefix = f"{prefix}.shared_expert.up_proj"
    down_prefix = f"{prefix}.shared_expert.down_proj"
    shared_gate_prefix = f"{prefix}.shared_expert_gate"

    rng = np.random.default_rng(args.seed)
    input_fp16 = rng.standard_normal((1, args.sequence_length, hidden_size)).astype(np.float16)
    input_bf16 = mx.array(input_fp16).astype(mx.bfloat16)
    gate = _quantized_linear(input_bf16, weights, gate_prefix)
    up = _quantized_linear(input_bf16, weights, up_prefix)
    hidden = nn.silu(gate) * up
    shared = _quantized_linear(hidden, weights, down_prefix)
    shared_gate = mx.sigmoid(_quantized_linear(input_bf16, weights, shared_gate_prefix))
    reference = shared * shared_gate
    mx.eval(reference)
    reference_fp16 = np.asarray(reference.astype(mx.float16))

    dense_gate = _dense_weight(weights, gate_prefix)
    dense_up = _dense_weight(weights, up_prefix)
    dense_down = _dense_weight(weights, down_prefix)
    dense_shared_gate = _dense_weight(weights, shared_gate_prefix)

    # Import lazily so normal vference development does not require the ANE group.
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                shape=(1, args.sequence_length, hidden_size),
                dtype=types.fp16,
            )
        ],
        opset_version=ct.target.macOS15,
    )
    def shared_expert(x):
        gate_value = mb.linear(x=x, weight=dense_gate, name="gate_projection")
        up_value = mb.linear(x=x, weight=dense_up, name="up_projection")
        activated = mb.silu(x=gate_value, name="gate_silu")
        hidden_value = mb.mul(x=activated, y=up_value, name="gated_hidden")
        down_value = mb.linear(x=hidden_value, weight=dense_down, name="down_projection")
        gate_logit = mb.linear(x=x, weight=dense_shared_gate, name="shared_gate_projection")
        gate_scale = mb.sigmoid(x=gate_logit, name="shared_gate_sigmoid")
        return mb.mul(x=down_value, y=gate_scale, name="shared_output")

    model = ct.convert(
        shared_expert,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    package = output_dir / f"shared-expert-layer-{args.layer}-s{args.sequence_length}.mlpackage"
    model.save(str(package))

    input_path = output_dir / "input-fp16.raw"
    reference_path = output_dir / "reference-mlx-bf16-qmm-output-fp16.raw"
    input_fp16.tofile(input_path)
    reference_fp16.tofile(reference_path)
    package_bytes = sum(path.stat().st_size for path in package.rglob("*") if path.is_file())
    manifest = {
        "format": "vference.coreml-shared-expert-probe.v1",
        "artifact": str(artifact),
        "artifact_manifest_sha256": _sha256(artifact / "manifest.json"),
        "coremltools_version": ct.__version__,
        "layer": args.layer,
        "sequence_length": args.sequence_length,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "seed": args.seed,
        "source_dtype": "bfloat16 activation with MLX group-64 affine 4-bit matmul",
        "coreml_boundary_dtype": "float16",
        "package": package.name,
        "package_bytes": package_bytes,
        "input": input_path.name,
        "input_sha256": _sha256(input_path),
        "reference_output": reference_path.name,
        "reference_output_sha256": _sha256(reference_path),
    }
    manifest_path = output_dir / "probe.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
