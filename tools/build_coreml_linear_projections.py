#!/usr/bin/env python3
"""Build a compressed Core ML probe for Qwen GatedDeltaNet input projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--input-raw", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _unpack_uint4(packed: np.ndarray) -> np.ndarray:
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    return ((packed[..., None] >> shifts) & 15).reshape(packed.shape[0], -1)


def main() -> None:
    args = _parse_args()
    artifact = args.artifact.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((artifact / "config.json").read_text())["text_config"]
    hidden_size = int(config["hidden_size"])
    layer_count = int(config["num_hidden_layers"])
    if not 0 <= args.layer < layer_count:
        raise ValueError(f"layer must be in [0, {layer_count})")
    if args.sequence_length <= 0:
        raise ValueError("sequence length must be positive")

    if args.input_raw is None:
        rng = np.random.default_rng(args.seed)
        input_fp16 = rng.standard_normal((1, args.sequence_length, hidden_size)).astype(
            np.float16
        )
        input_source = "synthetic deterministic normal"
    else:
        input_fp16 = np.fromfile(args.input_raw, dtype=np.float16)
        expected = args.sequence_length * hidden_size
        if input_fp16.size != expected:
            raise ValueError(f"input has {input_fp16.size} values; expected {expected}")
        input_fp16 = input_fp16.reshape(1, args.sequence_length, hidden_size)
        input_source = str(args.input_raw.resolve())

    weights = mx.load(artifact / "core.safetensors")
    prefix = f"language_model.model.layers.{args.layer}.linear_attn"
    input_bf16 = mx.array(input_fp16).astype(mx.bfloat16)
    reference_arrays = [
        _quantized_linear(input_bf16, weights, f"{prefix}.{name}") for name in PROJECTIONS
    ]
    reference = mx.concatenate(reference_arrays, axis=-1)
    mx.eval(reference)
    reference_fp16 = np.asarray(reference.astype(mx.float16))

    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    compressed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in PROJECTIONS:
        tensor = f"{prefix}.{name}"
        packed = np.asarray(weights[f"{tensor}.weight"])
        scales = np.asarray(weights[f"{tensor}.scales"].astype(mx.float16))
        biases = np.asarray(weights[f"{tensor}.biases"].astype(mx.float16))
        quantized = _unpack_uint4(packed).astype(types.np_uint4_dtype)
        offsets = (-biases.astype(np.float32) / scales.astype(np.float32)).astype(np.float16)
        compressed[name] = (quantized, scales, offsets)

    @mb.program(
        input_specs=[
            mb.TensorSpec(
                shape=(1, args.sequence_length, hidden_size),
                dtype=types.fp16,
            )
        ],
        opset_version=ct.target.macOS15,
    )
    def projections(x):
        outputs = []
        for name in PROJECTIONS:
            quantized, scales, offsets = compressed[name]
            packed_weight = mb.const(val=quantized, name=f"{name}_uint4")
            offset = mb.const(val=offsets, name=f"{name}_offset")
            weight = mb.constexpr_blockwise_shift_scale(
                data=packed_weight,
                scale=scales,
                offset=offset,
                name=f"{name}_dequantize",
            )
            outputs.append(mb.linear(x=x, weight=weight, name=name))
        return mb.concat(values=outputs, axis=-1, name="projection_outputs")

    model = ct.convert(
        projections,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    package = output_dir / f"linear-projections-layer-{args.layer}-s{args.sequence_length}.mlpackage"
    model.save(str(package))
    input_path = output_dir / "input-fp16.raw"
    reference_path = output_dir / "reference-mlx-bf16-qmm-output-fp16.raw"
    input_fp16.tofile(input_path)
    reference_fp16.tofile(reference_path)
    package_bytes = sum(path.stat().st_size for path in package.rglob("*") if path.is_file())
    manifest = {
        "format": "vference.coreml-linear-projections-probe.v1",
        "artifact": str(artifact),
        "artifact_manifest_sha256": _sha256(artifact / "manifest.json"),
        "coremltools_version": ct.__version__,
        "layer": args.layer,
        "sequence_length": args.sequence_length,
        "hidden_size": hidden_size,
        "output_size": int(reference_fp16.shape[-1]),
        "seed": args.seed,
        "input_source": input_source,
        "input": input_path.name,
        "input_sha256": _sha256(input_path),
        "reference_output": reference_path.name,
        "reference_output_sha256": _sha256(reference_path),
        "package": package.name,
        "package_bytes": package_bytes,
        "representation": "Core ML uint4 blockwise shift-scale reconstructed from MLX group-64 affine weights",
        "source_dtype": "bfloat16 activation with MLX group-64 affine 4-bit matmul",
        "coreml_boundary_dtype": "float16",
    }
    (output_dir / "probe.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
