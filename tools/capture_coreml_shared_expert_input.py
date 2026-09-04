#!/usr/bin/env python3
"""Capture a real Qwen shared-expert decode input for the Core ML probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vference.runtime.model import load_streaming_qwen


class CaptureSharedExpert(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.inputs: list[np.ndarray] = []
        self.outputs: list[np.ndarray] = []

    def __call__(self, x: mx.array) -> mx.array:
        output = self.inner(x)
        mx.eval(x, output)
        self.inputs.append(np.asarray(x.astype(mx.float16)).copy())
        self.outputs.append(np.asarray(output.astype(mx.float16)).copy())
        return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="Explain why exact expert routing matters in one concise sentence.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model, tokenizer, store = load_streaming_qwen(
        args.artifact,
        cache_capacity=8,
        nocache=True,
        store_kind="stable",
        cache_policy="global",
        demand_workers=8,
    )
    layer = model.language_model.layers[args.layer]
    capture = CaptureSharedExpert(layer.mlp.shared_expert)
    layer.mlp.shared_expert = capture
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    cache = model.make_cache()
    logits = model(mx.array(prompt_tokens)[None], cache=cache)
    mx.eval(logits)
    next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    decode_logits = model(mx.array([[next_token]]), cache=cache)
    mx.eval(decode_logits)
    if len(capture.inputs) != 2 or capture.inputs[-1].shape != (1, 1, 2048):
        raise RuntimeError(f"unexpected capture shapes: {[value.shape for value in capture.inputs]}")

    input_path = output_dir / "decode-input-fp16.raw"
    output_path = output_dir / "decode-shared-output-fp16.raw"
    capture.inputs[-1].tofile(input_path)
    capture.outputs[-1].tofile(output_path)
    record = {
        "format": "vference.coreml-shared-expert-capture.v1",
        "capture_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "artifact": str(args.artifact.resolve()),
        "layer": args.layer,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_tokens),
        "decode_token": next_token,
        "input_shape": list(capture.inputs[-1].shape),
        "input_dtype": "float16 converted exactly from the live bfloat16 layer input",
        "input": input_path.name,
        "input_sha256": _sha256(input_path),
        "reference_output": output_path.name,
        "reference_output_sha256": _sha256(output_path),
        "store_stats": store.stats(),
    }
    (output_dir / "capture.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
