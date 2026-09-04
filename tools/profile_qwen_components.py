#!/usr/bin/env python3
"""Profile Qwen component boundaries with explicit MLX evaluation fences."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vference.runtime.model import load_streaming_qwen


class TimedModule(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        *,
        category: str,
        layer: int | None,
        phase: list[str],
        records: list[dict[str, object]],
    ) -> None:
        super().__init__()
        self.inner = inner
        self.category = category
        self.layer = layer
        self.phase = phase
        self.records = records

    def __call__(self, *args: Any, **kwargs: Any) -> mx.array:
        started = time.perf_counter_ns()
        output = self.inner(*args, **kwargs)
        mx.eval(output)
        self.records.append(
            {
                "phase": self.phase[0],
                "category": self.category,
                "layer": self.layer,
                "seconds": (time.perf_counter_ns() - started) / 1_000_000_000,
            }
        )
        return output


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        groups[(str(record["phase"]), str(record["category"]))].append(
            float(record["seconds"])
        )
    result: dict[str, object] = {}
    for (phase, category), values in groups.items():
        ordered = sorted(values)
        result.setdefault(phase, {})[category] = {  # type: ignore[index]
            "calls": len(values),
            "total_seconds": sum(values),
            "mean_seconds": statistics.fmean(values),
            "p50_seconds": ordered[len(ordered) // 2],
            "p95_seconds": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--prompt", default="Hi.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, tokenizer, store = load_streaming_qwen(
        args.artifact,
        cache_capacity=320,
        nocache=True,
        store_kind="stable",
        cache_policy="layer",
        demand_workers=8,
    )
    phase = ["prefill"]
    records: list[dict[str, object]] = []
    for layer_id, layer in enumerate(model.language_model.layers):
        if layer.is_linear:
            layer.linear_attn = TimedModule(
                layer.linear_attn,
                category="linear_attention",
                layer=layer_id,
                phase=phase,
                records=records,
            )
        else:
            layer.self_attn = TimedModule(
                layer.self_attn,
                category="full_attention",
                layer=layer_id,
                phase=phase,
                records=records,
            )
        layer.mlp = TimedModule(
            layer.mlp,
            category="moe",
            layer=layer_id,
            phase=phase,
            records=records,
        )
    model.language_model.model.norm = TimedModule(
        model.language_model.model.norm,
        category="final_norm",
        layer=None,
        phase=phase,
        records=records,
    )
    model.language_model.lm_head = TimedModule(
        model.language_model.lm_head,
        category="lm_head",
        layer=None,
        phase=phase,
        records=records,
    )

    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    cache = model.make_cache()
    total_started = time.perf_counter()
    prefill_started = time.perf_counter()
    logits = model(mx.array(prompt_tokens)[None], cache=cache)
    mx.eval(logits)
    prefill_seconds = time.perf_counter() - prefill_started
    output_tokens: list[int] = []
    phase[0] = "decode"
    decode_started = time.perf_counter()
    for _ in range(args.max_tokens):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        output_tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
    decode_seconds = time.perf_counter() - decode_started
    total_seconds = time.perf_counter() - total_started

    artifact_manifest = json.loads((args.artifact / "manifest.json").read_text())
    result = {
        "experiment": "Qwen component timing with forced component-boundary evaluation",
        "recorded_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "artifact": {
            "repository": artifact_manifest["source"]["repository"],
            "model_revision": artifact_manifest["source"]["revision"],
            "expert_pack_sha256": artifact_manifest["output_sha256"]["experts.pack"],
        },
        "environment": {
            "hardware": "MacBook Air Mac14,2, Apple M2, 8 GiB unified memory",
            "operating_system": platform.platform(),
            "storage": "internal APFS Apple Fabric storage",
            "expert_reads": "F_NOCACHE",
        },
        "configuration": {
            "cache_capacity": 320,
            "cache_policy": "layer",
            "demand_workers": 8,
            "prefetch_policy": "none",
            "forced_evaluation": "after every attention, MoE, final norm, and LM-head boundary",
        },
        "workload": {
            "prompt": args.prompt,
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": len(output_tokens),
            "output_token_ids": output_tokens,
            "output_text": tokenizer.decode(output_tokens),
        },
        "timing": {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "decode_tokens_per_second": len(output_tokens) / decode_seconds,
            "components": _summary(records),
        },
        "memory": {
            "mlx_peak_bytes": mx.get_peak_memory(),
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_cache_bytes": mx.get_cache_memory(),
        },
        "expert_store": store.stats(),
        "interpretation": "Evaluation fences deliberately perturb scheduling and are for component attribution, not a throughput candidate.",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
