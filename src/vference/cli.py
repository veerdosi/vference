from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .artifacts.builder import build_qwen35_artifact, verify_qwen35_artifact
from .artifacts.safetensors import classify_tensor, scan_model
from .bench.storage import probe_storage
from .bench.cache import replay_trace
from .experiments import append_record, make_record
from .runtime.verify import (
    verify_multi_turn_state,
    verify_real_layer_math,
    verify_runtime_corpus,
)
from .runtime.generate import generate_greedy
from .system import mount_info, print_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_inspect(args: argparse.Namespace) -> None:
    model_dir = args.model.resolve()
    entries = scan_model(model_dir)
    counts: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    for entry in entries.values():
        category = classify_tensor(entry.name)
        counts[category] += 1
        byte_counts[category] += entry.nbytes
        dtype_counts[entry.dtype] += entry.nbytes
    config = json.loads((model_dir / "config.json").read_text())
    print_json(
        {
            "model_dir": str(model_dir),
            "model_type": config.get("model_type"),
            "quantization": config.get("quantization"),
            "tensor_count": len(entries),
            "tensor_counts_by_category": counts,
            "bytes_by_category": byte_counts,
            "bytes_by_dtype": dtype_counts,
        }
    )


def _storage_probe(args: argparse.Namespace) -> None:
    repo = _repo_root()
    artifact_identity = None
    manifest_path = args.file.resolve().parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        artifact_identity = {
            "format": manifest.get("format"),
            "file_sha256": manifest.get("output_sha256", {}).get(args.file.name),
            "manifest": str(manifest_path),
        }
    result = probe_storage(
        args.file.resolve(),
        read_size=args.read_size,
        reads=args.reads,
        pattern=args.pattern,
        nocache=args.nocache,
        seed=args.seed,
    )
    record = make_record(
        repo,
        "storage_probe",
        {
            "model": {
                "repo": "mlx-community/Qwen3.5-35B-A3B-4bit",
                "revision": "1e20fd8d42056f870933bf98ca6211024744f7ec",
                "runtime_artifact": artifact_identity,
            },
            "storage": mount_info(args.file),
            "workload": result,
            "conclusion": args.conclusion,
        },
    )
    if args.record:
        append_record(repo / "experiments/results.jsonl", record)
    print_json(record)


def _artifact_build(args: argparse.Namespace) -> None:
    result = build_qwen35_artifact(
        args.source, args.output, min_free_bytes=args.min_free_gib * 1024**3
    )
    print_json(result.__dict__)


def _artifact_verify(args: argparse.Namespace) -> None:
    result = verify_qwen35_artifact(args.source, args.artifact)
    print_json(result)
    if not result["verified"]:
        raise SystemExit(1)


def _expert_math_verify(args: argparse.Namespace) -> None:
    result = verify_real_layer_math(
        args.source,
        args.artifact,
        layer_id=args.layer,
        expert_ids=tuple(args.experts),
        seed=args.seed,
        store_kind=args.store,
    )
    print_json(result)
    if not result["bit_exact"]:
        raise SystemExit(1)


def _stream_generate(args: argparse.Namespace) -> None:
    print_json(
        generate_greedy(
            args.artifact,
            args.prompt,
            max_tokens=args.max_tokens,
            cache_capacity=args.cache_capacity,
            chat_template=not args.raw_prompt,
            enable_thinking=args.thinking,
            nocache=args.nocache,
            trace_output=args.trace_output,
            store_kind=args.store,
            prefill_chunk_size=args.prefill_chunk_size,
            repeat_raw_prompt_to_tokens=args.repeat_raw_prompt_to_tokens,
            cache_policy=args.cache_policy,
            decode_cache_policy=args.decode_cache_policy,
            clear_cache_between_prefill_chunks=args.clear_cache_between_prefill_chunks,
            needle=args.needle,
            needle_context_tokens=args.needle_context_tokens,
            max_mlx_memory_bytes=(
                int(args.max_mlx_memory_gib * 1024**3)
                if args.max_mlx_memory_gib is not None
                else None
            ),
        )
    )


def _multi_turn_verify(args: argparse.Namespace) -> None:
    result = verify_multi_turn_state(
        args.artifact,
        first=args.first,
        second=args.second,
        continuation_tokens=args.continuation_tokens,
        cache_capacity=args.cache_capacity,
    )
    print_json(result)
    if not result["initial_argmax_equal"] or not result["continuation_exact"]:
        raise SystemExit(1)


def _corpus_verify(args: argparse.Namespace) -> None:
    result = verify_runtime_corpus(
        args.artifact,
        args.corpus,
        cache_capacity=args.cache_capacity,
    )
    print_json(result)
    if not result["all_tokens_exact"]:
        raise SystemExit(1)


def _route_replay(args: argparse.Namespace) -> None:
    print_json(
        replay_trace(
            args.trace,
            capacities=tuple(args.capacities),
            record_size=args.record_size,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vference")
    commands = parser.add_subparsers(required=True)

    inspect_parser = commands.add_parser("artifact-inspect")
    inspect_parser.add_argument("model", type=Path)
    inspect_parser.set_defaults(func=_artifact_inspect)

    build_parser = commands.add_parser("artifact-build")
    build_parser.add_argument("source", type=Path)
    build_parser.add_argument("output", type=Path)
    build_parser.add_argument("--min-free-gib", type=int, default=30)
    build_parser.set_defaults(func=_artifact_build)

    verify_parser = commands.add_parser("artifact-verify")
    verify_parser.add_argument("source", type=Path)
    verify_parser.add_argument("artifact", type=Path)
    verify_parser.set_defaults(func=_artifact_verify)

    math_parser = commands.add_parser("expert-math-verify")
    math_parser.add_argument("source", type=Path)
    math_parser.add_argument("artifact", type=Path)
    math_parser.add_argument("--layer", type=int, default=0)
    math_parser.add_argument("--experts", type=int, nargs="+", default=list(range(8)))
    math_parser.add_argument("--seed", type=int, default=20260901)
    math_parser.add_argument("--store", choices=("python", "stable"), default="python")
    math_parser.set_defaults(func=_expert_math_verify)

    generate_parser = commands.add_parser("stream-generate")
    generate_parser.add_argument("artifact", type=Path)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-tokens", type=int, default=16)
    generate_parser.add_argument("--cache-capacity", type=int, default=64)
    generate_parser.add_argument("--raw-prompt", action="store_true")
    generate_parser.add_argument("--thinking", action="store_true")
    generate_parser.add_argument("--nocache", action="store_true")
    generate_parser.add_argument("--trace-output", type=Path)
    generate_parser.add_argument("--store", choices=("python", "stable"), default="python")
    generate_parser.add_argument("--prefill-chunk-size", type=int, default=1)
    generate_parser.add_argument("--repeat-raw-prompt-to-tokens", type=int)
    generate_parser.add_argument("--cache-policy", choices=("global", "layer"), default="global")
    generate_parser.add_argument("--decode-cache-policy", choices=("global", "layer"))
    generate_parser.add_argument("--clear-cache-between-prefill-chunks", action="store_true")
    generate_parser.add_argument("--needle")
    generate_parser.add_argument("--needle-context-tokens", type=int)
    generate_parser.add_argument("--max-mlx-memory-gib", type=float)
    generate_parser.set_defaults(func=_stream_generate)

    multi_turn_parser = commands.add_parser("multi-turn-verify")
    multi_turn_parser.add_argument("artifact", type=Path)
    multi_turn_parser.add_argument("--first", required=True)
    multi_turn_parser.add_argument("--second", required=True)
    multi_turn_parser.add_argument("--continuation-tokens", type=int, default=8)
    multi_turn_parser.add_argument("--cache-capacity", type=int, default=320)
    multi_turn_parser.set_defaults(func=_multi_turn_verify)

    corpus_parser = commands.add_parser("corpus-verify")
    corpus_parser.add_argument("artifact", type=Path)
    corpus_parser.add_argument("corpus", type=Path)
    corpus_parser.add_argument("--cache-capacity", type=int, default=320)
    corpus_parser.set_defaults(func=_corpus_verify)

    replay_parser = commands.add_parser("route-replay")
    replay_parser.add_argument("trace", type=Path)
    replay_parser.add_argument(
        "--capacities", type=int, nargs="+", default=[8, 64, 160, 320, 640, 1024]
    )
    replay_parser.add_argument("--record-size", type=int, default=1_769_472)
    replay_parser.set_defaults(func=_route_replay)

    storage_parser = commands.add_parser("storage-probe")
    storage_parser.add_argument("file", type=Path)
    storage_parser.add_argument("--read-size", type=int, default=1_769_472)
    storage_parser.add_argument("--reads", type=int, default=256)
    storage_parser.add_argument("--pattern", choices=("random", "sequential"), default="random")
    storage_parser.add_argument("--nocache", action="store_true")
    storage_parser.add_argument("--seed", type=int, default=20260831)
    storage_parser.add_argument("--record", action="store_true")
    storage_parser.add_argument("--conclusion", default="measurement pending analysis")
    storage_parser.set_defaults(func=_storage_probe)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
