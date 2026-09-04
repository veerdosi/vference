from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import psutil

from .artifacts.builder import build_qwen35_artifact, verify_qwen35_artifact
from .artifacts.safetensors import classify_tensor, scan_model
from .bench.storage import probe_storage
from .bench.cache import replay_prefetch, replay_trace
from .experiments import append_record, make_record
from .runtime.verify import (
    verify_multi_turn_state,
    verify_prefill_chunk_invariance,
    verify_real_layer_math,
    verify_runtime_corpus,
)
from .runtime.generate import generate_greedy
from .runtime.integrity import verify_artifact_integrity
from .runtime.model import runtime_adapter_for_artifact
from .native import extension
from .system import mount_info, print_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolved_demand_workers(store: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    return 8 if store == "stable" else 1


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


def _artifact_integrity(args: argparse.Namespace) -> None:
    print_json(verify_artifact_integrity(args.artifact, force=args.force))


def _doctor(args: argparse.Namespace) -> None:
    artifact = args.artifact.resolve()
    checks: dict[str, object] = {}
    failures: list[str] = []

    machine = platform.machine()
    platform_supported = platform.system() == "Darwin" and machine == "arm64"
    checks["platform"] = {
        "system": platform.system(),
        "machine": machine,
        "supported": platform_supported,
    }
    if not platform_supported:
        failures.append("vference requires Apple Silicon macOS")

    try:
        native = extension()
        probe = native.owned_zeros([1], "uint32")
        checks["native_extension"] = {"available": True, "probe_shape": list(probe.shape)}
    except Exception as error:
        checks["native_extension"] = {"available": False, "error": str(error)}
        failures.append("native stable-slot extension is unavailable")

    try:
        adapter = runtime_adapter_for_artifact(artifact)
        checks["architecture_adapter"] = {"name": adapter.name, "supported": True}
    except Exception as error:
        checks["architecture_adapter"] = {"supported": False, "error": str(error)}
        failures.append("artifact architecture is not supported")

    try:
        checks["integrity"] = verify_artifact_integrity(
            artifact, force=args.force_integrity
        )
    except Exception as error:
        checks["integrity"] = {"verified": False, "error": str(error)}
        failures.append("artifact integrity verification failed")

    pack = artifact / "experts.pack"
    if pack.is_file():
        checks["storage"] = mount_info(pack)
    memory = psutil.virtual_memory()
    checks["memory_bytes"] = {
        "total": memory.total,
        "available": memory.available,
    }
    result = {
        "ready": not failures,
        "artifact": str(artifact),
        "checks": checks,
        "failures": failures,
    }
    print_json(result)
    if failures:
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
    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text()
    print_json(
        generate_greedy(
            args.artifact,
            prompt,
            max_tokens=args.max_tokens,
            cache_capacity=args.cache_capacity,
            decode_cache_capacity=args.decode_cache_capacity,
            chat_template=not args.raw_prompt,
            enable_thinking=args.thinking,
            nocache=args.nocache,
            trace_output=args.trace_output,
            store_kind=args.store,
            prefill_chunk_size=args.prefill_chunk_size,
            repeat_raw_prompt_to_tokens=args.repeat_raw_prompt_to_tokens,
            truncate_raw_prompt_to_tokens=args.truncate_raw_prompt_to_tokens,
            raw_prompt_suffix=args.raw_prompt_suffix,
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
            prefetch_policy=args.prefetch_policy,
            prefetch_budget=args.prefetch_budget,
            prefetch_min_observations=args.prefetch_min_observations,
            demand_workers=_resolved_demand_workers(args.store, args.demand_workers),
            auto_cache_capacity=args.auto_cache_capacity,
            allow_unqualified_prefill_chunk_size=(args.allow_unqualified_prefill_chunk_size),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
    )


def _chat(args: argparse.Namespace) -> None:
    messages: list[dict[str, str]] = []
    print("vference chat — /reset clears history, /quit exits", file=sys.stderr)
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if not prompt:
            continue
        if prompt in {"/quit", "/exit"}:
            return
        if prompt == "/reset":
            messages.clear()
            print("history cleared", file=sys.stderr)
            continue
        messages.append({"role": "user", "content": prompt})
        try:
            result = generate_greedy(
                args.artifact,
                prompt,
                messages=messages,
                max_tokens=args.max_tokens,
                cache_capacity=args.cache_capacity,
                chat_template=True,
                enable_thinking=args.thinking,
                nocache=args.nocache,
                store_kind="stable",
                prefill_chunk_size=512,
                cache_policy="global",
                decode_cache_policy="layer",
                clear_cache_between_prefill_chunks=True,
                max_mlx_memory_bytes=(
                    int(args.max_mlx_memory_gib * 1024**3)
                    if args.max_mlx_memory_gib is not None
                    else None
                ),
                prefetch_policy="none",
                demand_workers=8,
                auto_cache_capacity=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
            )
        except Exception as error:
            messages.pop()
            print(f"request failed: {error}", file=sys.stderr)
            continue
        response = str(result["output_text"])
        print(f"assistant> {response}")
        messages.append({"role": "assistant", "content": response})
        print(
            f"[{result['prompt_tokens']} prompt tokens; "
            f"{len(result['output_tokens'])} output tokens; "
            f"{result['decode_tokens_per_second'] or 0:.3f} decode tok/s]",
            file=sys.stderr,
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
        prefetch_policy=args.prefetch_policy,
        prefetch_budget=args.prefetch_budget,
        prefetch_min_observations=args.prefetch_min_observations,
        demand_workers=args.demand_workers,
    )
    print_json(result)
    if not result["all_tokens_exact"] or not result["all_step_logits_exact"]:
        raise SystemExit(1)


def _prefill_chunk_verify(args: argparse.Namespace) -> None:
    result = verify_prefill_chunk_invariance(
        args.artifact,
        prompt=args.prompt,
        prompt_token_count=args.prompt_tokens,
        chunk_sizes=tuple(args.chunk_sizes),
        continuation_tokens=args.max_tokens,
        cache_capacity=args.cache_capacity,
        nocache=args.nocache,
    )
    print_json(result)
    if (
        not result["all_initial_logits_exact"]
        or not result["all_output_tokens_exact"]
        or not result["all_routes_exact"]
    ):
        raise SystemExit(1)


def _route_replay(args: argparse.Namespace) -> None:
    print_json(
        replay_trace(
            args.trace,
            capacities=tuple(args.capacities),
            record_size=args.record_size,
        )
    )


def _prefetch_replay(args: argparse.Namespace) -> None:
    print_json(
        replay_prefetch(
            args.trace,
            capacity_per_layer=args.capacity_per_layer,
            budgets=tuple(args.budgets),
            record_size=args.record_size,
            adaptive_min_observations=tuple(args.adaptive_min_observations),
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

    integrity_parser = commands.add_parser("artifact-integrity")
    integrity_parser.add_argument("artifact", type=Path)
    integrity_parser.add_argument("--force", action="store_true")
    integrity_parser.set_defaults(func=_artifact_integrity)

    doctor_parser = commands.add_parser(
        "doctor", help="verify the native runtime and artifact without loading the model"
    )
    doctor_parser.add_argument("artifact", type=Path)
    doctor_parser.add_argument("--force-integrity", action="store_true")
    doctor_parser.set_defaults(func=_doctor)

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
    prompt_group = generate_parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    generate_parser.add_argument("--max-tokens", type=int, default=16)
    generate_parser.add_argument("--cache-capacity", type=int, default=320)
    generate_parser.add_argument("--decode-cache-capacity", type=int)
    generate_parser.add_argument("--raw-prompt", action="store_true")
    generate_parser.add_argument("--thinking", action="store_true")
    generate_parser.add_argument("--nocache", action="store_true")
    generate_parser.add_argument("--trace-output", type=Path)
    generate_parser.add_argument("--store", choices=("python", "stable"), default="stable")
    generate_parser.add_argument("--prefill-chunk-size", type=int, default=512)
    generate_parser.add_argument(
        "--allow-unqualified-prefill-chunk-size",
        action="store_true",
        help="allow an experimental multi-chunk Qwen prefill boundary that may change routing/output",
    )
    generate_parser.add_argument("--repeat-raw-prompt-to-tokens", type=int)
    generate_parser.add_argument("--truncate-raw-prompt-to-tokens", type=int)
    generate_parser.add_argument("--raw-prompt-suffix")
    generate_parser.add_argument(
        "--cache-policy", choices=("global", "layer", "demand"), default="global"
    )
    generate_parser.add_argument("--decode-cache-policy", choices=("global", "layer", "demand"))
    generate_parser.add_argument(
        "--clear-cache-between-prefill-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate_parser.add_argument("--needle")
    generate_parser.add_argument("--needle-context-tokens", type=int)
    generate_parser.add_argument("--max-mlx-memory-gib", type=float)
    generate_parser.add_argument(
        "--prefetch-policy",
        choices=("none", "adaptive_cross"),
        default="none",
    )
    generate_parser.add_argument("--prefetch-budget", type=int, default=1)
    generate_parser.add_argument("--prefetch-min-observations", type=int, default=8)
    generate_parser.add_argument(
        "--demand-workers",
        type=int,
        help="exact-demand read workers (default: 8 for stable, 1 for python)",
    )
    generate_parser.add_argument(
        "--auto-cache-capacity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shrink the prefill expert pool when required to admit reserved context state",
    )
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.set_defaults(func=_stream_generate)

    chat_parser = commands.add_parser(
        "chat",
        help="interactive exact stateless-transcript chat using the qualified stable store",
    )
    chat_parser.add_argument("artifact", type=Path)
    chat_parser.add_argument("--max-tokens", type=int, default=256)
    chat_parser.add_argument("--cache-capacity", type=int, default=320)
    chat_parser.add_argument("--thinking", action="store_true")
    chat_parser.add_argument(
        "--nocache", action=argparse.BooleanOptionalAction, default=True
    )
    chat_parser.add_argument("--max-mlx-memory-gib", type=float)
    chat_parser.add_argument("--temperature", type=float, default=0.0)
    chat_parser.add_argument("--top-p", type=float, default=1.0)
    chat_parser.add_argument("--top-k", type=int, default=0)
    chat_parser.add_argument("--seed", type=int, default=0)
    chat_parser.set_defaults(func=_chat)

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
    corpus_parser.add_argument(
        "--prefetch-policy",
        choices=("none", "adaptive_cross"),
        default="none",
    )
    corpus_parser.add_argument("--prefetch-budget", type=int, default=1)
    corpus_parser.add_argument("--prefetch-min-observations", type=int, default=8)
    corpus_parser.add_argument("--demand-workers", type=int, default=1)
    corpus_parser.set_defaults(func=_corpus_verify)

    chunk_parser = commands.add_parser("prefill-chunk-verify")
    chunk_parser.add_argument("artifact", type=Path)
    chunk_parser.add_argument("--prompt", required=True)
    chunk_parser.add_argument("--prompt-tokens", type=int, default=512)
    chunk_parser.add_argument("--chunk-sizes", nargs="+", type=int, required=True)
    chunk_parser.add_argument("--max-tokens", type=int, default=16)
    chunk_parser.add_argument("--cache-capacity", type=int, default=320)
    chunk_parser.add_argument("--nocache", action="store_true")
    chunk_parser.set_defaults(func=_prefill_chunk_verify)

    replay_parser = commands.add_parser("route-replay")
    replay_parser.add_argument("trace", type=Path)
    replay_parser.add_argument(
        "--capacities", type=int, nargs="+", default=[8, 64, 160, 320, 640, 1024]
    )
    replay_parser.add_argument("--record-size", type=int, default=1_769_472)
    replay_parser.set_defaults(func=_route_replay)

    prefetch_parser = commands.add_parser("prefetch-replay")
    prefetch_parser.add_argument("trace", type=Path)
    prefetch_parser.add_argument("--capacity-per-layer", type=int, default=8)
    prefetch_parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 8])
    prefetch_parser.add_argument("--adaptive-min-observations", nargs="+", type=int, default=[1])
    prefetch_parser.add_argument("--record-size", type=int, default=1_769_472)
    prefetch_parser.set_defaults(func=_prefetch_replay)

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
