from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .artifacts.safetensors import classify_tensor, scan_model
from .bench.storage import probe_storage
from .experiments import append_record, make_record
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
            },
            "storage": mount_info(args.file),
            "workload": result,
            "conclusion": args.conclusion,
        },
    )
    if args.record:
        append_record(repo / "experiments/results.jsonl", record)
    print_json(record)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vference")
    commands = parser.add_subparsers(required=True)

    inspect_parser = commands.add_parser("artifact-inspect")
    inspect_parser.add_argument("model", type=Path)
    inspect_parser.set_defaults(func=_artifact_inspect)

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
