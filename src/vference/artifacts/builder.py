from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from vference.adapters.base import ExpertLayer
from vference.adapters.qwen35 import Qwen35MoeAdapter

from .safetensors import TensorEntry, classify_tensor, scan_model


COPY_CHUNK = 8 * 1024 * 1024
DEFAULT_RESERVE_BYTES = 30 * 1024**3


@dataclass(frozen=True)
class BuildResult:
    output_dir: str
    core_bytes: int
    expert_bytes: int
    expert_record_bytes: int
    expert_records: int
    elapsed_seconds: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact(handle: BinaryIO, size: int, *, path: Path) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise OSError(f"short read from {path}: wanted {size}, got {len(data)}")
    return data


def _core_header(entries: Iterable[TensorEntry]) -> tuple[bytes, list[TensorEntry]]:
    ordered = sorted(entries, key=lambda entry: entry.name)
    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": "mlx"}}
    for entry in ordered:
        header[entry.name] = {
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + entry.nbytes],
        }
        offset += entry.nbytes
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    return encoded, ordered


def _copy_core(entries: list[TensorEntry], output: Path) -> dict[str, str]:
    header, ordered = _core_header(entries)
    digests: dict[str, str] = {}
    with output.open("xb", buffering=0) as destination:
        destination.write(struct.pack("<Q", len(header)))
        destination.write(header)
        for entry in ordered:
            digest = hashlib.sha256()
            remaining = entry.nbytes
            with entry.shard.open("rb", buffering=0) as source:
                source.seek(entry.file_offset)
                while remaining:
                    data = _read_exact(
                        source, min(COPY_CHUNK, remaining), path=entry.shard
                    )
                    destination.write(data)
                    digest.update(data)
                    remaining -= len(data)
            digests[entry.name] = digest.hexdigest()
        destination.flush()
        os.fsync(destination.fileno())
    return digests


def _pack_experts(layers: tuple[ExpertLayer, ...], output: Path) -> dict[str, str]:
    layer_stride = layers[0].record_size * layers[0].expert_count
    total_size = layer_stride * len(layers)
    digests: dict[str, str] = {}
    with output.open("xb", buffering=0) as destination:
        destination.truncate(total_size)
        for layer_ordinal, layer in enumerate(layers):
            if layer.record_size != layers[0].record_size:
                raise ValueError("variable expert record sizes are not supported by format v1")
            layer_base = layer_ordinal * layer_stride
            for component in layer.components:
                digest = hashlib.sha256()
                with component.source.shard.open("rb", buffering=0) as source:
                    source.seek(component.source.file_offset)
                    for expert_id in range(layer.expert_count):
                        data = _read_exact(
                            source,
                            component.bytes_per_expert,
                            path=component.source.shard,
                        )
                        digest.update(data)
                        record_offset = layer_base + expert_id * layer.record_size
                        destination.seek(record_offset + component.record_offset)
                        destination.write(data)
                digests[component.name] = digest.hexdigest()
        destination.flush()
        os.fsync(destination.fileno())
    return digests


def _index_document(layers: tuple[ExpertLayer, ...], digests: dict[str, str]) -> dict:
    first = layers[0]
    return {
        "format": "vference.expert-pack.v1",
        "alignment": 4096,
        "layer_count": len(layers),
        "expert_count_per_layer": first.expert_count,
        "record_size": first.record_size,
        "layer_stride": first.record_size * first.expert_count,
        "layer_ids": [layer.layer_id for layer in layers],
        "components": [
            {
                "suffix": component.name.split(".switch_mlp.", 1)[1],
                "dtype": component.dtype,
                "source_shape": list(component.shape),
                "bytes_per_expert": component.bytes_per_expert,
                "record_offset": component.record_offset,
            }
            for component in first.components
        ],
        "source_tensor_sha256": digests,
    }


def _copy_support_files(source: Path, destination: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix == ".safetensors":
            continue
        if path.name in {"model.safetensors.index.json", ".DS_Store", ".gitattributes"}:
            continue
        target = destination / path.name
        shutil.copy2(path, target)
        copied[path.name] = _sha256_file(target)
    return copied


def build_qwen35_artifact(
    source: Path,
    output: Path,
    *,
    min_free_bytes: int = DEFAULT_RESERVE_BYTES,
) -> BuildResult:
    started = time.monotonic()
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config = json.loads((source / "config.json").read_text())
    text_config = config["text_config"]
    entries = scan_model(source)
    adapter = Qwen35MoeAdapter(
        layers=text_config["num_hidden_layers"], experts=text_config["num_experts"]
    )
    layers = adapter.expert_layers(entries)
    core_entries = [entry for entry in entries.values() if classify_tensor(entry.name) == "text_core"]
    expert_bytes = sum(
        component.source.nbytes for layer in layers for component in layer.components
    )
    core_payload_bytes = sum(entry.nbytes for entry in core_entries)
    required = expert_bytes + core_payload_bytes

    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if free - required < min_free_bytes:
        raise OSError(
            f"artifact needs about {required:,} bytes but would leave "
            f"{free - required:,} bytes; required reserve is {min_free_bytes:,} bytes"
        )

    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial build already exists: {partial}")
    partial.mkdir()
    try:
        expert_path = partial / "experts.pack"
        core_path = partial / "core.safetensors"
        expert_digests = _pack_experts(layers, expert_path)
        core_digests = _copy_core(core_entries, core_path)
        index = _index_document(layers, expert_digests)
        (partial / "experts.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        support_hashes = _copy_support_files(source, partial)
        manifest = {
            "format": "vference.runtime-model.v1",
            "architecture_adapter": adapter.name,
            "text_only": True,
            "source": {
                "path": str(source),
                "repository": "mlx-community/Qwen3.5-35B-A3B-4bit",
                "revision": "1e20fd8d42056f870933bf98ca6211024744f7ec",
            },
            "quantization": config.get("quantization"),
            "tensor_counts": {
                "core": len(core_entries),
                "routed_expert": len(expert_digests),
            },
            "payload_bytes": {
                "core": core_payload_bytes,
                "routed_expert": expert_bytes,
            },
            "source_tensor_sha256": {**core_digests, **expert_digests},
            "support_file_sha256": support_hashes,
            "output_sha256": {
                "core.safetensors": _sha256_file(core_path),
                "experts.pack": _sha256_file(expert_path),
                "experts.index.json": _sha256_file(partial / "experts.index.json"),
            },
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(partial, output)
    except BaseException:
        # Keep the explicitly named partial directory for diagnosis/resume tooling.
        raise

    return BuildResult(
        output_dir=str(output),
        core_bytes=(output / "core.safetensors").stat().st_size,
        expert_bytes=(output / "experts.pack").stat().st_size,
        expert_record_bytes=layers[0].record_size,
        expert_records=len(layers) * layers[0].expert_count,
        elapsed_seconds=time.monotonic() - started,
    )


def verify_qwen35_artifact(source: Path, artifact: Path) -> dict[str, object]:
    source = source.resolve()
    artifact = artifact.resolve()
    manifest = json.loads((artifact / "manifest.json").read_text())
    index = json.loads((artifact / "experts.index.json").read_text())
    entries = scan_model(source)
    config = json.loads((source / "config.json").read_text())
    text_config = config["text_config"]
    layers = Qwen35MoeAdapter(
        layers=text_config["num_hidden_layers"], experts=text_config["num_experts"]
    ).expert_layers(entries)

    failures: list[str] = []
    pack_path = artifact / "experts.pack"
    with pack_path.open("rb", buffering=0) as pack:
        for layer_ordinal, layer in enumerate(layers):
            layer_base = layer_ordinal * index["layer_stride"]
            for component in layer.components:
                source_digest = hashlib.sha256()
                rebuilt_digest = hashlib.sha256()
                with component.source.shard.open("rb", buffering=0) as source_handle:
                    source_handle.seek(component.source.file_offset)
                    for expert_id in range(layer.expert_count):
                        expected = _read_exact(
                            source_handle,
                            component.bytes_per_expert,
                            path=component.source.shard,
                        )
                        source_digest.update(expected)
                        pack.seek(
                            layer_base
                            + expert_id * layer.record_size
                            + component.record_offset
                        )
                        actual = _read_exact(pack, component.bytes_per_expert, path=pack_path)
                        rebuilt_digest.update(actual)
                        if actual != expected:
                            failures.append(f"{component.name}[{expert_id}]")
                            if len(failures) >= 10:
                                break
                expected_digest = manifest["source_tensor_sha256"].get(component.name)
                if source_digest.hexdigest() != expected_digest:
                    failures.append(f"source digest changed: {component.name}")
                if rebuilt_digest.hexdigest() != expected_digest:
                    failures.append(f"reconstructed digest mismatch: {component.name}")
                if len(failures) >= 10:
                    break
            if len(failures) >= 10:
                break

    core_entries = scan_model(artifact)
    source_core = {
        name: entry for name, entry in entries.items() if classify_tensor(name) == "text_core"
    }
    if core_entries.keys() != source_core.keys():
        failures.append("core tensor inventory mismatch")
    else:
        for name, source_entry in source_core.items():
            output_entry = core_entries[name]
            digest = hashlib.sha256()
            with output_entry.shard.open("rb", buffering=0) as handle:
                handle.seek(output_entry.file_offset)
                remaining = output_entry.nbytes
                while remaining:
                    chunk = _read_exact(handle, min(COPY_CHUNK, remaining), path=output_entry.shard)
                    digest.update(chunk)
                    remaining -= len(chunk)
            if digest.hexdigest() != manifest["source_tensor_sha256"].get(name):
                failures.append(f"core digest mismatch: {name}")
                if len(failures) >= 10:
                    break

    output_hash_failures = [
        name
        for name, expected in manifest["output_sha256"].items()
        if _sha256_file(artifact / name) != expected
    ]
    failures.extend(f"output hash mismatch: {name}" for name in output_hash_failures)
    return {
        "artifact": str(artifact),
        "source": str(source),
        "verified": not failures,
        "failures": failures,
        "expert_records_checked": len(layers) * layers[0].expert_count,
        "source_tensors_checked": len(source_core) + sum(len(layer.components) for layer in layers),
    }
