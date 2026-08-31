from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorEntry:
    name: str
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    header_size: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def file_offset(self) -> int:
        return 8 + self.header_size + self.data_start


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: truncated safetensors length")
        header_size = struct.unpack("<Q", raw_length)[0]
        if header_size <= 0 or header_size >= path.stat().st_size:
            raise ValueError(f"{path}: invalid safetensors header size {header_size}")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"{path}: truncated safetensors header")
    return header_size, json.loads(raw_header)


def scan_shard(path: Path) -> list[TensorEntry]:
    header_size, header = read_header(path)
    payload_size = path.stat().st_size - 8 - header_size
    entries: list[TensorEntry] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        start, end = metadata["data_offsets"]
        dtype = metadata["dtype"]
        shape = tuple(metadata["shape"])
        if not 0 <= start <= end <= payload_size:
            raise ValueError(f"{path}: invalid offsets for {name}: {(start, end)}")
        expected = _DTYPE_BYTES[dtype]
        for dimension in shape:
            expected *= dimension
        if expected != end - start:
            raise ValueError(
                f"{path}: byte-size mismatch for {name}: header={end - start}, shape={expected}"
            )
        entries.append(
            TensorEntry(name, path, dtype, shape, start, end, header_size)
        )
    return entries


def scan_model(model_dir: Path) -> dict[str, TensorEntry]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as handle:
            index = json.load(handle)
        weight_map: dict[str, str] = index["weight_map"]
    else:
        shard_names = sorted(path.name for path in model_dir.glob("*.safetensors"))
        if not shard_names:
            raise FileNotFoundError(f"no safetensors files found in {model_dir}")
        weight_map = {}
        for shard_name in shard_names:
            for entry in scan_shard(model_dir / shard_name):
                if entry.name in weight_map:
                    raise ValueError(f"duplicate tensor: {entry.name}")
                weight_map[entry.name] = shard_name
    shard_names = sorted(set(weight_map.values()))
    entries: dict[str, TensorEntry] = {}
    for shard_name in shard_names:
        shard = model_dir / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"indexed shard is missing: {shard}")
        for entry in scan_shard(shard):
            if entry.name in entries:
                raise ValueError(f"duplicate tensor: {entry.name}")
            entries[entry.name] = entry
    if entries.keys() != weight_map.keys():
        missing = sorted(weight_map.keys() - entries.keys())
        extra = sorted(entries.keys() - weight_map.keys())
        raise ValueError(f"index/header mismatch: missing={missing[:5]}, extra={extra[:5]}")
    for name, entry in entries.items():
        if entry.shard.name != weight_map[name]:
            raise ValueError(f"index points {name} to the wrong shard")
    return entries


def classify_tensor(name: str) -> str:
    if ".mlp.switch_mlp." in name:
        return "routed_expert"
    if name.startswith("vision_tower.") or name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    return "text_core"
