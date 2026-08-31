import json
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from vference.artifacts.builder import _copy_core
from vference.artifacts.safetensors import scan_model


def test_copy_core_is_payload_lossless(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    arrays = {
        "model.embed.weight": np.arange(24, dtype=np.float32).reshape(6, 4),
        "model.norm.weight": np.arange(4, dtype=np.float16),
    }
    save_file(arrays, source / "model.safetensors", metadata={"format": "mlx"})
    entries = scan_model(source)

    output = tmp_path / "core.safetensors"
    digests = _copy_core(list(entries.values()), output)
    rebuilt_dir = tmp_path / "rebuilt"
    rebuilt_dir.mkdir()
    output.rename(rebuilt_dir / output.name)
    rebuilt = scan_model(rebuilt_dir)

    assert rebuilt.keys() == entries.keys()
    assert digests.keys() == entries.keys()
    for name in entries:
        original = entries[name]
        copied = rebuilt[name]
        with original.shard.open("rb") as source_handle:
            source_handle.seek(original.file_offset)
            expected = source_handle.read(original.nbytes)
        with copied.shard.open("rb") as output_handle:
            output_handle.seek(copied.file_offset)
            actual = output_handle.read(copied.nbytes)
        assert actual == expected


def test_written_header_is_eight_byte_aligned(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    save_file({"x": np.arange(3, dtype=np.uint32)}, source / "model.safetensors")
    output = tmp_path / "core.safetensors"
    _copy_core(list(scan_model(source).values()), output)
    with output.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    assert header_size % 8 == 0
    assert header["x"]["data_offsets"] == [0, 12]
