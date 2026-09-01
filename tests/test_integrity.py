from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from vference.runtime.integrity import file_identity, verify_artifact_integrity


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_artifact_integrity_caches_identity_and_detects_corruption(tmp_path: Path) -> None:
    core = b"verified core payload"
    experts = b"verified expert payload"
    config = b'{"model_type":"test"}'
    (tmp_path / "core.safetensors").write_bytes(core)
    (tmp_path / "experts.pack").write_bytes(experts)
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "vference.runtime-model.v1",
                "output_sha256": {
                    "core.safetensors": _sha256(core),
                    "experts.pack": _sha256(experts),
                },
                "support_file_sha256": {"config.json": _sha256(config)},
            }
        )
    )

    first = verify_artifact_integrity(tmp_path)
    assert first["verified"]
    assert not first["cached"]
    assert first["bytes_hashed"] == len(core) + len(experts) + len(config)
    assert first["stamp_written"]
    second = verify_artifact_integrity(tmp_path)
    assert second["verified"]
    assert second["cached"]
    assert second["bytes_hashed"] == 0

    identity = file_identity(tmp_path / "experts.pack")
    corrupted = bytes([experts[0] ^ 1]) + experts[1:]
    (tmp_path / "experts.pack").write_bytes(corrupted)
    os.utime(
        tmp_path / "experts.pack",
        ns=(identity["mtime_ns"] + 1, identity["mtime_ns"] + 1),
    )
    with pytest.raises(OSError, match="SHA-256 mismatch"):
        verify_artifact_integrity(tmp_path)
