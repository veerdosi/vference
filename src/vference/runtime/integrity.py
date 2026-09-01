from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

CHUNK_BYTES = 8 * 1024**2
STAMP_FORMAT = "vference.integrity-stamp.v1"
STAMP_NAME = ".vference-integrity.json"


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_files(manifest: dict[str, object]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for field in ("output_sha256", "support_file_sha256"):
        values = manifest.get(field)
        if not isinstance(values, dict):
            raise ValueError(f"artifact manifest lacks {field}")
        for raw_name, raw_digest in values.items():
            name = str(raw_name)
            if Path(name).name != name:
                raise ValueError(f"artifact manifest contains unsafe file name: {name}")
            digest = str(raw_digest)
            if len(digest) != 64:
                raise ValueError(f"artifact manifest contains invalid SHA-256 for {name}")
            if name in expected and expected[name] != digest:
                raise ValueError(f"artifact manifest contains conflicting SHA-256 for {name}")
            expected[name] = digest
    if "core.safetensors" not in expected or "experts.pack" not in expected:
        raise ValueError("artifact manifest lacks required runtime payload hashes")
    return expected


def verify_artifact_integrity(artifact: Path, *, force: bool = False) -> dict[str, object]:
    artifact = artifact.resolve()
    manifest_path = artifact / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "vference.runtime-model.v1":
        raise ValueError(f"unsupported runtime artifact format: {manifest.get('format')}")
    expected = _expected_files(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    identities = {name: file_identity(artifact / name) for name in sorted(expected)}
    stamp_path = artifact / STAMP_NAME
    started = time.perf_counter()

    if not force:
        try:
            stamp = json.loads(stamp_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            stamp = None
        if (
            isinstance(stamp, dict)
            and stamp.get("format") == STAMP_FORMAT
            and stamp.get("manifest_sha256") == manifest_sha256
            and stamp.get("expected_sha256") == expected
            and stamp.get("file_identities") == identities
        ):
            current = {name: file_identity(artifact / name) for name in sorted(expected)}
            if current == identities:
                return {
                    "verified": True,
                    "cached": True,
                    "bytes_hashed": 0,
                    "elapsed_seconds": time.perf_counter() - started,
                    "manifest_sha256": manifest_sha256,
                    "file_count": len(expected),
                    "stamp": str(stamp_path),
                }

    bytes_hashed = 0
    verified_identities: dict[str, dict[str, int]] = {}
    for name, expected_digest in sorted(expected.items()):
        path = artifact / name
        before = file_identity(path)
        actual_digest = _sha256_file(path)
        after = file_identity(path)
        if before != after:
            raise OSError(f"artifact file changed while hashing: {path}")
        if actual_digest != expected_digest:
            raise OSError(
                f"artifact SHA-256 mismatch for {path}: {actual_digest} != {expected_digest}"
            )
        bytes_hashed += after["size"]
        verified_identities[name] = after
    identities = {name: file_identity(artifact / name) for name in sorted(expected)}
    if identities != verified_identities:
        raise OSError("artifact files changed before integrity verification completed")
    stamp = {
        "format": STAMP_FORMAT,
        "manifest_sha256": manifest_sha256,
        "expected_sha256": expected,
        "file_identities": identities,
    }
    stamp_written = False
    temporary = stamp_path.with_name(f"{STAMP_NAME}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(stamp, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, stamp_path)
        stamp_written = True
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
    return {
        "verified": True,
        "cached": False,
        "bytes_hashed": bytes_hashed,
        "elapsed_seconds": time.perf_counter() - started,
        "manifest_sha256": manifest_sha256,
        "file_count": len(expected),
        "stamp": str(stamp_path),
        "stamp_written": stamp_written,
    }
