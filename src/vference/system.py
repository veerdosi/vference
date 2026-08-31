from __future__ import annotations

import json
import os
import platform
import plistlib
import subprocess
from pathlib import Path
from typing import Any

import psutil


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def git_state(repo: Path) -> dict[str, Any]:
    try:
        commit = _run("git", "-C", str(repo), "rev-parse", "HEAD")
        dirty = bool(_run("git", "-C", str(repo), "status", "--porcelain"))
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def mount_info(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    target_device = resolved.stat().st_dev
    candidates: list[Path] = []
    for partition in psutil.disk_partitions(all=True):
        mount = Path(partition.mountpoint)
        try:
            if mount.stat().st_dev == target_device:
                candidates.append(mount)
        except OSError:
            continue
    if not candidates:
        raise ValueError(f"cannot resolve mount point for {path}")
    mount_point = max(candidates, key=lambda item: len(item.parts))
    payload = subprocess.check_output(
        ["diskutil", "info", "-plist", str(mount_point)], stderr=subprocess.DEVNULL
    )
    info = plistlib.loads(payload)
    return {
        "path": str(resolved),
        "mount_point": info.get("MountPoint"),
        "device": info.get("DeviceNode"),
        "filesystem": info.get("FilesystemType"),
        "protocol": info.get("BusProtocol"),
        "internal": info.get("Internal"),
        "solid_state": info.get("SolidState"),
        "volume_total_bytes": info.get("TotalSize"),
        "volume_free_bytes": info.get("APFSContainerFree") or info.get("FreeSpace"),
    }


def machine_info() -> dict[str, Any]:
    process = psutil.Process()
    swap = psutil.swap_memory()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model": _run("sysctl", "-n", "hw.model"),
        "chip": _run("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_count": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "process_rss_bytes": process.memory_info().rss,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
    }


def print_json(data: Any) -> None:
    print(json.dumps(data, sort_keys=True, indent=2))
