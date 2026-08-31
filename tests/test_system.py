from pathlib import Path

from vference.system import mount_info


def test_mount_info_accepts_file_path(tmp_path: Path) -> None:
    path = tmp_path / "probe.bin"
    path.write_bytes(b"vference")
    info = mount_info(path)
    assert info["mount_point"]
    assert info["device"]
    assert info["path"] == str(path.resolve())
