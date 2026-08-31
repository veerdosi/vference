import subprocess
from pathlib import Path

from vference.artifacts.safetensors import classify_tensor


def test_classify_tensor() -> None:
    assert classify_tensor("language_model.model.layers.0.mlp.switch_mlp.up_proj.weight") == (
        "routed_expert"
    )
    assert classify_tensor("language_model.model.embed_tokens.weight") == "text_core"
    assert classify_tensor("vision_tower.blocks.0.attn.qkv.weight") == "vision"
    assert classify_tensor("mtp.layers.0.mlp.experts.0.up_proj.weight") == "mtp"


def test_repository_does_not_contain_model_weights() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "*.safetensors", "*.pack"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not tracked
