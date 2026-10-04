"""Configuration shared by the downloader and worker; no GPU imports."""

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


OFFLOAD_MODES = {"none", "text_encoder", "model", "sequential"}


def _env_bool(name: str, default: str) -> bool:
    value = os.environ.get(name, default).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


@dataclass(frozen=True)
class Settings:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    model_path: str | None = None
    # The ~17.5 GB Qwen3-VL text encoder only runs once per image, so by default it lives in pinned host
    # RAM and is copied to the GPU just for prompt encoding. The transformer and VAE stay resident.
    offload: str = "text_encoder"
    vae_tiling: bool = True
    # Reject requests whose estimated VRAM need exceeds what is free, instead of OOMing mid-generation.
    memory_guard: bool = True
    max_output_bytes: int = 8_000_000

    @classmethod
    def from_env(cls) -> "Settings":
        offload = os.environ.get("CPU_OFFLOAD", "text_encoder").lower()
        if offload not in OFFLOAD_MODES:
            raise ValueError("CPU_OFFLOAD must be none, text_encoder, model, or sequential")
        return cls(
            model_id=os.environ.get("MODEL_ID", MODEL_ID),
            model_revision=os.environ.get("MODEL_REVISION", MODEL_REVISION),
            model_path=os.environ.get("MODEL_PATH") or None,
            offload=offload,
            vae_tiling=_env_bool("VAE_TILING", "true"),
            memory_guard=_env_bool("MEMORY_GUARD", "true"),
        )


def configure_cache() -> Path:
    """Call before importing huggingface_hub, which reads its paths at import time."""
    default = (
        "/runpod-volume/huggingface" if Path("/runpod-volume").is_dir() else "/cache/huggingface"
    )
    cache = Path(os.environ.setdefault("HF_HOME", default))
    cache.mkdir(parents=True, exist_ok=True)
    return cache
