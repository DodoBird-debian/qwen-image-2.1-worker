"""Pure-Python VRAM estimates so oversized requests fail fast instead of OOMing the worker.

The dominant per-request cost is the prefix KV cache: every text and condition-image token keeps a BF16 key
and value in each of the transformer's 32 blocks for the whole denoising loop, once per guidance branch.
That is 512 KiB per token, so ten 1024px references (~41k tokens) pin ~20 GiB, and the same ten references
at output_resolution=2048 (~164k tokens) would need ~80 GiB before any activations.
"""

import math
from dataclasses import dataclass

GIB = 1024**3

# Qwen-Image-2.1 transformer: 32 blocks x (K + V) x 4096 channels x 2 bytes (BF16).
KV_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2
# Transient prefill activations per token (residual, modulated input, FP32 RoPE copies, 3x SwiGLU). Conservative.
ACTIVATION_BYTES_PER_TOKEN = 16 * 4096 * 2
# VAE tiles, cuBLAS/cuDNN workspaces and allocator slack.
FIXED_OVERHEAD_BYTES = 3 * GIB
# One latent token covers a 16x16 pixel tile.
PIXELS_PER_TOKEN_SIDE = 16


def reference_tokens(width: int, height: int, output_resolution: int) -> int:
    """Latent tokens one condition image adds, mirroring the pipeline's `calculate_dimensions`."""
    ratio = width / height
    target_width = math.sqrt(output_resolution * output_resolution * ratio)
    target_height = target_width / ratio
    target_width = round(target_width / 32) * 32
    target_height = round(target_height / 32) * 32
    return (target_width // PIXELS_PER_TOKEN_SIDE) * (target_height // PIXELS_PER_TOKEN_SIDE)


@dataclass(frozen=True)
class MemoryEstimate:
    prefix_tokens: int
    target_tokens: int
    branches: int
    kv_cache_bytes: int
    total_bytes: int


def estimate(
    reference_sizes: list[tuple[int, int]],
    output_resolution: int,
    width: int,
    height: int,
    prompt_chars: int,
    true_cfg: bool,
    use_kv_cache: bool,
) -> MemoryEstimate:
    # Upper-bound text tokens; prompts are tiny next to image tokens anyway.
    text_tokens = prompt_chars // 2 + 128
    prefix = text_tokens + sum(reference_tokens(w, h, output_resolution) for w, h in reference_sizes)
    target = (width // PIXELS_PER_TOKEN_SIDE) * (height // PIXELS_PER_TOKEN_SIDE)
    branches = 2 if true_cfg else 1
    kv_cache = prefix * branches * KV_BYTES_PER_TOKEN if use_kv_cache else 0
    activations = (prefix + target) * ACTIVATION_BYTES_PER_TOKEN
    return MemoryEstimate(
        prefix_tokens=prefix,
        target_tokens=target,
        branches=branches,
        kv_cache_bytes=kv_cache,
        total_bytes=kv_cache + activations + FIXED_OVERHEAD_BYTES,
    )
