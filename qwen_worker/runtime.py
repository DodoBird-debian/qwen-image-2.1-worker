"""GPU initialization performed once, before the Runpod worker starts accepting jobs."""

import logging
from pathlib import Path

from qwen_worker.config import Settings, configure_cache
from qwen_worker.service import ImageService

logger = logging.getLogger(__name__)


def download_model(settings: Settings) -> str:
    configure_cache()
    if settings.model_path:
        if not (Path(settings.model_path) / "model_index.json").is_file():
            raise ValueError("MODEL_PATH must contain a complete Diffusers model snapshot")
        return settings.model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=settings.model_id,
        revision=settings.model_revision,
        allow_patterns=[
            "model_index.json",
            "processor/*",
            "scheduler/*",
            "text_encoder/*",
            "transformer/*",
            "vae/*",
        ],
    )


def load_service(settings: Settings) -> ImageService:
    configure_cache()
    import os
    import torch
    from diffusers import QwenImage21Pipeline

    # Prevent thread over-subscription across 26 vCPUs (stops CPU pinning at 100%)
    os.environ.setdefault("CUDA_DEVICE_SCHEDULE", "2")  # BLOCKING_SYNC / YIELD
    cpu_cores = os.cpu_count() or 4
    num_threads = min(8, max(2, cpu_cores // 2))
    torch.set_num_threads(num_threads)
    logger.info("Configured PyTorch CPU threads: %d (vCPUs available: %d)", num_threads, cpu_cores)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to run Qwen-Image-2.1 inference")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This worker requires a GPU with BF16 support (Ampere or newer)")
    logger.info("Loading %s at %s", settings.model_id, settings.model_revision)
    source = download_model(settings)
    pipe = QwenImage21Pipeline.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
    )
    if settings.vae_tiling:
        pipe.vae.enable_tiling()

    # Device placement based on offload mode
    if settings.offload == "model":
        pipe.enable_model_cpu_offload()
    elif settings.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif settings.offload == "text_encoder":
        # Keep 14.2 GB Transformer and 1.35 GB VAE resident on GPU.
        # Only offload the 17.5 GB Text Encoder to CPU, freeing 64.5+ GB VRAM for denoising.
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.text_encoder.to("cpu")
        logger.info("Placed transformer and VAE on GPU; text_encoder placed on CPU (dynamic swap).")

        def _offload_text_encoder_pre_hook(module, inputs):
            if next(pipe.text_encoder.parameters()).device.type == "cuda":
                pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()
                logger.debug("Prompt encoded; text_encoder offloaded to CPU for denoising loop.")

        pipe.transformer.register_forward_pre_hook(_offload_text_encoder_pre_hook)
    else:
        pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)

    def infer(**kwargs):
        try:
            with torch.inference_mode():
                if settings.offload == "text_encoder":
                    # Bring text encoder to GPU only for the prompt encoding step
                    pipe.text_encoder.to("cuda")
                return pipe(**kwargs)
        except Exception:
            # Failed calls bypass the pipeline's normal hook/cache cleanup.
            for cleanup in (pipe.vae.clear_cache, pipe.maybe_free_model_hooks):
                try:
                    cleanup()
                except Exception:
                    logger.exception("Pipeline cleanup failed")
            raise
        finally:
            if settings.offload == "text_encoder":
                if next(pipe.text_encoder.parameters()).device.type == "cuda":
                    pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()

    def generator(seed):
        return torch.Generator(device="cuda").manual_seed(seed)

    def get_memory_budget() -> int:
        if not torch.cuda.is_available() or not settings.memory_guard:
            return 1024**4  # Effectively unlimited if guard is disabled
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes

    def release_memory() -> None:
        if torch.cuda.is_available():
            if settings.offload == "text_encoder":
                if next(pipe.text_encoder.parameters()).device.type == "cuda":
                    pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    logger.info("Model ready; GPU=%s; offload=%s", torch.cuda.get_device_name(), settings.offload)
    return ImageService(
        infer,
        generator,
        settings,
        memory_budget=get_memory_budget if settings.memory_guard else None,
        release_memory=release_memory,
    )

