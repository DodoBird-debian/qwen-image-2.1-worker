"""GPU initialization performed once, before the Runpod worker starts accepting jobs."""

import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path

from qwen_worker.config import Settings, configure_cache
from qwen_worker.service import ImageService

logger = logging.getLogger("qwen_worker.runtime")


def _format_bytes(size_bytes: int) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GiB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    return f"{size_bytes} B"


def _get_dir_stats(path: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) for a directory."""
    if not path.is_dir():
        return 0, 0
    total_bytes = 0
    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(fp)
                count += 1
            except OSError:
                pass
    return count, total_bytes


def download_model(settings: Settings) -> str:
    if settings.model_path:
        model_path = Path(settings.model_path)
        logger.info("[MODEL RESOLVER] Using explicit MODEL_PATH: %s", model_path)
        if not (model_path / "model_index.json").is_file():
            raise ValueError(f"MODEL_PATH '{model_path}' must contain a complete Diffusers model snapshot (model_index.json missing)")
        count, size = _get_dir_stats(model_path)
        logger.info("[MODEL RESOLVER] Verified explicit model directory: %d files (%s)", count, _format_bytes(size))
        return str(model_path)

    # 1. Exact RunPod official Cached Models pattern:
    # /runpod-volume/huggingface-cache/hub/models--{org}--{name}/snapshots/{hash}/
    formatted_id = f"models--{settings.model_id.replace('/', '--')}"
    candidate_bases = [
        Path("/runpod-volume/huggingface-cache/hub"),
        Path("/runpod-volume/huggingface/hub"),
        Path("/root/.cache/huggingface/hub"),
    ]
    
    logger.info("[MODEL RESOLVER] Searching official RunPod network volume model cache...")
    for base in candidate_bases:
        snapshot_dir = base / formatted_id / "snapshots"
        logger.info("  -> Checking snapshot directory: %s", snapshot_dir)
        if snapshot_dir.is_dir():
            snapshots = [p for p in snapshot_dir.iterdir() if p.is_dir()]
            logger.info("     Found %d snapshot(s) in %s: %s", len(snapshots), snapshot_dir, [s.name for s in snapshots])
            if snapshots:
                # Try exact revision match
                for snap in snapshots:
                    if snap.name == settings.model_revision and (snap / "model_index.json").is_file():
                        count, size = _get_dir_stats(snap)
                        logger.info("  ==> [CACHE HIT] Exact revision match: %s (%d files, %s)", snap, count, _format_bytes(size))
                        return str(snap)
                # Fallback to any snapshot with model_index.json
                for snap in snapshots:
                    if (snap / "model_index.json").is_file():
                        count, size = _get_dir_stats(snap)
                        logger.info("  ==> [CACHE HIT] Usable snapshot match: %s (%d files, %s)", snap, count, _format_bytes(size))
                        return str(snap)

    # 2. General scan in /runpod-volume
    runpod_vol = Path("/runpod-volume")
    if runpod_vol.is_dir():
        logger.info("[MODEL RESOLVER] Scanning /runpod-volume for Diffusers model_index.json...")
        target_name = settings.model_id.split("/")[-1].lower()  # "qwen-image-2.1"
        for candidate in runpod_vol.glob("**/model_index.json"):
            if target_name in str(candidate).lower():
                parent = candidate.parent
                count, size = _get_dir_stats(parent)
                logger.info("  ==> [CACHE HIT] Found matching cached model in /runpod-volume: %s (%d files, %s)", parent, count, _format_bytes(size))
                return str(parent)
        for candidate in runpod_vol.glob("**/model_index.json"):
            parent = candidate.parent
            count, size = _get_dir_stats(parent)
            logger.info("  ==> [CACHE HIT] Found candidate model in /runpod-volume: %s (%d files, %s)", parent, count, _format_bytes(size))
            return str(parent)
    else:
        logger.info("[MODEL RESOLVER] /runpod-volume is not mounted or not a directory.")

    # 3. Check if weights are baked into the container image (/opt/huggingface)
    opt_hf = Path("/opt/huggingface")
    if opt_hf.is_dir():
        logger.info("[MODEL RESOLVER] Checking baked container weights at /opt/huggingface...")
        for candidate in opt_hf.glob("**/model_index.json"):
            parent = candidate.parent
            count, size = _get_dir_stats(parent)
            logger.info("  ==> [CACHE HIT] Found baked model in container: %s (%d files, %s)", parent, count, _format_bytes(size))
            return str(parent)

    # 4. Resolve/download via Hugging Face cache
    configure_cache()
    from huggingface_hub import snapshot_download

    logger.warning("[MODEL RESOLVER] No local cache found. Downloading model snapshot from Hugging Face Hub: %s (%s)...", settings.model_id, settings.model_revision)
    t0 = time.perf_counter()
    downloaded_path = snapshot_download(
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
    count, size = _get_dir_stats(Path(downloaded_path))
    logger.info("[MODEL RESOLVER] Download completed in %.2fs. Destination: %s (%d files, %s)", 
                time.perf_counter() - t0, downloaded_path, count, _format_bytes(size))
    return downloaded_path


def load_service(settings: Settings) -> ImageService:
    configure_cache()
    import torch
    from diffusers import QwenImage21Pipeline

    # =========================================================================
    # SYSTEM DIAGNOSTICS BANNER
    # =========================================================================
    logger.info("=" * 80)
    logger.info("     QWEN-IMAGE-2.1 RUNPOD SERVERLESS WORKER - INITIALIZATION BANNER")
    logger.info("=" * 80)
    logger.info(" [SYSTEM & PLATFORM]")
    logger.info("   * Platform: %s (%s %s)", platform.system(), platform.release(), platform.machine())
    logger.info("   * Python: %s | Executable: %s", sys.version.split()[0], sys.executable)
    logger.info("   * Working Directory: %s", os.getcwd())
    logger.info("   * PyTorch: %s | CUDA Runtime: %s | cuDNN: %s", 
                torch.__version__, torch.version.cuda or "N/A", torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A")

    if not torch.cuda.is_available():
        logger.error("<!> FATAL: CUDA is not available. This worker requires an NVIDIA GPU.")
        raise RuntimeError("A CUDA GPU is required to run Qwen-Image-2.1 inference")
    if not torch.cuda.is_bf16_supported():
        logger.error("<!> FATAL: BF16 is not supported on this device. Ampere (RTX 30xx/Axx) or newer required.")
        raise RuntimeError("This worker requires a GPU with BF16 support (Ampere or newer)")

    device_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_cap = torch.cuda.get_device_capability(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    logger.info(" [GPU COMPUTE & VRAM]")
    logger.info("   * CUDA Devices: %d (Active: 0 - %s, Compute %d.%d)", device_count, gpu_name, gpu_cap[0], gpu_cap[1])
    logger.info("   * Initial VRAM: %.2f GiB Free / %.2f GiB Total", free_mem / (1024**3), total_mem / (1024**3))
    logger.info("   * Native BF16 Support: %s", torch.cuda.is_bf16_supported())

    # Host Memory and CPU
    cpu_cores = os.cpu_count() or 4
    num_threads = min(8, max(2, cpu_cores // 2))
    torch.set_num_threads(num_threads)
    os.environ.setdefault("CUDA_DEVICE_SCHEDULE", "2")  # BLOCKING_SYNC / YIELD

    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_info = f"{vm.available / (1024**3):.1f} GiB Available / {vm.total / (1024**3):.1f} GiB Total"
    except ImportError:
        ram_info = "psutil not available"

    logger.info(" [CPU & HOST SYSTEM RAM]")
    logger.info("   * CPU Logical Cores: %d | PyTorch CPU Threads: %d", cpu_cores, num_threads)
    logger.info("   * Host System RAM: %s", ram_info)

    # Storage & Volumes
    logger.info(" [STORAGE & VOLUMES]")
    for path_str in ["/runpod-volume", "/opt/huggingface", "/tmp", str(Path.home() / ".cache")]:
        p = Path(path_str)
        if p.exists():
            try:
                du = shutil.disk_usage(p)
                logger.info("   * %s: Mounted / Available (Free: %.1f GB, Total: %.1f GB)", path_str, du.free / (1000**3), du.total / (1000**3))
            except OSError:
                logger.info("   * %s: Exists", path_str)
        else:
            logger.info("   * %s: [Not Found]", path_str)

    # Worker Settings
    logger.info(" [WORKER SETTINGS]")
    logger.info("   * MODEL_ID: %s", settings.model_id)
    logger.info("   * MODEL_REVISION: %s", settings.model_revision)
    logger.info("   * MODEL_PATH: %s", settings.model_path or "(Dynamic auto-discovery)")
    logger.info("   * CPU_OFFLOAD: %s", settings.offload)
    logger.info("   * VAE_TILING: %s", settings.vae_tiling)
    logger.info("   * MEMORY_GUARD: %s", settings.memory_guard)
    logger.info("=" * 80)

    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    t_resolve = time.perf_counter()
    logger.info(">>> Step 1/3: Resolving model snapshot...")
    source = download_model(settings)
    logger.info("<<< Step 1/3 Completed: Model source ready in %.2fs from '%s'", time.perf_counter() - t_resolve, source)

    logger.info(">>> Step 2/3: Loading Diffusers QwenImage21Pipeline (torch_dtype=bfloat16)...")
    t_pipe = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        local_files_only=True,
    )
    logger.info("<<< Step 2/3 Completed: Pipeline components instantiated in %.2fs", time.perf_counter() - t_pipe)

    logger.info(">>> Step 3/3: Applying hardware placement & memory offload strategy...")
    if settings.vae_tiling:
        pipe.vae.enable_tiling()
        logger.info("   * VAE tiling enabled for large-resolution decoding memory optimization.")

    # Device placement based on offload mode
    if settings.offload == "model":
        pipe.enable_model_cpu_offload()
        logger.info("   * Enabled standard Diffusers model CPU offload.")
    elif settings.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        logger.info("   * Enabled sequential CPU offload.")
    elif settings.offload == "text_encoder":
        # Keep 14.2 GB Transformer and 1.35 GB VAE resident on GPU.
        # Only offload the 17.5 GB Text Encoder to CPU, freeing 64.5+ GB VRAM for denoising.
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.text_encoder.to("cpu")
        logger.info("   * Custom Dynamic Offload: Transformer & VAE resident in CUDA VRAM; Text Encoder resident in CPU RAM.")

        def _offload_text_encoder_pre_hook(module, inputs):
            if next(pipe.text_encoder.parameters()).device.type == "cuda":
                pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()
                logger.debug("   * Text encoder offloaded back to CPU host RAM after prompt encoding.")

        pipe.transformer.register_forward_pre_hook(_offload_text_encoder_pre_hook)
    else:
        pipe.to("cuda")
        logger.info("   * Placed full pipeline (all components) resident on CUDA GPU.")

    alloc_mem = torch.cuda.memory_allocated() / (1024**3)
    reserved_mem = torch.cuda.memory_reserved() / (1024**3)
    free_mem_after, _ = torch.cuda.mem_get_info(0)
    free_gib_after = free_mem_after / (1024**3)

    logger.info("=" * 80)
    logger.info(">>> QWEN-IMAGE-2.1 SERVICE READY TO SERVE RUNPOD SERVERLESS JOBS <<<")
    logger.info("    Baseline GPU VRAM: %.2f GiB Allocated | %.2f GiB Reserved | %.2f GiB Free", alloc_mem, reserved_mem, free_gib_after)
    logger.info("=" * 80)

    pipe.set_progress_bar_config(disable=True)

    def infer(**kwargs):
        try:
            with torch.inference_mode():
                if settings.offload == "text_encoder":
                    # Bring text encoder to GPU only for the prompt encoding step
                    t_te = time.perf_counter()
                    pipe.text_encoder.to("cuda")
                    logger.debug("   • Text encoder moved to CUDA in %.1fms", (time.perf_counter() - t_te) * 1000)
                return pipe(**kwargs)
        except Exception:
            # Failed calls bypass the pipeline's normal hook/cache cleanup.
            for cleanup in (pipe.vae.clear_cache, pipe.maybe_free_model_hooks):
                try:
                    cleanup()
                except Exception:
                    logger.exception("Pipeline cleanup failed during error handling")
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

    return ImageService(
        infer,
        generator,
        settings,
        memory_budget=get_memory_budget if settings.memory_guard else None,
        release_memory=release_memory,
    )


