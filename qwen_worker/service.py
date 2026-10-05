"""Request orchestration, independent of GPU loading so it can be tested on CPU."""

import inspect
import json
import logging
import threading
import time
from collections.abc import Callable

try:
    import torch
except ImportError:
    torch = None

from qwen_worker.config import Settings
from qwen_worker.images import decode_images, encode_image
from qwen_worker.memory import GIB, MemoryEstimate, estimate
from qwen_worker.requests import MAX_SEED, GenerationRequest, InputError

logger = logging.getLogger("qwen_worker.service")


def _format_bytes(size_bytes: int | float) -> str:
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GiB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KiB"
    return f"{int(size_bytes)} B"


class ImageService:
    def __init__(
        self,
        pipeline,
        generator_factory: Callable,
        settings: Settings,
        memory_budget: Callable[[], int] | None = None,
        release_memory: Callable[[], None] | None = None,
    ):
        self.pipeline = pipeline
        self.generator_factory = generator_factory
        self.settings = settings
        # Returns the bytes available for per-request tensors (KV cache, activations); None disables the guard.
        self.memory_budget = memory_budget
        self.release_memory = release_memory
        self._lock = threading.Lock()

    def check_memory(self, request: GenerationRequest, references: list) -> MemoryEstimate:
        need = estimate(
            [image.size for image in references],
            request.output_resolution,
            request.width,
            request.height,
            len(request.effective_prompt) + len(request.negative_prompt or ""),
            request.true_cfg_scale > 1,
            request.use_kv_cache,
        )
        need_gib = need.total_bytes / GIB
        kv_gib = need.kv_cache_bytes / GIB
        act_gib = need.activation_bytes / GIB
        ovh_gib = need.overhead_bytes / GIB

        if self.memory_budget is not None:
            budget = self.memory_budget()
            budget_gib = budget / GIB
            logger.info("  [VRAM PRE-FLIGHT CHECK]")
            logger.info("    * Tokens: %d prefix (%d text, %d image) | %d target latent | %d branches", 
                        need.prefix_tokens, need.text_tokens, need.image_tokens, need.target_tokens, need.branches)
            logger.info("    * Breakdown: KV Cache: %.2f GiB | Activations: %.2f GiB | Overhead: %.2f GiB | Total Need: %.2f GiB",
                        kv_gib, act_gib, ovh_gib, need_gib)
            logger.info("    * GPU Free VRAM Available: %.2f GiB", budget_gib)

            if need.total_bytes > budget:
                logger.error("  <!> [VRAM REJECTED] Needed %.2f GiB exceeds available %.2f GiB", need_gib, budget_gib)
                raise InputError(
                    f"Request needs an estimated {need_gib:.1f} GiB of free VRAM "
                    f"(KV cache {kv_gib:.1f} GiB for {need.prefix_tokens} prefix tokens x "
                    f"{need.branches} branch(es)) but only {budget_gib:.1f} GiB is available. Use fewer "
                    "reference images, output_resolution 512/1024, true_cfg_scale 1, or use_kv_cache false."
                )
            logger.info("    * Verdict: [PASSED] (Estimated surplus: %.2f GiB)", budget_gib - need_gib)
        else:
            logger.info("  [VRAM PRE-FLIGHT] Memory guard disabled; Estimated requirement: %.2f GiB (KV cache: %.2f GiB)", need_gib, kv_gib)

        return need

    def handle(self, job: dict) -> dict:
        if not isinstance(job, dict):
            raise InputError("job must contain an input object")
        
        job_id = job.get("id", "unspecified_job")
        job_input = job.get("input", {})
        request = GenerationRequest.parse(job_input)

        # 1. Log Request Overview
        logger.info("  [REQUEST SPECIFICATION]")
        logger.info("    * Prompt (%d chars): '%s%s'", len(request.prompt), request.prompt[:120], "..." if len(request.prompt) > 120 else "")
        if request.negative_prompt:
            logger.info("    * Negative Prompt (%d chars): '%s%s'", len(request.negative_prompt), request.negative_prompt[:80], "..." if len(request.negative_prompt) > 80 else "")
        total_mp = (request.width * request.height) / 1_000_000
        logger.info("    * Target Canvas: %dx%d (%.2f MP) | Format: %s (Quality: %d, Transparent: %s)", 
                    request.width, request.height, total_mp, request.output_format.upper(), request.quality, request.transparent)
        logger.info("    * Sampling: %d steps | True CFG: %.1f | Seed: %d | Batch: %d image(s)", 
                    request.steps, request.true_cfg_scale, request.seed, request.num_images)
        logger.info("    * Optimization: use_kv_cache=%s | output_resolution=%d", 
                    request.use_kv_cache, request.output_resolution)

        # 2. Decode Reference Images
        t_dec = time.perf_counter()
        references = decode_images(request.images)
        if references:
            ref_pixels = sum(img.width * img.height for img in references)
            logger.info("  [REFERENCE IMAGES]")
            logger.info("    * Decoded %d reference image(s) (total %.2f MP) in %.1fms:", len(references), ref_pixels / 1_000_000, (time.perf_counter() - t_dec) * 1000)
            for idx, img in enumerate(references, 1):
                logger.info("      - Image #%d: %dx%d (%s, aspect ratio %.2f)", idx, img.width, img.height, img.mode, img.width / max(1, img.height))
        else:
            logger.info("  [REFERENCE IMAGES] None (pure text-to-image mode)")

        # 3. Memory Guard Check
        self.check_memory(request, references)

        # 4. Pipeline Parameters
        kwargs = {
            "prompt": request.effective_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "true_cfg_scale": request.true_cfg_scale,
            "output_resolution": request.output_resolution,
            "use_kv_cache": request.use_kv_cache,
        }
        if references:
            kwargs["image"] = references
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt

        started = time.perf_counter()
        result = {
            "model": self.settings.model_id,
            "revision": None if self.settings.model_path else self.settings.model_revision,
            "parameters": {
                "num_inference_steps": request.steps,
                "true_cfg_scale": request.true_cfg_scale,
                "output_resolution": request.output_resolution,
                "use_kv_cache": request.use_kv_cache,
            },
            "images": [],
        }

        # 5. Denoising & Image Generation
        # Diffusers mutates its scheduler/attention state. Never overlap calls to one pipeline.
        logger.info("  [EXECUTION PIPELINE]")
        with self._lock:
            try:
                for index in range(request.num_images):
                    seed = (request.seed + index) % (MAX_SEED + 1)
                    logger.info("    >>> Generating image %d/%d (Seed: %d, Resolution: %dx%d, Steps: %d)...", 
                                index + 1, request.num_images, seed, request.width, request.height, request.steps)
                    
                    t_gen = time.perf_counter()
                    output = self.pipeline(**kwargs, generator=self.generator_factory(seed))
                    gen_time = time.perf_counter() - t_gen
                    ms_per_step = (gen_time / request.steps) * 1000 if request.steps else 0
                    logger.info("    <<< Denoising & VAE decode for %d/%d completed in %.2fs (avg %.1f ms/step)", 
                                index + 1, request.num_images, gen_time, ms_per_step)

                    t_enc = time.perf_counter()
                    encoded = encode_image(output.images[0], request.output_format, request.quality, seed)
                    enc_time = (time.perf_counter() - t_enc) * 1000
                    b64_len = len(encoded.get("image_base64", ""))
                    approx_bytes = int(b64_len * 3 / 4)
                    logger.info("    * Encoded output image to %s (%s, took %.1fms)", 
                                request.output_format.upper(), _format_bytes(approx_bytes), enc_time)

                    result["images"].append(encoded)

            finally:
                if self.release_memory is not None:
                    try:
                        self.release_memory()
                    except Exception:
                        pass


        total_infer_sec = round(time.perf_counter() - started, 3)
        result["inference_seconds"] = total_infer_sec

        if torch is not None and torch.cuda.is_available():
            alloc_mem = torch.cuda.memory_allocated() / (1024**3)
            free_mem, _ = torch.cuda.mem_get_info(0)
            free_gib = free_mem / (1024**3)
            logger.info("  [POST-INFERENCE STATE] VRAM: %.2f GiB Allocated | %.2f GiB Free", alloc_mem, free_gib)

        logger.info("  [BATCH SUMMARY] Successfully generated %d image(s) in %.3fs total", len(result["images"]), total_infer_sec)
        return result


