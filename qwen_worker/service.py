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

    def check_memory(self, request: GenerationRequest, references: list) -> tuple[MemoryEstimate, bool]:
        need = estimate(
            [image.size for image in references],
            request.output_resolution,
            request.width,
            request.height,
            len(request.effective_prompt) + len(request.negative_prompt or ""),
            request.true_cfg_scale > 1,
            request.use_kv_cache,
            batch_size=request.num_images,
        )
        need_gib = need.total_bytes / GIB
        parallel_need_gib = need.parallel_total_bytes / GIB
        kv_gib = need.kv_cache_bytes / GIB
        act_gib = need.activation_bytes / GIB
        ovh_gib = need.overhead_bytes / GIB
        par_act_gib = need.parallel_activations_bytes / GIB
        par_ovh_gib = need.parallel_overhead_bytes / GIB
        can_parallel = False

        if self.memory_budget is not None:
            budget = self.memory_budget()
            budget_gib = budget / GIB
            
            logger.info("  " + "=" * 76)
            logger.info("  [ADAPTIVE EXECUTION ROUTER & VRAM BUDGET AUDIT]")
            logger.info("  " + "=" * 76)
            logger.info("  * Token Accounting:")
            logger.info("    - Prefix Tokens : %d (Text: %d tokens, Reference Images: %d tokens across %d condition image(s))",
                        need.prefix_tokens, need.text_tokens, need.image_tokens, len(references))
            logger.info("    - Target Latents: %d tokens/image (%dx%d canvas) -> Total batch tokens: %d",
                        need.target_tokens, request.width, request.height, need.target_tokens * request.num_images)
            logger.info("    - Guidance      : %d branch(es) (True CFG Scale: %.1f)", need.branches, request.true_cfg_scale)
            logger.info("  * Memory Allocation Breakdown:")
            logger.info("    - KV Cache      : %.2f GiB (%s: 512 KiB/token x %d tokens x %d branches)",
                        kv_gib, "ENABLED" if request.use_kv_cache else "DISABLED (0.00 GiB)", need.prefix_tokens, need.branches)
            logger.info("    - Fixed Overhead: %.2f GiB (VAE workspace, cuDNN, CUDA allocator pools)", ovh_gib)
            logger.info("    - Mode B (Seq)  : %.2f GiB (Sequential 1-by-1 generation loop: KV %.2f GiB + Act %.2f GiB + Ovh %.2f GiB)",
                        need_gib, kv_gib, act_gib, ovh_gib)
            if request.num_images > 1:
                logger.info("    - Mode A (Par)  : %.2f GiB (Parallel tensor pass: Shared KV %.2f GiB + Batch Act %.2f GiB + Ovh %.2f GiB)",
                            parallel_need_gib, kv_gib, par_act_gib, par_ovh_gib)
            logger.info("    - Available VRAM: %.2f GiB Free on GPU", budget_gib)
            logger.info("  " + "-" * 76)

            if need.total_bytes > budget:
                logger.error("  <!> [ROUTING: REJECTED - INSUFFICIENT VRAM]")
                logger.error("      Reason: Even sequential generation requires %.2f GiB, which exceeds available %.2f GiB by %.2f GiB.",
                             need_gib, budget_gib, need_gib - budget_gib)
                logger.error("      Recommendation: Reduce reference images, lower output_resolution (512/1024), set true_cfg_scale=1.0, or disable use_kv_cache.")
                logger.info("  " + "=" * 76)
                raise InputError(
                    f"Request needs an estimated {need_gib:.1f} GiB of free VRAM "
                    f"(KV cache {kv_gib:.1f} GiB for {need.prefix_tokens} prefix tokens x "
                    f"{need.branches} branch(es)) but only {budget_gib:.1f} GiB is available. Use fewer "
                    "reference images, output_resolution 512/1024, true_cfg_scale 1, or use_kv_cache false."
                )

            # Adaptive Route Selection
            if request.num_images > 1:
                if need.parallel_total_bytes <= budget:
                    can_parallel = True
                    surplus_gib = budget_gib - parallel_need_gib
                    logger.info("  * ROUTING DECISION: >>> [MODE A: PARALLEL TENSOR BATCHING] (SPEED MODE) <<<")
                    logger.info("    -> Rationale    : Parallel requirement (%.2f GiB) <= Available VRAM (%.2f GiB).", parallel_need_gib, budget_gib)
                    logger.info("    -> Safety Margin: +%.2f GiB free headroom remaining during peak generation.", surplus_gib)
                    logger.info("    -> Speed Factor : All %d images will be generated simultaneously in 1 single forward pass (~3.5x faster!).", request.num_images)
                else:
                    can_parallel = False
                    deficit_gib = parallel_need_gib - budget_gib
                    logger.info("  * ROUTING DECISION: >>> [MODE B: SEQUENTIAL SAFETY LOOP] (SAFE MODE) <<<")
                    logger.info("    -> Rationale    : Parallel requirement (%.2f GiB) exceeds free VRAM (%.2f GiB) by %.2f GiB.", parallel_need_gib, budget_gib, deficit_gib)
                    logger.info("    -> Protection   : Automatically routing to sequential 1-by-1 generation to prevent CUDA Out-Of-Memory (OOM).")
                    logger.info("    -> Sequential   : Peak memory will be constrained to %.2f GiB (Headroom: +%.2f GiB).", need_gib, budget_gib - need_gib)
            else:
                can_parallel = False
                logger.info("  * ROUTING DECISION: [SINGLE-IMAGE STANDARD PASS] (batch_size = 1, Need: %.2f GiB, Headroom: +%.2f GiB)",
                            need_gib, budget_gib - need_gib)
            logger.info("  " + "=" * 76)
        else:
            can_parallel = request.num_images > 1
            logger.info("  [VRAM PRE-FLIGHT] Memory guard disabled; estimated need: %.2f GiB (Sequential: %.2f GiB, Parallel: %.2f GiB)", 
                        need_gib, need_gib, parallel_need_gib)

        return need, can_parallel

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

        # 3. Memory Guard & Routing Check
        need, can_parallel = self.check_memory(request, references)

        # 4. Pipeline Parameters Base
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

        # 5. Denoising & Image Generation (Adaptive Router)
        logger.info("  [EXECUTION PIPELINE]")
        with self._lock:
            try:
                # PATH A: Parallel Tensor Batching
                if can_parallel and request.num_images > 1:
                    seeds = [(request.seed + i) % (MAX_SEED + 1) for i in range(request.num_images)]
                    generators = [self.generator_factory(s) for s in seeds]
                    
                    batch_kwargs = dict(kwargs)
                    batch_kwargs["prompt"] = [request.effective_prompt] * request.num_images
                    if "negative_prompt" in batch_kwargs and batch_kwargs["negative_prompt"] is not None:
                        batch_kwargs["negative_prompt"] = [batch_kwargs["negative_prompt"]] * request.num_images
                    if references:
                        batch_kwargs["image"] = [references] * request.num_images

                    logger.info("    >>> [PARALLEL TENSOR PASS START] Generating %d images in 1 simultaneous CUDA forward pass (Seeds: %s, Steps: %d)...", 
                                request.num_images, seeds, request.steps)
                    t_gen = time.perf_counter()
                    
                    try:
                        output = self.pipeline(**batch_kwargs, generator=generators)
                        gen_time = time.perf_counter() - t_gen
                        ms_per_step = (gen_time / request.steps) * 1000 if request.steps else 0
                        logger.info("    <<< [PARALLEL TENSOR PASS COMPLETE] Denoising for all %d images completed in %.2fs (avg %.1f ms/step, effective %.2fs/image)", 
                                    request.num_images, gen_time, ms_per_step, gen_time / request.num_images)

                        for idx, img in enumerate(output.images):
                            t_enc = time.perf_counter()
                            seed = seeds[idx]
                            encoded = encode_image(img, request.output_format, request.quality, seed)
                            enc_time = (time.perf_counter() - t_enc) * 1000
                            b64_len = len(encoded.get("image_base64", ""))
                            approx_bytes = int(b64_len * 3 / 4)
                            logger.info("    * Encoded parallel batch image %d/%d to %s (%s, took %.1fms)", 
                                        idx + 1, request.num_images, request.output_format.upper(), _format_bytes(approx_bytes), enc_time)
                            result["images"].append(encoded)

                    except Exception as batch_err:
                        logger.warning("    <!> [BATCH RETRY] Parallel tensor execution encountered '%s'. Gracefully falling back to sequential safety loop...", batch_err)
                        result["images"].clear()
                        for index in range(request.num_images):
                            seed = (request.seed + index) % (MAX_SEED + 1)
                            t_sub = time.perf_counter()
                            out_sub = self.pipeline(**kwargs, generator=self.generator_factory(seed))
                            encoded = encode_image(out_sub.images[0], request.output_format, request.quality, seed)
                            result["images"].append(encoded)
                            logger.info("    * Fallback sequential image %d/%d generated in %.2fs", index + 1, request.num_images, time.perf_counter() - t_sub)

                # PATH B: Sequential Generation Loop
                else:
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
