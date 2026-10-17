"""Request orchestration, independent of GPU loading so it can be tested on CPU."""

import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext

from qwen_worker.config import Settings
from qwen_worker.images import decode_images, encode_image
from qwen_worker.memory import GIB, MemoryEstimate, estimate, text_token_bound
from qwen_worker.output_budget import parallel_output_fits
from qwen_worker.requests import MAX_SEED, GenerationRequest, InputError

logger = logging.getLogger("qwen_worker.service")
OUTPUT_ENVELOPE_HEADROOM = 4096


class ImageService:
    def __init__(
        self, pipeline, generator_factory: Callable, settings: Settings,
        memory_budget: Callable[[], int] | None = None,
        release_memory: Callable[[], None] | None = None,
        phase_budgets: Callable[[], dict[str, int]] | None = None,
        job_context: Callable | None = None,
        preflight: Callable | None = None,
    ):
        self.pipeline = pipeline
        self.generator_factory = generator_factory
        self.settings = settings
        self.memory_budget = memory_budget
        self.release_memory = release_memory
        self.phase_budgets = phase_budgets
        self.job_context = job_context
        self.preflight = preflight
        self._lock = threading.Lock()

    def check_memory(self, request: GenerationRequest, references: list) -> tuple[MemoryEstimate, bool]:
        need = estimate(
            [image.size for image in references], request.output_resolution,
            request.width, request.height, len(request.effective_prompt),
            request.true_cfg_scale > 1, request.use_kv_cache, request.num_images,
            positive_text_tokens=text_token_bound(request.effective_prompt, len(references)),
            negative_text_tokens=text_token_bound(request.negative_prompt or "", len(references)),
        )
        if self.phase_budgets is not None:
            budgets = self.phase_budgets()
        elif self.memory_budget is not None:
            budget = self.memory_budget()
            budgets = dict.fromkeys(("conditioning", "denoising", "decoding"), budget)
        else:
            logger.info("Memory guard disabled; estimates remain advisory")
            return need, request.num_images > 1

        sequential = {"conditioning": need.conditioning_bytes,
                      "denoising": need.total_bytes, "decoding": need.decoding_bytes}
        parallel = {"conditioning": need.conditioning_bytes,
                    "denoising": need.parallel_total_bytes, "decoding": need.parallel_decoding_bytes}
        for phase, required in sequential.items():
            available = budgets[phase]
            logger.info("Memory estimate %s: sequential %.2f GiB, batch %.2f GiB, tensor budget %.2f GiB",
                        phase, required / GIB, parallel[phase] / GIB, available / GIB)
            if required > available:
                raise InputError(
                    f"Request needs an estimated {required / GIB:.1f} GiB for {phase} tensors "
                    f"but only {available / GIB:.1f} GiB is available after model placement. "
                    "Use fewer references, smaller dimensions/output_resolution, a shorter prompt, "
                    "or a GPU/offload configuration with more headroom."
                )
        can_parallel = request.num_images > 1 and all(
            required <= budgets[phase] for phase, required in parallel.items()
        )
        logger.info("Routing %d output(s): %s; prefix KV %.2f GiB/sample, %.2f GiB/batch (heuristic admission)",
                    request.num_images, "parallel" if can_parallel else "sequential",
                    need.kv_cache_bytes / GIB, need.parallel_kv_cache_bytes / GIB)
        return need, can_parallel

    def _check_output_size(self, result: dict) -> None:
        # Restore the documented original cap with headroom for Runpod's envelope
        # and the final timing field. Measure JSON bytes, not container size.
        size = len(json.dumps(result, ensure_ascii=True).encode("utf-8"))
        if size > self.settings.max_output_bytes - OUTPUT_ENVELOPE_HEADROOM:
            raise InputError(
                "Encoded images exceed the inline output limit. Use fewer images, "
                "smaller dimensions, or JPEG when transparency is not required."
            )

    def _append_images(self, result: dict, output, seeds: list[int], request: GenerationRequest) -> None:
        images = getattr(output, "images", None)
        if images is None or len(images) != len(seeds):
            actual = None if images is None else len(images)
            raise RuntimeError(f"Pipeline returned {actual} images; expected {len(seeds)}")
        started = time.perf_counter()
        for image, seed in zip(images, seeds):
            result["images"].append(encode_image(image, request.output_format, request.quality, seed))
            self._check_output_size(result)
        logger.info("Output encoding: %d image(s), %.3fs", len(seeds), time.perf_counter() - started)

    def handle(self, job: dict) -> dict:
        if not isinstance(job, dict):
            raise InputError("job must contain an input object")
        request = GenerationRequest.parse(job.get("input", {}))
        decoded_at = time.perf_counter()
        references = decode_images(request.images)
        logger.info("Job %s: %dx%d, outputs=%d, steps=%d, CFG=%s, refs=%d, decode=%.3fs",
                    job.get("id", "unspecified_job"), request.width, request.height,
                    request.num_images, request.steps, request.true_cfg_scale,
                    len(references), time.perf_counter() - decoded_at)
        kwargs = {
            "prompt": request.effective_prompt, "width": request.width, "height": request.height,
            "num_inference_steps": request.steps, "true_cfg_scale": request.true_cfg_scale,
            "output_resolution": request.output_resolution, "use_kv_cache": request.use_kv_cache,
        }
        if references:
            kwargs["image"] = references  # One flat reference set shared by all outputs.
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        result = {
            "model": self.settings.model_id,
            "revision": None if self.settings.model_path else self.settings.model_revision,
            "parameters": {"num_inference_steps": request.steps, "true_cfg_scale": request.true_cfg_scale,
                           "output_resolution": request.output_resolution, "use_kv_cache": request.use_kv_cache},
            "images": [],
        }
        started = time.perf_counter()
        with self._lock:
            logger.info("Inference lock wait: %.3fs", time.perf_counter() - started)
            try:
                if self.preflight is not None:
                    self.preflight(request, references)
                # Observe available memory only after the previous job has
                # relinquished the shared mutable pipeline.
                _, can_parallel = self.check_memory(request, references)
                if can_parallel and not parallel_output_fits(
                    request, result, self.settings.max_output_bytes, OUTPUT_ENVELOPE_HEADROOM
                ):
                    can_parallel = False
                    logger.info("Using sequential outputs: inline transport fit is uncertain; check each output before the next")
                context = self.job_context() if self.job_context is not None else nullcontext()
                with context:
                    seeds = [(request.seed + index) % (MAX_SEED + 1) for index in range(request.num_images)]
                    groups = [seeds] if can_parallel else [[seed] for seed in seeds]
                    for group in groups:
                        call_kwargs = dict(kwargs)
                        if can_parallel:
                            # Encode a scalar prompt once, then repeat embeddings.
                            call_kwargs["num_images_per_prompt"] = len(group)
                            generator = [self.generator_factory(seed) for seed in group]
                        else:
                            generator = self.generator_factory(group[0])
                        inference_at = time.perf_counter()
                        output = self.pipeline(**call_kwargs, generator=generator)
                        logger.info("Pipeline call: outputs=%d, elapsed=%.3fs", len(group), time.perf_counter() - inference_at)
                        self._append_images(result, output, group, request)
                        del output
                # Errors (including OOM) propagate to the existing handler;
                # retries here retain failed tensors and hide deterministic bugs.
            finally:
                if self.release_memory is not None:
                    try:
                        self.release_memory()
                    except Exception:
                        logger.exception("Post-job memory cleanup failed")
        result["inference_seconds"] = round(time.perf_counter() - started, 3)
        self._check_output_size(result)
        logger.info("Job complete: %d image(s), %.3fs", len(result["images"]), result["inference_seconds"])
        return result
