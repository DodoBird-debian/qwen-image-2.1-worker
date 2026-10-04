"""Request orchestration, independent of GPU loading so it can be tested on CPU."""

import json
import threading
import time
from collections.abc import Callable

from qwen_worker.config import Settings
from qwen_worker.images import decode_images, encode_image
from qwen_worker.memory import GIB, estimate
from qwen_worker.requests import MAX_SEED, GenerationRequest, InputError


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

    def check_memory(self, request: GenerationRequest, references: list) -> None:
        if self.memory_budget is None:
            return
        need = estimate(
            [image.size for image in references],
            request.output_resolution,
            request.width,
            request.height,
            len(request.effective_prompt) + len(request.negative_prompt or ""),
            request.true_cfg_scale > 1,
            request.use_kv_cache,
        )
        budget = self.memory_budget()
        if need.total_bytes > budget:
            raise InputError(
                f"Request needs an estimated {need.total_bytes / GIB:.1f} GiB of free VRAM "
                f"(KV cache {need.kv_cache_bytes / GIB:.1f} GiB for {need.prefix_tokens} prefix tokens x "
                f"{need.branches} branch(es)) but only {budget / GIB:.1f} GiB is available. Use fewer "
                "reference images, output_resolution 512/1024, true_cfg_scale 1, or use_kv_cache false."
            )

    def handle(self, job: dict) -> dict:
        if not isinstance(job, dict):
            raise InputError("job must contain an input object")
        request = GenerationRequest.parse(job.get("input"))
        references = decode_images(request.images)
        self.check_memory(request, references)
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
        # Diffusers mutates its scheduler/attention state. Never overlap calls to one pipeline.
        with self._lock:
            try:
                for index in range(request.num_images):
                    seed = (request.seed + index) % (MAX_SEED + 1)
                    output = self.pipeline(**kwargs, generator=self.generator_factory(seed))
                    result["images"].append(
                        encode_image(output.images[0], request.output_format, request.quality, seed)
                    )
                    if len(json.dumps(result).encode("utf-8")) > self.settings.max_output_bytes - 4096:
                        raise InputError(
                            "Encoded images exceed the response limit. Request fewer/smaller images "
                            "or use webp/jpeg output."
                        )
            finally:
                if self.release_memory is not None:
                    try:
                        self.release_memory()
                    except Exception:
                        pass
        result["inference_seconds"] = round(time.perf_counter() - started, 3)
        return result

