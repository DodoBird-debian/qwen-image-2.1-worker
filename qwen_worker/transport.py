"""Fit completed images into the serialized inline response without regenerating."""

import json
import logging
import math
import time
from itertools import combinations

from PIL import Image

from qwen_worker.images import encode_image
from qwen_worker.requests import GenerationRequest, InputError

logger = logging.getLogger("qwen_worker.transport")
MAX_INLINE_BYTES = 10_000_000
MIN_OUTPUT_SIDE = 256


def json_bytes(result: dict) -> int:
    # ASCII escaping is at least as large as the SDK's ensure_ascii=False.
    # Count base64 AND JSON metadata; the caller reserves envelope headroom.
    return len(json.dumps(result, ensure_ascii=True).encode("utf-8"))


def image_json_bytes(image: dict) -> int:
    return json_bytes({**image, "image_base64": ""}) + len(image["image_base64"])


def fit_output(result: dict, originals: list[tuple[Image.Image, int]],
               request: GenerationRequest, budget: int, *, final: bool = True,
               generation_failed: bool = False) -> int:
    """Optimize, resize within a usable floor, try JPEG, then maximize count.

    Every encoding uses original CPU pixels. JPEG fallback may flatten alpha on
    white; delivery metadata explains changes and partial batches. Never retry
    inference. Candidate sizes include delivery metadata before accepting them.
    """
    by_seed = dict((seed, image) for image, seed in originals)
    original_encodings = {item["seed"]: item for item in result["images"]}
    alpha = {}
    quality = request.quality
    fmt = request.output_format

    def has_alpha(seed: int) -> bool:
        if seed not in alpha:
            original = by_seed[seed]
            alpha[seed] = "A" in original.getbands() and original.getchannel("A").getextrema()[0] < 255
        return alpha[seed]

    def delivery(*, lossy: bool = False) -> None:
        warnings = []
        jpeg = [item for item in result["images"] if item["mime_type"] == "image/jpeg"]
        if jpeg and request.output_format != "jpeg":
            warnings.append("JPEG fallback was used to fit the response size limit.")
        if any(has_alpha(item["seed"]) for item in jpeg):
            warnings.append("JPEG does not support transparency; alpha was flattened on white.")
        changed = [item for item in result["images"] if item is not original_encodings[item["seed"]]]
        if changed:
            if fmt in {"jpeg", "webp"} and quality < request.quality:
                warnings.append(f"{fmt.upper()} encoding quality was reduced to {quality} to fit the response size limit.")
            elif fmt == "webp" and lossy and any("A" in by_seed[item["seed"]].getbands() for item in changed):
                warnings.append("Lossy WebP encoding was used to fit the response size limit.")
        if any((item["width"], item["height"]) != by_seed[item["seed"]].size for item in result["images"]):
            warnings.append("Image dimensions were reduced to fit the response size limit.")
        count = len(result["images"])
        partial = count < len(originals) or ((final or generation_failed) and count < request.num_images)
        if generation_failed:
            warnings.append(f"Returned {count} of {request.num_images} requested images after a generation pass failed. No inference was retried.")
        elif partial:
            warnings.append(f"Returned {count} of {request.num_images} requested images to fit the response size limit. Missing images were not retried.")
        if not warnings:
            result.pop("delivery", None)
            return
        result["delivery"] = {"requested_images": request.num_images,
                              "returned_images": count, "partial": partial,
                              "warnings": warnings}

    delivery()
    before = json_bytes(result)
    if before <= budget:
        return before
    started = time.perf_counter()
    # Leave room for the final elapsed-time digits when fitting between passes.
    target = budget - 64
    # Do not reduce an image to a meaningless 1x1 just to claim the batch fits.
    # Images already smaller than the floor are never enlarged.
    floor = max(min(1.0, MIN_OUTPUT_SIDE / min(image.size)) for image, _ in originals)

    def attempt(scale: float, *, lossy: bool = False) -> int:
        images = []
        for original, seed in originals:
            size = (max(1, int(original.width * scale)), max(1, int(original.height * scale)))
            image = original if size == original.size else original.resize(size, Image.Resampling.LANCZOS)
            candidate = encode_image(image, fmt, quality, seed, optimize=True, lossy=lossy)
            original_encoding = original_encodings[seed]
            images.append(min((original_encoding, candidate), key=image_json_bytes))
        result["images"] = images
        delivery(lossy=lossy)
        return json_bytes(result)

    def finish(stage: str, after: int, *, lossy: bool = False) -> int:
        # Low-complexity frames can often keep their original encoding
        # once the larger frames fit the cap.
        images = result["images"]
        for index, candidate in enumerate(images):
            original = original_encodings[candidate["seed"]]
            if candidate is original:
                continue
            delta = image_json_bytes(original) - image_json_bytes(candidate)
            if after + delta > target:
                continue
            images[index] = original
            delivery(lossy=lossy)
            size = json_bytes(result)
            if size <= target:
                after = size
            else:
                images[index] = candidate
                delivery(lossy=lossy)
        dimensions = ", ".join(f"{by_seed[item['seed']].width}x{by_seed[item['seed']].height}->{item['width']}x{item['height']}"
                               for item in result["images"])
        logger.warning("[OUTPUT COMPRESSION] %s: %d -> %d JSON bytes (budget=%d), format=%s, quality=%s, "
                       "images=%d/%d requested, dimensions=%s, elapsed=%.3fs", stage, before, after, budget,
                       fmt, quality if fmt != "png" else "n/a (PNG)", len(result["images"]),
                       request.num_images, dimensions, time.perf_counter() - started)
        return after

    size = attempt(1.0)
    if size <= target:
        return finish("encoder optimization at requested quality", size)

    lossy = fmt in {"jpeg", "webp"}
    if lossy:
        for quality in dict.fromkeys(min(request.quality, q) for q in (85, 70, 55, 40)):
            if fmt == "jpeg" and quality == request.quality:
                continue
            size = attempt(1.0, lossy=True)
            if size <= target:
                return finish("quality reduction", size, lossy=True)

    scale = 1.0
    for _ in range(6):
        if scale <= floor:
            break
        scale = max(floor, scale * min(0.8, math.sqrt(max(1, target) / max(1, size)) * 0.94))
        size = attempt(scale, lossy=lossy)
        if size <= target:
            return finish("resized to fit", size, lossy=lossy)
    if scale > floor:
        scale = floor
        size = attempt(scale, lossy=lossy)
        if size <= target:
            return finish("resized to minimum usable size", size, lossy=lossy)

    # Format/quality fallback happens before dropping any generated images.
    previous_quality = quality
    fmt = "jpeg"
    for quality in dict.fromkeys(min(request.quality, q) for q in (85, 70, 55, 40, 25, 15, 5, 1)):
        if request.output_format == "jpeg" and quality >= previous_quality:
            continue  # Already tried this quality (or better) at this size.
        size = attempt(scale, lossy=True)
        if size <= target:
            return finish("JPEG fallback", size, lossy=True)
    # The last encoded batch is JPEG at the lowest permitted quality.
    # Search every subset (at most 15) so a large first image cannot hide a
    # larger fitting count of later, smaller images. Preserve their seed order.
    encoded = result["images"]
    for count in range(len(encoded) - 1, 0, -1):
        for subset in combinations(encoded, count):
            result["images"] = list(subset)
            delivery(lossy=True)
            size = json_bytes(result)
            if size <= target:
                return finish("partial batch", size, lossy=True)
    raise InputError("Encoded images exceed the inline output limit even after JPEG compression. "
                     "Not even one image and its metadata fit the configured limit.")
