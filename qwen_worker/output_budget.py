"""Conservative transport routing; never reject a request from an upper bound.

Only metadata-free, 8-bit RGB(A) PNGs from the pinned pipeline/encoder have a
bound here. Unknown formats/encoders route sequentially and retain the final
exact JSON check. This does not claim that an uncertain batch will overflow.
"""

import json

from PIL import ImageFile, __version__ as pillow_version

from qwen_worker.requests import MAX_SEED, GenerationRequest


def png_byte_bound(width: int, height: int) -> int | None:
    if pillow_version != "12.3.0" or ImageFile.MAXBLOCK != 65536:
        return None
    # At most four 8-bit channels plus a filter byte per scanline. Pillow uses
    # deflateInit2(windowBits=15, memLevel=9, Z_FILTERED), with no intermediate
    # flushes. The general zlib/zlib-ng deflateBound envelope is larger than
    # the tighter default-settings bound (which must not be used at memLevel=9).
    source = height * (4 * width + 1)
    compressed = source + (source + 7) // 8 + (source + 63) // 64 + 11
    # Signature, IHDR, IEND, plus 12 bytes per IDAT chunk. ImageFile._save uses
    # max(MAXBLOCK, width*4) output buffers, emitting full buffers until finish.
    chunk_size = max(ImageFile.MAXBLOCK, width * 4)
    chunks = (compressed + chunk_size - 1) // chunk_size
    return 8 + 25 + 12 + compressed + chunks * 12


def parallel_output_fits(request: GenerationRequest, result: dict, limit: int, headroom: int) -> bool:
    if request.output_format != "png":
        return False  # No justified JPEG/WebP worst-case bound in this worker.
    image_bytes = png_byte_bound(request.width, request.height)
    if image_bytes is None:
        return False
    payload = dict(result)
    payload["images"] = [
        {"image_base64": "", "mime_type": "image/png", "width": request.width,
         "height": request.height, "seed": MAX_SEED}
        for _ in range(request.num_images)
    ]
    payload["inference_seconds"] = 0.0
    metadata = len(json.dumps(payload, ensure_ascii=True).encode("utf-8"))
    encoded_images = request.num_images * 4 * ((image_bytes + 2) // 3)
    # One allowance for the actual final check/envelope, another for timing
    # digits and future metadata. The final serialized check is authoritative.
    return metadata + encoded_images <= limit - 2 * headroom
