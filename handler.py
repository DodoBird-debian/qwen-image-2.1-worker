"""Runpod queue entrypoint; load a single pipeline before accepting jobs."""

import argparse
import base64
import json
import logging
from pathlib import Path

import runpod
import torch

from qwen_worker.config import Settings
from qwen_worker.requests import InputError
from qwen_worker.runtime import load_service

logger = logging.getLogger(__name__)

_service = None


def get_service():
    global _service
    if _service is None:
        _service = load_service(Settings.from_env())
    return _service


def handler(job):
    service = get_service()
    try:
        return service.handle(job)
    except InputError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        # Recoverable validation errors never reset the loaded model. CUDA failures do.
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            logger.exception("CUDA memory exhausted")
            return {
                "error": "CUDA out of memory. Set CPU_OFFLOAD=text_encoder (or model), reduce dimensions or reference images.",
                "refresh_worker": True,
            }
        logger.exception("Inference failed")
        return {"error": "Inference failed; inspect worker logs", "refresh_worker": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, help="Run one input JSON on a CUDA GPU and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args, sdk_args = parser.parse_known_args()

    if args.local:
        if sdk_args:
            parser.error(f"Unrecognized local arguments: {' '.join(sdk_args)}")
        result = handler(json.loads(args.local.read_text(encoding="utf-8")))
        if "error" in result:
            raise SystemExit(result["error"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(result["images"]):
            extension = image["mime_type"].split("/")[1]
            destination = args.output_dir / f"image-{index}-{image['seed']}.{extension}"
            destination.write_bytes(base64.b64decode(image.pop("image_base64")))
            image["path"] = str(destination)
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        get_service()
        runpod.serverless.start({"handler": handler})

