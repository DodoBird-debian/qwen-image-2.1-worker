"""Runpod queue entrypoint; load a single pipeline before accepting jobs."""

import time

_PROCESS_STARTED = time.perf_counter()

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import threading
from pathlib import Path

# Runtime resolves the required local snapshot before importing GPU libraries.
torch = None


class FlushStreamHandler(logging.StreamHandler):
    """Guarantees immediate, unbuffered stdout flush for RunPod serverless log streaming."""
    def emit(self, record):
        super().emit(record)
        self.flush()


# Configure root logger to output unbuffered to stdout immediately
_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler = FlushStreamHandler(sys.stdout)
_handler.setFormatter(_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler],
    force=True,
)
logger = logging.getLogger("qwen_worker.handler")
logger.info("[STARTUP] Python process started; measuring container-side initialization")
_sdk_started = time.perf_counter()
try:
    import runpod
except ImportError:
    runpod = None
logger.info("[STARTUP] RunPod SDK import=%.3fs; process elapsed=%.3fs",
            time.perf_counter() - _sdk_started, time.perf_counter() - _PROCESS_STARTED)

def configure_delivery_retry():
    """Retry result uploads using the same job ID after transient failures."""
    from aiohttp import ClientConnectionError, ServerTimeoutError
    from aiohttp_retry import FibonacciRetry
    from runpod.serverless.modules import rp_http

    def retry_options(*args, **kwargs):
        kwargs["exceptions"] = {ClientConnectionError, ServerTimeoutError, asyncio.TimeoutError}
        return FibonacciRetry(*args, **kwargs)

    rp_http.FibonacciRetry = retry_options
    original = rp_http._transmit

    async def transmit(session, url, data):
        job_id = session.headers.get("X-Request-ID", "unknown")
        await original(session, url, data)
        logger.info("[JOB %s] RunPod acknowledged result upload", job_id)

    rp_http._transmit = transmit

from qwen_worker.config import Settings
from qwen_worker.requests import InputError
from qwen_worker.runtime import load_service, prepare_startup_model, source_fingerprint
from qwen_worker.service import GenerationStopped, check_stop
from qwen_worker.recovery import job_lock
logger.info("[STARTUP] Worker source fingerprint=%s", source_fingerprint())

_service = None
_service_lock = threading.Lock()


def get_service():
    global _service, torch
    if _service is None:
        with _service_lock:
            if _service is None:
                logger.info("[INIT] Initializing Qwen-Image-2.1 service runtime...")
                t0 = time.perf_counter()
                settings = prepare_startup_model(Settings.from_env())
                _service = load_service(settings)
                torch = sys.modules.get("torch")
                logger.info("[INIT] Service initialization completed in %.2fs. Listening for RunPod serverless jobs.", time.perf_counter() - t0)
                logger.info("[STARTUP] Process-to-service-ready=%.3fs. RunPod queue/allocation time before process launch is not included.",
                            time.perf_counter() - _PROCESS_STARTED)
    return _service


def handler(job, *, stop_event: threading.Event | None = None):
    job_id = job.get("id", "unknown_job") if isinstance(job, dict) else "raw_input"
    logger.info("=" * 80)
    logger.info(">>> [RUNPOD JOB RECEIVED] ID: %s", job_id)
    logger.info("-" * 80)
    t0 = time.perf_counter()

    try:
        check_stop(stop_event)
        with job_lock(job):
            check_stop(stop_event)
            service = get_service()
            result = service.handle(job, stop_event=stop_event)
        check_stop(stop_event)
        elapsed = time.perf_counter() - t0
        logger.info("-" * 80)
        logger.info("<<< [INFERENCE COMPLETE] ID: %s in %.2fs; result delivery pending", job_id, elapsed)
        logger.info("=" * 80)
        return result
    except GenerationStopped:
        logger.info("[JOB %s] Stopped after inference cleanup", job_id)
        raise
    except InputError as exc:
        elapsed = time.perf_counter() - t0
        logger.warning("-" * 80)
        logger.warning("<!> [INPUT VALIDATION REJECTED] Job %s rejected in %.2fs: %s", job_id, elapsed, exc)
        logger.warning("=" * 80)
        return {"error": str(exc)}
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("-" * 80)
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**3)
                res = torch.cuda.memory_reserved() / (1024**3)
                logger.error("<X> [CUDA OOM FATAL] Job %s failed after %.2fs. VRAM Allocated: %.2f GiB, Reserved: %.2f GiB", job_id, elapsed, alloc, res)
                try:
                    logger.error("    VRAM Summary:\n%s", torch.cuda.memory_summary(abbreviated=True))
                except Exception:
                    pass
            else:
                logger.error("<X> [CUDA OOM FATAL] Job %s failed after %.2fs", job_id, elapsed)
            logger.error("=" * 80)
            return {
                "error": "CUDA out of memory. Reduce output dimensions or reference processing, or use CPU_OFFLOAD=model or CPU_OFFLOAD=sequential.",
                "refresh_worker": True,
            }
        logger.exception("<X> [INFERENCE ERROR] Job %s failed after %.2fs: %s", job_id, elapsed, exc)
        logger.error("=" * 80)
        return {"error": f"Inference error: {str(exc)}", "refresh_worker": True}


async def serverless_handler(job):
    # RunPod cancels the awaiting task. Shield the thread so the job stays
    # active until cooperative cancellation has unwound the GPU work.
    stop_event = threading.Event()
    work = asyncio.create_task(asyncio.to_thread(handler, job, stop_event=stop_event))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        stop_event.set()
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not work.cancelled():
            try:
                work.result()
            except GenerationStopped:
                pass
            except Exception:
                logger.exception("Inference failed while the cancelled job was unwinding")
        raise


if __name__ == "__main__":
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
        # Pre-warm service before registering with RunPod serverless queue
        get_service()
        configure_delivery_retry()
        runpod.serverless.start({"handler": serverless_handler})

