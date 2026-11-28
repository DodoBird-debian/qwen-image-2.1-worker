# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "Pillow==12.3.0",
#   "runpod==1.12.0",
#   "aiohttp==3.14.5",
#   "aiohttp-retry==2.9.1",
# ]
# ///
"""Reproduce the reviewed worker's delivery failures with synthetic CPU images."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

from PIL import Image


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
FROZEN_REV = "99c28f5d8a7fe0a5fa489ef4d619b1453198fccb"


def job(count=1, size=(256, 256)):
    return {"id": "local-delivery-check", "input": {
        "prompt": "fixture", "width": size[0], "height": size[1],
        "num_images": count, "seed": 42,
    }}


def solid(seed):
    return Image.new("RGBA", (256, 256), (seed % 256, 21, 7, 255))


def noise(size, rng):
    return Image.frombytes("RGBA", size, rng.randbytes(size[0] * size[1] * 4))


def fixture_service(groups, image_for_seed=solid, **options):
    from qwen_worker.config import Settings
    from qwen_worker.service import ImageService

    def pipeline(*, generator, **kwargs):
        seeds = generator if isinstance(generator, list) else [generator]
        groups.append(seeds)
        return SimpleNamespace(images=[image_for_seed(seed) for seed in seeds])

    return ImageService(pipeline, int, Settings(), **options)


def handle(service, request):
    import handler

    with patch.object(handler, "get_service", return_value=service):
        return handler.handler(request)


def returned_seeds(result):
    return [item["seed"] for item in result.get("images", [])]


def compare(observed, expected, matched_status="reproduced"):
    differences = {k: v for k, v in expected.items() if observed[k] != v}
    if differences:
        return {"status": "changed", "observed": observed, "expected": differences}
    return {"status": matched_status, "observed": observed}


def upload():
    import handler
    from aiohttp import ClientConnectionError
    from runpod.serverless.modules import rp_http, rp_job

    groups, posts, events = [], [], []
    service = fixture_service(groups)
    request = job()
    del request["input"]["seed"]

    async def send(method, url, *, data, **kwargs):
        events.append(f"SDK {method}")
        posts.append(returned_seeds(json.loads(data)["output"]))
        if len(posts) == 1:
            raise ClientConnectionError("fixture disconnected before delivery")
        return SimpleNamespace(status=200, method=method, closed=True,
                               text=AsyncMock(return_value="OK"), raise_for_status=lambda: None)

    def worker_info(message, *args):
        if "[RUNPOD JOB COMPLETED]" in message:
            events.append("worker logged JOB COMPLETED")

    def sdk_error(message, *args):
        if "Failed to return job results" in message:
            events.append("SDK logged upload failure")

    def sdk_info(message, *args):
        if message == "Finished.":
            events.append("SDK logged Finished.")

    async def deliver_twice():
        session = SimpleNamespace(headers={}, request=send)
        config = {"handler": handler.serverless_handler}
        # Keep the SDK retry client and result serializer. Only the HTTP session is fake.
        with (patch.object(handler, "get_service", return_value=service),
              patch("qwen_worker.requests.secrets.randbelow", side_effect=[42, 142]),
              patch.object(handler.logger, "info", side_effect=worker_info),
              patch.object(rp_http.log, "error", side_effect=sdk_error),
              patch.object(rp_http.log, "info", side_effect=sdk_info)):
            await rp_job.handle_job(session, config, request)
            attempts = len(posts)
            first_events = events.copy()
            await rp_job.handle_job(session, config, request)
        return compare(
            {"first_delivery_post_attempts": attempts, "first_delivery_events": first_events,
             "posted_seed_lists": posts, "pipeline_calls_for_same_job_id": groups},
            {"first_delivery_post_attempts": 1, "first_delivery_events": [
                "worker logged JOB COMPLETED", "SDK POST", "SDK logged upload failure", "SDK logged Finished."],
             "posted_seed_lists": [[42], [142]], "pipeline_calls_for_same_job_id": [[42], [142]]})

    return asyncio.run(deliver_twice())


def parallel():
    from qwen_worker.images import encode_image

    groups, attempted, encoded = [], [], []

    def encode(image, fmt, quality, seed, **options):
        attempted.append(seed)
        if seed == 44:
            raise OSError("fixture encoder failed on seed 44")
        result = encode_image(image, fmt, quality, seed, **options)
        encoded.append(seed)
        return result

    with patch("qwen_worker.service.encode_image", side_effect=encode):
        result = handle(fixture_service(groups), job(4))
    return compare(
        {"pipeline_calls": groups, "encoding_attempts": attempted, "encoded_seeds": encoded,
         "returned_seeds": returned_seeds(result), "refresh_worker": result.get("refresh_worker"),
         "error": result.get("error")},
        {"pipeline_calls": [[42, 43, 44, 45]], "encoding_attempts": [42, 43, 44],
         "encoded_seeds": [42, 43], "returned_seeds": [], "refresh_worker": True})


def sequential():
    from qwen_worker.images import encode_image
    from qwen_worker.memory import GIB
    from qwen_worker.service import OUTPUT_ENVELOPE_HEADROOM
    from qwen_worker.transport import json_bytes

    groups, fits = [], []
    frame = noise((2048, 1024), random.Random(42))
    service = fixture_service(groups, lambda seed: frame, memory_budget=lambda: 5 * GIB)
    fit = service._fit_output

    def observe_fit(result, originals, request, **options):
        attempt = {"seeds": [seed for image, seed in originals],
                   "recovery": options.get("generation_failed", False)}
        fits.append(attempt)
        size = fit(result, originals, request, **options)
        item = result["images"][0]
        with Image.open(io.BytesIO(base64.b64decode(item["image_base64"], validate=True))) as decoded:
            decoded.load()
            attempt.update(json_bytes=json_bytes(result), first_decoded_size=list(decoded.size))
        return size

    def optimize(*args, **kwargs):
        if len(groups) >= 2:
            raise OSError("fixture optimizer unavailable from pass two onward")
        return encode_image(*args, **kwargs)

    with (patch.object(service, "_fit_output", side_effect=observe_fit),
          patch("qwen_worker.transport.encode_image", side_effect=optimize)):
        result = handle(service, job(2, frame.size))
    budget = service.settings.max_output_bytes - OUTPUT_ENVELOPE_HEADROOM
    first_fit = fits[0].get("json_bytes") if fits else None
    return compare(
        {"pipeline_calls": groups, "fit_attempts": fits, "inline_body_budget": budget,
         "prior_fit_decoded_and_within_budget": first_fit is not None and first_fit <= budget,
         "repacked_first_pass": any(a["recovery"] and a["seeds"] == [42] for a in fits),
         "returned_seeds": returned_seeds(result), "error": result.get("error")},
        {"pipeline_calls": [[42], [43]], "prior_fit_decoded_and_within_budget": True,
         "repacked_first_pass": True, "returned_seeds": []})


def late_stop():
    import handler

    groups, fitted = [], []
    in_cleanup, release_cleanup = threading.Event(), threading.Event()

    def cleanup():
        in_cleanup.set()
        if not release_cleanup.wait(10):
            raise TimeoutError("fixture cleanup gate timed out")

    service = fixture_service(groups, release_memory=cleanup)
    fit = service._fit_output

    def observe_fit(result, *args, **kwargs):
        size = fit(result, *args, **kwargs)
        fitted.extend(returned_seeds(result))
        return size

    async def cancel_after_fit():
        with (patch.object(handler, "get_service", return_value=service),
              patch.object(service, "_fit_output", side_effect=observe_fit)):
            task = asyncio.create_task(handler.serverless_handler(job(2)))
            try:
                if not await asyncio.to_thread(in_cleanup.wait, 10):
                    raise TimeoutError("fixture did not reach final cleanup")
                task.cancel()
                await asyncio.sleep(0)  # Let the wrapper signal the inference thread.
            finally:
                release_cleanup.set()
                outcome, = await asyncio.gather(task, return_exceptions=True)
        if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
            raise outcome
        return compare(
            {"pipeline_calls": groups, "fitted_seeds_before_stop": fitted, "cancelled": task.cancelled(),
             "returned_seeds": returned_seeds(outcome) if isinstance(outcome, dict) else []},
            {"pipeline_calls": [[42, 43]], "fitted_seeds_before_stop": [42, 43],
             "cancelled": True, "returned_seeds": []})

    return asyncio.run(cancel_after_fit())


def packing():
    from qwen_worker.service import OUTPUT_ENVELOPE_HEADROOM
    from qwen_worker.transport import json_bytes

    def memory_mib(field):
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text().splitlines():
                if line.startswith(field + ":"):
                    return round(int(line.split()[1]) / 1024, 1)
        return None

    groups = []
    size, rng = (2048, 2048), random.Random(42)
    frames = [noise(size, rng) for _ in range(4)]
    service = fixture_service(groups, lambda seed: frames[seed - 42])
    rss = memory_mib("VmRSS")
    previous_peak = memory_mib("VmHWM")
    start = time.monotonic()
    result = service.handle(job(4, size))
    seconds = time.monotonic() - start
    peak = memory_mib("VmHWM")
    response_size = json_bytes(result)
    return compare(
        {"pipeline_calls": groups, "rss_with_input_frames_mib": rss, "peak_before_packing_mib": previous_peak,
         "peak_after_packing_mib": peak, "seconds": round(seconds, 3), "json_bytes": response_size,
         "returned_dimensions": [[i["width"], i["height"]] for i in result["images"]],
         "returned_seeds": returned_seeds(result),
         "within_cap": response_size + OUTPUT_ENVELOPE_HEADROOM <= service.settings.max_output_bytes},
        {"pipeline_calls": [[42, 43, 44, 45]], "returned_seeds": [42, 43, 44, 45],
         "within_cap": True}, matched_status="measured")


def frozen():
    from qwen_worker import service as service_module
    from qwen_worker.images import encode_image

    groups = []
    frame = noise((2048, 1024), random.Random(42))
    service = fixture_service(groups, lambda seed: frame)
    result = handle(service, job(size=frame.size))
    base64_bytes = len(encode_image(frame, "png", 95, 42)["image_base64"])
    return compare(
        {"service_sha256": hashlib.sha256(Path(service_module.__file__).read_bytes()).hexdigest(),
         "pipeline_calls": groups, "inline_cap_bytes": service.settings.max_output_bytes,
         "png_base64_bytes": base64_bytes, "returned_seeds": returned_seeds(result),
         "size_rejection": "inline output limit" in result.get("error", ""), "error": result.get("error")},
        {"pipeline_calls": [[42]], "returned_seeds": [], "size_rejection": True})


def cli():
    from qwen_worker.images import encode_image
    from scripts import client

    submitted, polled, errors = [], [], []
    output = {"images": [encode_image(solid(42), "png", 95, 42)]}

    def urlopen(request, **options):
        last = request.full_url.rsplit("/", 1)[-1]
        if request.get_method() == "POST" and last == "run":
            submitted.append(f"local-{len(submitted) + 1}")
            response = {"id": submitted[-1], "status": "IN_QUEUE"}
        elif request.get_method() == "GET" and last in submitted:
            first_poll = last not in polled
            polled.append(last)
            if first_poll:
                raise HTTPError(request.full_url, 503, "fixture unavailable", {}, None)
            response = {"id": last, "status": "COMPLETED", "output": output}
        else:
            raise AssertionError(f"unexpected fixture operation {request.get_method()} {last}")
        return io.BytesIO(json.dumps(response).encode())

    with tempfile.TemporaryDirectory() as temp:
        dest = Path(temp)
        argv = ["client", "local-endpoint", "--input", str(ROOT / "examples/smoke.json"),
                "--output-dir", temp]
        with (patch.object(client, "urlopen", side_effect=urlopen), patch.object(client.time, "sleep"),
              patch.dict(os.environ, {"RUNPOD_API_KEY": "local-fixture"}), patch.object(sys, "argv", argv)):
            for _ in range(2):
                try:
                    client.main()
                except RuntimeError as exc:
                    errors.append(str(exc))
            files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
            cli_polls = polled.copy()
            original = client.api_call(f"{client.API_ROOT}/local-endpoint/status/{submitted[0]}", "local-fixture")
            recovered = client.save_images(original, dest / "recovered")
    return compare(
        {"submitted_ids": submitted, "cli_status_polls": cli_polls, "cli_errors": errors,
         "files_before_manual_recovery": files, "original_seeds_recovered_by_get": returned_seeds(recovered)},
        {"submitted_ids": ["local-1", "local-2"], "cli_status_polls": ["local-1", "local-2"],
         "cli_errors": ["Runpod returned HTTP 503"] * 2, "files_before_manual_recovery": [],
         "original_seeds_recovered_by_get": [42]})


CASES = {"upload": upload, "parallel": parallel, "sequential": sequential, "late-stop": late_stop,
         "packing": packing, "frozen": frozen, "cli": cli}


@contextmanager
def source_tree(case, frozen_root):
    if case != "frozen":
        yield ROOT
        return
    sibling = ROOT.parent / "qwen-image-2.1-worker-frozen"
    if frozen_root or sibling.is_dir():
        source = (frozen_root or sibling).resolve()
        if not (source / "qwen_worker/service.py").is_file() or not (source / "handler.py").is_file():
            raise ValueError(f"Missing worker source in {source}")
        yield source
        return
    # A standalone checkout can read the original frozen source from its own Git objects.
    git = ["git", "-C", str(ROOT)]
    tree = subprocess.run([*git, "ls-tree", "-r", "--name-only", FROZEN_REV, "--", "handler.py", "qwen_worker"],
                          capture_output=True, text=True)
    if tree.returncode or not tree.stdout.strip():
        raise RuntimeError(f"Supply --frozen-root or make commit {FROZEN_REV} available in the local checkout")
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp)
        for name in tree.stdout.splitlines():
            dest = source / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(subprocess.check_output([*git, "show", f"{FROZEN_REV}:{name}"]))
        yield source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="all", choices=["all", *CASES])
    parser.add_argument("--frozen-root", type=Path, help="path to an existing frozen worker checkout")
    args = parser.parse_args()
    if args.case == "all":
        reports = {}
        for case in CASES:
            cmd = [sys.executable, "-B", str(SCRIPT), case]
            if args.frozen_root and case == "frozen":
                cmd += ["--frozen-root", str(args.frozen_root)]
            child = subprocess.run(cmd, capture_output=True, text=True)
            try:
                reports[case] = json.loads(child.stdout)
            except ValueError:
                reports[case] = {"status": "error", "error": child.stderr.strip() or f"child exit {child.returncode}"}
        sys.path.insert(0, str(ROOT))
        from qwen_worker.runtime import source_fingerprint
        report = {"python": platform.python_version(), "platform": platform.system(),
                  "packages": {p: version(p) for p in ("Pillow", "runpod", "aiohttp", "aiohttp-retry")},
                  "worker_source_fingerprint": source_fingerprint(), "cases": reports}
        success = all(r["status"] in ("reproduced", "measured") for r in reports.values())
    else:
        try:
            with source_tree(args.case, args.frozen_root) as source, open(os.devnull, "w") as quiet:
                sys.path.insert(0, str(source))
                with redirect_stdout(quiet):
                    logging.disable(logging.CRITICAL)
                    report = CASES[args.case]()
        except Exception as exc:
            report = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        success = report["status"] in ("reproduced", "measured")
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
