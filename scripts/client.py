"""Submit a job, wait for completion, and save images without printing base64 or API keys."""

import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import http.client
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

API_ROOT = "https://api.runpod.ai/v2"
MAX_RUN_REQUEST_BYTES = 10_000_000
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
JOB_ID = re.compile(r"[A-Za-z0-9_-]+")


class RunpodHTTPError(RuntimeError):
    def __init__(self, status: int):
        super().__init__(f"Runpod returned HTTP {status}")
        self.status = status


def transient_status(error: RunpodHTTPError) -> bool:
    return error.status in (408, 429) or error.status >= 500


def api_call(url: str, key: str, body: dict | None = None, *, timeout: float = 60) -> dict:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    if payload is not None and url.rstrip("/").endswith("/run") and len(payload) > MAX_RUN_REQUEST_BYTES:
        raise ValueError("Request exceeds the 10,000,000-byte Runpod /run payload limit")
    request = Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        # Avoid logging headers, the auth key, or a server body that could echo input images.
        exc.close()
        raise RunpodHTTPError(exc.code) from None


def atomic_write(path: Path, value: bytes) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    atomic_write(path, json.dumps(value, indent=2).encode())


@contextmanager
def job_receipt(destination: Path, endpoint: str):
    receipts = destination / ".runpod"
    receipts.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (receipts / f"{endpoint}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another client is managing endpoint {endpoint} in {destination}") from None
        yield receipts / f"{endpoint}.json"


def get_status(endpoint: str, key: str, job_id: str, deadline: float) -> dict:
    for attempt in range(5):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Client wait expired for job {job_id}. Resume with --resume {job_id}.")
        try:
            result = api_call(f"{endpoint}/status/{job_id}", key, timeout=min(60, remaining))
        except RunpodHTTPError as error:
            if not transient_status(error):
                raise
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            pass
        else:
            if not isinstance(result, dict) or result.get("id") != job_id:
                raise RuntimeError(f"Runpod returned an invalid status for job {job_id}")
            if result.get("status") not in TERMINAL | {"IN_QUEUE", "IN_PROGRESS"}:
                raise RuntimeError(f"Unexpected Runpod status for job {job_id}")
            return result
        if attempt < 4:
            time.sleep(min(2 ** attempt, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"Status temporarily unavailable for job {job_id}. Resume with --resume {job_id}.")


def poll_job(endpoint: str, key: str, job_id: str, deadline: float, result=None) -> dict:
    while not result or result.get("status") not in TERMINAL:
        result = get_status(endpoint, key, job_id, deadline)
        if result["status"] not in TERMINAL:
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    return result


def collect_job(args, key: str, payload: dict | None) -> dict | None:
    endpoint_id = args.endpoint_id
    endpoint = f"{API_ROOT}/{endpoint_id}"
    with job_receipt(args.output_dir, endpoint_id) as path:
        pending = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        known = pending.get("id") if pending else None
        if pending and (pending.get("endpoint") != endpoint_id or (known is not None and
                         (not isinstance(known, str) or not JOB_ID.fullmatch(known)))):
            raise RuntimeError(f"Unexpected receipt in {path}; inspect it before sending another job")
        if args.discard_uncertain:
            if not pending or known:
                raise RuntimeError("There is no unconfirmed submission to discard")
            path.unlink()
            print(f"Cleared the unconfirmed submission for {endpoint_id}.")
            return None
        result = None
        deadline = time.monotonic() + args.timeout
        if args.resume:
            if known and known != args.resume:
                raise RuntimeError(f"Use --resume {known} to recover the job already recorded in {path}")
            job_id = args.resume
            # Confirm a manually supplied ID before replacing an uncertain receipt.
            if known is None:
                result = get_status(endpoint, key, job_id, deadline)
                atomic_json(path, {"endpoint": endpoint_id, "id": job_id})
        else:
            if pending:
                how = f"--resume {known}" if known else "--resume JOB_ID after finding it in Runpod"
                raise RuntimeError(f"An existing submission is recorded in {path}. Use {how}.")
            atomic_json(path, {"endpoint": endpoint_id})
            try:
                result = api_call(f"{endpoint}/run", key, payload)
            except RunpodHTTPError as error:
                if 400 <= error.status < 500 and not transient_status(error):
                    path.unlink()
                    raise
                raise RuntimeError(f"Submission may have been accepted. Check Runpod and use --resume JOB_ID. Receipt: {path}") from None
            except (OSError, http.client.HTTPException, json.JSONDecodeError):
                raise RuntimeError(f"Submission may have been accepted. Check Runpod and use --resume JOB_ID. Receipt: {path}") from None
            job_id = result.get("id") if isinstance(result, dict) else None
            if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
                raise RuntimeError(f"Submission may have been accepted without a valid job ID. Receipt: {path}")
            atomic_json(path, {"endpoint": endpoint_id, "id": job_id})
            print(f"Submitted job {job_id}", flush=True)
        try:
            result = poll_job(endpoint, key, job_id, deadline, result)
        except RunpodHTTPError as error:
            if error.status == 404 and (known == job_id or not args.resume):
                path.unlink(missing_ok=True)
                raise RuntimeError(f"Job {job_id} is unavailable through Runpod status. Check the worker recovery store if configured.") from None
            raise
        if result["status"] != "COMPLETED":
            path.unlink(missing_ok=True)
            raise RuntimeError(f"Job {job_id}: {result.get('status')}: {result.get('error', '')}")
        metadata = save_images(result, args.output_dir / job_id)
        path.unlink(missing_ok=True)
        return metadata


def save_images(result: dict, destination: Path) -> dict:
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    output = result.get("output", result)
    if isinstance(output, dict) and "error" in output:
        raise RuntimeError(str(output["error"]))
    if not isinstance(output, dict) or not isinstance(output.get("images"), list) or not output["images"]:
        raise RuntimeError("The response contains no generated images")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in output.items() if key != "images"}
    metadata["images"] = []
    warnings = []
    extensions = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp"}
    for index, image in enumerate(output["images"]):
        mime = image.get("mime_type") if isinstance(image, dict) else None
        extension = extensions.get(mime) if isinstance(mime, str) else None
        if extension is None:
            warnings.append(f"Image {index + 1}: unexpected MIME type.")
            continue
        try:
            image_bytes = base64.b64decode(image["image_base64"], validate=True)
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                if Image.MIME.get(decoded.format) != mime:
                    raise ValueError("Image format differs from the response")
                decoded.verify()
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                decoded.load()
        except (KeyError, TypeError, ValueError, OSError, SyntaxError, Image.DecompressionBombError):
            warnings.append(f"Image {index + 1}: invalid image data.")
            continue
        path = destination / f"image-{index}.{extension}"
        atomic_write(path, image_bytes)
        metadata["images"].append(
            {
                **{key: value for key, value in image.items() if key != "image_base64"},
                "path": str(path),
            }
        )
    if not metadata["images"]:
        raise RuntimeError("The response contains no usable generated images. " + " ".join(warnings))
    if warnings:
        metadata["client_warnings"] = warnings
    atomic_json(destination / "result.json", metadata)
    return metadata


def recover_worker(root: Path, job_id: str, destination: Path) -> dict:
    """Read original frames as an operator of the mounted recovery volume."""
    directory = root / hashlib.sha256(job_id.encode()).hexdigest()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("job_id") != job_id or not isinstance(manifest.get("images"), list):
        raise RuntimeError(f"Unexpected worker recovery manifest for {job_id}")
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"job_id": job_id, "images": []}
    for entry in manifest["images"]:
        seed = entry["seed"]
        if type(seed) is not int or not 0 <= seed < 2 ** 63 or entry.get("file") != f"{seed}.png":
            raise RuntimeError(f"Unexpected worker recovery filename for {job_id}")
        pixels = (directory / entry["file"]).read_bytes()
        if hashlib.sha256(pixels).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Recovery file checksum failed for seed {seed}")
        with Image.open(io.BytesIO(pixels)) as decoded:
            if decoded.format != "PNG" or decoded.size != (entry["width"], entry["height"]):
                raise RuntimeError(f"Recovery file metadata differs for seed {seed}")
            decoded.load()
        path = destination / entry["file"]
        atomic_write(path, pixels)
        metadata["images"].append({**entry, "path": str(path)})
    if not metadata["images"]:
        raise RuntimeError(f"No worker frames have been saved for {job_id}")
    atomic_json(destination / "result.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint_id", nargs="?")
    parser.add_argument("--input", type=Path, default=Path("examples/generate.json"))
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Local editing reference; repeat for multiple images",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--timeout", type=int, default=1800)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--decode", type=Path, help="Decode an existing /status or worker result JSON"
    )
    action.add_argument("--resume", metavar="JOB_ID", help="Fetch an already submitted job by its Runpod ID")
    action.add_argument("--discard-uncertain", action="store_true",
                        help="Clear a receipt with no known ID after checking the endpoint")
    action.add_argument("--recover-worker", metavar="JOB_ID",
                        help="Copy original images using authorized filesystem access to the worker recovery volume")
    parser.add_argument("--recovery-dir", type=Path, help="Private mounted directory configured as QWEN_RECOVERY_DIR")
    args = parser.parse_args()
    if args.decode:
        result = json.loads(args.decode.read_text(encoding="utf-8-sig"))
        print(json.dumps(save_images(result, args.output_dir), indent=2))
        return
    if args.recover_worker:
        if not JOB_ID.fullmatch(args.recover_worker) or not args.recovery_dir:
            parser.error("--recover-worker needs a valid job ID and --recovery-dir")
        print(json.dumps(recover_worker(args.recovery_dir, args.recover_worker, args.output_dir / args.recover_worker), indent=2))
        return
    if not args.endpoint_id or not JOB_ID.fullmatch(args.endpoint_id):
        parser.error("Supply a valid endpoint id")
    if args.resume and not JOB_ID.fullmatch(args.resume):
        parser.error("Supply a valid Runpod job id")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    key = os.environ.get("RUNPOD_API_KEY", "")
    if args.discard_uncertain:
        collect_job(args, key, None)
        return
    if not key:
        parser.error("Set RUNPOD_API_KEY in the environment")
    if args.resume:
        print(json.dumps(collect_job(args, key, None), indent=2))
        return
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), dict):
        parser.error("The input file must contain an input object")
    if args.image:
        if len(args.image) > 10:
            parser.error("At most 10 reference images are supported")
        sizes = [path.stat().st_size for path in args.image]
        if sum(4 * ((size + 2) // 3) for size in sizes) > 10_000_000:
            parser.error("Reference images exceed the 10 MB base64 upload limit")
        payload["input"]["images"] = [
            base64.b64encode(path.read_bytes()).decode("ascii") for path in args.image
        ]
    if len(json.dumps(payload).encode("utf-8")) > MAX_RUN_REQUEST_BYTES:
        parser.error("Request exceeds the 10,000,000-byte Runpod /run payload limit")
    print(json.dumps(collect_job(args, key, payload), indent=2))


if __name__ == "__main__":
    main()
