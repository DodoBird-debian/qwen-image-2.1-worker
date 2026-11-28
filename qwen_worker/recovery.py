"""Optional recovery on a private, shared filesystem mounted into workers."""
import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def job_lock(job):
    root = os.environ.get("QWEN_RECOVERY_DIR")
    if not root or not isinstance(job, dict) or not isinstance(job.get("id"), str):
        yield
        return
    directory = Path(root)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = hashlib.sha256(job["id"].encode()).hexdigest()
    with (directory / (key + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def open_job(job):
    root = os.environ.get("QWEN_RECOVERY_DIR")
    if not root or not isinstance(job, dict) or not isinstance(job.get("id"), str):
        return None
    key = hashlib.sha256(job["id"].encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(job.get("input", {}), sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    directory = Path(root) / key
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["request_sha256"] != fingerprint:
            raise ValueError("Recovery job ID was reused for a different input")
    else:
        manifest = {"job_id": job["id"], "request_sha256": fingerprint, "images": []}
        atomic_write(manifest_path, json.dumps(manifest).encode())
    return directory, manifest


def atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_image(recovery, image, seed):
    if recovery is None:
        return
    directory, manifest = recovery
    output = io.BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    name = f"{seed}.png"
    atomic_write(directory / name, data)
    manifest["images"] = [item for item in manifest["images"] if item["seed"] != seed]
    manifest["images"].append({"seed": seed, "file": name, "width": image.width,
                               "height": image.height, "mime_type": "image/png",
                               "sha256": hashlib.sha256(data).hexdigest()})
    atomic_write(directory / "manifest.json", json.dumps(manifest).encode())


def save_result(recovery, result):
    if recovery is not None:
        atomic_write(recovery[0] / "result.json", json.dumps(result).encode())


def load_result(recovery):
    if recovery is None or not (recovery[0] / "result.json").exists():
        return None
    return json.loads((recovery[0] / "result.json").read_text())


def prune(root, older_than_seconds, *, apply=False):
    """List or remove old job directories while holding their persistent locks."""
    if not math.isfinite(older_than_seconds) or older_than_seconds <= 0:
        raise ValueError("Retention period must be finite and positive")
    root = Path(root)
    now = time.time()
    selected = []
    for directory in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{64}", directory.name) or directory.is_symlink() or not directory.is_dir():
            continue
        with (root / (directory.name + ".lock")).open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                modified = max([directory.stat().st_mtime] +
                               [item.stat().st_mtime for item in directory.iterdir() if not item.is_symlink()])
                if now - modified < older_than_seconds:
                    continue
                selected.append(directory.name)
                if apply:
                    shutil.rmtree(directory)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune old worker recovery data")
    parser.add_argument("--root", default=os.environ.get("QWEN_RECOVERY_DIR"), required=not os.environ.get("QWEN_RECOVERY_DIR"))
    parser.add_argument("--older-than-hours", type=float, required=True)
    parser.add_argument("--apply", action="store_true", help="Delete listed directories. The default is a dry run.")
    args = parser.parse_args()
    for key in prune(args.root, args.older_than_hours * 3600, apply=args.apply):
        print(key)
