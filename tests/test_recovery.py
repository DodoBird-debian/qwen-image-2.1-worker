"""Regression coverage for shared worker recovery."""
import asyncio
import base64
import io
import json
import os
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import handler
from qwen_worker.config import Settings
from qwen_worker.memory import GIB
from qwen_worker.images import encode_image
from qwen_worker.service import ImageService, OUTPUT_ENVELOPE_HEADROOM
from qwen_worker.recovery import job_lock, open_job, prune
from scripts.client import recover_worker


class RecoveryTests(unittest.TestCase):
    def test_parallel_encoder_failure_preserves_decodable_siblings(self):
        calls = []

        def pipeline(*, generator, **kwargs):
            calls.append(generator)
            return SimpleNamespace(images=[Image.new("RGB", (256, 256), (seed, 0, 0)) for seed in generator])

        def encode(image, format, quality, seed, **options):
            if seed == 44:
                raise OSError("fixture could not encode seed 44")
            return encode_image(image, format, quality, seed, **options)

        service = ImageService(pipeline, int, Settings())
        job = {"input": {"prompt": "fixture", "width": 256, "height": 256, "seed": 42, "num_images": 4}}
        with patch("qwen_worker.service.encode_image", side_effect=encode):
            result = service.handle(job)
        self.assertEqual(calls, [[42, 43, 44, 45]])
        self.assertEqual([image["seed"] for image in result["images"]], [42, 43, 45])
        self.assertFalse(result.get("refresh_worker"))
        self.assertTrue(result["delivery"]["partial"])
        self.assertIn("Could not encode seeds: 44.", result["delivery"]["warnings"])
        for image in result["images"]:
            with Image.open(io.BytesIO(base64.b64decode(image["image_base64"]))) as decoded:
                self.assertEqual(decoded.getpixel((0, 0)), (image["seed"], 0, 0))

    def test_later_packing_failure_keeps_the_previous_encoded_frame(self):
        calls = []
        size = (2048, 1024)
        frame = Image.frombytes("RGBA", size, random.Random(42).randbytes(size[0] * size[1] * 4))

        def pipeline(*, generator, **kwargs):
            calls.append(generator)
            return SimpleNamespace(images=[frame])

        def optimize(*args, **kwargs):
            if len(calls) > 1:
                raise OSError("second-pass optimizer failed")
            return encode_image(*args, **kwargs)

        settings = Settings()
        service = ImageService(pipeline, int, settings, memory_budget=lambda: 5 * GIB)
        job = {"input": {"prompt": "fixture", "width": size[0], "height": size[1],
                         "seed": 42, "num_images": 2}}
        with patch("qwen_worker.transport.encode_image", side_effect=optimize):
            result = service.handle(job)
        self.assertEqual(calls, [42, 43])
        self.assertEqual([image["seed"] for image in result["images"]], [42])
        self.assertTrue(result["delivery"]["partial"])
        self.assertLessEqual(len(json.dumps(result).encode()), settings.max_output_bytes - OUTPUT_ENVELOPE_HEADROOM)
        with Image.open(io.BytesIO(base64.b64decode(result["images"][0]["image_base64"]))) as decoded:
            decoded.load()

    def test_cancel_after_fit_leaves_originals_retrievable(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def cleanup():
            entered.set()
            release.wait(5)

        def pipeline(*, generator, **kwargs):
            calls.append(generator)
            return SimpleNamespace(images=[Image.new("RGBA", (256, 256), (seed, 0, 0, 127)) for seed in generator])

        service = ImageService(pipeline, int, Settings(), release_memory=cleanup)
        job = {"id": "late-stop", "input": {"prompt": "fixture", "width": 256, "height": 256,
                                               "seed": 42, "num_images": 2}}

        async def run():
            task = asyncio.create_task(handler.serverless_handler(job))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                task.cancel()
                await asyncio.sleep(0)
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"QWEN_RECOVERY_DIR": directory}), \
                patch.object(handler, "get_service", return_value=service):
            asyncio.run(run())
            recovered = recover_worker(Path(directory), job["id"], Path(directory) / "export")
            self.assertEqual([image["seed"] for image in recovered["images"]], [42, 43])
            with Image.open(recovered["images"][1]["path"]) as original:
                self.assertEqual(original.getpixel((0, 0)), (43, 0, 0, 127))
        self.assertEqual(calls, [[42, 43]])

    def test_prune_is_dry_run_and_skips_locked_jobs(self):
        job = {"id": "active", "input": {}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"QWEN_RECOVERY_DIR": directory}):
            with job_lock(job):
                path, _ = open_job(job)
                old = time.time() - 7200
                os.utime(path / "manifest.json", (old, old))
                os.utime(path, (old, old))
                self.assertEqual(prune(directory, 3600, apply=True), [])
                self.assertTrue(path.exists())
            with self.assertRaises(ValueError):
                prune(directory, float("nan"), apply=True)
            self.assertEqual(prune(directory, 3600), [path.name])
            self.assertTrue(path.exists())
            self.assertEqual(prune(directory, 3600, apply=True), [path.name])
            self.assertFalse(path.exists())
            self.assertTrue((path.parent / (path.name + ".lock")).exists())

    def test_replacement_resumes_missing_seed(self):
        calls = []
        failed = False

        def pipeline(*, generator, **kwargs):
            nonlocal failed
            calls.append(generator)
            if generator == 43 and not failed:
                failed = True
                raise RuntimeError("worker failure")
            return SimpleNamespace(images=[Image.new("RGB", (256, 256))])

        job = {"id": "replace", "input": {
            "prompt": "fixture", "width": 256, "height": 256, "seed": 42, "num_images": 2,
        }}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"QWEN_RECOVERY_DIR": directory}):
            first = ImageService(pipeline, int, Settings(), memory_budget=lambda: 4 * GIB)
            first.check_memory = lambda *_: (None, False)
            self.assertEqual([i["seed"] for i in first.handle(job)["images"]], [42])
            second = ImageService(pipeline, int, Settings(), memory_budget=lambda: 4 * GIB)
            second.check_memory = lambda *_: (None, False)
            self.assertEqual([i["seed"] for i in second.handle(job)["images"]], [42, 43])
            self.assertEqual(calls, [42, 43, 43])

    def test_concurrent_duplicate_delivery_runs_once(self):
        calls = []
        entered = threading.Event()
        release = threading.Event()

        def pipeline(*, generator, **kwargs):
            calls.append(generator)
            entered.set()
            self.assertTrue(release.wait(5))
            return SimpleNamespace(images=[Image.new("RGB", (256, 256))])

        job = {"id": "duplicate", "input": {"prompt": "fixture", "width": 256, "height": 256, "seed": 42}}
        service = ImageService(pipeline, int, Settings())
        results = []
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"QWEN_RECOVERY_DIR": directory}), \
                patch.object(handler, "get_service", return_value=service):
            first = threading.Thread(target=lambda: results.append(handler.handler(job)))
            second = threading.Thread(target=lambda: results.append(handler.handler(job)))
            first.start()
            self.assertTrue(entered.wait(5))
            second.start()
            release.set()
            first.join(5)
            second.join(5)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(calls, [42])
            self.assertEqual(results[0], results[1])

    def test_storage_failure_preserves_inline_result(self):
        service = ImageService(lambda **kwargs: SimpleNamespace(images=[Image.new("RGB", (256, 256))]),
                               int, Settings())
        job = {"id": "storage-failure", "input": {
            "prompt": "fixture", "width": 256, "height": 256, "seed": 42,
        }}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"QWEN_RECOVERY_DIR": directory}), \
                patch("qwen_worker.service.save_image", side_effect=OSError("disk full")):
            result = service.handle(job)
            self.assertEqual([image["seed"] for image in result["images"]], [42])
            self.assertIn("Recovery storage failed", result["delivery"]["warnings"][0])


if __name__ == "__main__":
    unittest.main()
