"""Exercise the registered async worker handler and cooperative cleanup on CPU."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import handler
from qwen_worker.config import Settings
from qwen_worker.memory import GIB
from qwen_worker.service import ImageService, check_stop


def job(seed, count=1):
    return {"id": str(seed), "input": {
        "prompt": "fixture", "width": 256, "height": 256, "seed": seed, "num_images": count,
    }}


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stopping_a_later_pass_waits_for_cleanup_and_skips_other_passes(self):
        second_pass = threading.Event()
        cleaning = threading.Event()
        allow_cleanup = threading.Event()
        next_job_entered = threading.Event()
        events = []

        def pipeline(*, generator, stop_event, **kwargs):
            events.append(generator)
            if generator == 42:
                second_pass.set()
                if not stop_event.wait(4):
                    raise AssertionError("the event loop could not deliver the stop")
                check_stop(stop_event)
            return SimpleNamespace(images=[Image.new("RGB", (256, 256))])

        def release_memory():
            if not cleaning.is_set():
                cleaning.set()
                if not allow_cleanup.wait(4):
                    raise AssertionError("the test did not release cleanup")
                events.append("cleanup")

        service = ImageService(pipeline, lambda seed: seed, Settings(),
                               memory_budget=lambda: 4 * GIB, release_memory=release_memory)

        def with_service(job, *, stop_event):
            if job["id"] == "50":
                next_job_entered.set()
            return service.handle(job, stop_event=stop_event)

        with patch.object(handler, "get_service", return_value=SimpleNamespace(handle=with_service)):
            running = asyncio.create_task(handler.serverless_handler(job(41, 3)))
            next_job = None
            try:
                self.assertTrue(await asyncio.to_thread(second_pass.wait, 4))
                running.cancel()
                self.assertTrue(await asyncio.to_thread(cleaning.wait, 4))
                running.cancel()
                next_job = asyncio.create_task(handler.serverless_handler(job(50)))
                self.assertTrue(await asyncio.to_thread(next_job_entered.wait, 4))
                self.assertFalse(running.done())
                self.assertEqual(events, [41, 42])
            finally:
                allow_cleanup.set()
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
                if next_job is not None:
                    await asyncio.gather(next_job, return_exceptions=True)

        self.assertTrue(running.cancelled())
        self.assertEqual(events, [41, 42, "cleanup", 50])
        self.assertEqual([item["seed"] for item in next_job.result()["images"]], [50])


if __name__ == "__main__":
    unittest.main()
