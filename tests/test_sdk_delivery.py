"""Exercise result uploads through the installed Runpod SDK using fake HTTP."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientConnectionError
from PIL import Image
from runpod.serverless.modules import rp_http, rp_job

import handler
from qwen_worker.config import Settings
from qwen_worker.service import ImageService


class ResultDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def exercise_disconnect(self, accepted_before_disconnect):
        calls = []
        attempts = []
        accepted = []

        def pipeline(*, generator, **kwargs):
            calls.append(generator)
            return SimpleNamespace(images=[Image.new("RGB", (256, 256), "red")])

        async def send(method, url, *, data, **kwargs):
            payload = json.loads(data)
            attempts.append(payload)
            if len(attempts) == 1:
                if accepted_before_disconnect:
                    accepted.append(payload)
                raise ClientConnectionError("simulated connection loss")
            accepted.append(payload)
            return SimpleNamespace(
                status=200, method=method, closed=True,
                text=AsyncMock(return_value="OK"), raise_for_status=lambda: None,
            )

        service = ImageService(pipeline, int, Settings())
        job = {"id": "delivery-fixture", "input": {
            "prompt": "fixture", "width": 256, "height": 256, "seed": 42,
        }}
        session = SimpleNamespace(headers={}, request=send)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"QWEN_RECOVERY_DIR": directory}), \
                patch.object(handler, "get_service", return_value=service), \
                patch.object(rp_http, "FibonacciRetry", rp_http.FibonacciRetry), \
                patch.object(rp_http, "_transmit", rp_http._transmit), \
                patch.object(handler.logger, "info") as info:
            handler.configure_delivery_retry()
            await rp_job.handle_job(session, {"handler": handler.serverless_handler}, job)
            self.assertEqual(len(attempts), 2)
            await rp_job.handle_job(session, {"handler": handler.serverless_handler}, job)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(calls, [42])
            self.assertEqual(len(accepted), 3 if accepted_before_disconnect else 2)
            self.assertTrue(all(entry["output"] == attempts[0]["output"] for entry in attempts))
            acknowledgements = [call for call in info.call_args_list
                                if "RunPod acknowledged result upload" in call.args[0]]
            # Other startup tests can wrap the SDK transmitter earlier in this process.
            self.assertGreaterEqual(len(acknowledgements), 2)
            self.assertTrue(all(call.args[1] == job["id"] for call in acknowledgements))

    async def test_disconnect_before_acceptance(self):
        await self.exercise_disconnect(False)

    async def test_disconnect_after_acceptance(self):
        await self.exercise_disconnect(True)


if __name__ == "__main__":
    unittest.main()
