import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from scripts import client


def generate_png():
    with io.BytesIO() as out:
        Image.new("RGB", (4, 3)).save(out, "PNG")
        return out.getvalue()


class ClientRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.args = SimpleNamespace(endpoint_id="test-endpoint", output_dir=self.output,
                                    discard_uncertain=False, timeout=60, resume=None)
        self.receipt = self.output / ".runpod/test-endpoint.json"
        self.png = generate_png()
        self.result = {"id": "accepted", "status": "COMPLETED", "output": {
            "images": [{"seed": 42, "mime_type": "image/png",
                        "image_base64": base64.b64encode(self.png).decode()}]}}

    def test_transient_status_retries_without_repeating_post(self):
        with (patch.object(client, "api_call", side_effect=[
            {"id": "accepted", "status": "IN_QUEUE"}, client.RunpodHTTPError(503), self.result,
        ]) as api, patch.object(client.time, "sleep")):
            client.collect_job(self.args, "test-key", {"input": {}})
        self.assertEqual([call.args[0].rsplit("/", 1)[-1] for call in api.call_args_list],
                         ["run", "accepted", "accepted"])
        self.assertEqual((self.output / "accepted/image-0.png").read_bytes(), self.png)
        self.assertFalse(self.receipt.exists())

    def test_restart_resumes_after_status_retry_budget(self):
        with (patch.object(client, "api_call", side_effect=[
            {"id": "accepted", "status": "IN_QUEUE"}, *[client.RunpodHTTPError(503)] * 5,
        ]), patch.object(client.time, "sleep")):
            with self.assertRaisesRegex(RuntimeError, "--resume accepted"):
                client.collect_job(self.args, "test-key", {"input": {}})
        self.assertEqual(json.loads(self.receipt.read_text())["id"], "accepted")
        with patch.object(client, "api_call", return_value=self.result) as api:
            with self.assertRaisesRegex(RuntimeError, "existing submission"):
                client.collect_job(self.args, "test-key", {"input": {}})
            api.assert_not_called()
            self.args.resume = "accepted"
            client.collect_job(self.args, "test-key", None)
            self.assertEqual(api.call_count, 1)
            self.assertTrue(api.call_args.args[0].endswith("/status/accepted"))
        self.assertEqual((self.output / "accepted/image-0.png").read_bytes(), self.png)

    def test_new_cli_process_resumes_original_job_after_503(self):
        requests = []
        complete = threading.Event()
        fixture_result = self.result

        class RunpodFixture(BaseHTTPRequestHandler):
            def respond(self, status, body):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_POST(self):
                requests.append((self.command, self.path))
                self.rfile.read(int(self.headers["Content-Length"]))
                self.respond(200, {"id": "accepted", "status": "IN_QUEUE"})

            def do_GET(self):
                requests.append((self.command, self.path))
                if complete.is_set():
                    self.respond(200, fixture_result)
                else:
                    self.respond(503, {"error": "fixture outage"})

            def log_message(self, *args):
                pass

        payload = self.output / "input.json"
        payload.write_text('{"input": {}}')
        with ThreadingHTTPServer(("127.0.0.1", 0), RunpodFixture) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                entry = ("from scripts import client; "
                         f"client.API_ROOT = 'http://127.0.0.1:{server.server_port}'; client.main()")
                command = [sys.executable, "-c", entry, "test-endpoint", "--output-dir", str(self.output)]
                env = {**os.environ, "RUNPOD_API_KEY": "fixture-only"}
                original = subprocess.run(command + ["--input", str(payload), "--timeout", "2"],
                                          env=env, capture_output=True, text=True, timeout=12)
                self.assertNotEqual(original.returncode, 0)
                self.assertIn("--resume accepted", original.stderr)
                self.assertEqual(json.loads(self.receipt.read_text())["id"], "accepted")
                complete.set()
                resumed = subprocess.run(command + ["--resume", "accepted"], env=env,
                                         capture_output=True, text=True, timeout=12)
                self.assertEqual(resumed.returncode, 0, resumed.stderr)
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual([request for request in requests if request[0] == "POST"],
                         [("POST", "/test-endpoint/run")])
        self.assertTrue(all(path == "/test-endpoint/status/accepted" for method, path in requests if method == "GET"))
        self.assertGreaterEqual(sum(method == "GET" for method, _ in requests), 2)
        self.assertEqual((self.output / "accepted/image-0.png").read_bytes(), self.png)

    def test_lost_post_reply_survives_restart_and_wrong_lookup(self):
        with patch.object(client, "api_call", side_effect=ConnectionResetError):
            with self.assertRaisesRegex(RuntimeError, "may have been accepted"):
                client.collect_job(self.args, "test-key", {"input": {}})
        self.assertEqual(json.loads(self.receipt.read_text()), {"endpoint": "test-endpoint"})
        self.args.resume = "typo"
        with patch.object(client, "api_call", side_effect=client.RunpodHTTPError(404)):
            with self.assertRaises(client.RunpodHTTPError):
                client.collect_job(self.args, "test-key", None)
        self.assertEqual(json.loads(self.receipt.read_text()), {"endpoint": "test-endpoint"})

    def test_operator_recovers_original_after_queue_result_expires(self):
        recovery = self.output / hashlib.sha256(b"accepted").hexdigest()
        recovery.mkdir()
        checksum = hashlib.sha256(self.png).hexdigest()
        (recovery / "42.png").write_bytes(self.png)
        (recovery / "manifest.json").write_text(json.dumps({
            "job_id": "accepted", "images": [{"seed": 42, "file": "42.png",
                "width": 4, "height": 3, "sha256": checksum, "mime_type": "image/png"}]}))
        exported = self.output / "recovered"
        client.recover_worker(self.output, "accepted", exported)
        self.assertEqual((exported / "42.png").read_bytes(), self.png)
        (recovery / "42.png").write_bytes(b"broken")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            client.recover_worker(self.output, "accepted", exported)
