"""CPU-only runtime contract tests; no models, CUDA or network are required."""

import json
import errno
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_worker.config import MODEL_REVISION, Settings, configure_cache
from qwen_worker import runtime


def make_snapshot(path):
    path.mkdir(parents=True, exist_ok=True)
    components = ("processor", "scheduler", "text_encoder", "transformer", "vae")
    index = {"_class_name": "QwenImage21Pipeline"}
    for component in components:
        index[component] = ["diffusers", component]
        directory = path / component
        directory.mkdir(exist_ok=True)
        filename = "scheduler_config.json" if component == "scheduler" else "config.json"
        (directory / filename).write_text("{}", encoding="utf-8")
        if component in {"text_encoder", "transformer", "vae"}:
            (directory / "model.safetensors").write_bytes(b"mock weights")
    (path / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
    return path


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "hub"
        self.model_cache = self.cache / "models--Qwen--Qwen-Image-2.1"

    def test_explicit_complete_snapshot(self):
        path = make_snapshot(self.root / "explicit")
        self.assertEqual(runtime.download_model(Settings(model_path=str(path))), str(path))

    def test_readonly_cache_downloads_exact_revision_to_writable_fallback(self):
        path = make_snapshot(self.root / "downloaded")
        download = Mock(side_effect=[PermissionError(errno.EROFS, "Read-only mount"), str(path)])
        with patch.object(runtime, "configure_cache", return_value=self.root), \
                patch.object(runtime, "_cache_roots", return_value=[]), \
                patch.object(runtime, "FALLBACK_HUB_CACHE", self.root / "fallback"), \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(runtime.download_model(Settings()), str(path))
        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args.kwargs["cache_dir"], str(self.root / "fallback"))
        for call in download.call_args_list:
            self.assertEqual(call.kwargs["revision"], MODEL_REVISION)

    def test_download_disk_full_is_not_retried_as_readonly_cache(self):
        download = Mock(side_effect=OSError(errno.ENOSPC, "No space"))
        with patch.object(runtime, "configure_cache", return_value=self.root), \
                patch.object(runtime, "_cache_roots", return_value=[]), \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaises(OSError):
                runtime.download_model(Settings())
        download.assert_called_once()

    def test_cache_configuration_does_not_require_directory_creation(self):
        destination = self.root / "readonly-parent" / "absent"
        with patch.dict(os.environ, {"HF_HOME": str(destination)}, clear=True), \
                patch.object(Path, "mkdir", side_effect=PermissionError(errno.EROFS, "read only")) as mkdir:
            self.assertEqual(configure_cache(), destination)
        mkdir.assert_not_called()

    def test_complete_readonly_cache_hit_requires_no_writes_or_network(self):
        path = make_snapshot(self.model_cache / "snapshots" / MODEL_REVISION)
        download = Mock(side_effect=AssertionError("must not download"))
        with patch.object(runtime, "configure_cache", return_value=self.root / "absent"), \
                patch.object(runtime, "_cache_roots", return_value=[self.cache]), \
                patch.object(Path, "mkdir", side_effect=PermissionError(errno.EROFS, "read only")) as mkdir, \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(runtime.download_model(Settings()), str(path))
        mkdir.assert_not_called()
        download.assert_not_called()

    def test_explicit_snapshot_requires_no_cache_writes_or_network(self):
        path = make_snapshot(self.root / "explicit")
        with patch.object(runtime, "configure_cache", side_effect=AssertionError("must not select download cache")), \
                patch.object(Path, "mkdir", side_effect=PermissionError(errno.EROFS, "read only")):
            self.assertEqual(runtime.download_model(Settings(model_path=str(path))), str(path))

    def test_missing_readonly_directory_falls_back_before_network_call(self):
        path = make_snapshot(self.root / "downloaded")
        readonly = self.root / "readonly" / "hub"
        fallback = self.root / "fallback"
        download = Mock(return_value=str(path))
        real_mkdir = Path.mkdir
        def mkdir(target, *args, **kwargs):
            if target == readonly:
                raise PermissionError(errno.EROFS, "read only")
            return real_mkdir(target, *args, **kwargs)
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(runtime, "configure_cache", return_value=readonly.parent), \
                patch.object(runtime, "_cache_roots", return_value=[]), \
                patch.object(runtime, "FALLBACK_HUB_CACHE", fallback), \
                patch.object(Path, "mkdir", autospec=True, side_effect=mkdir), \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(runtime.download_model(Settings()), str(path))
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["cache_dir"], str(fallback))
        self.assertEqual(download.call_args.kwargs["revision"], MODEL_REVISION)

    def test_unwritable_fallback_does_not_retry_forever(self):
        download = Mock(side_effect=AssertionError("no writable destination"))
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(runtime, "configure_cache", return_value=self.root), \
                patch.object(runtime, "_cache_roots", return_value=[]), \
                patch.object(runtime, "FALLBACK_HUB_CACHE", self.root / "fallback"), \
                patch.object(Path, "mkdir", side_effect=PermissionError(errno.EACCES, "denied")) as mkdir, \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaises(PermissionError):
                runtime.download_model(Settings())
        self.assertEqual(mkdir.call_count, 2)
        download.assert_not_called()

    def test_disk_full_during_directory_creation_is_not_permission_fallback(self):
        download = Mock()
        with patch.object(runtime, "configure_cache", return_value=self.root), \
                patch.object(runtime, "_cache_roots", return_value=[]), \
                patch.object(Path, "mkdir", side_effect=OSError(errno.ENOSPC, "disk full")) as mkdir, \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaises(OSError) as caught:
                runtime.download_model(Settings())
        self.assertEqual(caught.exception.errno, errno.ENOSPC)
        mkdir.assert_called_once()
        download.assert_not_called()

    def test_already_using_fallback_only_attempts_it_once(self):
        fallback = self.root / "hub"
        download = Mock(side_effect=PermissionError(errno.EROFS, "read only"))
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(runtime, "configure_cache", return_value=self.root), \
                patch.object(runtime, "_cache_roots", return_value=[]), \
                patch.object(runtime, "FALLBACK_HUB_CACHE", fallback), \
                patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaises(PermissionError):
                runtime.download_model(Settings())
        download.assert_called_once()

    def test_nonpermission_download_failure_has_no_fallback(self):
        for failure in (RuntimeError("authentication"), ConnectionError("network")):
            download = Mock(side_effect=failure)
            with self.subTest(failure=failure), patch.object(runtime, "configure_cache", return_value=self.root), \
                    patch.object(runtime, "_cache_roots", return_value=[]), \
                    patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                with self.assertRaises(type(failure)):
                    runtime.download_model(Settings())
            download.assert_called_once()

    def test_model_index_alone_is_not_complete(self):
        path = self.root / "partial"
        path.mkdir()
        (path / "model_index.json").write_text('{"_class_name":"QwenImage21Pipeline"}')
        with self.assertRaisesRegex(runtime.ModelCacheError, "no processor"):
            runtime.download_model(Settings(model_path=str(path)))

    def test_wrong_pipeline_class_is_rejected(self):
        path = make_snapshot(self.root / "wrong")
        (path / "model_index.json").write_text('{"_class_name":"OtherPipeline"}')
        with self.assertRaisesRegex(ValueError, "not a QwenImage21Pipeline"):
            runtime._validate_snapshot(path)

    def test_missing_shard_rejected(self):
        path = make_snapshot(self.root / "partial")
        manifest = {"weight_map": {"weight": "model-00002-of-00002.safetensors"}}
        (path / "transformer" / "model.safetensors.index.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Missing or empty weight shard"):
            runtime._validate_snapshot(path)

    def test_malformed_weight_map_rejected(self):
        path = make_snapshot(self.root / "malformed")
        manifest = path / "vae" / "model.safetensors.index.json"
        for value in ({}, {"weight": ["bad"]}, {"weight": "../outside.safetensors"}, {"weight": "..\\outside.safetensors"}):
            with self.subTest(value=value):
                manifest.write_text(json.dumps({"weight_map": value}))
                with self.assertRaises(ValueError):
                    runtime._validate_snapshot(path)

    def test_empty_weight_file_rejected(self):
        path = make_snapshot(self.root / "empty")
        (path / "vae" / "model.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "no safetensors weights"):
            runtime._validate_snapshot(path)

    def test_exact_revision_cache_hit(self):
        exact = make_snapshot(self.model_cache / "snapshots" / MODEL_REVISION)
        make_snapshot(self.model_cache / "snapshots" / ("a" * 40))
        with patch.object(runtime, "configure_cache", return_value=self.root), patch.object(runtime, "_cache_roots", return_value=[self.cache]):
            self.assertEqual(runtime.download_model(Settings()), str(exact))

    def test_serving_refuses_missing_wrong_incomplete_or_broken_cache_without_downloads(self):
        wrong = make_snapshot(self.model_cache / "snapshots" / ("a" * 40))
        for state in ("missing", "empty", "incomplete", "broken"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temp:
                cache = Path(temp) / "absent-mount" / "hub"
                snapshot = cache / self.model_cache.name / "snapshots" / MODEL_REVISION
                if state == "empty":
                    snapshot.mkdir(parents=True)
                elif state in {"incomplete", "broken"}:
                    make_snapshot(snapshot)
                    shard = snapshot / "transformer" / "model.safetensors"
                    shard.unlink()
                    if state == "broken":
                        shard.symlink_to(Path(temp) / "unmounted-blobs" / "weight")
                download = Mock(side_effect=AssertionError("must not download"))
                with patch.object(runtime, "configure_cache", return_value=cache.parent), \
                        patch.object(runtime, "_cache_roots", return_value=[cache, wrong.parents[2]]), \
                        patch.object(Path, "mkdir", side_effect=AssertionError("must not write")), \
                        patch.object(runtime, "download_model", side_effect=AssertionError("serving cannot provision")), \
                        patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                    with self.assertRaisesRegex(runtime.ModelCacheError, "Downloads are disabled"):
                        runtime.load_service(Settings())
                download.assert_not_called()

    def test_cached_only_resolution_requires_complete_explicit_path(self):
        with self.assertRaisesRegex(runtime.ModelCacheError, "Invalid cached MODEL_PATH"):
            runtime.resolve_cached_model(Settings(model_path=str(self.root / "missing")))

    def test_previously_provisioned_fallback_is_a_known_cache(self):
        self.assertIn(runtime.FALLBACK_HUB_CACHE, runtime._cache_roots(self.root))

    def test_container_entrypoint_panics_before_gpu_imports_or_queue_registration(self):
        start = Mock()
        real_import = __import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "huggingface_hub"}:
                raise AssertionError(f"must fail before importing {name}")
            return real_import(name, *args, **kwargs)
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["handler.py"]), \
                patch.dict(sys.modules, {"runpod": SimpleNamespace(serverless=SimpleNamespace(start=start))}), \
                patch.object(runtime, "configure_cache", return_value=self.root), \
                patch.object(runtime, "_cache_roots", return_value=[self.cache]), \
                patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(runtime.ModelCacheError, "Downloads are disabled"):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / "handler.py"), run_name="__main__")
        start.assert_not_called()

    def test_wrong_revision_is_never_selected(self):
        make_snapshot(self.model_cache / "snapshots" / ("a" * 40))
        downloaded = make_snapshot(self.root / "downloaded")
        download = Mock(return_value=str(downloaded))
        with patch.object(runtime, "configure_cache", return_value=self.root), patch.object(runtime, "_cache_roots", return_value=[self.cache]), patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(runtime.download_model(Settings()), str(downloaded))
        self.assertEqual(download.call_args.kwargs["revision"], MODEL_REVISION)
        self.assertEqual(download.call_args.kwargs["repo_id"], "Qwen/Qwen-Image-2.1")

    def test_incomplete_cache_is_repaired_through_hub(self):
        partial = make_snapshot(self.model_cache / "snapshots" / MODEL_REVISION)
        (partial / "transformer" / "model.safetensors").unlink()
        downloaded = make_snapshot(self.root / "downloaded")
        download = Mock(return_value=str(downloaded))
        with patch.object(runtime, "configure_cache", return_value=self.root), patch.object(runtime, "_cache_roots", return_value=[self.cache]), patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(runtime.download_model(Settings()), str(downloaded))
        download.assert_called_once()

    def test_download_is_validated(self):
        download = Mock(return_value=str(self.root / "absent"))
        with patch.object(runtime, "configure_cache", return_value=self.root), patch.object(runtime, "_cache_roots", return_value=[]), patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaisesRegex(ValueError, "metadata"):
                runtime.download_model(Settings())

    def test_named_revision_resolves_only_its_ref(self):
        refs = self.model_cache / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(MODEL_REVISION)
        self.assertEqual(runtime._cached_snapshot(self.cache, Settings(model_revision="main")), self.model_cache / "snapshots" / MODEL_REVISION)
        self.assertIsNone(runtime._cached_snapshot(self.cache, Settings(model_revision="missing")))
        self.assertIsNone(runtime._cached_snapshot(self.cache, Settings(model_revision="../../outside")))

    def test_explicit_hub_cache_takes_precedence(self):
        with patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "custom")}, clear=True), patch.object(Path, "home", return_value=self.root):
            self.assertEqual(runtime._cache_roots(self.root)[0], self.root / "custom")


class FakeTensor:
    def __init__(self, size, device="cpu"):
        self.size = size
        self.device = SimpleNamespace(type=device)

    def numel(self):
        return self.size

    def element_size(self):
        return 1


def module(size, device="cpu"):
    tensor = FakeTensor(size, device)
    return SimpleNamespace(parameters=lambda: iter([tensor]), buffers=lambda: iter([]))


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.pipe = SimpleNamespace(text_encoder=module(30), transformer=module(20, "cuda"), vae=module(5, "cuda"))
        self.torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (100, 200), memory_reserved=lambda: 30, memory_allocated=lambda: 20))
        self.weights = {"text_encoder": 30, "transformer": 20, "vae": 5}

    def test_encoder_mode_reserves_future_encoder(self):
        self.assertEqual(runtime._phase_tensor_budgets(self.pipe, self.torch, "text_encoder", self.weights), {"conditioning": 80, "denoising": 110, "decoding": 110})

    def test_model_mode_reserves_overlapping_components(self):
        self.pipe.transformer = module(20)
        self.pipe.vae = module(5)
        expected = {"conditioning": 75, "denoising": 85, "decoding": 105}
        for mode in ("model", "sequential"):
            with self.subTest(mode=mode):
                self.assertEqual(runtime._phase_tensor_budgets(self.pipe, self.torch, mode, self.weights), expected)

    def test_existing_resident_weights_are_not_double_reserved(self):
        self.pipe.text_encoder = module(30, "cuda")
        self.assertEqual(runtime._phase_tensor_budgets(self.pipe, self.torch, "model", self.weights), dict.fromkeys(("conditioning", "denoising", "decoding"), 110))

    def test_budget_clamped_to_zero(self):
        self.torch.cuda.mem_get_info = lambda: (0, 200)
        self.torch.cuda.memory_reserved = lambda: 20
        self.assertEqual(runtime._phase_tensor_budgets(self.pipe, self.torch, "text_encoder", self.weights)["conditioning"], 0)

    def test_module_bytes_deduplicates_and_filters_device(self):
        tensor = FakeTensor(7, "cuda")
        cpu = FakeTensor(4)
        fake = SimpleNamespace(parameters=lambda: iter([tensor, tensor]), buffers=lambda: iter([cpu]))
        self.assertEqual(runtime._module_bytes(fake), 11)
        self.assertEqual(runtime._module_bytes(fake, "cuda"), 7)


class EncoderHookTests(unittest.TestCase):
    def test_hook_changes_only_unused_language_outputs(self):
        encoder = SimpleNamespace(forward=lambda logits_to_keep=0, **kwargs: None, register_forward_pre_hook=Mock())
        runtime._install_conditioning_encoder_hook(encoder)
        hook = encoder.register_forward_pre_hook.call_args.args[0]
        self.assertTrue(encoder.register_forward_pre_hook.call_args.kwargs["with_kwargs"])
        pixels = object()
        original = {"output_hidden_states": True, "pixel_values": pixels, "use_cache": True}
        args, kwargs = hook(encoder, (), original)
        self.assertEqual(args, ())
        self.assertEqual(kwargs["logits_to_keep"], 1)
        self.assertIs(kwargs["use_cache"], False)
        self.assertIs(kwargs["output_hidden_states"], True)
        self.assertIs(kwargs["pixel_values"], pixels)
        self.assertIs(original["use_cache"], True)

    def test_incompatible_encoder_rejected(self):
        encoder = SimpleNamespace(forward=lambda **kwargs: None)
        with self.assertRaisesRegex(RuntimeError, "logits_to_keep"):
            runtime._install_conditioning_encoder_hook(encoder)

    def test_architecture_guard(self):
        transformer = SimpleNamespace(config=SimpleNamespace(num_layers=32, num_attention_heads=32, attention_head_dim=128, patch_size=1, in_channels=64))
        vae = SimpleNamespace(config=SimpleNamespace(z_dim=64), spatial_compression_ratio=16)
        pipeline = SimpleNamespace(transformer=transformer, vae=vae)
        runtime._validate_architecture(pipeline)
        transformer.config.num_layers = 40
        with self.assertRaisesRegex(ValueError, "num_layers"):
            runtime._validate_architecture(pipeline)
        transformer.config.num_layers = 32
        vae.spatial_compression_ratio = 8
        with self.assertRaisesRegex(ValueError, "VAE"):
            runtime._validate_architecture(pipeline)


class RuntimeIntegrationTests(unittest.TestCase):
    def load_fake_service(self, settings=None):
        events = []

        class FakeModule:
            def __init__(self, name, size):
                self.name = name
                self.tensor = FakeTensor(size)
                self.config = SimpleNamespace(num_layers=32, num_attention_heads=32,
                                              attention_head_dim=128, patch_size=1, in_channels=64, z_dim=64)
                self.spatial_compression_ratio = 16

            def parameters(self):
                return iter([self.tensor])

            def buffers(self):
                return iter([])

            def to(self, device):
                self.tensor.device.type = device
                events.append((self.name, device))
                return self

            def forward(self, logits_to_keep=0, **kwargs):
                return None

            def register_forward_pre_hook(self, hook, with_kwargs=False):
                self.hook = hook

            def enable_slicing(self):
                events.append("slicing")

            def enable_tiling(self):
                events.append("tiling")

            def clear_cache(self):
                events.append("clear_vae")

        class FakePipeline:
            def __init__(self):
                self.text_encoder = FakeModule("text_encoder", 30)
                self.transformer = FakeModule("transformer", 20)
                self.vae = FakeModule("vae", 5)
                self.fail = False

            @classmethod
            def from_pretrained(cls, source, **kwargs):
                return cls()

            def set_progress_bar_config(self, **kwargs):
                pass

            def _get_qwen_prompt_embeds(self, prompt, image, device):
                _, kwargs = self.text_encoder.hook(self.text_encoder, (), {"output_hidden_states": True})
                events.append(("encode", kwargs))
                if self.fail:
                    raise RuntimeError("encoder failure")
                return object(), object(), object()

            def prepare_latents(self, *args):
                events.append("reference_vae")
                return None

            def __call__(self, prompt, callback_on_step_end, callback_on_step_end_tensor_inputs):
                self._get_qwen_prompt_embeds(prompt, None, "cuda")
                self.prepare_latents(None, 1, 64, 32, 32, "bf16", "cuda", None)
                self.callback_inputs = callback_on_step_end_tensor_inputs
                for step in range(40):
                    self.callback_result = callback_on_step_end(self, step, None, {})
                return SimpleNamespace(images=[])

            def maybe_free_model_hooks(self):
                events.append("free_hooks")

        cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                               device_count=lambda: 1, get_device_name=lambda index: "mock GPU",
                               get_device_capability=lambda index: (8, 0), mem_get_info=lambda *args: (100, 200),
                               memory_allocated=lambda: 20, memory_reserved=lambda: 30,
                               reset_peak_memory_stats=lambda: events.append("reset_peaks"), max_memory_allocated=lambda: 50,
                               max_memory_reserved=lambda: 60,
                               empty_cache=lambda: events.append("empty_cache"), ipc_collect=lambda: None)
        torch = SimpleNamespace(__version__="mock", version=SimpleNamespace(cuda="mock"),
                                backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 0)),
                                cuda=cuda, set_num_threads=lambda threads: None, bfloat16="bf16",
                                inference_mode=nullcontext)
        with patch.dict(os.environ), patch.dict(sys.modules, {"torch": torch, "diffusers": SimpleNamespace(QwenImage21Pipeline=FakePipeline)}), patch.object(runtime, "resolve_cached_model", return_value="mock"), patch.object(runtime, "RequestPreflight", return_value=Mock()) as preflight:
            service = runtime.load_service(settings or Settings())
        self.assertIs(service.preflight, preflight.return_value)
        events.clear()
        return service, events

    def test_encoder_evicted_before_vae_and_not_loaded_on_cache_hit(self):
        service, events = self.load_fake_service()
        with service.job_context():
            service.pipeline(prompt="a")
            service.pipeline(prompt="a")
        self.assertEqual(events.count(("text_encoder", "cuda")), 1)
        self.assertEqual(events.count(("text_encoder", "cpu")), 1)
        self.assertLess(events.index(("text_encoder", "cpu")), events.index("reference_vae"))
        encode_events = [entry for entry in events if isinstance(entry, tuple) and entry[0] == "encode"]
        self.assertEqual(len(encode_events), 1)
        self.assertEqual(encode_events[0][1], {"output_hidden_states": True, "logits_to_keep": 1, "use_cache": False})
        self.assertEqual(service.phase_budgets(), {"conditioning": 80, "denoising": 110, "decoding": 110})

    def test_startup_summary_separates_loading_stages_and_fingerprints_source(self):
        with self.assertLogs(runtime.logger, level="INFO") as logs:
            self.load_fake_service()
        summary = next(line for line in logs.output if "[STARTUP SUMMARY]" in line)
        for field in ("cache=", "imports=", "device/diagnostics=", "model_load=", "placement=", "service_total=", "source="):
            self.assertIn(field, summary)
        fingerprint = runtime.source_fingerprint()
        self.assertRegex(fingerprint, r"^[0-9a-f]{16}$")
        self.assertEqual(fingerprint, runtime.source_fingerprint())

    def test_encoder_failure_releases_placement_and_job_cache(self):
        service, events = self.load_fake_service()
        # The infer closure captures the derived pipeline instance.
        pipeline = next(cell.cell_contents for cell in service.pipeline.__closure__ if hasattr(cell.cell_contents, "conditioning_context"))
        pipeline.fail = True
        with self.assertRaisesRegex(RuntimeError, "encoder failure"), service.job_context():
            service.pipeline(prompt="a")
        self.assertIsNone(pipeline._conditioning_cache)
        self.assertEqual(pipeline.text_encoder.tensor.device.type, "cpu")
        self.assertIn("clear_vae", events)
        self.assertIn("free_hooks", events)

    def test_progress_callback_does_not_transfer_tensors_and_peaks_cover_whole_job(self):
        service, events = self.load_fake_service()
        with self.assertLogs(runtime.logger, level="INFO") as logs:
            with service.job_context() as stats:
                service.pipeline(prompt="a")
                service.pipeline(prompt="a")
        self.assertEqual(events.count("reset_peaks"), 1)
        self.assertEqual(stats, {"allocated": 50, "reserved": 60})
        pipeline = next(cell.cell_contents for cell in service.pipeline.__closure__ if hasattr(cell.cell_contents, "conditioning_context"))
        self.assertEqual(pipeline.callback_inputs, [])
        self.assertEqual(pipeline.callback_result, {})
        self.assertTrue(any("step 1/40" in line for line in logs.output))
        self.assertTrue(any("step 40/40" in line for line in logs.output))

    def test_positional_preflight_still_runs_when_memory_guard_is_disabled(self):
        from qwen_worker.requests import InputError
        service, events = self.load_fake_service(Settings(memory_guard=False))
        self.assertIsNone(service.memory_budget)
        self.assertIsNone(service.phase_budgets)
        service.preflight.side_effect = InputError("unsupported layout")
        with self.assertRaisesRegex(InputError, "unsupported layout"):
            service.handle({"input": {"prompt": "test"}})
        service.preflight.assert_called_once()
        self.assertNotIn(("text_encoder", "cuda"), events)


if __name__ == "__main__":
    unittest.main()
