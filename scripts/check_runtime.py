"""Check real imports and pipeline call compatibility without downloading model weights."""

import inspect
import json
import sys
from pathlib import Path
from importlib.metadata import version

import torch
import torchvision
from diffusers import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from diffusers import AutoencoderKLQwenImage21

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_worker.conditioning import make_pipeline_class

worker_pipeline = make_pipeline_class(QwenImage21Pipeline)
if inspect.signature(worker_pipeline.__init__) != inspect.signature(QwenImage21Pipeline.__init__):
    raise RuntimeError("Worker adapter changed the Diffusers component constructor")
if inspect.signature(worker_pipeline.__call__) != inspect.signature(QwenImage21Pipeline.__call__):
    raise RuntimeError("Worker adapter changed the upstream inference signature")
if worker_pipeline._get_signature_keys(worker_pipeline) != QwenImage21Pipeline._get_signature_keys(QwenImage21Pipeline):
    raise RuntimeError("Worker adapter changed the Diffusers component discovery")

required = {
    "prompt",
    "image",
    "negative_prompt",
    "true_cfg_scale",
    "height",
    "width",
    "num_inference_steps",
    "generator",
    "output_resolution",
    "use_kv_cache",
    "num_images_per_prompt",
}
missing = required - set(inspect.signature(QwenImage21Pipeline.__call__).parameters)
if missing:
    raise RuntimeError(f"Unsupported pipeline API; missing parameters: {sorted(missing)}")
adapter_methods = {
    "_get_qwen_prompt_embeds": {"prompt", "image", "device"},
    "_encode_vae_image": {"image", "generator"},
    "prepare_latents": {"images", "batch_size", "generator"},
}
for method, required_parameters in adapter_methods.items():
    actual = set(inspect.signature(getattr(QwenImage21Pipeline, method)).parameters)
    if required_parameters - actual:
        raise RuntimeError(f"Unsupported conditioning adapter API: {method}")
if "logits_to_keep" not in inspect.signature(Qwen3VLForConditionalGeneration.forward).parameters:
    raise RuntimeError("Conditioning encoder does not support logits_to_keep")
if "with_kwargs" not in inspect.signature(torch.nn.Module.register_forward_pre_hook).parameters:
    raise RuntimeError("PyTorch does not support conditioning keyword hooks")
for method in ("enable_slicing", "enable_tiling", "clear_cache"):
    if not callable(getattr(AutoencoderKLQwenImage21, method, None)):
        raise RuntimeError(f"Unsupported VAE API: {method}")
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "packages": {
                name: version(name)
                for name in [
                    "diffusers",
                    "transformers",
                    "runpod",
                    "accelerate",
                    "huggingface-hub",
                    "safetensors",
                ]
            },
            "pipeline": QwenImage21Pipeline.__name__,
            "worker_pipeline": worker_pipeline.__name__,
            "encoder": Qwen3VLForConditionalGeneration.__name__,
            "processor": Qwen3VLProcessor.__name__,
        },
        indent=2,
    )
)
