# Qwen-Image-2.1 Runpod Worker v0.0.1.2
[![Runpod](https://api.runpod.io/badge/DodoBird-debian/qwen-image-2.1-worker)](https://console.runpod.io/hub/listing/DodoBird-debian/qwen-image-2.1-worker)

A queue-based Runpod Serverless worker for **Qwen/Qwen-Image-2.1** (v0.0.1.2). It loads one BF16
pipeline before accepting jobs and reuses it for text-to-image generation, image
editing with up to ten references, and transparent image generation.

## Send a request

`RUNPOD_API_KEY` is needed by the client. The worker receives its queue credentials
from Runpod automatically.

```bash
export RUNPOD_API_KEY='your-key'
python scripts/client.py YOUR_ENDPOINT_ID --input examples/generate.json
```

PowerShell:

```powershell
$env:RUNPOD_API_KEY = 'your-key'
python scripts/client.py YOUR_ENDPOINT_ID --input examples/generate.json
```

The client submits to `/run`, waits on the returned job id, and saves images and
metadata under `outputs/<job-id>/`. A client timeout does not cancel the remote job.
Cold starts include pulling the container and loading the cached model. Startup
is strictly cache-only: a missing, incomplete, or wrong-revision snapshot is fatal,
and no weights are downloaded by the serving process. Use `examples/smoke.json`
for a 512×512, four-step boot test; it is not a
quality benchmark.

A minimal request:

```json
{
  "input": {
    "prompt": "A ceramic fox on a blue table, soft window light",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 40,
    "seed": 42,
    "output_format": "png"
  }
}
```

### Editing and transparency

Use local reference files with the client:

```bash
python scripts/client.py YOUR_ENDPOINT_ID --input examples/edit.json --image photo.png
python scripts/client.py YOUR_ENDPOINT_ID --input examples/edit.json --image first.png --image second.png
python scripts/client.py YOUR_ENDPOINT_ID --input examples/transparent.json
```

In raw API requests, `images` is a list of base64 strings or
`data:image/png;base64,...` data URIs. URL and filesystem inputs are not accepted by
the worker. Reference alpha is preserved. `transparent: true` applies Qwen's
recommended RGBA prompt format and requires PNG or WebP output. Transparency is a
model capability prompted through text; the worker does not synthesize an alpha mask.

### Input fields

| Field | Default / validation |
|---|---|
| `prompt` | Required non-empty string, at most 20,000 characters |
| `width`, `height` | 1024 each; supply both; multiples of 32 from 256 to 4096; total ≤4,500,000 pixels |
| `aspect_ratio` | Alternative to dimensions: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16; uses Qwen's native approximately 2K sizes |
| `num_inference_steps` | 40; integer 1–100 |
| `seed` | Random if omitted or -1; otherwise integer 0 to 2^63−1 |
| `num_images` | 1; integer 1–4; consecutive seeds, parallel only when phase estimates admit the batch |
| `images` | Empty; at most 10 references, combined base64 ≤10,000,000 characters; the entire `/run` JSON must also fit 10,000,000 bytes |
| `transparent` | false |
| `output_format` | png; png, webp, or jpeg |
| `quality` | 95; integer 1–100, used by JPEG and non-alpha WebP |
| `true_cfg_scale` | 1.0; finite number 1–20; values above 1 add a second transformer pass |
| `negative_prompt` | Requires `true_cfg_scale > 1`; omitted with CFG enabled becomes an empty string |
| `use_kv_cache` | true; set false to trade speed for lower reference-cache memory |
| `output_resolution` | 1024; 512, 1024, or 2048; controls reference preprocessing, independently of explicit output dimensions |

CPU offloading does not move the request KV cache to CPU. Multiple high-resolution
references can still exhaust VRAM; reduce `output_resolution` or disable `use_kv_cache`.
The cache setting is included in returned parameters because it can change BF16 results.

The actual pipeline uses `true_cfg_scale`, not `guidance_scale`. Unknown fields are
rejected to catch misspellings. Reference images are limited to 20 million pixels
each and 40 million pixels total, with aspect ratios at most 100:1; animations are rejected.
The aspect-ratio bound prevents upstream preprocessing from rounding a dimension to zero.

### Output

Runpod wraps the worker result in its job `output` field:

```json
{
  "model": "Qwen/Qwen-Image-2.1",
  "revision": "790c92633540aa0cb11d9abf19eb46d861714758",
  "images": [
    {
      "image_base64": "...",
      "mime_type": "image/png",
      "width": 1024,
      "height": 1024,
      "seed": 42
    }
  ],
  "inference_seconds": 12.34
}
```

The time shown here is illustrative. PNG and alpha WebP preserve transparency;
JPEG composites alpha on white. The worker caps serialized inline output at
8,000,000 bytes, including base64 and JSON metadata, with 4,096 bytes reserved for
Runpod's envelope. The final check also enforces a hard ceiling of 10,000,000 bytes.
When a result is too large, the worker optimizes the encoding, lowers JPEG/WebP
quality if necessary, then proportionally resizes down to a minimum 256-pixel
short side (never enlarging smaller outputs). If it still does not fit, it switches
to JPEG and tries progressively lower quality, down to 1. JPEG flattens transparency
on white. If the full batch still cannot fit, the worker returns the largest fitting
subset, preserving the selected images' seeds and order. It checks all subsets,
so a large first image cannot prevent sending more smaller images. In sequential
mode, a partial result stops remaining generation passes to avoid wasting compute.
No inference is retried. A top-level `delivery` object reports `requested_images`,
`returned_images`, `partial`, and human-readable `warnings` about JPEG, transparency,
and omitted images. Its bytes are included in the size check. Clients must accept
fewer images and inspect each actual MIME type. Returned width and height always
describe the encoded image, which may be smaller than the request. Compression
uses the original CPU pixels. Compression stages, sizes, quality, and dimensions are logged. Clients that
map results onto an edit canvas must scale coordinates using the returned dimensions.
`scripts/client.py --decode response.json`
also decodes previously saved API results.

## Configuration

| Environment variable | Default / effect |
|---|---|
| `MODEL_ID` | `Qwen/Qwen-Image-2.1`; alternate checkpoints must use the same pipeline |
| `MODEL_REVISION` | `790c92633540aa0cb11d9abf19eb46d861714758`; change together with MODEL_ID |
| `MODEL_PATH` | Optional complete local Diffusers snapshot; must already exist; bypasses Hub revision lookup |
| `HF_HOME` | Automatic persistent-volume or container-cache path |
| `HF_TOKEN` | Optional, handled by huggingface_hub |
| `CPU_OFFLOAD` | `text_encoder` (default), `model`, `none`, or `sequential`; select from measured phase peaks on the target GPU |
| `VAE_TILING` | `true`; reduces decoding memory at large resolutions |
| `MEMORY_GUARD` | `true` (default); conservative phase estimates reject oversize jobs; does not guarantee freedom from OOM |

## Hardware and verification status

The Runpod listing targets one 80 GB Ampere GPU. This repository does not contain
reproducible evidence for the earlier GPU performance and stability claims, so
those claims have been removed. The current implementation has CPU regression
coverage; image quality, throughput and memory peaks need validation on the
deployment GPU. Smaller GPUs may require `model` or `sequential` offload and
smaller requests. The guard intentionally reserves whole component weights for
sequential offload until its submodule peaks have been measured.

Parallel KV storage grows with both image count and CFG branches. Sequential
outputs reuse prompt embeddings, image layout masks and deterministic reference
latents within the job; random target noise and denoising caches remain independent.
The encoder computes only the final unused language-logit position and disables
its autoregressive cache. VAE slicing bounds batch decode work per image.

Checkpoint discovery accepts the configured revision only, or a complete explicit
`MODEL_PATH`. An unrelated or incomplete cached model cannot silently satisfy it.
The worker validates metadata and every referenced weight shard before loading
GPU libraries, enables Hub/Transformers offline mode, and uses local-only component
loading. There is no serving-mode download fallback. Cache validation checks file
presence and nonempty shards; the model loader still checks tensor contents.

Provision the cache separately, before starting a GPU worker. Runpod cached-model
mounts must contain the exact `MODEL_ID` and `MODEL_REVISION` configured on the
endpoint. If using your own persistent cache, the explicit provisioning command is:

```bash
docker run --rm -v qwen-cache:/cache qwen-image-2.1-worker:0.0.1.2 \
  python scripts/download_model.py
```

This command intentionally permits downloads and does not need a GPU. Set the same
model/revision environment variables on provisioning and serving containers. The
serving runtime never invokes this downloader. A fatal cache error exits before
queue registration; correct the mount/revision before restarting workers.

### Generation logs

Each job logs dimensions, references, actual sequential/parallel routing, CPU
offload mode, pass number, and seeds. Denoising logs the first/last step and at
most one update per five seconds between them, with progress, elapsed time and an
estimated remaining time. The completion/failure summary includes total elapsed
time, lock wait, pipeline/encoding time, image count, response bytes, and peak
PyTorch CUDA allocated/reserved VRAM across **all** passes of the job. These are
allocator peaks, not device-wide driver/process memory measurements. Sequential
generation resets peak counters once per job, rather than once per image.

For example (illustrative values):

```text
[JOB example] GENERATION START: route=sequential, outputs=2, steps=40, offload=model
[JOB example] Generation pass 1/2: 1 image(s), seeds=[42]
[DENOISING] step 20/40 (50%), pipeline elapsed=29.00s, estimated remaining=29.00s
[JOB example] COMPLETE: route=sequential, images=2/2, elapsed=117.000s, lock_wait=0.000s, pipeline=115.000s, encoding=2.000s, peak VRAM allocated=16.37 GiB, reserved=16.54 GiB, response=4500000 bytes
```

The worker uses BF16 and requires an Ampere-or-newer CUDA GPU. It keeps one model
instance per process and serializes inference. CUDA OOM returns an actionable job
error and requests a worker restart. Other inference failures also reset the worker;
input errors keep the model warm.

## Verify a deployment

Container dependency check:

```bash
docker run --rm qwen-image-2.1-worker:0.0.1.2 python scripts/check_runtime.py
```

Real inference smoke test, on a Docker host with NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all -v qwen-cache:/cache -v "$PWD/outputs:/outputs" \
  qwen-image-2.1-worker:0.0.1.2 python handler.py \
  --local examples/smoke.json --output-dir /outputs
```

Import checks do not demonstrate image quality or GPU memory use. Real GPU
inference and performance are not yet verified. Before production, run the smoke
request and representative editing/2K jobs on the chosen Runpod GPU.

## Source pins and model license

- [Qwen-Image-2.1 model and usage](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [Pinned model snapshot](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/790c92633540aa0cb11d9abf19eb46d861714758)
- [Pinned Diffusers pipeline](https://github.com/huggingface/diffusers/blob/e0118ade2f60234c41bacf40330a7e2f61108849/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- [Transformers 5.17.0](https://github.com/huggingface/transformers/releases/tag/v5.17.0)
- [Qwen Research License for model weights](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

The container pins its direct dependencies and the Diffusers source commit.
The build resolves transitive dependencies; record `pip freeze` and deploy an image
digest to reproduce the exact tested environment.
