# Qwen-Image-2.1-Turbo migration

The `trubo` branches in the worker and WebUI use Qwen's official fixed eight-step
schedule, rather than creating an experimental seven-step schedule.

## Runpod configuration

Select this cached model in the endpoint's Model field:

```text
https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo:fc13ba1986fcb905854dc8de2f52eee395f49b5b
```

Set the matching environment:

```text
MODEL_ID=Qwen/Qwen-Image-2.1-Turbo
MODEL_REVISION=fc13ba1986fcb905854dc8de2f52eee395f49b5b
CPU_OFFLOAD=model
```

Use the image built from the worker's `trubo` branch. Leave `MODEL_PATH` unset
unless it identifies a complete Turbo snapshot. Keep production downloads
disabled; `QWEN_HUB_TEST_PROVISION` remains a test-only exception. A working cache
of the base model does not satisfy the Turbo model ID/revision. Switching models
does not itself repair a missing Runpod cache mount.

Deploy the WebUI's `trubo` branch against that endpoint after its cached-model
startup succeeds. Existing browser history remains available. Imported base-model
step settings normalize to eight; comparison grids offer seed and CFG axes because
changing only the requested step count does not change Turbo's saved schedule.

## Runtime compatibility

Turbo retains `QwenImage21Pipeline`, the same transformer configuration, VAE
latent channels and spatial compression, reference-image API, and RGBA support.
The existing memory formulas and CPU-offload strategy therefore remain applicable;
fewer denoising iterations do not shrink the model weights or persistent KV cache.
Actual GPU peaks and output quality still need validation on the deployed hardware.

Diffusers is pinned to `da1d3829cf08d4f329b526d89e17cc035c049d8d`, the merge of
[PR 14950](https://github.com/huggingface/diffusers/pull/14950). The former pin
ignored the checkpoint's `sample_sigmas`. The new pin loads that configuration;
our build check verifies constructor support and startup rejects a missing or
invalid Turbo schedule. The transformer implementation is unchanged between the
pins; the QwenImage21 VAE diff contains annotations, comments, and whitespace.

The API keeps its existing request fields and limits. `num_inference_steps`
defaults to 8. Valid older values from 1 to 100 normalize to the checkpoint's
actual schedule length before inference and response construction, so logs and
`parameters.num_inference_steps` report the actual eight steps. CFG defaults to 1;
values above 1 still add a negative branch. The four-image request limit, serial
request batching, transport fitting, and cache-only startup remain in place.

## Validation

CPU tests cover saved-schedule validation, normalization of legacy requests, actual
step metadata, and eight-step progress. WebUI unit/browser checks cover the eight-
step default, imported settings, and seed/CFG comparisons. Hub smoke inputs use
Turbo at eight steps for text generation, transparency, and reference editing.
No local GPU quality or latency benchmark is implied by those CPU checks.

Sources: [model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo),
[pinned model configuration](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/model_index.json).
