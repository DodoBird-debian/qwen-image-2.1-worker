# Losing a completed image on a Runpod worker

The worker can finish inference, log `RUNPOD JOB COMPLETED`, and lose the only copy of the image before Runpod accepts the result. The most important fix is to save each finished image under its job ID and seed while the worker still has the pixels. Two smaller changes can immediately protect batches from encoder failures.

This review covers the current worker at `09d4b5414025f706b6b2b22bb727e56d44c0ae56`, its CLI, and the frozen worker at `99c28f5d8a7fe0a5fa489ef4d619b1453198fccb`. The current source fingerprint was `618017a8a336ef8a`. The evidence was collected locally on 2026-11-27 with synthetic pipeline outputs, real Pillow encoders, and the pinned Runpod SDK. Attributing a particular charge requires the deployed build, job ID, worker logs, and usage for that attempt. [Recorded observations](baseline.json) · [Worker fingerprint](../qwen_worker/runtime.py#L28-L39)

## Run the reproductions

From the root of `qwen-image-2.1-worker`, run all seven cases or select one by name. The script declares its Python dependencies inline. `uv` installs them on first use. The cases use local fixtures and a fake HTTP session. They require no GPU, model weights, API key, or paid endpoint.

```sh
uv run --python 3.12 astra62newfix/reproduce.py
uv run --python 3.12 astra62newfix/reproduce.py sequential
```

The case names are `upload`, `parallel`, `sequential`, `late-stop`, `packing`, `frozen`, and `cli`. The full run starts each case in a fresh process and prints one JSON report. In an existing environment with the four pinned packages, `python astra62newfix/reproduce.py` also works. After `uv` caches the dependencies, the run can be repeated offline. [Reproduction script](reproduce.py)

```sh
uv run --offline --python 3.12 astra62newfix/reproduce.py > /tmp/qwen-delivery-review.json
```

`reproduced` means the failure matched the reviewed behavior. `measured` means the packing workload completed and its returned seeds and response size were checked. `changed` prints the observations and the expectations that differed, which is useful after a fix. `error` indicates a harness or setup failure. A changed or errored case makes the command exit nonzero. Timing and memory vary by host. The memory fields use Linux `/proc/self/status` and are `null` where it is unavailable. Once durable storage exists, also check its contents in the acceptance tests below. A cancelled job can still return an empty inline response after its files have been saved successfully.

The frozen case uses the sibling `../qwen-image-2.1-worker-frozen` when present. Otherwise it reads the frozen commit from local Git history into a temporary directory. All 12 worker source and requirements files in the sibling used for this review matched that commit. In a shallow checkout, an existing frozen source tree can be supplied explicitly.

```sh
uv run --python 3.12 astra62newfix/reproduce.py frozen --frozen-root /path/to/frozen-worker
```

## 1. The final upload can fail after the completion log

**Reproduction.** `upload` fails the first result POST with `aiohttp.ClientConnectionError`. Every later POST is allowed to succeed. The real SDK made one attempt on the first delivery. The trace was `worker logged JOB COMPLETED` → `SDK POST` → `SDK logged upload failure` → `SDK logged Finished.` The call returned normally. Delivering the same job ID to the handler again ran inference again. The unseeded request drew 42 and then 142 from the controlled random source.

**Root cause.** The handler returns an in-memory dictionary. The SDK owns the subsequent result POST. Runpod 1.12.0 configures up to three attempts. The retry library's default exception set is empty, so this connection failure exits on the first attempt. The SDK catches the error and logs `Finished.` in `finally`. On replay the worker has no stored result or seed allocation to look up. [Handler return](../handler.py#L84-L94) · [SDK result POST](https://github.com/runpod/runpod-python/blob/v1.12.0/runpod/serverless/modules/rp_http.py#L29-L88) · [Retry exception defaults](https://github.com/inyutin/aiohttp_retry/blob/v2.9.1/aiohttp_retry/retry_options.py#L13-L36) · [Seed allocation](../qwen_worker/requests.py#L133-L135)

**Fix and acceptance.** Persist the selected seed and completed images as described below. Retry delivery of the saved result with bounded backoff for transient connection failures. Record upload acknowledgement at the transport layer and label the handler log as inference complete. Test both a disconnect before the POST arrives and a lost acknowledgement after acceptance. Replaying the same ID should recover identical image bytes with one total inference call.

## 2. One encoder failure wipes out a parallel batch

**Reproduction.** `parallel` returns four generated images from one pipeline call. Seeds 42 and 43 encode successfully. An injected encoder error on 44 prevents the worker from attempting 45. The handler returns zero images and `refresh_worker=true`.

**Root cause.** `_append_images` collects each encoding in a temporary list. The service commits the group to `result` only after every image encodes and fits. The exception handler sees an empty committed result and rethrows. The outer handler reports the encoder exception as an inference error. [Per-image encoding](../qwen_worker/service.py#L106-L116) · [Batch commit and exception handling](../qwen_worker/service.py#L198-L229) · [Refresh response](../handler.py#L118-L124)

**Fix and acceptance.** Track successful encodings by seed, continue encoding unaffected siblings, and pack those successes as a partial response. An alternate encoder can also recover the affected seed from its original pixels. Keep the existing `delivery` counts and warnings, and reserve worker refresh for failures that impair the runtime. With the injected error, seeds 42, 43, and 45 should remain decodable after one pipeline call. Report any missing seed explicitly.

## 3. Sequential recovery overwrites a usable response

**Reproduction.** `sequential` uses a 5 GiB synthetic tensor budget to select two passes. The first noisy 2048×1024 PNG fits as a decodable 1626×813 image in a 7,016,189-byte response. The body budget is 7,995,904 bytes. From the second pass onward the fixture makes the optimization encoder unavailable. The worker tries to fit both images, then tries to recompress the first one alone. It returns zero images.

**Root cause.** Recovery overwrites `result["images"]` with the original, larger encodings before calling `_fit_output` again. An error in that call escapes after the already-fitted copy has been displaced. The existing later-pass regression test uses small images and a pipeline failure. Its first-pass output needs no recompression. [Recovery path](../qwen_worker/service.py#L235-L245) · [Existing regression test](../tests/test_transport.py#L171-L190)

**Fix and acceptance.** Keep the last validated response intact while building the next candidate. Reserve room for partial-delivery metadata and use the saved encoding directly if fitting fails. Under the fixture fault, seed 42 should still decode, the final JSON including warnings should fit the cap, and the pipeline should run only the two original passes.

## 4. A late timeout can discard a fully fitted response

**Reproduction.** `late-stop` blocks final cleanup after seeds 42 and 43 have both fit. It then cancels the actual `serverless_handler` task, the mechanism the SDK uses for job stops and execution timeouts. The task waits for cleanup, exits cancelled, and returns neither image. This is a worker stop. The CLI's own polling deadline has separate behavior.

**Root cause.** The async wrapper sets the stop event and propagates cancellation after the thread unwinds. The handler also checks the event after `service.handle` returns. Completed passes exist only in process memory throughout this interval. [Async cancellation](../handler.py#L127-L150) · [Final stop check](../handler.py#L84-L93) · [Runpod stop handling](https://github.com/runpod/runpod-python/blob/v1.12.0/runpod/serverless/modules/rp_scale.py#L275-L282)

**Fix and acceptance.** Checkpoint each finished image before optional compression and final cleanup. Continue honoring cancellation and skip future GPU passes. In the late-stop fixture, the queue task should remain cancelled while an authorized lookup by job ID can retrieve the already-saved seeds. Apply the deployment's retention policy to explicitly cancelled work.

## 5. Large responses increase host RAM use after inference

**Measurement.** `packing` starts with four independent, noisy 2048×2048 RGBA frames already in memory. All four outputs fit as 575×575 images in 5,870,379 JSON bytes. A host OOM diagnosis for a real job requires the container or host termination record. [Rerun measurements](baseline.json)

| Local run | Host RSS with frames loaded | Peak host RSS | Post-processing time |
| --- | ---: | ---: | ---: |
| Original review | 101.8 MiB | 550.3 MiB | 6.46 s |
| Recorded script run | 108.7 MiB | 558.1 MiB | 6.63 s |

**Root cause.** The service retains pixel copies and the original base64 encodings while fitting candidates. The encoder makes another pixel copy, and size accounting serializes the full response into a string and UTF-8 bytes. These allocations occur on the host while the inference lock is held. VRAM admission does not account for them. [Retained originals](../qwen_worker/service.py#L106-L116) · [Pixel and base64 buffers](../qwen_worker/images.py#L59-L82) · [Response sizing and candidate encoding](../qwen_worker/transport.py#L19-L29) · [Packing loop](../qwen_worker/transport.py#L78-L99)

**Fix and acceptance.** Save original images one at a time, release redundant buffers, and keep compact metadata for candidate encodings. Count base64 lengths and serialized metadata incrementally, then verify the final serialized response once. Compare peak host RSS and elapsed time on the same fixture and host. Retain all four seeds, accurate delivery metadata and the 8,000,000-byte cap including the 4,096-byte envelope allowance.

## 6. The frozen worker rejects large output after generation

**Reproduction.** `frozen` generates one 2048×1024 RGBA fixture. Its PNG is 11,202,008 bytes after base64 encoding, already over the 8,000,000-byte cap. The frozen handler returns an inline-size error and zero images after one completed pipeline call. The first pass in `sequential` exercises the current worker's ability to fit the same pixels.

**Root cause.** The frozen service appends the original encoding and immediately raises `InputError` if the serialized result exceeds the cap. It has no output-fitting or durable-storage fallback. Sequential routing limits subsequent passes. The first completed image alone exceeded the cap here. [Frozen size check](https://github.com/DodoBird-debian/qwen-image-2.1-worker/blob/99c28f5d8a7fe0a5fa489ef4d619b1453198fccb/qwen_worker/service.py#L79-L98)

**Fix and acceptance.** For deployments still using the frozen build, port the current fitting and delivery metadata or deploy a compatible newer worker. Where exact output dimensions or transparency matter, retain the full-resolution original and supply a retrieval path alongside a fitted preview. The fixture should yield a decodable result after one pipeline call, with any size or format change disclosed. [Current delivery contract](../README.md#output)

## 7. A transient status error can lead to a second submission

**Reproduction.** `cli` accepts a fake `/run` and returns HTTP 503 on the first status poll for each job. The CLI exits after that poll and leaves no local files. Relaunching the same command creates a second ID. A subsequent GET for the *original* ID recovers seed 42. The second submission requires the relaunch. The simulated server kept the original result available.

**Root cause.** The CLI prints the accepted ID, polls without retrying a transient HTTP failure, and first writes job files after receiving a completed response. Its submission command always starts with `POST /run`. `--decode` reads a previously saved response file. [CLI request errors](../scripts/client.py#L20-L34) · [Submit, poll and save](../scripts/client.py#L125-L142)

**Fix and acceptance.** Atomically record the endpoint and accepted job ID immediately. Add a resume-by-ID command and bounded backoff for transient GET failures. Preserve an uncertain submission outcome when the initial POST acknowledgement is lost. Test a 503 followed by success, including a client restart. Recovery should save the original image with one submission total.

## Durable recovery across worker replacement

1. On acceptance, create a job record on a shared network volume or in object storage. Scope it to the endpoint and job ID. Store the request fingerprint, model revision and resolved seeds so a replacement worker resumes the same work. A conditional create or lease can coordinate concurrent deliveries of one job.
2. As each output becomes available, write its full-resolution bytes. Publish its manifest entry only after the write succeeds, using an atomic rename on a volume or a conditional manifest update in object storage. Include the seed, dimensions, MIME type and checksum. An interrupted write should leave earlier entries readable.
3. Before starting inference on replay, read the manifest. Reuse completed outputs and restrict any remaining inference to missing seeds. An independently accessible lookup by job ID lets the client retrieve saved images even if the final queue callback never arrives. Keep the existing inline response usable for current clients.
4. Exercise replacement after a saved image, duplicate delivery, and a lost upload acknowledgement. Verify the returned bytes and count actual pipeline calls. A process death between producing pixels and the first durable write remains a loss window. Set an expiry and cleanup policy for stored outputs and expose that expiry to clients.

The parallel and sequential fixes can ship independently of storage. They preserve usable images while a worker is alive. Durable storage also covers replacement and allows recovery after the queue result expires.

## Recover a recent job

Use the endpoint and job ID from the original submission. Runpod documents 30 minutes of result retention after completion for asynchronous `/run` jobs. Its `/retry` operation requeues an eligible failed or timed-out job with the same ID and clears previous output. Start with a status lookup. [Runpod operation reference](https://docs.runpod.io/serverless/endpoints/operation-reference#run) · [Retry behavior](https://docs.runpod.io/serverless/endpoints/operation-reference#retry)

```sh
status_file=$(mktemp "qwen-${RUNPOD_JOB_ID}.XXXXXX")
curl --fail --silent --show-error \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --output "$status_file" \
  "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/status/$RUNPOD_JOB_ID"
```

For `IN_QUEUE` or `IN_PROGRESS`, keep polling the same ID. For `COMPLETED` with `output.images`, decode the saved response locally. If a result is missing, correlate the job ID and source fingerprint with the worker's generation, encoding, cancellation and SDK upload logs.

```sh
uv run --no-project --with 'Pillow==12.3.0' python scripts/client.py \
  --decode "$status_file" --output-dir "outputs/$RUNPOD_JOB_ID"
```

The existing worker suite also passed all 134 tests on the reviewed source. To run it separately from the failure reproductions, use the following command from the worker root.

```sh
uv run --no-project --python 3.12 --with 'Pillow==12.3.0' \
  --with 'runpod==1.12.0' --with 'huggingface-hub==1.32.0' \
  python -m unittest discover -s tests
```
