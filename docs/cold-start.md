# Cold-start investigation

RunPod `delayTime` includes waiting for a worker and cold startup. It is not the
same measurement as Python initialization. [RunPod's optimization guide](https://docs.runpod.io/serverless/development/optimization)
describes the distinction.

## Evidence from this checkout and the supplied log

- On October 18, the supplied log reported cache resolution **0.00 s**, pipeline
  component loading **1.04 s**, and service initialization **3.41 s**.
- Those logs do not contain the complete platform queue/allocation interval.
  They cannot identify the cause of the reported change from 13–17 s to 30–50 s.
- GitHub main remained `99c28f5` when checked. The later cache-only/response-fitting
  worker changes were local at that time. A local checkout is not evidence of
  the image RunPod is actually running; compare its source fingerprint/image tag.
- The current Dockerfile installs dependencies and performs import checks during
  the image build, not at worker startup. Added documentation is approximately
  600 KiB. There is no startup compilation, warm-up inference, or paid keepalive.
- Snapshot validation reads metadata and checks shards, without scanning weight
  contents. A cache miss now fails rather than downloading weights in serving mode.

## New measurements

The first startup log precedes the SDK import. Logs report SDK import time,
source fingerprint, cache validation, Torch/Diffusers imports, device/diagnostic
setup, model loading, placement, and process-to-service-ready time. Compare these
with the **same job's** RunPod delay and worker logs. A long delay with short
process startup points outside Python initialization; a slow import/load/placement
stage identifies the code/storage stage to investigate. These logs are diagnostics,
not evidence that startup has become faster.

## Changes to consider after identifying the slow stage

1. Verify FlashBoot is enabled and the exact configured model revision is cached.
   Managed cached models schedule workers onto hosts with preloaded files.
2. Check worker availability and allowed GPU/data-center/CUDA combinations.
   A network volume restricts scheduling to its data center and adds network I/O.
   Broadening choices requires checking each GPU's memory and compatibility.
3. For interactive sessions, a modest idle timeout can reuse a running worker
   between requests. Waiting time is billed. Keeping an active worker avoids
   baseline cold starts but is billed continuously. No such settings were changed.
4. If imports/loading dominate, benchmark shard loading and CPU/RAM allocation
   on the actual GPU host. Do not blindly load the entire model onto a 24 GiB GPU:
   this checkpoint has about 30.2 GiB of BF16 parameters. Avoid unmeasured parallel
   loading that could exhaust host RAM. Keep the existing offload strategy.

See [endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)
and [worker lifecycle](https://docs.runpod.io/serverless/workers/overview) for
FlashBoot, scheduling constraints, and billing behavior. An earlier management API check in this session returned HTTP 403; current
endpoint settings have not been verified for this investigation. No paid benchmark jobs were submitted.
