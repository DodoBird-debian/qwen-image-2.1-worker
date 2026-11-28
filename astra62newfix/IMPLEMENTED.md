# Paid result recovery changes

The current worker, frozen worker, command-line client, and WebUI all have local fixes for the two audits. The audit reproduction scripts and their original measurements document the failures before the changes. [Worker audit](README.md) · [WebUI audit](UI.md)

## Worker and command-line client

| Audit finding | Implementation |
| --- | --- |
| Result upload interrupted | The worker configures bounded retries for transient SDK upload failures. It logs inference completion and upload acknowledgement separately. The same job ID can reuse the saved result when shared recovery is configured. |
| One encoder error wipes out siblings | The worker processes the remaining seeds and returns the usable images with a delivery warning. It checkpoints original pixels before transport encoding when recovery is enabled. |
| Later packing corrupts an earlier usable response | Fitting builds a candidate separately and retains the previous fitted response until a candidate passes the byte budget. |
| Cancellation loses finished pixels | Completed images are checkpointed before optional compression and cleanup. Cancellation still stops later GPU passes. An operator can retrieve the checkpointed PNGs. |
| Packing raises peak host memory | The size check counts base64 lengths without repeatedly serializing the full payload. The four-image fixture measured a reduction from 541.4 MiB to 308.5 MiB peak RSS in fresh processes on the same host. Both runs returned seeds 42–45 and serialized to 5,870,379 bytes. The comparison ran without the optional recovery directory. |
| Frozen worker rejects oversized output | The frozen source now fits oversized images and reports delivery changes. With recovery enabled it also stores the full-resolution originals. The 2048 × 1024 fixture returned a decodable seed 42 after one pipeline call. |
| CLI restarts can submit the same work again | The CLI writes an endpoint receipt before submission, records the acknowledged ID atomically, retries temporary status failures, and supports `--resume JOB_ID`. An ambiguous submission remains in the receipt until explicitly resolved. Use the same output directory when resuming. |

For recovery after worker replacement or queue expiry, configure `QWEN_RECOVERY_DIR` for each endpoint on a private shared mount that supports atomic rename and cross-worker POSIX locks. The worker records request identity, selected seeds, PNG checksums, and finished results. Use the CLI on a machine authorized to read the mount. Files are available after their checkpoint commits, and retention is controlled by the operator. [Worker setup and retention](../README.md#recovery-storage)

~~~sh
python scripts/client.py YOUR_ENDPOINT_ID --resume JOB_ID --output-dir outputs
python scripts/client.py --recover-worker JOB_ID --recovery-dir /mnt/qwen-recovery --output-dir recovered
~~~

## WebUI

| Audit finding | Implementation |
| --- | --- |
| Quota failure while running a series | Keep the current result downloadable, pause future requests, and retry saving before continuing. Reopened series can check the existing accepted job. |
| Partial tab overwrites complete history | Merge verified frames and pending decode bytes in IndexedDB. A verified image replaces the pending bytes for its seed. |
| History trimming deletes a recovered job | Check current status and ownership of saved images in the deletion transaction. |
| One-time decode failure causes repeat compute | Save header-checked bytes for local decode retry and consult the original job before offering new generation for a decode failure. Cached bytes can recover after the provider result expires. |
| Closing the tab loses an unconfirmed single request | With browser persistence enabled, claim a shared receipt before the POST. The receipt is restored in a fresh tab and shown in History. |
| Mistyped job ID unlocks a series retry | Verify the candidate through status before linking it. A failed lookup leaves the submission unresolved. |
| An unexpected seed puts valid siblings in the retry | Complete independently matched seeds and omit them from subsequent requests. |
| Interrupted GET body stops polling | Keep body-read transport errors eligible for bounded GET retries. |
| Local history read blocks remote status | Fall back to the active or manually provided ID when history fails to load. |

Run the browser acceptance checks from the worker repository root. The harness uses synthetic Runpod responses and actual Chromium storage. The worker tests also use CPU fixtures and a simulated SDK receiver. Activation on a deployed endpoint and shared-volume compatibility require validation in that environment.

## Independent audit

| Requirement | Verification |
| --- | --- |
| Worker 1 · Upload interrupted before or after acceptance | The installed SDK ran through its retry client against a simulated HTTP session. Both tests asserted the acknowledgement, equal payloads, and one inference across replay. |
| Worker 2 · Siblings survive an encoder failure | A parallel pipeline fixture fails seed 44 at the real encoder boundary. The test decodes 42, 43, and 45, checks the missing-seed warning, and verifies one pipeline call. |
| Worker 3 · Prior response survives later fitting failure | The second-pass encoding fault test decodes the retained PNG and measures the complete serialized response against the cap. |
| Worker 4 · Finished originals survive cancellation | The test cancels the real async handler after fitting and before cleanup finishes. The operator CLI recovery function reads and verifies the original files and their alpha channel. A separate retention test verifies that locked jobs are skipped. |
| Worker 5 · Packing memory and response size | The independent repeat measured 307.2 MiB peak RSS and 6.342 seconds. The prior same-host unfixed run measured 541.4 MiB and 6.681 seconds. Both versions retained four seeds and fit the 8 MB cap. |
| Worker 6 · Frozen output fits | The original 2048 × 1024 fixture produced 11,202,008 base64 bytes and returned seed 42 after one pipeline call. Frozen tests also exercised warnings, stored originals, and replay. |
| Worker 7 · CLI recovery after restart | Two actual CLI processes used a loopback HTTP fixture. The first stopped while status returned 503, and the second resumed the recorded ID. The test counted one POST and checked the saved PNG byte for byte. |
| UI 1–3 · Quota, merging, and cleanup | Chromium exercised real IndexedDB transactions. The focused suite checks resume after a failed write, exact bytes after both orders of disjoint writes, and preservation of pending decode bytes while an empty failure is deleted. |
| UI 4–5 · Decode recovery and uncertain submissions | Chromium made real Blob decodes with injected faults. It recovered cached pixels after a simulated expiry and checked that a new tab preserved the unresolved submission with one POST total. |
| UI 6–7 · Wrong IDs and seed mismatch | A 404 and a valid ID with unrelated seeds both leave the series unconfirmed. The tests also check that a uniquely matched sibling completes, preserves its original bytes, and stays out of the paid retry. Conflicting images with the same seed remain flagged. |
| UI 8–9 · Failed body or browser history read | Injection at the response stream and IndexedDB read boundaries drives the normal UI recovery flow. Both tests count the status and generation requests and verify recovery of the original image. |

The independent run used synthetic CPU images and simulated provider responses. The provider tests establish client and worker behavior under these failures. Confirming production receiver semantics, filesystem locking on the deployed volume, and the actual paid-job outcome requires evidence from the deployment. For a manually entered series ID, the UI uses the endpoint, returned ID, and matching seed set. Different jobs can use the same seeds, so the user still needs to identify the original job in Runpod. [Chromium acceptance evidence](ui-verified.json)

Two negative controls confirmed the tests detect the failures. The parallel encoder test raised on seed 44 against the archived pre-fix worker. In an isolated browser copy with the pending-frame retention check removed, Chromium deleted the pending result and failed the retention assertion.

The final rerun passed 148 current-worker tests, 81 frozen-worker tests, 111 WebUI unit tests, all 79 browser regression tests, and all 15 targeted Chromium acceptance cases. TypeScript also passed.

~~~sh
node astra62newfix/verify_ui_fixes.mjs
~~~
