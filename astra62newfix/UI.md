# Where the WebUI can lose completed images or repeat paid work

The WebUI can erase an image already saved in browser history. In one reproduction, two tabs collected the same job. The first saved both seeds, the second saved one, and only one survived after both tabs closed. In another, the UI submitted three consecutive jobs while image storage was failing. It showed all three as complete, but reopening the series recovered zero images.

Nine cases below reproduce on the local WebUI working tree. They cover loss of the browser copy, repeat submissions, and interruptions to collecting an existing job. The tests used synthetic Runpod responses and incurred zero GPU charges. Establishing what happened to a particular paid job requires its endpoint, job ID, deployed UI version, and provider records. [Recorded observations](ui-baseline.json) · [Browser reproducer](reproduce_ui.mjs)

**If a result is missing now**, keep any tab that can still display it open and download the raw image there. If you know the endpoint and job ID, check the original job status before confirming **Continue missing images** or **Retry failed images**. Both buttons can create new jobs. Runpod documents a 30-minute result window after an asynchronous job completes. A status lookup during that window can still return pixels the UI lost locally. [Runpod operation reference](https://docs.runpod.io/serverless/endpoints/operation-reference)

## Run the reproductions

From the worker repository root, use the neighboring `qwen-web-ui` checkout with its dependencies installed. The reproducer starts its own loopback server, opens disposable Chromium profiles, and intercepts every provider request. It creates valid 512 × 512 PNGs, drives the real UI controls, and checks the bytes of stored results against their original SHA-256 digests.

~~~sh
node astra62newfix/reproduce_ui.mjs
node astra62newfix/reproduce_ui.mjs --list
node astra62newfix/reproduce_ui.mjs partial-resume concurrent-save
node astra62newfix/reproduce_ui.mjs --ui-root /path/to/qwen-web-ui quota-series
~~~

For a fresh UI checkout, install its locked dependencies with `npm ci` there. Use Node 22.12+ or 24 and Chromium. The script accepts `CHROMIUM_PATH`, uses `/usr/bin/chromium` when present, and otherwise uses Playwright’s Chromium installation. The repository’s setup also describes the browser requirements. [WebUI development setup](../../qwen-web-ui/README.md#L298)

A zero exit means every selected failure matched its recorded observations. After a fix, a changed outcome or a setup failure produces a nonzero exit. The JSON identifies the case, failed assertion, and requests observed. Use the acceptance condition below to judge the change. [Baseline JSON](ui-baseline.json)

## 1. A series keeps spending after browser storage fails

**Reproduce with `quota-series`.** Enable browser storage and start a three-image series with one seed per request. The fixture rejects each IndexedDB image write with `QuotaExceededError` while allowing small metadata writes. Seeds 42, 43, and 44 all arrived in memory. After the first failed image write, the UI submitted the next two jobs automatically. The series reached 3/3 and history still said “Saved in this browser.” After closing the tab and reopening the saved series, all three cells were unavailable.

**Root cause.** The repository retains the image in memory when a disk write throws. The job manager shows a storage warning, clears the pending marker before the terminal save, and returns a completed job to the runner. The runner can then dispatch the next request. The checkpoint records each cell’s status and job ID, while the history label reflects the global storage preference. Neither establishes that the image bytes committed. [Storage fallback](../../qwen-web-ui/src/storage/db.ts#L38) · [Warning](../../qwen-web-ui/src/api/jobs.ts#L97) · [Completion order](../../qwen-web-ui/src/api/jobs.ts#L218) · [Checkpoint](../../qwen-web-ui/src/generation/series-checkpoint.ts#L55) · [History label](../../qwen-web-ui/src/components/image-history.ts#L55)

**Fix and acceptance.** Track delivery and local save success separately for each job. When an opted-in image write fails, keep the blob downloadable, retain the job ID for recovery, and pause before the next submission. Offer to retry saving or explicitly continue in memory. With this fixture, one job should be submitted before the pause and the UI should identify seed 42 as available only in the tab. After storage recovers, saving it should require zero additional generation requests.

## 2. A second tab can overwrite a more complete saved job

**Reproduce with `concurrent-save`.** Two tabs follow the same accepted job before either receives its response. Both receive the same two valid PNGs. Tab A commits seeds 42 and 43. Tab B experiences a one-time decoder failure on seed 43 and commits seed 42. After closing both tabs, a fresh tab reads only seed 42 from IndexedDB. This required one generation request.

**Root cause.** The write guard protects a nonempty result from an empty update. An incoming array containing even one frame passes the guard and replaces the entire stored array and summary. Session reads also prefer a nonempty cached copy even when disk holds additional frames. [Write guard](../../qwen-web-ui/src/storage/indexeddb.ts#L151) · [Frame replacement](../../qwen-web-ui/src/storage/indexeddb.ts#L50) · [Cached reads](../../qwen-web-ui/src/storage/db.ts#L52)

**Fix and acceptance.** Merge verified frames for the same endpoint, job, and generation attempt by exact seed in one IndexedDB transaction. Derive the summary from the committed set and refresh the in-memory view from it. Preserve each distinct valid seed when two partial results contain different images. Either write order in this reproduction should leave both original PNGs readable in a fresh tab.

## 3. Stale history cleanup can delete a recovered job

**Reproduce with `stale-trim`.** Set the history limit to one. Tab A sees job 1 time out. Model a provider retry, then collect the same job ID in tab B and verify that seed 42 is saved. Close B. When A finishes an unrelated job for seed 80, its automatic cleanup deletes job 1’s record and PNG. A fresh tab finds only job 2. The small limit exposes the deletion quickly. Runpod’s retry operation reuses the ID of a failed or timed-out job. [Runpod retry behavior](https://docs.runpod.io/serverless/endpoints/operation-reference#retry)

**Root cause.** Cleanup reads disk summaries, then overwrites them with this tab’s session summaries. Tab A’s old `TIMED_OUT` status displaces the saved `COMPLETED` status for job 1. The delete transaction removes the job, frames, and summary without checking whether it now owns images. [Candidate selection](../../qwen-web-ui/src/storage/db.ts#L85) · [Deletion](../../qwen-web-ui/src/storage/indexeddb.ts#L187)

**Fix and acceptance.** Give automatic cleanup its own conditional deletion operation. Re-read the current terminal status and check for stored frames inside the same write transaction that performs deletion. Keep explicit user deletion available. The original test should retain seeds 42 and 80 even with a history limit of one. An old terminal failure with zero frames should still be eligible for cleanup.

## 4. A local decode failure can lead to paying again for a delivered seed

**Reproduce with `partial-resume`.** The original status response contains valid PNGs for seeds 42 and 43. Make the browser decoder fail once on 43. Only 42 is saved. Entering the original ID in **History → Resume polling** leaves the total status GET count at one. Explicitly confirming **Continue missing images** submits a second job for 43 even though the original response still contains it.

**Root cause.** A transient decode exception discards that frame’s raw bytes. One successful sibling leaves the job marked completed. Both manual resume and the job manager treat any saved image as sufficient. The runner labels the missing cell as omitted for response size even when the first response delivered it. [Frame decoding](../../qwen-web-ui/src/api/jobs.ts#L60) · [Per-frame error handling](../../qwen-web-ui/src/api/jobs.ts#L181) · [History shortcut](../../qwen-web-ui/src/app.ts#L1567) · [Job manager shortcut](../../qwen-web-ui/src/api/jobs.ts#L143) · [Missing cell](../../qwen-web-ui/src/generation/runner.ts#L171)

**Fix and acceptance.** Record which seeds the response contained and which the browser decoded. Keep size- and header-checked raw bytes available for another decode, with a separate validation state. For an incomplete cache, allow **Resume polling** to fetch the original ID and merge recovered frames. Use worker delivery metadata to explain actual omissions. With the one-time fault, both seeds should be recovered from the original bytes or status response after exactly one generation request.

## 5. Closing a tab can lose the only record of an unconfirmed single submission

**Reproduce with `lost-single-receipt`.** Enable browser storage. The fixture accepts a single request for seed 42 and loses the response before the UI receives its job ID. The original tab blocks another submission. Close it and open a fresh tab in the same browser profile. **Generate** is enabled for the same input. Clicking it creates a second job for 42. History has no record of the first job, and only the second ID is polled.

**Root cause.** A single submission records its outstanding attempt in tab-scoped `sessionStorage`. The durable job record is created only after a valid response ID arrives. The saved series checkpoint applies to planned series. A fresh tab therefore has no single-job receipt or warning to consult. [Tab guard](../../qwen-web-ui/src/generation/submission-guard.ts#L87) · [Record creation](../../qwen-web-ui/src/api/jobs.ts#L113) · [Single and series submission](../../qwen-web-ui/src/app.ts#L1482)

**Fix and acceptance.** With persistence enabled, commit a small attempt record before sending a single-job POST. Share it across tabs, attach the returned job ID when available, and expose unresolved attempts in History. If the response is lost, let the user reconcile the attempt with Runpod or explicitly authorize another run. After closing the first tab in this test, the fresh tab should show that unresolved attempt before accepting the same request again.

## 6. Checking a mistyped job ID can unlock a duplicate series request

**Reproduce with `unverified-id`.** The fixture accepts seeds 42 and 43 under `job-1`, then drops the submission response. In **Resolve unconfirmed submission**, enter `job-typo` and choose **Check this job ID**. Its status lookup returns 404. The saved grid assigns both cells to the typo, marks them failed, and clears the pending guard. Confirming **Retry failed images** submits both seeds again. The first job still contains both images.

**Root cause.** The UI links and saves the candidate ID and acknowledges the unknown submission before checking the candidate with the provider. The ordinary known-job 404 handler marks the candidate expired. Series reconciliation treats that terminal result as a failure for both cells. [Link and save](../../qwen-web-ui/src/app.ts#L1427) · [Subsequent status check](../../qwen-web-ui/src/app.ts#L1463) · [404 handling](../../qwen-web-ui/src/api/jobs.ts#L222) · [Cell reconciliation](../../qwen-web-ui/src/generation/runner.ts#L89)

**Fix and acceptance.** Check the candidate with a read-only status request before replacing the unresolved attempt. Verify the endpoint, returned ID, and any available evidence that it belongs to these cells. Keep the original guard on a 404 or an inconclusive response. In this case the user should be able to correct the ID to `job-1`, collect both images, and finish after one POST.

## 7. One unexpected seed puts a valid sibling into the paid retry

**Reproduce with `seed-mismatch`.** Request 42 and 43. Supply a reply with two decodable images labeled 42 and 99. The UI saves both images in history, but the series reports 0/2 and marks both planned cells failed. Confirming the retry submits 42 and 43 together. This case requires anomalous seed metadata from the provider or worker.

**Root cause.** The runner validates the entire returned seed set before completing any cell. Its error path attaches the matching seed 42 to its cell but still marks that cell failed. Retry selects cells by status, so it includes the already saved image. [Batch validation](../../qwen-web-ui/src/generation/runner.ts#L163) · [Failure assignment](../../qwen-web-ui/src/generation/runner.ts#L201) · [Retry selection](../../qwen-web-ui/src/app.ts#L1366)

**Fix and acceptance.** Complete each uniquely matched expected seed, flag extra or ambiguous images separately, and keep raw downloads available. The first response should complete 42. A confirmed replacement should request only 43. A duplicate seed with conflicting images should remain flagged for review.

## 8. An interrupted status body stops automatic collection

**Reproduce with `interrupted-status`.** The first status GET receives successful headers, then its response stream raises a read error. The UI reports invalid JSON and pauses after one GET. The accepted ID survives. One click on **Check server status** sends a second GET and retrieves seed 42 under the original ID. This was a recoverable interruption with one generation request.

**Root cause.** The transport reads the body and parses JSON in the same catch block. A body-read `TypeError` becomes a generic parsing error, which bypasses the poller’s existing transient-error backoff. [Body read](../../qwen-web-ui/src/api/runpod.ts#L74) · [Polling retries](../../qwen-web-ui/src/api/runpod.ts#L145)

**Fix and acceptance.** Read the body separately so transport errors retain their type. Use the existing bounded retry for a failed status GET and preserve the POST ambiguity handling. One interrupted read followed by a valid response should collect seed 42 automatically with two status GETs and one generation request.

## 9. A local history read can block a remote status check

**Reproduce with `history-read-error`.** Pause polling an accepted job and let the fixture finish it. Make IndexedDB fail one job-record read. The first click on **Check server status** shows the local storage error and sends zero GETs, although the active tab still knows the ID. A second click after the read fault clears sends one GET and recovers seed 42.

**Root cause.** Resume awaits `getJob` before using the active in-memory record or constructing the status client. When persistence is enabled, `getJob` reads IndexedDB even with a cached record. An exception exits the recovery path before the remote lookup. [Cache and disk lookup](../../qwen-web-ui/src/storage/db.ts#L52) · [Resume ordering](../../qwen-web-ui/src/app.ts#L1565) · [Error exit](../../qwen-web-ui/src/app.ts#L1596)

**Fix and acceptance.** Treat history lookup as optional when the user already has the endpoint and job ID. Fall back to the current record or the known ID, show any storage warning, and expose the recovered raw result for download. With the same one-time read fault, the first click should fetch the result and the generation POST count should remain one.

## Verification

Tested on 2026-11-27 with UI version 1.2.1, Node 24.22.2, and Chromium 156.0.8078.79. The UI checkout contained local edits and untracked source files on top of `f9f00506e007d6701358a8f20b4976c15af680c0`. The baseline fingerprints 73 files in total, covering the entire `src/` directory, `index.html`, `package.json`, and `package-lock.json`. Source links assume the UI and worker repositories are sibling checkouts. [Exact fingerprint and case results](ui-baseline.json)

The existing 107 UI unit tests and seven browser tests covering recovery, partial delivery, history migration, and cross-tab series coordination also passed. TypeScript completed with `--noEmit`. The failure cases above exercise the additional storage, timing, and response conditions described for each. [Existing recovery tests](../../qwen-web-ui/tests/browser/recovery-and-crops.spec.ts) · [History tests](../../qwen-web-ui/tests/browser/history.spec.ts) · [Polling and migration tests](../../qwen-web-ui/tests/browser/audit.spec.ts)
