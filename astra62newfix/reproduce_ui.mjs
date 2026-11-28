#!/usr/bin/env node
// Runs the checked-out WebUI in Chromium. All provider traffic is handled in memory.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scenarios = {
  "partial-resume": partialResume,
  "concurrent-save": concurrentSave,
  "stale-trim": staleTrim,
  "quota-series": quotaSeries,
  "interrupted-status": interruptedStatus,
  "history-read-error": historyReadError,
  "seed-mismatch": seedMismatch,
  "unverified-id": unverifiedId,
  "lost-single-receipt": lostSingleReceipt,
};
const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("--list")) {
  console.log("node reproduce_ui.mjs [--ui-root PATH] [CASE ...]\n" + Object.keys(scenarios).join("\n"));
  process.exit(0);
}
let uiRoot = fileURLToPath(new URL("../../qwen-web-ui/", import.meta.url));
const rootOption = args.indexOf("--ui-root");
if (rootOption !== -1) {
  if (!args[rootOption + 1]) throw new Error("--ui-root needs a path");
  uiRoot = resolve(args.splice(rootOption, 2)[1]);
}
const selected = args.length ? args : Object.keys(scenarios);
for (const name of selected) assert.ok(Object.hasOwn(scenarios, name), "Unknown case " + name);
const uiRequire = createRequire(join(uiRoot, "package.json"));
const { chromium, expect } = uiRequire("@playwright/test");
const { createServer } = await import(pathToFileURL(uiRequire.resolve("vite")).href);
const endpoint = "ui-audit";
let browser, server, origin, pictures;

async function sourceFingerprint() {
  const files = ["index.html", "package.json", "package-lock.json"];
  async function walk(dir) {
    for (const item of await readdir(join(uiRoot, dir), { withFileTypes: true })) {
      const path = dir + "/" + item.name;
      if (item.isDirectory()) await walk(path);
      else if (item.isFile()) files.push(path);
    }
  }
  await walk("src");
  const hash = createHash("sha256");
  for (const path of files.sort()) {
    hash.update(path).update("\0").update(await readFile(join(uiRoot, path))).update("\0");
  }
  let head;
  try { head = execFileSync("git", ["rev-parse", "HEAD"], { cwd: uiRoot, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(); }
  catch { head = null; }
  return { version: uiRequire("./package.json").version, head, fileCount: files.length, sha256: hash.digest("hex") };
}

async function makePictures() {
  const page = await browser.newPage();
  try {
    return await page.evaluate(async () => {
      const images = [];
      for (let i = 0; i < 3; i++) {
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 512;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#849ac7";
        ctx.fillRect(0, 0, 512, 512);
        ctx.fillStyle = "#be653e";
        for (let j = 0; j < 9 * i; j++) ctx.fillRect(j * 19, j * 17, 13, 21);
        const blob = await new Promise(resolveBlob => canvas.toBlob(resolveBlob, "image/png"));
        const bitmap = await createImageBitmap(blob);
        if (bitmap.width !== 512 || bitmap.height !== 512) throw new Error("Invalid fixture");
        bitmap.close();
        images.push({ base64: canvas.toDataURL("image/png").split(",")[1], size: blob.size });
      }
      return images;
    });
  } finally { await page.close(); }
}

function complete(id, seeds) {
  return { id, status: "COMPLETED", output: { images: seeds.map(seed => ({
    seed, width: 512, height: 512, mime_type: "image/png",
    image_base64: pictures[Math.abs(Number(seed) - 42) % pictures.length].base64,
  })) } };
}

// One isolated browser profile per scenario. Unknown paths or off-origin requests
// cannot reach a provider. The fixture retains the original result for each ID.
async function fixture() {
  const context = await browser.newContext({ serviceWorkers: "block" });
  const f = { context, jobs: new Map(), posts: [], polls: [], confirmations: [], unexpectedDialogs: [], historyLimit: 12 };
  await context.route("**/*", async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== origin) return route.abort();
    if (url.pathname === "/config.json")
      return route.fulfill({ json: { endpointId: endpoint, serverAuth: true, historyLimit: f.historyLimit } });
    if (!url.pathname.startsWith("/api/")) return route.continue();
    const [, , api, ep, op, id] = url.pathname.split("/");
    if (api !== "runpod" || ep !== endpoint) return route.fulfill({ status: 404, body: "" });
    if (op === "run" && request.method() === "POST") {
      const { input } = request.postDataJSON();
      const seeds = Array.from({ length: input.num_images }, (_, i) => Number(input.seed) + i);
      const jobId = "job-" + (f.posts.length + 1);
      f.posts.push(seeds);
      f.jobs.set(jobId, complete(jobId, seeds));
      if (f.onSubmit) return f.onSubmit(route, jobId);
      return route.fulfill({ json: { id: jobId, status: "IN_QUEUE" } });
    }
    if (op === "status" && request.method() === "GET") {
      f.polls.push(id);
      if (f.onStatus) return f.onStatus(route, id, request.frame().page());
      return serveJob(f, route, id);
    }
    return route.fulfill({ status: 404, body: "" });
  });
  return f;
}

function serveJob(f, route, id) {
  const job = f.jobs.get(id);
  return job ? route.fulfill({ json: job }) : route.fulfill({ status: 404, json: { error: "Unknown fixture ID" } });
}
function loseFirstReply(route, id) {
  return id === "job-1" ? route.abort("connectionreset") : route.fulfill({ json: { id, status: "IN_QUEUE" } });
}
async function openPage(f, init) {
  const page = await f.context.newPage();
  page.setDefaultTimeout(8000);
  page.on("dialog", async dialog => {
    if (dialog.type() === "beforeunload") return dialog.accept();
    if (dialog.type() === "confirm" && f.expectedConfirmation && dialog.message().startsWith(f.expectedConfirmation)) {
      f.confirmations.push(dialog.message());
      f.expectedConfirmation = undefined;
      return dialog.accept();
    }
    f.unexpectedDialogs.push(dialog.message());
    await dialog.dismiss();
  });
  if (init) await page.addInitScript(init);
  await page.goto(origin);
  await expect(page.locator("#connection-dot")).toHaveClass(/connected/);
  return page;
}
async function persist(page) {
  await page.locator("#persist-browser").check();
  await expect(page.locator("#storage-mode-note")).toContainText("Saved in this browser");
}
async function configureRun(page, { count = 1, batch, seed = 42 } = {}) {
  const options = page.locator("#run-options");
  if (!await options.evaluate(el => el.open)) await options.locator("summary").click();
  await page.locator("#run-kind").selectOption(batch ? "series" : "single");
  if (batch) {
    await page.locator("#series-total").fill(String(count));
    await page.locator("#series-batch").selectOption(String(batch));
  } else await page.locator("#image-count").selectOption(String(count));
  await page.locator("#width").fill("512");
  await page.locator("#height").fill("512");
  await page.locator("#seed").fill(String(seed));
  await page.locator("#prompt").fill("A ceramic cup");
}
async function generate(page) {
  await expect(page.locator("#generate")).toBeEnabled();
  await page.locator("#generate").click();
}
async function confirmSeriesAction(f, page, selector, expectedText) {
  const before = f.confirmations.length;
  f.expectedConfirmation = expectedText + " will create 1 new RunPod job";
  try {
    await page.locator(selector).click();
    assert.equal(f.confirmations.length, before + 1, "Expected a new-job confirmation");
  } finally { f.expectedConfirmation = undefined; }
}
async function manualResume(page, id) {
  await page.locator("#history-open").click();
  await page.locator("#manual-job-id").fill(id);
  await page.locator(".resume-manual").getByRole("button", { name: "Resume polling" }).click();
  await expect(page.locator("#content-dialog")).toBeHidden();
}
async function diskJob(page, id) {
  const stored = await page.evaluate(async key => {
    const disk = await import("/src/storage/indexeddb.ts");
    const job = await disk.get("jobs", key);
    if (!job) return null;
    const frames = [];
    for (const { seed, blob } of job.results ?? []) {
      const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
      const sha256 = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
      frames.push({ seed, sha256 });
    }
    return { status: job.status, frames };
  }, endpoint + ":" + id);
  if (!stored) return null;
  for (const frame of stored.frames) {
    const expected = pictures[Math.abs(Number(frame.seed) - 42) % pictures.length].sha256;
    assert.equal(frame.sha256, expected, "Stored PNG differs from fixture for seed " + frame.seed);
  }
  return { status: stored.status, seeds: stored.frames.map(frame => frame.seed) };
}
async function diskSeeds(page, id) { return (await diskJob(page, id))?.seeds ?? []; }
async function checkpoint(page) {
  return page.evaluate(async () => {
    const { savedSeries } = await import("/src/generation/series-checkpoint.ts");
    const [state] = await savedSeries();
    return Object.values(state?.value.cells ?? {}).map(cell => ({ status: cell.status, job: cell.job?.runpodId }));
  });
}
async function seed43DecodeError(page) {
  await page.evaluate(size => {
    const native = window.createImageBitmap.bind(window);
    window.auditDecoderRejections = 0;
    window.createImageBitmap = async (...args) => {
      const [blob] = args;
      if (window.auditDecoderRejections === 0 && blob instanceof Blob && blob.type === "image/png" && blob.size === size) {
        window.auditDecoderRejections++;
        throw new DOMException("One-time local decode failure", "InvalidStateError");
      }
      return native(...args);
    };
  }, pictures[1].size);
}
async function partialResume(f) {
  const page = await openPage(f);
  await persist(page);
  await seed43DecodeError(page);
  await configureRun(page, { count: 2, batch: 2 });
  await generate(page);
  await expect(page.locator("#run-progress")).toContainText("stopped · 1 / 2");
  await expect.poll(() => diskSeeds(page, "job-1")).toEqual(["42"]);
  const retained = f.jobs.get("job-1").output.images.map(img => img.seed);
  assert.deepEqual(retained, [42, 43]);
  assert.equal(await page.evaluate(() => window.auditDecoderRejections), 1);
  const initialGets = f.polls.length;
  await manualResume(page, "job-1");
  await expect(page.locator("#series-continue")).toBeEnabled();
  const getsAfterResume = f.polls.length;
  assert.equal(initialGets, 1);
  assert.equal(getsAfterResume, 1);
  await confirmSeriesAction(f, page, "#series-continue", "Continuing missing images");
  await expect(page.locator("#run-progress")).toContainText("completed · 2 / 2");
  assert.deepEqual(f.posts, [[42, 43], [43]]);
  assert.deepEqual(f.polls, ["job-1", "job-2"]);
  return { originallyDelivered: retained, locallySavedForOriginalId: await diskSeeds(page, "job-1"),
    initialGets, getsAfterManualResume: getsAfterResume, submittedSeedGroups: f.posts, confirmedNewJobs: f.confirmations.length };
}

async function concurrentSave(f) {
  const a = await openPage(f);
  await persist(a);
  const b = await openPage(f);
  await persist(b);
  await seed43DecodeError(b);
  let releaseA, releaseB;
  const holdA = new Promise(resolveGate => { releaseA = resolveGate; });
  const holdB = new Promise(resolveGate => { releaseB = resolveGate; });
  f.onStatus = async (route, id, page) => {
    await (page === a ? holdA : holdB);
    await serveJob(f, route, id);
  };
  try {
    await configureRun(a, { count: 2 });
    await generate(a);
    await expect.poll(() => f.polls.length).toBe(1);
    await manualResume(b, "job-1");
    await expect.poll(() => f.polls.length).toBe(2);
    releaseA();
    await expect.poll(() => diskSeeds(a, "job-1")).toEqual(["42", "43"]);
    const firstWrite = await diskSeeds(a, "job-1");
    releaseB();
    await expect.poll(() => diskSeeds(b, "job-1")).toEqual(["42"]);
    assert.equal(await b.evaluate(() => window.auditDecoderRejections), 1);
    await a.close();
    await b.close();
    const cold = await openPage(f);
    const afterClosingBoth = await diskSeeds(cold, "job-1");
    assert.deepEqual(afterClosingBoth, ["42"]);
    assert.deepEqual(f.posts, [[42, 43]]);
    return { firstDurableWrite: firstWrite, durableAfterClosingBothTabs: afterClosingBoth,
      submittedSeedGroups: f.posts, statusGets: f.polls };
  } finally { releaseA(); releaseB(); }
}

async function staleTrim(f) {
  f.historyLimit = 1;
  const a = await openPage(f);
  await persist(a);
  let timedOut = true;
  f.onStatus = (route, id) => id === "job-1" && timedOut
    ? route.fulfill({ json: { id, status: "TIMED_OUT" } }) : serveJob(f, route, id);
  await configureRun(a);
  await generate(a);
  await expect.poll(() => diskJob(a, "job-1")).toEqual({ status: "TIMED_OUT", seeds: [] });
  timedOut = false; // Model a retry at the provider under the same job ID.
  const b = await openPage(f);
  await persist(b);
  await manualResume(b, "job-1");
  await expect.poll(() => diskSeeds(b, "job-1")).toEqual(["42"]);
  const recovered = await diskJob(b, "job-1");
  await b.close();
  await configureRun(a, { seed: 80 });
  await generate(a);
  await expect.poll(() => diskSeeds(a, "job-2")).toEqual(["80"]);
  await expect.poll(() => diskJob(a, "job-1")).toBeNull();
  await a.close();
  const cold = await openPage(f);
  const remaining = await diskJob(cold, "job-2");
  const lost = await diskJob(cold, "job-1");
  assert.equal(lost, null);
  assert.deepEqual(remaining.seeds, ["80"]);
  assert.deepEqual(f.posts, [[42], [80]]);
  return { historyLimit: 1, recoveredBeforeTrim: recovered, originalAfterTrim: lost, newerJob: remaining,
    submittedSeedGroups: f.posts, statusGets: f.polls };
}

async function quotaSeries(f) {
  const page = await openPage(f, () => {
    const put = IDBObjectStore.prototype.put;
    window.auditFrameWriteFailures = 0;
    IDBObjectStore.prototype.put = function (value, ...rest) {
      if (this.name === "job-frames" && value?.length) {
        window.auditFrameWriteFailures++;
        throw new DOMException("Fixture image-storage quota", "QuotaExceededError");
      }
      return put.call(this, value, ...rest);
    };
  });
  await persist(page);
  await configureRun(page, { count: 3, batch: 1 });
  await generate(page);
  await expect(page.locator("#run-progress")).toContainText("completed · 3 / 3");
  await expect.poll(async () => (await checkpoint(page)).map(cell => cell.status)).toEqual(Array(3).fill("completed"));
  const ids = [...f.jobs.keys()];
  const inMemory = await page.evaluate(async ids => {
    const db = await import("/src/storage/db.ts");
    return Promise.all(ids.map(async id => (await db.jobFrames(id)).map(img => img.seed)));
  }, ids.map(id => endpoint + ":" + id));
  const onDisk = await Promise.all(ids.map(id => diskSeeds(page, id)));
  assert.deepEqual(inMemory, [["42"], ["43"], ["44"]]);
  assert.deepEqual(onDisk, [[], [], []]);
  const failedWrites = await page.evaluate(() => window.auditFrameWriteFailures);
  assert.equal(failedWrites, 3);
  const historyNote = await page.locator("#image-history-note").textContent();
  assert.match(historyNote, /Saved in this browser/);
  await page.close();
  const cold = await openPage(f);
  const saved = cold.locator("#saved-series");
  await expect(saved).toBeVisible();
  if (!await saved.evaluate(el => el.open)) await saved.locator("summary").click();
  await saved.getByRole("button", { name: "Reopen series" }).click();
  await expect(cold.locator("#run-batches")).toContainText("unavailable");
  const reopened = await cold.evaluate(async () => {
    const mod = await import("/src/generation/series-checkpoint.ts");
    const [row] = await mod.savedSeries();
    const restored = await mod.openSeries(row);
    return [...restored.cells.values()].map(cell => cell.status);
  });
  assert.deepEqual(reopened, Array(3).fill("unavailable"));
  assert.deepEqual(f.posts, [[42], [43], [44]]);
  return { submittedSeedGroups: f.posts, failedFrameWrites: failedWrites, inMemorySeeds: inMemory,
    durableSeeds: onDisk, historyNote, reopenedCellStates: reopened };
}

async function interruptedStatus(f) {
  const page = await openPage(f, () => {
    const originalFetch = window.fetch.bind(window);
    window.auditReadFailures = 0;
    window.fetch = async (...args) => {
      const response = await originalFetch(...args);
      if (String(args[0]).includes("/status/") && !window.auditReadFailures) {
        window.auditReadFailures++;
        return new Response(new ReadableStream({
          start(controller) { controller.error(new TypeError("Status response body interrupted")); },
        }), { status: response.status, headers: response.headers });
      }
      return response;
    };
  });
  await persist(page);
  await configureRun(page);
  await generate(page);
  const check = page.locator("#check-generation");
  await expect(check).toBeVisible();
  await expect(check).toBeEnabled();
  await expect(page.locator("#toast")).toContainText("invalid JSON");
  const guard = await page.evaluate(() => JSON.parse(sessionStorage.getItem("qwen-pending-submission")));
  assert.equal(guard?.id, "job-1");
  assert.equal(await page.evaluate(() => window.auditReadFailures), 1);
  assert.deepEqual(f.polls, ["job-1"]);
  await check.click();
  await expect.poll(() => diskSeeds(page, "job-1")).toEqual(["42"]);
  assert.deepEqual(f.posts, [[42]]);
  assert.deepEqual(f.polls, ["job-1", "job-1"]);
  return { automaticStatusGetsBeforePause: 1, pausedWithKnownId: guard.id,
    statusGetsAfterManualCheck: f.polls, recoveredSeeds: await diskSeeds(page, "job-1"), submittedSeedGroups: f.posts };
}

async function historyReadError(f) {
  const page = await openPage(f);
  await persist(page);
  let running = true;
  f.onStatus = (route, id) => running ? route.fulfill({ json: { id, status: "IN_PROGRESS" } }) : serveJob(f, route, id);
  await configureRun(page);
  await generate(page);
  await expect(page.locator("#job-status")).toContainText("Generating on RunPod");
  await page.locator("#pause-job").click();
  const check = page.locator("#check-generation");
  await expect(check).toBeEnabled();
  running = false; // The same ID now has a completed result ready for collection.
  await page.evaluate(() => {
    const get = IDBObjectStore.prototype.get;
    window.auditReadErrors = 0;
    IDBObjectStore.prototype.get = function (...args) {
      if (this.name === "jobs" && !window.auditReadErrors) {
        window.auditReadErrors++;
        throw new DOMException("One-time local history read failure", "UnknownError");
      }
      return get.apply(this, args);
    };
  });
  const getsBeforeCheck = f.polls.length;
  await check.click();
  await expect(page.locator("#toast")).toContainText("local history read failure");
  assert.equal(await page.evaluate(() => window.auditReadErrors), 1);
  const getsAfterFailedCheck = f.polls.length;
  assert.equal(getsAfterFailedCheck, getsBeforeCheck);
  const marker = await page.evaluate(() => JSON.parse(sessionStorage.getItem("qwen-pending-submission")));
  assert.equal(marker.id, "job-1");
  await check.click();
  await expect.poll(() => diskSeeds(page, "job-1")).toEqual(["42"]);
  assert.equal(f.polls.length, getsBeforeCheck + 1);
  assert.deepEqual(f.posts, [[42]]);
  return { knownId: marker.id, statusGetsSentByFirstManualCheck: getsAfterFailedCheck - getsBeforeCheck,
    statusGetsSentBySecondManualCheck: f.polls.length - getsAfterFailedCheck,
    recoveredSeeds: await diskSeeds(page, "job-1"), submittedSeedGroups: f.posts };
}

async function seedMismatch(f) {
  const page = await openPage(f);
  await persist(page);
  f.onSubmit = (route, id) => {
    if (id === "job-1") f.jobs.set(id, complete(id, [42, 99]));
    return route.fulfill({ json: { id, status: "IN_QUEUE" } });
  };
  await configureRun(page, { count: 2, batch: 2 });
  await generate(page);
  await expect(page.locator("#run-progress")).toContainText("failed · 0 / 2");
  await expect(page.locator("#run-message")).toContainText("unexpected seed 99");
  await expect.poll(async () => (await checkpoint(page)).map(cell => cell.status)).toEqual(["failed", "failed"]);
  const states = (await checkpoint(page)).map(cell => cell.status);
  const firstSaved = await diskSeeds(page, "job-1");
  assert.deepEqual(firstSaved, ["42", "99"]);
  await confirmSeriesAction(f, page, "#series-retry", "Retrying failed or cancelled images");
  await expect(page.locator("#run-progress")).toContainText("completed · 2 / 2");
  assert.deepEqual(f.posts, [[42, 43], [42, 43]]);
  return { requestedSeeds: f.posts[0], firstDurableSeeds: firstSaved, cellStatesBeforeRetry: states,
    confirmedRetrySeeds: f.posts[1], matchingSeedGeneratedTwice: 42 };
}

async function unverifiedId(f) {
  const page = await openPage(f);
  await persist(page);
  f.onSubmit = loseFirstReply;
  await configureRun(page, { count: 2, batch: 2 });
  await generate(page);
  await expect(page.locator("#run-progress")).toContainText("submission-uncertain");
  await page.locator("#series-verify").click();
  await page.getByLabel("RunPod job ID, if you found the submitted request").fill("job-typo");
  await page.getByRole("button", { name: "Check this job ID" }).click();
  const retry = page.locator("#series-retry");
  await expect(retry).toBeEnabled();
  await expect.poll(async () => (await checkpoint(page)).map(cell => cell.status)).toEqual(["failed", "failed"]);
  const beforeRetry = await checkpoint(page);
  assert.ok(beforeRetry.every(cell => cell.job === "job-typo"));
  assert.deepEqual(f.polls, ["job-typo"]);
  const guardCleared = await page.evaluate(() => sessionStorage.getItem("qwen-pending-submission") === null);
  assert.equal(guardCleared, true);
  const original = f.jobs.get("job-1").output.images.map(img => img.seed);
  await confirmSeriesAction(f, page, "#series-retry", "Retrying failed or cancelled images");
  await expect(page.locator("#run-progress")).toContainText("completed · 2 / 2");
  assert.deepEqual(original, [42, 43]);
  assert.deepEqual(f.posts, [[42, 43], [42, 43]]);
  return { acceptedIdWhoseReplyWasLost: "job-1", originalRetainedSeeds: original, cellsAfterCandidate404: beforeRetry,
    guardCleared, statusGets: f.polls, submittedSeedGroups: f.posts };
}

async function lostSingleReceipt(f) {
  const first = await openPage(f);
  await persist(first);
  f.onSubmit = loseFirstReply;
  await configureRun(first);
  await generate(first);
  await expect(first.locator(".submission-notice")).toContainText("submission response was lost");
  await expect(first.locator("#generate")).toBeDisabled();
  const marker = await first.evaluate(() => JSON.parse(sessionStorage.getItem("qwen-pending-submission")));
  assert.equal(marker.endpoint, endpoint);
  assert.equal(marker.id, undefined);
  assert.equal(await diskJob(first, "job-1"), null);
  await first.close();
  const fresh = await openPage(f);
  await expect(fresh.locator("#persist-browser")).toBeChecked();
  await configureRun(fresh);
  await expect(fresh.locator("#generate")).toBeEnabled();
  assert.equal(await fresh.evaluate(() => sessionStorage.getItem("qwen-pending-submission")), null);
  await generate(fresh);
  await expect.poll(() => diskSeeds(fresh, "job-2")).toEqual(["42"]);
  const originalSeeds = f.jobs.get("job-1").output.images.map(img => img.seed);
  assert.deepEqual(originalSeeds, [42]);
  assert.deepEqual(f.posts, [[42], [42]]);
  assert.deepEqual(f.polls, ["job-2"]);
  return { persistenceEnabled: true, firstTabBlocked: true, freshTabOfferedGenerate: true,
    originalRetainedSeeds: originalSeeds, originalDurableRecord: await diskJob(fresh, "job-1"),
    submittedSeedGroups: f.posts, statusGets: f.polls };
}

const cache = await mkdtemp(join(tmpdir(), "astra62-ui-"));
try {
  server = await createServer({ root: uiRoot, configFile: false, envDir: false, mode: "test",
    cacheDir: cache, logLevel: "error", server: { host: "127.0.0.1", port: 0, hmr: false },
    optimizeDeps: { noDiscovery: true, include: ["lossless-json", "fflate"] } });
  await server.listen();
  origin = "http://127.0.0.1:" + server.httpServer.address().port;
  const executablePath = process.env.CHROMIUM_PATH || (existsSync("/usr/bin/chromium") ? "/usr/bin/chromium" : undefined);
  browser = await chromium.launch({ executablePath, headless: true });
  pictures = await makePictures();
  assert.equal(new Set(pictures.map(p => p.size)).size, pictures.length, "PNG fixtures need distinct sizes");
  for (const pic of pictures) pic.sha256 = createHash("sha256").update(Buffer.from(pic.base64, "base64")).digest("hex");
  const report = {
    ui: await sourceFingerprint(),
    node: process.version,
    chromium: browser.version(),
    playwright: uiRequire("@playwright/test/package.json").version,
    vite: uiRequire("vite/package.json").version,
    transport: "Loopback Vite with an in-memory Runpod fixture",
    cases: {},
  };
  for (const name of selected) {
    const f = await fixture();
    try {
      const observed = await scenarios[name](f);
      assert.deepEqual(f.unexpectedDialogs, [], "Unexpected browser confirmation");
      report.cases[name] = { reproduced: true, observed };
    } catch (error) {
      report.cases[name] = { reproduced: false, error: error.message, submittedSeedGroups: f.posts, statusGets: f.polls };
      process.exitCode = 1;
    } finally { await f.context.close(); }
  }
  console.log(JSON.stringify(report, null, 2));
} finally {
  try {
    await browser?.close();
  } finally {
    try {
      await server?.close();
    } finally {
      await rm(cache, { recursive: true, force: true });
    }
  }
}
