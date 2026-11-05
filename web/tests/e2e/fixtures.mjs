import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test as base, expect } from "@playwright/test";

const repoRoot = fileURLToPath(new URL("../../../", import.meta.url));
export const problemId = "cml/vector-operations/p01-vector-projection";
export const problemPath = `/problem/${problemId}`;
export const workspacePath = `/api/v1/workspace/${problemId}`;
export const reference = await readFile(
  new URL(
    `../../../content/cml/vector-operations/problems/p01-vector-projection/solution.py`,
    import.meta.url,
  ),
  "utf8",
);

async function startServer(child, diagnostics) {
  const address = new Promise((resolve, reject) => {
    let output = "";
    const timer = setTimeout(
      () => reject(new Error(`Server address timed out: ${diagnostics()}`)),
      20_000,
    );
    child.stdout.on("data", (chunk) => {
      output += chunk;
      const line = output
        .split("\n")
        .slice(0, -1)
        .find((entry) => entry.startsWith('{"baseURL":'));
      if (line) {
        clearTimeout(timer);
        resolve(JSON.parse(line).baseURL);
      }
    });
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`Scratch API exited (${code}): ${diagnostics()}`));
    });
  });
  const baseURL = await address;
  await expect
    .poll(
      async () => {
        if (child.exitCode !== null) throw new Error(diagnostics());
        try {
          return (await fetch(`${baseURL}/api/v1/catalog`, { signal: AbortSignal.timeout(1000) }))
            .status;
        } catch {
          return 0;
        }
      },
      { timeout: 20_000, message: "Fresh scratch API becomes ready" },
    )
    .toBe(200);
  return baseURL;
}

export const test = base.extend({
  // biome-ignore lint/correctness/noEmptyPattern: Playwright requires a destructured fixture argument.
  baseURL: async ({}, use) => {
    const scratch = await mkdtemp(join(tmpdir(), "dogma-browser-"));
    const child = spawn("uv", ["run", "--frozen", "python", "web/tests/e2e/serve.py", scratch], {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let diagnostics = "";
    child.stderr.on("data", (chunk) => {
      diagnostics = `${diagnostics}${chunk}`.slice(-12_000);
    });
    try {
      await use(await startServer(child, () => diagnostics));
    } finally {
      if (child.pid !== undefined && child.exitCode === null && child.signalCode === null) {
        const exited = once(child, "exit");
        child.kill("SIGTERM");
        const force = setTimeout(() => child.kill("SIGKILL"), 10_000);
        try {
          await exited;
        } finally {
          clearTimeout(force);
        }
      }
      await rm(scratch, { recursive: true, force: true });
    }
  },
});

export { expect };
