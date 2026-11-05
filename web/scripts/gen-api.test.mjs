// Exercise the real generator CLI in a scratch tree. Only uv's Python export is replaced;
// openapi-typescript, comparison, file writes and exit codes are the production path.
import { spawnSync } from "node:child_process";
import {
  chmod,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, it } from "vitest";

it("writes deterministic API artifacts and check rejects either stale artifact without changing it", async () => {
  const scratch = await mkdtemp(join(tmpdir(), "dogma-web-api-"));
  try {
    const web = join(scratch, "web");
    const scripts = join(web, "scripts");
    const api = join(web, "src", "api");
    const bin = join(scratch, "bin");
    await Promise.all([
      mkdir(scripts, { recursive: true }),
      mkdir(api, { recursive: true }),
      mkdir(bin),
    ]);
    const originalWeb = resolve(dirname(fileURLToPath(import.meta.url)), "..");
    await copyFile(join(originalWeb, "scripts", "gen-api.mjs"), join(scripts, "gen-api.mjs"));
    await symlink(join(originalWeb, "node_modules"), join(web, "node_modules"), "dir");
    const spec = `${JSON.stringify({ openapi: "3.1.0", info: { title: "Scratch", version: "1" }, paths: {}, components: { schemas: { Scratch: { type: "object", properties: { count: { type: "integer" } }, required: ["count"] } } } }, null, 2)}\n`;
    await writeFile(join(scratch, "spec.json"), spec);
    const uv = join(bin, "uv");
    await writeFile(
      uv,
      '#!/usr/bin/env node\nimport { readFileSync } from "node:fs";\nif (JSON.stringify(process.argv.slice(2, 5)) !== JSON.stringify(["run", "--frozen", "python"])) process.exit(9);\nprocess.stdout.write(readFileSync(process.env.SCRATCH_SPEC, "utf8"));\n',
    );
    await chmod(uv, 0o755);
    const run = (...args) =>
      spawnSync(process.execPath, [join(scripts, "gen-api.mjs"), ...args], {
        encoding: "utf8",
        env: {
          ...process.env,
          PATH: `${bin}:${process.env.PATH}`,
          SCRATCH_SPEC: join(scratch, "spec.json"),
        },
      });
    const generated = run();
    expect(generated.status, generated.stderr).toBe(0);
    expect(await readFile(join(api, "openapi.json"), "utf8")).toBe(spec);
    const types = await readFile(join(api, "schema.d.ts"), "utf8");
    expect(types).toContain("count: number");
    expect(run("--check").status).toBe(0);
    for (const name of ["openapi.json", "schema.d.ts"]) {
      await writeFile(join(api, name), "stale");
      const checked = run("--check");
      expect(checked.status).toBe(1);
      expect(checked.stderr).toContain(name);
      expect(await readFile(join(api, name), "utf8")).toBe("stale");
      expect(run().status).toBe(0);
    }
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }
}, 30_000); // spawns the generator four times; the 5 s default is too tight under load
