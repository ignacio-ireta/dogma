import { fileURLToPath } from "node:url";
import { expect, problemPath, reference, test, workspacePath } from "./fixtures.mjs";

async function openProblem(page) {
  await page.goto(problemPath);
  await expect(page.getByRole("textbox", { name: "Your python solution" })).toBeVisible();
  await expect(page.locator(".save-status")).toContainText("Saved");
}

async function replaceSolution(page, text) {
  const editor = page.getByRole("textbox", { name: "Your python solution" });
  await editor.fill(text);
  await expect(editor).toContainText(text.trim().split("\n").at(-1));
}

test("a learner day survives saves, grading, closing, review and reload", async ({
  page,
  baseURL,
}) => {
  const errors = [];
  const offOriginRequests = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (new URL(request.url()).origin !== new URL(baseURL).origin) {
      offOriginRequests.push(request.url());
    }
  });
  // A returning learner: the first-visit introduction has its own spec.
  await page.addInitScript(() => window.localStorage.setItem("dogma.welcome.seen", "1"));
  await page.goto("/");
  await expect(page).toHaveURL(/\/today$/);
  await expect(page.getByRole("heading", { name: "Today", exact: true })).toBeVisible();
  await expect(page.getByRole("table", { name: "Your session blocks" })).toBeVisible();
  await page.screenshot({
    path: fileURLToPath(
      new URL(
        "../../../.superpowers/sdd/2026-10-25-learning-pages/browser-screenshot.png",
        import.meta.url,
      ),
    ),
    fullPage: true,
  });
  await page.getByRole("link", { name: "Start session", exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`${problemPath}$`));
  await expect(page.getByRole("textbox", { name: "Your python solution" })).toBeVisible();
  await replaceSolution(page, reference);
  await expect(page.locator(".save-status")).toContainText("Saved");

  await page.reload();
  await expect(page.getByRole("textbox", { name: "Your python solution" })).toContainText("return");
  const saved = await page.request.get(workspacePath);
  expect((await saved.json()).content).toBe(reference);
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByText("All tests passed", { exact: false })).toBeVisible();
  await expect(page.locator(".verdict__counts")).toContainText(/public [1-9]\d*\/[1-9]\d*/);
  await page.getByRole("button", { name: "Submit", exact: true }).click();
  await expect(page.locator(".verdict__counts")).toContainText(/hidden [1-9]\d*\/[1-9]\d*/);
  await expect(page.getByText("All tests passed", { exact: false })).toBeVisible();
  await expect(page.getByText("Recorded in your history.", { exact: true })).toBeVisible();

  await page
    .getByRole("navigation", { name: "Main" })
    .getByRole("link", { name: "Today", exact: true })
    .click();
  await page
    .getByLabel("Where I stopped", { exact: true })
    .fill("Finished the vector projection problem.");
  await page.getByLabel("Exact next step", { exact: true }).fill("Recall the projection formula.");
  await page
    .getByLabel("60-second answer (optional)", { exact: true })
    .fill("A projection keeps the component along a direction.");
  await page.getByRole("button", { name: "Save note and close", exact: true }).click();
  await expect(
    page.getByText("Session closed. Your transition note is saved.", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByText("Finished the vector projection problem.", { exact: true }),
  ).toBeVisible();

  await page
    .getByRole("navigation", { name: "Main" })
    .getByRole("link", { name: "Review", exact: true })
    .click();
  await expect(page.getByRole("button", { name: "3 Good", exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "Reveal answer", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Answer", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "3 Good", exact: true }).click();
  await expect(page.getByText(/1 reviewed/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Reveal answer", exact: true })).toBeVisible();

  await page
    .getByRole("navigation", { name: "Main" })
    .getByRole("link", { name: "Map", exact: true })
    .click();
  const vector = page.getByRole("article", { name: "Vector operations", exact: true });
  await expect(vector.getByText("Content: Reviewed", { exact: true })).toBeVisible();
  await expect(vector.getByText("Mastery: Solved", { exact: true })).toBeVisible();
  await expect(vector.getByRole("link")).toHaveAttribute("href", problemPath);
  await page
    .getByRole("navigation", { name: "Main" })
    .getByRole("link", { name: "Progress", exact: true })
    .click();
  await expect(page.getByText(/1 of \d+ core topics solved/)).toBeVisible();
  await expect(
    page
      .getByRole("article", { name: "Vector operations", exact: true })
      .getByText("Core problem solved", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByText(/1 of \d+ core topics solved/)).toBeVisible();
  expect(errors).toEqual([]);
  expect(offOriginRequests).toEqual([]);
});

test("failed departure save recovers the editor buffer and retries to durable storage", async ({
  page,
}) => {
  await openProblem(page);
  const durable = (await (await page.request.get(workspacePath)).json()).content;
  await page.route(`**${workspacePath}`, async (route) => {
    if (route.request().method() !== "PUT") return route.continue();
    await route.fulfill({
      status: 503,
      contentType: "application/problem+json",
      body: JSON.stringify({
        title: "Save unavailable",
        status: 503,
        detail: "Scratch save failure",
        code: "API_WORKSPACE_WRITE_FAILED",
      }),
    });
  });
  const recovered = `${reference}\n# Recovered after leaving the page.\n`;
  await replaceSolution(page, recovered);
  await expect(page.locator(".save-status")).toContainText("Not saved:");
  await page
    .getByRole("navigation", { name: "Main" })
    .getByRole("link", { name: "Map", exact: true })
    .click();
  await expect(page.getByRole("heading", { name: "Map", exact: true })).toBeVisible();
  await page
    .getByRole("article", { name: "Vector operations", exact: true })
    .getByRole("link")
    .click();
  await expect(page.getByRole("textbox", { name: "Your python solution" })).toContainText(
    "# Recovered after leaving the page.",
  );
  await expect(page.locator(".save-status")).toContainText("Not saved:");
  expect((await (await page.request.get(workspacePath)).json()).content).toBe(durable);
  await page.unroute(`**${workspacePath}`);
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByText("All tests passed", { exact: false })).toBeVisible();
  await expect(page.locator(".save-status")).toContainText("Saved");
  await page.reload();
  await expect(page.getByRole("textbox", { name: "Your python solution" })).toContainText(
    "# Recovered after leaving the page.",
  );
  expect((await (await page.request.get(workspacePath)).json()).content).toBe(recovered);
});
