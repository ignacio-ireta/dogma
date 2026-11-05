import { expect, problemId, problemPath, reference, test } from "./fixtures.mjs";

// A timed re-solve from the window to the verdict. Today only plans a re-solve once the schedule
// is due, and the browser smoke clock is fixed on the day of the first pass, so the window is
// opened through the API here. Everything after the start happens in the browser.
test("a timed re-solve is started, submitted and shows its verdict and next due date", async ({
  page,
}) => {
  await page.goto(problemPath);
  const editor = page.getByRole("textbox", { name: "Your python solution" });
  await expect(editor).toBeVisible();
  await editor.fill(reference);
  await expect(page.locator(".save-status")).toContainText("Saved");
  await page.getByRole("button", { name: "Submit", exact: true }).click();
  await expect(page.getByText("Recorded in your history.", { exact: true })).toBeVisible();

  const started = await page.request.post("/api/v1/resolves", { data: { problem_id: problemId } });
  expect(started.status()).toBe(201);
  await page.goto(problemPath);
  await expect(page.getByRole("heading", { name: "Timed re-solve open" })).toBeVisible();
  await expect(page.getByText("Time budget 20 min")).toBeVisible();
  await expect(page.getByText(/Run is unlimited/)).toBeVisible();

  // Starting restored the starter, so the learner pastes the solution again and submits.
  await editor.fill(reference);
  await expect(page.locator(".save-status")).toContainText("Saved");
  await page.getByRole("button", { name: "Submit", exact: true }).click();
  await expect(page.getByText(/^Timed re-solve passed:/)).toBeVisible();
  await expect(page.getByText(/^Next timed re-solve: .* stage 1 of 3\./)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Timed re-solve open" })).toHaveCount(0);

  await page
    .getByRole("navigation", { name: "Main" })
    .getByRole("link", { name: "Progress", exact: true })
    .click();
  const readiness = page.getByRole("heading", { name: "Readiness", exact: true });
  await expect(readiness).toBeVisible();
  await expect(page.getByRole("list", { name: "Last timed results" })).toContainText("passed");
  await expect(page.locator(".readiness")).not.toContainText(/streak|overdue|debt/i);
});
