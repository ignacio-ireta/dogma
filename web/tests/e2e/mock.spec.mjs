import { expect, test } from "./fixtures.mjs";

for (const width of [320, 390]) {
  test(`all 30 mock items can be scored at ${width}px phone width`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 844 });
    await page.goto("/mock");
    await page.getByRole("button", { name: "Start a mock", exact: true }).click();
    const radios = page.locator('.score-item input[type="radio"]');
    await expect(radios).toHaveCount(120);
    const overflow = await page.evaluate(() =>
      Array.from(document.querySelectorAll("main *"))
        .filter((el) => el.getBoundingClientRect().right > window.innerWidth)
        .slice(-20)
        .map((el) => `${el.tagName}.${el.className}: ${el.getBoundingClientRect().width}`),
    );
    expect(overflow).toEqual([]);
    const choices = page.locator('.score-item input[value="3"]');
    for (let index = 0; index < 30; index += 1) {
      const choice = choices.nth(index);
      const label = choice.locator("..");
      await label.scrollIntoViewIfNeeded();
      const box = await label.boundingBox();
      expect(box.width).toBeGreaterThanOrEqual(44);
      expect(box.height).toBeGreaterThanOrEqual(44);
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(width);
      await label.click();
      await expect(choice).toBeChecked();
    }
    await page.screenshot({ path: testInfo.outputPath("phone-scoring.png"), fullPage: true });
    await page.getByRole("button", { name: "Score the mock", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Debrief", exact: true })).toBeVisible();
    await expect(page.getByText(/^90 of \d+$/)).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    );
  });
}

test("a mock runs start, reveal, score and debrief on a scratch repo", async ({
  page,
  baseURL,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("/mock");
  await expect(page.getByRole("heading", { name: "Mock interview", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Start a mock", exact: true }).click();

  await expect(page.getByRole("heading", { name: "The case", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Planned timings", exact: true })).toBeVisible();

  await page
    .getByRole("textbox", { name: "Your question", exact: true })
    .fill("What does the current retention process do?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(
    page.getByText("Subscribers who call to cancel get a 20 percent discount"),
  ).toBeVisible();

  const save = page.getByRole("button", { name: "Score the mock", exact: true });
  await expect(save).toBeDisabled();
  const threes = page.locator('input[type="radio"][value="3"]');
  const count = await threes.count();
  expect(count).toBe(30);
  for (let index = 0; index < count; index += 1) {
    await threes.nth(index).check();
  }
  await expect(page.getByText("All 30 items scored.", { exact: true })).toBeVisible();
  await expect(save).toBeEnabled();
  await save.click();

  await expect(page.getByRole("heading", { name: "Debrief", exact: true })).toBeVisible();
  await expect(page.getByText(/^90 of \d+$/)).toBeVisible();
  await expect(page.getByText("Strongest part", { exact: true })).toBeVisible();
  await expect(page.getByText("Area to improve", { exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "Case debrief" })).toBeVisible();

  expect(errors).toEqual([]);
  expect(baseURL).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
});
