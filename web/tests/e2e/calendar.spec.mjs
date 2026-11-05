import { expect, test } from "./fixtures.mjs";

test("the calendar previews a go-to-week move and shows the new week on Progress", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("/progress");
  await expect(page.getByRole("heading", { name: "Calendar", exact: true })).toBeVisible();
  await expect(page.getByText(/This week is week 1 of 14/)).toBeVisible();

  await page.getByLabel("Go to week").selectOption("3");
  await page.getByRole("button", { name: "Preview the move", exact: true }).click();
  await expect(page.getByText("week 1 → week 3")).toBeVisible();

  await page.getByRole("button", { name: "Move the calendar", exact: true }).click();
  await expect(page.getByText("Calendar moved. This week is week 3.")).toBeVisible();
  await expect(page.getByText(/This week is week 3 of 14/)).toBeVisible();
  await expect(page.getByText(/Current week: 3/)).toBeVisible();

  expect(errors).toEqual([]);
});

test("the Progress page, with the calendar and start-over sections, fits a 390px phone", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/progress");
  await expect(page.getByRole("heading", { name: "Calendar", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Start over", exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
