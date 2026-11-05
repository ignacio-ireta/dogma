import { expect, test } from "./fixtures.mjs";

test("a first visit opens the introduction, and the next opens Today", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/welcome$/);
  await expect(page.getByRole("heading", { name: "Welcome to dogma", level: 1 })).toBeVisible();
  await expect(page.getByRole("link", { name: "Start here" })).toHaveAttribute(
    "aria-current",
    "page",
  );

  await page.getByRole("link", { name: "Try the first problem" }).click();
  await expect(page).toHaveURL(/p01-vector-projection\?tour=1$/);
  await expect(page.getByRole("heading", { name: "Vector projection" }).first()).toBeVisible();
  const guide = page.getByRole("region", { name: "Your first problem" });
  await expect(guide).toBeVisible();

  await guide.getByRole("button", { name: "Hide this guide" }).click();
  await expect(guide).toBeHidden();
  await expect(page).toHaveURL(/p01-vector-projection$/);

  await page.goto("/");
  await expect(page).toHaveURL(/\/today$/);
});

test("every card on the introduction opens a real page", async ({ page }) => {
  await page.goto("/welcome");
  const links = page.getByRole("region", { name: "Where things live" }).getByRole("link");
  const hrefs = await links.evaluateAll((items) => items.map((item) => item.getAttribute("href")));
  expect(hrefs).toHaveLength(6);
  for (const href of hrefs) {
    await page.goto(href);
    await expect(page.getByRole("heading", { name: "Not found" })).toHaveCount(0);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  }
});

test("a deep link is never redirected on a first visit", async ({ page }) => {
  await page.goto("/map");
  await expect(page).toHaveURL(/\/map$/);
});

for (const width of [320, 390]) {
  test(`the introduction fits a ${width}px phone`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    await page.goto("/welcome");
    await expect(page.getByRole("heading", { name: "Words you'll see" })).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBe(0);
  });

  test(`the guide strip fits a ${width}px phone`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    await page.goto("/problem/cml/vector-operations/p01-vector-projection?tour=1");
    await expect(page.getByRole("region", { name: "Your first problem" })).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBe(0);
  });
}

test("the introduction works from the keyboard alone", async ({ page }) => {
  await page.goto("/welcome");
  const skip = page.getByRole("link", { name: "Skip to Today" });
  await skip.focus();
  await expect(skip).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/today$/);
});
