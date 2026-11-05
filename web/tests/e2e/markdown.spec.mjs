import { expect, problemPath, test } from "./fixtures.mjs";

test("small bundled fonts load under the site's self-only font policy", async ({ page }) => {
  const response = await page.goto(problemPath);
  expect(response.headers()["content-security-policy"]).toContain("font-src 'self';");
  const loaded = await page.evaluate(async () => {
    const loads = [
      document.fonts.load('400 16px "KaTeX_Size3"'),
      document.fonts.load('400 16px "JetBrains Mono Variable"', "Ѡ"),
    ];
    return Promise.all(
      loads.map((load) => load.then((faces) => faces.length > 0).catch(() => false)),
    );
  });
  expect(loaded).toEqual([true, true]);
});

for (const width of [1440, 390, 320]) {
  test(`the vector lesson renders readable math and a scrollable table at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto(problemPath);
    await page.getByRole("tab", { name: "Lesson", exact: true }).click();
    const lesson = page.getByRole("region", { name: "Lesson", exact: true });
    await expect(lesson.getByRole("heading", { name: "The system" })).toBeVisible();
    await page.evaluate(() => document.fonts.ready);

    const table = lesson.getByRole("table");
    await expect(table.getByRole("row")).toHaveCount(6);
    await expect(table.getByRole("columnheader")).toHaveText([
      "Operation",
      "Formula",
      "Shape in → out",
      "Meaning",
    ]);
    await expect(lesson.locator(".katex-display math")).toHaveCount(2);
    await expect(lesson.locator(".katex-error")).toHaveCount(0);
    await expect(lesson.locator(".prose")).toHaveCSS("font-family", /Inter Variable/);
    await expect(lesson.locator(".prose")).toHaveCSS("font-weight", "400");
    await expect(lesson.locator("strong").first()).toHaveCSS("font-weight", "700");
    expect(await page.evaluate(() => document.fonts.check('400 16px "Inter Variable"'))).toBe(true);

    const scrollRegion = lesson.getByRole("region", { name: "Scrollable table" });
    expect(await scrollRegion.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
    await scrollRegion.focus();
    await page.keyboard.press("ArrowRight");
    await expect.poll(() => scrollRegion.evaluate((el) => el.scrollLeft)).toBeGreaterThan(0);

    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      ),
    ).toBe(0);
    for (const equation of await lesson.locator(".katex-display").all()) {
      const bounds = await equation.boundingBox();
      expect(bounds.x).toBeGreaterThanOrEqual(0);
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(width);
    }
  });
}
