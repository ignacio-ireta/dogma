import { expect, test } from "./fixtures.mjs";

const exercises = [
  {
    id: "cml/vector-operations/p01-vector-projection",
    scene: "vectors",
  },
  {
    id: "sql/sql-select-filter-and-sort/p01-top-five-completed-orders-2024",
    scene: "tables",
  },
  {
    id: "llm/word-and-character-tokenizers/p01-word-level-vocabulary",
    scene: "tokens",
  },
];

test("Today follows its API topic and falls back safely for missing or unknown topics", async ({
  page,
  baseURL,
}) => {
  const response = await page.request.get(`${baseURL}/api/v1/today`);
  expect(response.ok()).toBe(true);
  const today = await response.json();
  expect(today.current).not.toBeNull();

  let topicId = "cml/vector-operations";
  await page.route("**/api/v1/today*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ...today,
        current: { ...today.current, topic_id: topicId },
      }),
    }),
  );

  for (const [id, scene] of [
    ["cml/vector-operations", "vectors"],
    [null, "study"],
    ["cml/future-topic", "study"],
  ]) {
    topicId = id;
    await page.goto(`${baseURL}/today?illustration-case=${scene}`);
    await expect(page.locator(".page-heading svg.study-art")).toHaveAttribute(
      "data-topic-scene",
      scene,
    );
  }
});

test("exercise navigation selects decorative topic scenes without changing the workspace", async ({
  page,
  baseURL,
}) => {
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 900 });
    for (const exercise of exercises) {
      await page.goto(`${baseURL}/problem`);
      const link = page.locator(`a[href="/problem/${exercise.id}"]`);
      await expect(link).toBeVisible();
      await link.click();
      await expect(page).toHaveURL(`${baseURL}/problem/${exercise.id}`);

      const illustration = page.locator("svg.study-art");
      await expect(illustration).toHaveAttribute("data-topic-scene", exercise.scene);
      await expect(illustration).toHaveAttribute("aria-hidden", "true");
      await expect(illustration.locator("image, img")).toHaveCount(0);
      await expect(page.getByRole("tablist", { name: "Problem material" })).toBeVisible();
      await expect(page.getByRole("textbox", { name: /Your .* solution/ })).toBeVisible();

      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      expect(overflow, `${width}px ${exercise.id}`).toBe(0);

      if (exercise.scene === "vectors") {
        await page.getByRole("tab", { name: "Lesson", exact: true }).click();
        const lesson = page.getByRole("region", { name: "Lesson", exact: true });
        await expect(lesson).toBeVisible();
        await expect(lesson.locator(".katex").first()).toBeVisible();
        await expect(page.getByRole("textbox", { name: /Your .* solution/ })).toBeVisible();
        const lessonOverflow = await page.evaluate(
          () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
        );
        expect(lessonOverflow, `${width}px vector lesson`).toBe(0);
      }
    }
  }
});
