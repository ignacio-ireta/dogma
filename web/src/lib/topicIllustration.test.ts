import { existsSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { topicScene } from "./topicIllustration";

const contentRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../../content");

function authoredTopicIds(): string[] {
  return readdirSync(contentRoot, { withFileTypes: true }).flatMap((track) => {
    if (!track.isDirectory()) return [];
    const trackDir = join(contentRoot, track.name);
    return readdirSync(trackDir, { withFileTypes: true })
      .filter(
        (topic) => topic.isDirectory() && existsSync(join(trackDir, topic.name, "topic.yaml")),
      )
      .map((topic) => `${track.name}/${topic.name}`);
  });
}

describe("topicScene", () => {
  it("maps every authored topic package to an explicit non-fallback scene", () => {
    const ids = authoredTopicIds();
    expect(ids.length).toBeGreaterThan(0);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ids) expect(topicScene(id), id).not.toBe("study");
  });

  it("chooses distinct scenes for representative topic families", () => {
    expect(topicScene("cml/vector-operations")).toBe("vectors");
    expect(topicScene("cml/matrix-multiplication")).toBe("matrices");
    expect(topicScene("cml/derivatives")).toBe("calculus");
    expect(topicScene("cml/gradient-descent")).toBe("optimization");
    expect(topicScene("cml/probability-basics")).toBe("probability");
    expect(topicScene("cml/logistic-regression")).toBe("classification");
    expect(topicScene("cml/confusion-matrix-and-count-based-scores")).toBe("classification");
    expect(topicScene("sql/sql-joins-and-unions")).toBe("tables");
    expect(topicScene("sql/pandas-groupby-aggregate-and-transform")).toBe("tables");
    expect(topicScene("dl/multi-head-attention")).toBe("attention");
    expect(topicScene("llm/word-and-character-tokenizers")).toBe("tokens");
    expect(topicScene("llm/retrieval-augmented-generation-pipeline")).toBe("retrieval");
    expect(topicScene("sys/churn-prediction-case")).toBe("systems");
    expect(topicScene("safe/split-conformal-prediction")).toBe("safety");
  });

  it("uses the neutral study scene for null, empty, unknown and un-authored IDs", () => {
    expect(topicScene()).toBe("study");
    expect(topicScene(null)).toBe("study");
    expect(topicScene("")).toBe("study");
    expect(topicScene("cml/future-topic")).toBe("study");
    expect(topicScene("gen/diffusion-models")).toBe("study");
    expect(topicScene("constructor")).toBe("study");
    expect(topicScene("__proto__")).toBe("study");
  });
});
