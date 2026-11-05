import { encodeId } from "../../api/client";
import type { SessionOut } from "../../api/learning";

export const kindLabel = {
  learn: "Learn",
  consolidate: "Consolidate",
  minimum: "Minimum viable",
  interview: "Interview",
};

/** The planner supplies CLI prose; only its explicit problem ids become browser targets. */
export function actionProblem(action: string): string | undefined {
  return action.match(/(?:dogma open|Open)\s+([\w-]+\/[\w-]+\/[\w-]+)(?![\w/.-])/i)?.[1];
}

export function firstActionTarget(session: SessionOut): string {
  if (session.kind === "interview") return "/mock";
  if (/dogma review\b/i.test(session.first_action)) return "/review";
  if (/dogma close\b/i.test(session.first_action)) return "#session-close";
  const problem = actionProblem(session.first_action);
  if (problem || session.must_do) return `/problem/${encodeId(problem ?? session.must_do ?? "")}`;
  return "#session-close";
}

/** Adapt the planner's known command templates without rewriting learner goals or unknown text. */
export function browserInstruction(text: string, session: SessionOut): string {
  const escapePattern = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // Timed re-solves are started and submitted from the Problem page, so the CLI steps read as that.
  const id = "[\\w-]+/[\\w-]+/[\\w-]+";
  let instruction = text
    .replace(new RegExp(`dogma resolve ${id} --abandon`, "g"), "abandon it on the Problem page")
    .replace(new RegExp(`Start with dogma resolve ${id}`, "g"), "Start on the Problem page")
    .replace(
      new RegExp(`Start the timed re-solve: dogma resolve ${id}`, "g"),
      "Start the timed re-solve on the Problem page",
    )
    .replace(new RegExp(`dogma submit ${id}`, "g"), "Submit")
    // Mocks run on the Mock page (the planner's CLI wording is kept for the terminal).
    .replace(/Start the mock: dogma mock start/g, "Start the mock on the Mock page")
    .replace(
      /Start with dogma mock start, then score with dogma mock score\./g,
      "Start on the Mock page, then score there.",
    )
    .replace(/dogma mock debrief shows/g, "The Mock page's debrief shows");
  if (session.topic_id) {
    const lesson = escapePattern(`content/${session.topic_id}/lesson.md`);
    instruction = instruction.replace(
      new RegExp(`\\b([Rr]ead|open) ${lesson}(?![\\w/-]|\\.[\\w])`, "g"),
      "$1 the Lesson tab",
    );
  }
  const problems = new Set([
    session.must_do,
    actionProblem(session.first_action),
    ...session.blocks.map((block) => actionProblem(block.detail)),
  ]);
  for (const problem of problems) {
    if (!problem) continue;
    instruction = instruction.replace(
      new RegExp(`\\bdogma (open|run|submit) ${escapePattern(problem)}(?![\\w/-]|\\.[\\w])`, "g"),
      (_command, action: string) => {
        if (action === "open") return `open ${problem} in the Problem page`;
        return action === "run" ? "Run" : "Submit";
      },
    );
  }
  return instruction
    .replace(
      /(?:Run )?dogma review --limit ([1-9]\d*)(?=\.?$)/g,
      "Review up to $1 due cards on the Review page",
    )
    .replace(/^dogma close:/, "Use the close form below:")
    .replace(/\bdogma close\b/g, "the close form below");
}
