// The copy of the introduction. Labels here must match what the app really shows: Run, Submit,
// Hints, Low energy and the nav items.
import type { IconName } from "../components/Icon";

export const FIRST_PROBLEM_TOUR_PATH =
  "/problem/cml/vector-operations/p01-vector-projection?tour=1";

export interface Place {
  readonly to: string;
  readonly label: string;
  readonly icon: IconName;
  readonly text: string;
}

export const PLACES: readonly Place[] = [
  {
    to: "/today",
    label: "Today",
    icon: "today",
    text: "Your next step. It plans your session, names the first action and shows your last note.",
  },
  {
    to: "/problem",
    label: "Problem",
    icon: "problem",
    text: "Every exercise by track. Each has a statement, a lesson, hints and tests.",
  },
  {
    to: "/review",
    label: "Review",
    icon: "review",
    text: "Short recall cards, spaced out over time. Cards appear after your first closed Learn session.",
  },
  {
    to: "/mock",
    label: "Mock",
    icon: "mock",
    text: "A timed practice interview, with coding, theory and a system design case.",
  },
  {
    to: "/map",
    label: "Map",
    icon: "map",
    text: "Every topic in the curriculum, and how far along each one is.",
  },
  {
    to: "/progress",
    label: "Progress",
    icon: "progress",
    text: "Your path week by week: topics solved, attempts and readiness.",
  },
];

export const SESSION_BLOCKS: readonly (readonly [name: string, minutes: number, text: string])[] = [
  ["Retrieve", 5, "Recall something you already know."],
  ["Predict", 3, "Guess before you read. Being wrong is fine."],
  ["Learn", 12, "Read the lesson for one new idea."],
  ["Practice", 20, "Solve a problem on it."],
  ["Close", 5, "Write down where you stopped."],
];

export const FIRST_STEPS: readonly (readonly [title: string, text: string])[] = [
  ["Read the Statement tab", "It says exactly what to build and what done looks like."],
  ["Write your code and press Run", "Run checks your file against the public tests."],
  [
    "Stuck? Open the Hints tab",
    "Hints come one at a time: the idea, then the formula, then the shape of the code.",
  ],
  [
    "Press Submit",
    "Hidden tests check the edge cases, and a failure names the kind of mistake to look for.",
  ],
];

export const GLOSSARY: readonly (readonly [term: string, meaning: string])[] = [
  [
    "Transition note",
    "What you write when you close a session: where you stopped and your next step. It comes first next time, so you never have to work out where you were.",
  ],
  ["Must-do", "The one thing that makes today a success. Everything else is a bonus."],
  ["Learn session", "One new topic, from recall to a solved problem."],
  ["Consolidate session", "Review cards, then timed re-solves of older problems."],
  [
    "Timed re-solve",
    "Solving an earlier problem again, on a clock and without hints, to check that it stuck.",
  ],
  ["Low energy", "A shorter session with familiar practice and nothing new. It still counts."],
  [
    "Mastery",
    "How far a topic has come: introduced, solved, consolidating, retained, then interview-ready. It never resets to zero.",
  ],
];
