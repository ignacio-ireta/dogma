// Named aliases over the generated contract. Never hand-write API shapes here: add them to the
// API and run `pnpm gen:api`.
import type { components } from "./schema";

type Schemas = components["schemas"];

export type CatalogOut = Schemas["CatalogOut"];
export type CatalogTopicOut = Schemas["CatalogTopicOut"];
export type CatalogProblemOut = Schemas["CatalogProblemOut"];
export type TopicOut = Schemas["TopicOut"];
export type ProblemOut = Schemas["ProblemOut"];
export type WorkspaceFile = Schemas["WorkspaceFile"];
export type AttemptCreated = Schemas["AttemptCreated"];
export type LearnerReport = Schemas["LearnerReport"];
export type LearnerTest = Schemas["LearnerTest"];
export type ProgressOut = Schemas["ProgressOut"];
export type HintOut = Schemas["HintOut"];
export type ReviewedHintOut = Schemas["ReviewedHintOut"];
export type ReviewedEventOut = Schemas["ReviewedEventOut"];
export type TextDeltaOut = Schemas["TextDeltaOut"];
export type UnavailableOut = Schemas["UnavailableOut"];
export type ResolveWindowOut = Schemas["ResolveWindowOut"];
export type ResolveStartedOut = Schemas["ResolveStartedOut"];
export type ReSolveDueOut = Schemas["ReSolveDueOut"];
export type TimedVerdictOut = Schemas["TimedVerdictOut"];
export type TimedResultOut = Schemas["TimedResultOut"];
export type ReadinessOut = Schemas["ReadinessOut"];
export type TimedResolveOut = Schemas["TimedResolveOut"];
export type InterviewOut = Schemas["InterviewOut"];
export type InterviewQuestionOut = Schemas["InterviewQuestionOut"];
export type AnswerRequest = Schemas["AnswerRequest"];
export type AnswerOut = Schemas["AnswerOut"];
export type RubricOut = Schemas["RubricOut"];
export type InterviewReadyOut = Schemas["InterviewReadyOut"];
export type MockStatusOut = Schemas["MockStatusOut"];
export type MockPlanOut = Schemas["MockPlanOut"];
export type MockReportOut = Schemas["MockReportOut"];
export type MockDebriefOut = Schemas["MockDebriefOut"];
export type PartScoreOut = Schemas["PartScoreOut"];
export type RevealOut = Schemas["RevealOut"];
export type AbandonedOut = Schemas["AbandonedOut"];

export type RunMode = "run" | "submit";
