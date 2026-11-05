import { python } from "@codemirror/lang-python";
import { sql } from "@codemirror/lang-sql";
import type { Extension } from "@codemirror/state";
import type { ProblemOut } from "../api/types";

export type EditorLanguage = ProblemOut["language"];

export function languageExtension(language: EditorLanguage): Extension {
  return language === "sql" ? sql() : python();
}
