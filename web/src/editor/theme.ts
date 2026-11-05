// Editor colours follow the light page palette in styles/tokens.css.
// No colour literals here (ENGINEERING_STANDARDS §7).
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import type { Extension } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import { tags } from "@lezer/highlight";

export const editorTheme: Extension = EditorView.theme({
  "&": {
    color: "var(--code-text)",
    backgroundColor: "var(--code-bg)",
    fontSize: "var(--code-size)",
    height: "100%",
  },
  ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "1.85" },
  ".cm-content": { caretColor: "var(--code-caret)", padding: "var(--space-3) 0" },
  ".cm-gutters": {
    backgroundColor: "var(--code-gutter-bg)",
    color: "var(--code-gutter-text)",
    border: "none",
  },
  ".cm-activeLine": { backgroundColor: "var(--code-active-line)" },
  ".cm-activeLineGutter": { backgroundColor: "var(--code-active-line)" },
  "&.cm-focused": { outline: "var(--focus-ring)", outlineOffset: "-2px" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": {
    backgroundColor: "var(--code-selection)",
  },
});

const highlight = HighlightStyle.define([
  { tag: [tags.keyword, tags.modifier, tags.operatorKeyword], color: "var(--syntax-keyword)" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--syntax-string)" },
  { tag: [tags.number, tags.bool, tags.null, tags.atom], color: "var(--syntax-number)" },
  { tag: tags.comment, color: "var(--syntax-comment)", fontStyle: "italic" },
  {
    tag: [tags.function(tags.variableName), tags.function(tags.propertyName)],
    color: "var(--syntax-function)",
  },
  { tag: [tags.typeName, tags.className], color: "var(--syntax-type)" },
  { tag: [tags.operator, tags.punctuation], color: "var(--syntax-operator)" },
]);

export const editorHighlight: Extension = syntaxHighlighting(highlight);
