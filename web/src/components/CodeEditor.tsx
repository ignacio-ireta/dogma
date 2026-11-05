// The CodeMirror 6 editor for the learner's file. The parent owns the text; this component mirrors
// it and reports every edit. Created once per mount: the page is keyed by problem id.
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { bracketMatching } from "@codemirror/language";
import { Compartment, EditorState } from "@codemirror/state";
import {
  drawSelection,
  EditorView,
  highlightActiveLine,
  keymap,
  lineNumbers,
} from "@codemirror/view";
import { useEffect, useRef, useState } from "react";
import { shortcutKeymap } from "../editor/keymap";
import { type EditorLanguage, languageExtension } from "../editor/language";
import { editorHighlight, editorTheme } from "../editor/theme";
import type { ShortcutBinding } from "../lib/shortcuts";

export interface CodeEditorProps {
  readonly value: string;
  readonly language: EditorLanguage;
  readonly label: string;
  readonly bindings: readonly ShortcutBinding[];
  readonly onChange: (next: string) => void;
}

export function CodeEditor({ value, language, label, bindings, onChange }: CodeEditorProps) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const [languageSlot] = useState(() => new Compartment());
  const [initial] = useState(() => ({ value, label, language }));
  const latest = useRef({ bindings, onChange });

  useEffect(() => {
    latest.current = { bindings, onChange };
  });

  useEffect(() => {
    if (host.current === null) {
      return undefined;
    }
    const instance = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: initial.value,
        extensions: [
          shortcutKeymap(() => latest.current.bindings),
          lineNumbers(),
          history(),
          drawSelection(),
          highlightActiveLine(),
          bracketMatching(),
          editorTheme,
          editorHighlight,
          languageSlot.of(languageExtension(initial.language)),
          keymap.of([...defaultKeymap, ...historyKeymap]),
          EditorView.contentAttributes.of({ "aria-label": initial.label }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) {
              latest.current.onChange(update.state.doc.toString());
            }
          }),
        ],
      }),
    });
    view.current = instance;
    return () => {
      instance.destroy();
      view.current = null;
    };
  }, [initial, languageSlot]);

  useEffect(() => {
    view.current?.dispatch({ effects: languageSlot.reconfigure(languageExtension(language)) });
  }, [language, languageSlot]);

  useEffect(() => {
    const instance = view.current;
    if (instance !== null && instance.state.doc.toString() !== value) {
      instance.dispatch({
        changes: { from: 0, to: instance.state.doc.length, insert: value },
      });
    }
  }, [value]);

  return <div ref={host} className="code-editor" />;
}
