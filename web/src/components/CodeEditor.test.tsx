import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { useShortcuts } from "../hooks/useShortcuts";
import type { ShortcutBinding } from "../lib/shortcuts";
import { CodeEditor } from "./CodeEditor";

function Editor({
  bindings,
  value = "def solve():\n    pass",
  onChange = () => {},
}: {
  bindings: readonly ShortcutBinding[];
  value?: string;
  onChange?: (text: string) => void;
}) {
  useShortcuts(bindings);
  return (
    <CodeEditor
      value={value}
      label="Your solution"
      language="python"
      bindings={bindings}
      onChange={onChange}
    />
  );
}

it.each([
  { shift: false, expected: "run" },
  { shift: true, expected: "submit" },
])(
  "uses the editor shortcut once without inserting a newline ($expected)",
  ({ shift, expected }) => {
    const calls: string[] = [];
    const bindings = [
      { id: "run", key: "Mod-Enter", label: "Run", run: () => calls.push("run") },
      { id: "submit", key: "Mod-Shift-Enter", label: "Submit", run: () => calls.push("submit") },
    ];
    const onChange = vi.fn();
    render(<Editor bindings={bindings} onChange={onChange} />);
    const editor = screen.getByRole("textbox", { name: "Your solution" });
    editor.focus();
    fireEvent.keyDown(editor, { key: "Enter", code: "Enter", ctrlKey: true, shiftKey: shift });
    expect(calls).toEqual([expected]);
    expect(onChange).not.toHaveBeenCalled();
  },
);

it("uses the latest callbacks inside the editor and handles page-level Cmd shortcuts", () => {
  const old = vi.fn();
  const latest = vi.fn();
  const { rerender } = render(
    <Editor bindings={[{ id: "hint", key: "Mod-h", label: "Hint", run: old }]} />,
  );
  rerender(<Editor bindings={[{ id: "hint", key: "Mod-h", label: "Hint", run: latest }]} />);
  fireEvent.keyDown(screen.getByRole("textbox"), { key: "h", code: "KeyH", ctrlKey: true });
  expect(latest).toHaveBeenCalledTimes(1);
  fireEvent.keyDown(window, { key: "h", metaKey: true });
  expect(latest).toHaveBeenCalledTimes(2);
  expect(old).not.toHaveBeenCalled();
});
