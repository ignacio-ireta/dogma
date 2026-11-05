import type { SaveStatus } from "../lib/autosave";
import { SaveStatusText } from "./SaveStatusText";
import type { ShortcutItem } from "./ShortcutHelp";
import { ShortcutHelp } from "./ShortcutHelp";

export interface EditorToolbarProps {
  readonly fileName: string;
  readonly status: SaveStatus;
  readonly running: "run" | "submit" | null;
  readonly disabled: boolean;
  readonly onRun: () => void;
  readonly onSubmit: () => void;
  readonly shortcuts: readonly ShortcutItem[];
}

export function EditorToolbar({
  fileName,
  status,
  running,
  disabled,
  onRun,
  onSubmit,
  shortcuts,
}: EditorToolbarProps) {
  const busy = running !== null;
  return (
    <div className="toolbar">
      <div className="toolbar__file">
        <code>{fileName}</code>
        <SaveStatusText status={status} />
      </div>
      <div className="toolbar__actions">
        <button
          type="button"
          className="button"
          onClick={onRun}
          disabled={disabled || busy}
          aria-keyshortcuts="Meta+Enter Control+Enter"
        >
          {running === "run" ? "Running…" : "Run"}
        </button>
        <button
          type="button"
          className="button button--primary"
          onClick={onSubmit}
          disabled={disabled || busy}
          aria-keyshortcuts="Meta+Shift+Enter Control+Shift+Enter"
        >
          {running === "submit" ? "Submitting…" : "Submit"}
        </button>
        <ShortcutHelp items={shortcuts} />
      </div>
    </div>
  );
}
