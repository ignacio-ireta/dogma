import type { SaveStatus } from "../lib/autosave";

/** The autosave state in words (and a symbol, so colour is never the only signal). */
export function SaveStatusText({ status }: { readonly status: SaveStatus }) {
  const text = saveStatusText(status);
  return (
    <span className={`save-status save-status--${status.kind}`} role="status" aria-live="polite">
      <span aria-hidden="true">{saveStatusSymbol(status)}</span> {text}
    </span>
  );
}

export function saveStatusText(status: SaveStatus): string {
  switch (status.kind) {
    case "clean":
      return "Saved";
    case "dirty":
      return "Unsaved changes";
    case "saving":
      return "Saving…";
    case "error":
      return `Not saved: ${status.message}`;
    case "too-large":
      return "Too large to save (256 KiB limit). Shorten the file to run it.";
  }
}

function saveStatusSymbol(status: SaveStatus): string {
  switch (status.kind) {
    case "clean":
      return "✓";
    case "dirty":
    case "saving":
      return "•";
    case "error":
    case "too-large":
      return "✗";
  }
}
