// Keyboard shortcuts (ENGINEERING_STANDARDS §7). One table serves the page (window listener) and
// the editor (CodeMirror keymap), so both spell the keys the same way: "Mod-Shift-Enter".
// `Mod` is Cmd on macOS and Ctrl elsewhere; either modifier key counts as Mod.

export interface KeyEvent {
  readonly key: string;
  readonly metaKey: boolean;
  readonly ctrlKey: boolean;
  readonly shiftKey: boolean;
  readonly altKey: boolean;
}

export interface Combo {
  readonly mod: boolean;
  readonly shift: boolean;
  readonly alt: boolean;
  readonly key: string;
}

export function parseCombo(combo: string): Combo {
  const parts = combo.split("-");
  const key = parts.pop() ?? "";
  return {
    mod: parts.includes("Mod"),
    shift: parts.includes("Shift"),
    alt: parts.includes("Alt"),
    key,
  };
}

export function matchesCombo(event: KeyEvent, combo: string): boolean {
  const wanted = parseCombo(combo);
  const modDown = event.metaKey || event.ctrlKey;
  return (
    wanted.mod === modDown &&
    wanted.shift === event.shiftKey &&
    wanted.alt === event.altKey &&
    wanted.key.toLowerCase() === event.key.toLowerCase()
  );
}

export interface ShortcutBinding {
  readonly id: string;
  readonly key: string;
  readonly label: string;
  readonly run: () => void;
}

/** The binding whose combo matches the event, if any. */
export function findBinding(
  bindings: readonly ShortcutBinding[],
  event: KeyEvent,
): ShortcutBinding | undefined {
  return bindings.find((binding) => matchesCombo(event, binding.key));
}
