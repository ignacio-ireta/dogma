// Page-level shortcuts. A key that CodeMirror already handled (defaultPrevented) is not run twice.
import { useEffect, useRef } from "react";
import { findBinding, type ShortcutBinding } from "../lib/shortcuts";

export function useShortcuts(bindings: readonly ShortcutBinding[]): void {
  const latest = useRef(bindings);

  useEffect(() => {
    latest.current = bindings;
  });

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) {
        return;
      }
      const binding = findBinding(latest.current, event);
      if (binding === undefined) {
        return;
      }
      event.preventDefault();
      binding.run();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
