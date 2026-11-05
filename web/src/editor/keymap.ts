import { type Extension, Prec } from "@codemirror/state";
import { keymap } from "@codemirror/view";
import type { ShortcutBinding } from "../lib/shortcuts";

/**
 * The page's shortcuts, also active inside the editor. Mod-Enter would otherwise insert a blank
 * line there, so these bindings run first, and a handled key is marked as handled so the page's
 * window listener does not run it a second time.
 */
export function shortcutKeymap(bindings: () => readonly ShortcutBinding[]): Extension {
  return Prec.highest(
    keymap.of(
      ["Mod-Enter", "Mod-Shift-Enter", "Mod-h"].map((key) => ({
        key,
        preventDefault: true,
        run: () => {
          const binding = bindings().find((item) => item.key === key);
          if (binding === undefined) {
            return false;
          }
          binding.run();
          return true;
        },
      })),
    ),
  );
}
