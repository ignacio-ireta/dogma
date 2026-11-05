import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";

export interface ShortcutItem {
  readonly keys: string;
  readonly label: string;
}

/** A button that opens the list of keyboard shortcuts. Escape closes it and returns focus. */
export function ShortcutHelp({ items }: { readonly items: readonly ShortcutItem[] }) {
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        button.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const onButtonKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="shortcut-help">
      <button
        ref={button}
        type="button"
        className="button button--quiet"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
        onKeyDown={onButtonKey}
      >
        Keyboard shortcuts
      </button>
      {open ? (
        <section id={panelId} className="shortcut-help__panel" aria-label="Keyboard shortcuts">
          <dl>
            {items.map((item) => (
              <div key={item.label} className="shortcut-help__row">
                <dt>
                  <kbd>{item.keys}</kbd>
                </dt>
                <dd>{item.label}</dd>
              </div>
            ))}
          </dl>
          <p className="muted">Press Escape to close.</p>
        </section>
      ) : null}
    </div>
  );
}
