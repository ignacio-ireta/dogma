// biome-ignore-all lint/a11y/noNoninteractiveTabindex: WAI-ARIA tab panels are keyboard focus destinations.
import { type KeyboardEvent, type ReactNode, useRef } from "react";

export interface TabDef<T extends string> {
  readonly id: T;
  readonly label: string;
}

interface TabsProps<T extends string> {
  readonly label: string;
  readonly tabs: readonly TabDef<T>[];
  readonly active: T;
  readonly onSelect: (id: T) => void;
  readonly idPrefix: string;
  /** Every panel stays mounted; inactive ones are hidden, so their state survives a tab switch. */
  readonly panels: Readonly<Record<T, ReactNode>>;
}

/**
 * A tab list with the WAI-ARIA keyboard pattern: arrows move between tabs, Home and End jump to
 * the ends, and the active tab's panel is the only one shown.
 */
export function Tabs<T extends string>({
  label,
  tabs,
  active,
  onSelect,
  idPrefix,
  panels,
}: TabsProps<T>) {
  const buttons = useRef<Map<T, HTMLButtonElement>>(new Map());

  const focusAt = (index: number) => {
    const next = tabs[(index + tabs.length) % tabs.length];
    if (next === undefined) {
      return;
    }
    onSelect(next.id);
    buttons.current.get(next.id)?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const moves: Record<string, () => void> = {
      ArrowRight: () => focusAt(index + 1),
      ArrowLeft: () => focusAt(index - 1),
      Home: () => focusAt(0),
      End: () => focusAt(tabs.length - 1),
    };
    const move = moves[event.key];
    if (move !== undefined) {
      event.preventDefault();
      move();
    }
  };

  return (
    <div className="tabs">
      <div role="tablist" aria-label={label} className="tabs__list">
        {tabs.map((tab, index) => {
          const selected = tab.id === active;
          return (
            <button
              key={tab.id}
              ref={(node) => {
                if (node) {
                  buttons.current.set(tab.id, node);
                } else {
                  buttons.current.delete(tab.id);
                }
              }}
              type="button"
              role="tab"
              id={`${idPrefix}-tab-${tab.id}`}
              aria-selected={selected}
              aria-controls={`${idPrefix}-panel-${tab.id}`}
              tabIndex={selected ? 0 : -1}
              className="tabs__tab"
              onClick={() => onSelect(tab.id)}
              onKeyDown={(event) => onKeyDown(event, index)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          role="tabpanel"
          id={`${idPrefix}-panel-${tab.id}`}
          aria-labelledby={`${idPrefix}-tab-${tab.id}`}
          className="tabs__panel"
          tabIndex={0}
          hidden={tab.id !== active}
        >
          {panels[tab.id]}
        </div>
      ))}
    </div>
  );
}
