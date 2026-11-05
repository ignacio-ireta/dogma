import { useEffect } from "react";

/** Keys 1-4 rate the revealed card. Chords, auto-repeat and typing fields are left alone. */
export function useRatingKeys(enabled: boolean, onRate: (rating: number) => void) {
  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (
        event.repeat ||
        event.ctrlKey ||
        event.altKey ||
        event.metaKey ||
        event.shiftKey ||
        event.defaultPrevented
      )
        return;
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.isContentEditable || target.closest("input, textarea, select"))
      )
        return;
      if (/^[1-4]$/.test(event.key) && enabled) {
        event.preventDefault();
        onRate(Number(event.key));
      }
    };
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  });
}
