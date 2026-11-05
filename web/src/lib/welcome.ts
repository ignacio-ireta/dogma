// Remembers, per browser, that the introduction has been shown. This is only a convenience: when
// storage is unavailable the learner goes straight to Today, and Start here stays in the nav.
const SEEN_KEY = "dogma.welcome.seen";

export function hasSeenWelcome(): boolean {
  try {
    return window.localStorage.getItem(SEEN_KEY) !== null;
  } catch {
    return true;
  }
}

export function markWelcomeSeen(): void {
  try {
    window.localStorage.setItem(SEEN_KEY, "1");
  } catch {
    // Blocked storage only means the root keeps going to Today, so there is nothing to recover.
  }
}
