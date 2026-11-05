// Debounced autosave for the learner's file (ROADMAP I-15). Pure logic with injected timing, so it
// is tested with fake timers. Saves never overlap, and an edit made during a save is saved next.
import { describeError } from "../api/client";

/** The API refuses larger files (ARCHITECTURE §4.7): 256 KiB of UTF-8. */
export const WORKSPACE_LIMIT_BYTES = 256 * 1024;
export const SAVE_DELAY_MS = 800;

export type SaveStatus =
  | { readonly kind: "clean" }
  | { readonly kind: "dirty" }
  | { readonly kind: "saving" }
  | { readonly kind: "error"; readonly message: string }
  | { readonly kind: "too-large" };

export function utf8Length(text: string): number {
  return new TextEncoder().encode(text).length;
}

export interface AutosaveOptions {
  /** Write the text to the workspace file. Throws when the write fails. */
  readonly save: (text: string) => Promise<void>;
  readonly onStatus: (status: SaveStatus) => void;
  readonly delayMs?: number;
  readonly limitBytes?: number;
}

export class Autosave {
  private readonly save: (text: string) => Promise<void>;
  private readonly onStatus: (status: SaveStatus) => void;
  private readonly delayMs: number;
  private readonly limitBytes: number;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private lastSaved: string | null = "";
  private queued: string | null = null;
  private inFlight: string | null = null;
  private chain: Promise<boolean> = Promise.resolve(true);

  constructor(options: AutosaveOptions) {
    this.save = options.save;
    this.onStatus = options.onStatus;
    this.delayMs = options.delayMs ?? SAVE_DELAY_MS;
    this.limitBytes = options.limitBytes ?? WORKSPACE_LIMIT_BYTES;
  }

  /** The saved file content, or null when the file does not exist yet. */
  reset(saved: string | null): void {
    this.clearTimer();
    this.lastSaved = saved;
    this.queued = null;
    this.onStatus({ kind: "clean" });
  }

  /** The learner edited the buffer. The save waits for a pause of `delayMs`. */
  edit(text: string): void {
    this.clearTimer();
    if (utf8Length(text) > this.limitBytes) {
      this.queued = text;
      this.onStatus({ kind: "too-large" });
      return;
    }
    // Back to the saved text with nothing pending and nothing on its way: nothing to write.
    if (text === this.lastSaved && this.queued === null && this.inFlight === null) {
      this.onStatus({ kind: "clean" });
      return;
    }
    this.queued = text;
    this.onStatus({ kind: "dirty" });
    this.timer = setTimeout(() => {
      void this.flush();
    }, this.delayMs);
  }

  /** Save now (Run and Submit call this first, because the grader reads the file on disk). */
  flush(): Promise<boolean> {
    this.clearTimer();
    this.chain = this.chain.then(() => this.drain());
    return this.chain;
  }

  private async drain(): Promise<boolean> {
    while (this.queued !== null) {
      const text = this.queued;
      if (utf8Length(text) > this.limitBytes) {
        this.onStatus({ kind: "too-large" });
        return false;
      }
      this.queued = null;
      this.onStatus({ kind: "saving" });
      this.inFlight = text;
      try {
        await this.save(text);
        this.lastSaved = text;
      } catch (error) {
        // Keep the text, so the next flush tries again instead of dropping the edit.
        this.queued = this.queued ?? text;
        this.onStatus(
          utf8Length(this.queued) > this.limitBytes
            ? { kind: "too-large" }
            : { kind: "error", message: describeError(error) },
        );
        return false;
      } finally {
        this.inFlight = null;
      }
    }
    this.onStatus({ kind: "clean" });
    return true;
  }

  private clearTimer(): void {
    if (this.timer !== undefined) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }
  }
}
