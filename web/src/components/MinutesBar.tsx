// A session drawn to scale: one segment per block, as wide as its share of the minutes.
import type { CSSProperties } from "react";

export interface MinutesBlock {
  readonly name: string;
  readonly minutes: number;
}

/**
 * The picture of a session's blocks. It repeats what the list or table beside it says in words
 * (segment n is row n), so it is hidden from assistive technology rather than described twice.
 * Nothing is drawn when the blocks carry no time.
 */
export function MinutesBar({ blocks }: { readonly blocks: readonly MinutesBlock[] }) {
  const total = blocks.reduce((sum, block) => sum + Math.max(0, block.minutes), 0);
  if (total <= 0) {
    return null;
  }
  return (
    <figure className="minutes-bar" aria-hidden="true">
      <div className="minutes-bar__track">
        {blocks.map((block, index) => (
          <span
            className="minutes-bar__segment"
            key={block.name}
            style={{ "--minutes": Math.max(0, block.minutes) } as CSSProperties}
          >
            {index + 1}
          </span>
        ))}
      </div>
      <figcaption>
        <span>0</span>
        <span>{total} min, drawn to scale</span>
      </figcaption>
    </figure>
  );
}
