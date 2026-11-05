import type { ReactNode } from "react";

export type NoticeTone = "info" | "success" | "warning" | "error";

const ICONS: Record<NoticeTone, string> = {
  info: "i",
  success: "✓",
  warning: "!",
  error: "✗",
};

/** A calm message. Colour is never the only signal: each tone has an icon and a text label. */
export function Notice({
  tone,
  children,
}: {
  readonly tone: NoticeTone;
  readonly children: ReactNode;
}) {
  return (
    <p className={`notice notice--${tone}`} role={tone === "error" ? "alert" : "status"}>
      <span className="notice__icon" aria-hidden="true">
        {ICONS[tone]}
      </span>
      <span>{children}</span>
    </p>
  );
}
