// A calm question that waits for an answer before anything happens. The cancel button has focus,
// so a stray Enter keeps the current state. It is not a modal: the page behind stays readable.
import { useEffect, useId, useRef } from "react";
import "../styles/resolves.css";

export interface ConfirmDialogProps {
  readonly title: string;
  readonly body: string;
  readonly confirmLabel: string;
  readonly cancelLabel: string;
  readonly onConfirm: () => void;
  readonly onCancel: () => void;
}

export function ConfirmDialog({
  title,
  body,
  confirmLabel,
  cancelLabel,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const id = useId();
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    cancel.current?.focus();
  }, []);
  return (
    <div
      className="confirm"
      role="alertdialog"
      aria-labelledby={`${id}-title`}
      aria-describedby={`${id}-body`}
    >
      <h3 className="confirm__title" id={`${id}-title`}>
        {title}
      </h3>
      <p className="confirm__body" id={`${id}-body`}>
        {body}
      </p>
      <div className="confirm__actions">
        <button type="button" className="button" ref={cancel} onClick={onCancel}>
          {cancelLabel}
        </button>
        <button type="button" className="button button--primary" onClick={onConfirm}>
          {confirmLabel}
        </button>
      </div>
    </div>
  );
}
