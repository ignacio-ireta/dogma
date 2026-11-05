import { Markdown } from "./Markdown";

export function StatementPanel({ markdown }: { readonly markdown: string | null | undefined }) {
  if (markdown === undefined) {
    return <p className="muted">Loading the statement…</p>;
  }
  if (markdown === null || markdown.trim() === "") {
    return <p className="muted">This problem has no statement yet.</p>;
  }
  return <Markdown source={markdown} />;
}
