// Markdown tables and maths (KaTeX, bundled). Raw HTML in the source is dropped, and link
// targets retain react-markdown's URL checks.
import "katex/dist/katex.min.css";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { remarkDisplayMath } from "../lib/remarkDisplayMath";

const components: Components = {
  table: ({ node: _node, ...props }) => (
    // biome-ignore lint/a11y/noNoninteractiveTabindex: Keyboard users need to scroll wide tables.
    <section className="prose-table" aria-label="Scrollable table" tabIndex={0}>
      <table {...props} />
    </section>
  ),
  th: ({ node: _node, ...props }) => <th scope="col" {...props} />,
};

export function Markdown({ source }: { readonly source: string }) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkDisplayMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
        skipHtml
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
