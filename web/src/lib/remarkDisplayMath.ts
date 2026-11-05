import type { Root } from "mdast";
import { visit } from "unist-util-visit";
import type { VFile } from "vfile";

// Our lessons also use $$equation$$ on one line. remark-math treats that as inline math;
// promote it only when it occupies an entire paragraph, without touching prose or code.
export function remarkDisplayMath() {
  return (tree: Root, file: VFile) => {
    const source = String(file);
    visit(tree, "paragraph", (paragraph, index, parent) => {
      const math = paragraph.children[0];
      if (
        parent === undefined ||
        index === undefined ||
        paragraph.children.length !== 1 ||
        math?.type !== "inlineMath"
      ) {
        return;
      }

      const start = math.position?.start.offset;
      const end = math.position?.end.offset;
      if (start === undefined || end === undefined || !source.slice(start, end).startsWith("$$")) {
        return;
      }

      parent.children[index] = {
        type: "math",
        value: math.value,
        position: paragraph.position,
        data: {
          hName: "div",
          hProperties: { className: ["math-display"] },
          hChildren: [{ type: "text", value: math.value }],
        },
      };
    });
  };
}
