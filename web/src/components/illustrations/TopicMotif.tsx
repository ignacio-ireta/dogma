import type { ReactNode } from "react";
import type { TopicScene } from "../../lib/topicIllustration";
import { dataMotifs } from "./dataMotifs";
import { mathMotifs } from "./mathMotifs";
import { modelMotifs } from "./modelMotifs";
import { systemMotifs } from "./systemMotifs";

const motifs = {
  ...mathMotifs,
  ...dataMotifs,
  ...modelMotifs,
  ...systemMotifs,
} satisfies Record<TopicScene, ReactNode>;

export function TopicMotif({ scene }: { readonly scene: TopicScene }) {
  return motifs[scene];
}
