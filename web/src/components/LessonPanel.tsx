import type { TopicOut } from "../api/types";
import { Markdown } from "./Markdown";

export function LessonPanel({ topic }: { readonly topic: TopicOut | undefined }) {
  if (topic === undefined) {
    return <p className="muted">Loading the lesson…</p>;
  }
  if (topic.lesson === null) {
    return <p className="muted">The lesson for this topic is not written yet.</p>;
  }
  return (
    <section aria-label="Lesson">
      <p className="muted">
        {topic.lesson.minutes} min · minimum version: {topic.lesson.min_viable}
      </p>
      <Markdown source={topic.lesson.markdown} />
    </section>
  );
}
