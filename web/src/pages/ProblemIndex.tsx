// The topic picker shown at /problem: topics grouped by track, each with its problems.
import { Link } from "react-router";
import { describeError } from "../api/client";
import type { CatalogTopicOut } from "../api/types";
import { Loading } from "../components/Loading";
import { Notice } from "../components/Notice";
import { PageHeading } from "../components/PageHeading";
import { useCatalog } from "../hooks/useCatalog";

export function ProblemIndex() {
  const catalog = useCatalog();
  if (catalog.isPending) {
    return <Loading what="the catalogue" />;
  }
  if (catalog.isError) {
    return <Notice tone="error">{describeError(catalog.error)}</Notice>;
  }
  const tracks = groupByTrack(catalog.data.topics);
  return (
    <section className="page problem-catalog">
      <PageHeading
        title="Choose a problem"
        eyebrow="Build your understanding"
        description="Small problems, lasting knowledge. Pick a topic and make it yours."
      />
      {tracks.map(([track, topics]) => (
        <section key={track} className="track" aria-labelledby={`track-${track}`}>
          <h2 id={`track-${track}`} className="track__title">
            {track.toUpperCase()}
          </h2>
          <div className="problem-catalog__topics">
            {topics.map((topic) => (
              <article key={topic.id} className="topic-card">
                <h3>{topic.title}</h3>
                <ul className="problem-list">
                  {topic.problems.map((problem) => (
                    <li key={problem.id}>
                      <Link to={`/problem/${problem.id}`}>{problem.title}</Link>{" "}
                      <span className="muted">
                        {problem.minutes} min · {problem.language}
                      </span>
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </section>
      ))}
    </section>
  );
}

function groupByTrack(topics: readonly CatalogTopicOut[]): [string, CatalogTopicOut[]][] {
  const groups = new Map<string, CatalogTopicOut[]>();
  for (const topic of topics) {
    if (topic.problems.length === 0) {
      continue;
    }
    const bucket = groups.get(topic.track) ?? [];
    bucket.push(topic);
    groups.set(topic.track, bucket);
  }
  return [...groups.entries()];
}
