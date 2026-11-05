import { useState } from "react";
import { Link } from "react-router";
import { describeError, encodeId } from "../api/client";
import type { CoverageRowOut, Tier } from "../api/overview";
import { Loading } from "../components/Loading";
import { Notice } from "../components/Notice";
import { PageHeading } from "../components/PageHeading";
import { useCatalog } from "../hooks/useCatalog";
import { recordedLabel, useCoverage, useOverviewProgress } from "../hooks/useOverview";
import "../styles/overview.css";

/** Say what is known: a failed catalog is not the same as a verified core problem being absent. */
function catalogMessage(
  catalog: { isPending: boolean; isError: boolean },
  hasProblems: boolean,
): string {
  if (catalog.isError) return "Core problem link unavailable.";
  if (catalog.isPending) return hasProblems ? "Loading core problem…" : "Checking core problems…";
  return "No verified core problem available.";
}

export function MapPage() {
  const [tier, setTier] = useState<Tier>("A");
  const [track, setTrack] = useState("");
  const coverage = useCoverage(tier, track || undefined);
  const tracks = useCoverage();
  const progress = useOverviewProgress();
  const catalog = useCatalog();
  const mastery = new Map(
    progress.data?.weeks.flatMap((week) =>
      week.topics.map((topic) => [topic.topic_id, topic.mastery]),
    ),
  );
  const trackOptions = tracks.data?.by_track ?? coverage.data?.by_track ?? [];
  const rows = coverage.data?.rows ?? [];
  const groups = new Map<string, CoverageRowOut[]>();
  for (const row of rows) {
    const topics = groups.get(row.track) ?? [];
    topics.push(row);
    groups.set(row.track, topics);
  }

  return (
    <section className="page overview">
      <PageHeading
        title="Map"
        eyebrow="See the connections"
        description="A landscape of what you can learn. Start with the core, then follow your curiosity."
      />
      <div className="overview__filters">
        <label>
          Tier
          <select value={tier} onChange={(event) => setTier(event.target.value as Tier)}>
            <option value="A">A · Core</option>
            <option value="B">B</option>
            <option value="C">C</option>
          </select>
        </label>
        <label>
          Track
          <select value={track} onChange={(event) => setTrack(event.target.value)}>
            <option value="">All tracks</option>
            {trackOptions.map((item) => (
              <option key={item.key} value={item.key}>
                {item.title ?? item.key.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
      </div>
      {coverage.isPending && <Loading what="the topic map" />}
      {coverage.isError && (
        <>
          <Notice tone="error">{describeError(coverage.error)}</Notice>
          <button className="button" type="button" onClick={() => void coverage.refetch()}>
            Retry topic map
          </button>
        </>
      )}
      {progress.isError && <Notice tone="error">{describeError(progress.error)}</Notice>}
      {catalog.isError && <Notice tone="error">{describeError(catalog.error)}</Notice>}
      {coverage.data && !coverage.isError && (
        <>
          {coverage.data.unreadable_topics.length > 0 && (
            <Notice tone="warning">
              Some topic packages could not be read: {coverage.data.unreadable_topics.join(", ")}.
              They are not counted as authored.
            </Notice>
          )}
          {rows.length === 0 && <p>No topics match these filters.</p>}
          {[...groups].map(([code, topics]) => (
            <section className="track" key={code} aria-labelledby={`map-track-${code}`}>
              <h2 id={`map-track-${code}`}>
                {trackOptions.find((item) => item.key === code)?.title ?? code.toUpperCase()}
              </h2>
              <div className="overview__topics">
                {topics.map((row) => {
                  const state = progress.isError
                    ? "Unavailable"
                    : progress.isPending
                      ? "Loading…"
                      : mastery.has(row.id)
                        ? recordedLabel(mastery.get(row.id) ?? "")
                        : "Not recorded in core path";
                  const authored = row.has_package;
                  const contentStatus = authored ? row.status : "missing";
                  const topic = catalog.data?.topics.find((item) => item.id === row.id);
                  // CONTENT_SPEC §1/6.2: the validated p01 problem is the sole core problem.
                  const problem = topic?.problems.find(
                    (item) =>
                      item.id.split("/").at(-1)?.startsWith("p01-") &&
                      (item.status === "verified" || item.status === "reviewed"),
                  );
                  const linkable =
                    authored && (contentStatus === "verified" || contentStatus === "reviewed");
                  return (
                    <article className="topic-card" key={row.id} aria-labelledby={`map-${row.id}`}>
                      <h3 id={`map-${row.id}`}>{row.title}</h3>
                      <p className="eyebrow">
                        {row.id} · Tier {row.tier}
                      </p>
                      <p>Content: {recordedLabel(contentStatus)}</p>
                      <p
                        className="overview__mastery"
                        data-mastery={progress.isError ? undefined : mastery.get(row.id)}
                      >
                        Mastery: {state}
                      </p>
                      <ul className="overview__content">
                        <li>Lesson: {authored && row.has_lesson ? "Available" : "Missing"}</li>
                        <li>Review cards: {authored && row.has_cards ? "Available" : "Missing"}</li>
                        <li>
                          Interview material:{" "}
                          {authored && row.has_interview ? "Available" : "Missing"}
                        </li>
                        <li>{authored ? row.n_problems : 0} authored problems</li>
                      </ul>
                      {linkable && problem ? (
                        <Link to={`/problem/${encodeId(problem.id)}`}>{problem.title}</Link>
                      ) : (
                        <p className="muted">
                          {catalogMessage(catalog, authored && row.n_problems > 0)}
                        </p>
                      )}
                    </article>
                  );
                })}
              </div>
            </section>
          ))}
        </>
      )}
    </section>
  );
}
