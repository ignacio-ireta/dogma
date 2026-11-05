import { useState } from "react";
import { describeError } from "../api/client";
import { Loading } from "../components/Loading";
import { Notice } from "../components/Notice";
import { PageHeading } from "../components/PageHeading";
import { recordedLabel, useOverviewProgress } from "../hooks/useOverview";
import "../styles/overview.css";
import { CalendarPanel } from "./progress/CalendarPanel";
import { InterviewReadyPanel } from "./progress/InterviewReadyPanel";
import { ReadinessPanel } from "./progress/ReadinessPanel";
import { StartOverPanel } from "./progress/StartOverPanel";

export function ProgressPage() {
  const [week, setWeek] = useState<number | undefined>();
  const all = useOverviewProgress();
  const selected = useOverviewProgress(week);

  return (
    <section className="page overview">
      <PageHeading
        title="Progress"
        eyebrow="Look how far you’ve come"
        description="Every attempt adds understanding. Here’s the picture of your practice so far."
      />
      {all.isPending && <Loading what="your progress" />}
      {all.isError && (
        <>
          <Notice tone="error">{describeError(all.error)}</Notice>
          <button type="button" className="button" onClick={() => void all.refetch()}>
            Retry progress
          </button>
        </>
      )}
      {all.data && !all.isError && (
        <>
          <h2>{all.data.course.title}</h2>
          <p className="muted">
            {all.data.course.start} – {all.data.course.end}
          </p>
          <div className="overview__summary">
            <p>
              {all.data.summary.solved_topics} of {all.data.summary.total_topics} core topics solved
            </p>
            <p>{all.data.summary.attempts_this_week} attempts this week</p>
            <p>
              Current week: {all.data.summary.current_week} · starts {all.data.summary.week_start}
            </p>
          </div>
          <CalendarPanel />
          <ReadinessPanel readiness={all.data.readiness} />
          <InterviewReadyPanel ready={all.data.readiness.interview_ready} />
          <div className="overview__filters">
            <label>
              Study week
              <select
                value={week ?? ""}
                onChange={(event) =>
                  setWeek(event.target.value ? Number(event.target.value) : undefined)
                }
              >
                <option value="">All weeks</option>
                {all.data.weeks.map((item) => (
                  <option key={item.week} value={item.week}>
                    Week {item.week} · {item.theme}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {selected.isPending && week !== undefined && <Loading what="the selected week" />}
          {selected.isError && week !== undefined && (
            <>
              <Notice tone="error">{describeError(selected.error)}</Notice>
              <button type="button" className="button" onClick={() => void selected.refetch()}>
                Retry selected week
              </button>
            </>
          )}
          {selected.data && !selected.isError && (
            <>
              {selected.data.weeks.length === 0 && <p>No study weeks are available.</p>}
              {selected.data.weeks.map((item) => (
                <section className="track" key={item.week} aria-labelledby={`week-${item.week}`}>
                  <h2 id={`week-${item.week}`}>
                    Week {item.week} · {item.theme}
                  </h2>
                  <p>
                    {item.solved} of {item.total} topics solved · starts {item.starts}
                  </p>
                  {item.topics.length === 0 && <p>No topics are scheduled for this week.</p>}
                  {item.topics.map((topic) => (
                    <article className="topic-card" key={topic.topic_id} aria-label={topic.title}>
                      <h3>{topic.title}</h3>
                      <p className="eyebrow">{topic.topic_id}</p>
                      <p>Content: {recordedLabel(topic.status)}</p>
                      <p className="overview__mastery" data-mastery={topic.mastery}>
                        Mastery: {recordedLabel(topic.mastery)}
                      </p>
                      <p>{topic.solved ? "Core problem solved" : "Core problem not solved"}</p>
                      <p>{topic.attempts} attempts</p>
                      {topic.best ? (
                        <p>
                          Best result: {topic.best.passed}/{topic.best.total} tests ·{" "}
                          {topic.best.mode}
                        </p>
                      ) : (
                        <p className="muted">
                          {topic.attempts === 0
                            ? "No attempts recorded."
                            : "No test result recorded."}
                        </p>
                      )}
                      {topic.last_attempt_at && (
                        <p className="muted">
                          Last attempt:{" "}
                          <time dateTime={topic.last_attempt_at}>{topic.last_attempt_at}</time>
                        </p>
                      )}
                    </article>
                  ))}
                </section>
              ))}
            </>
          )}
        </>
      )}
      <StartOverPanel />
    </section>
  );
}
