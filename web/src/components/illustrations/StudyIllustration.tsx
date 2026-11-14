import { topicScene } from "../../lib/topicIllustration";
import { GuideFigure } from "./GuideFigure";
import { ReaderFigure } from "./ReaderFigure";
import { TopicMotif } from "./TopicMotif";
import "../../styles/illustrations.css";

/** Hand-authored, decorative SVG. No IDs: several scenes can safely share a page. */
export function StudyIllustration({
  topicId,
  compact = false,
}: {
  readonly topicId?: string | null;
  readonly compact?: boolean;
}) {
  const scene = topicScene(topicId);
  return (
    <svg
      className={`study-art${compact ? " study-art--compact" : ""}`}
      viewBox={compact ? "0 0 330 190" : "0 0 640 400"}
      width={compact ? 330 : 640}
      height={compact ? 190 : 400}
      aria-hidden="true"
      focusable="false"
      data-topic-scene={scene}
    >
      {compact ? (
        <>
          <path
            className="art-violet-soft"
            d="M93 43q76-46 174-19 62 28 42 99-16 47-98 42L84 157Z"
          />
          <g transform="translate(129 28) rotate(4 90 60)">
            <rect className="art-white" x="-5" y="-5" width="190" height="130" rx="12" />
            <TopicMotif scene={scene} />
          </g>
          <g transform="translate(35 13) scale(.54)">
            <GuideFigure />
          </g>
          <path className="art-line art-muted-line" d="M19 177h291" />
          <path className="art-coral" d="m295 8 4 8 9 1-7 6 2 9-8-5-8 5 2-9-7-6 9-1Z" />
        </>
      ) : (
        <>
          <path
            className="art-violet-soft"
            d="M130 110C171 30 309 18 414 58s177 103 139 198-197 118-321 83S85 205 130 110"
          />
          <path className="art-mint" d="M58 207q17-52 65-24t17 112q-26 39-68 3t-14-91" />
          <g transform="rotate(-5 348 177)">
            <rect className="art-ink" x="205" y="76" width="298" height="210" rx="16" />
            <rect
              className="art-white art-outlined"
              x="197"
              y="68"
              width="298"
              height="210"
              rx="16"
            />
            <path className="art-line art-muted-line" d="M215 96h261" />
            <circle className="art-coral" cx="218" cy="83" r="3" />
            <circle className="art-violet" cx="230" cy="83" r="3" />
            <circle className="art-cobalt" cx="242" cy="83" r="3" />
            <g transform="translate(236 109) scale(1.23)">
              <TopicMotif scene={scene} />
            </g>
          </g>
          <g transform="translate(89 74)">
            <GuideFigure />
          </g>
          <g transform="translate(425 162)">
            <ReaderFigure />
          </g>
          <path className="art-line" d="M51 374h549" />
          <path className="art-coral" d="m49 41 7 16 17 2-13 11 4 17-15-9-15 9 4-17-13-11 17-2Z" />
          <path className="art-line art-cobalt-line" d="M528 62q9-20 20-5t20-6 18 1" />
          <path className="art-line" d="m298 325 10-9 10 9 10-9 10 9" />
          <circle className="art-coral" cx="368" cy="343" r="11" />
          <path className="art-line art-violet-line" d="M577 130v16m-8-8h16" />
          <path className="art-cobalt" d="m252 337 13 24h-26Z" />
        </>
      )}
    </svg>
  );
}
