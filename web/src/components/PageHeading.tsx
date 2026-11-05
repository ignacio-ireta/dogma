import { Icon } from "./Icon";
import { StudyIllustration } from "./illustrations/StudyIllustration";

export function PageHeading({
  title,
  eyebrow,
  description,
  illustrated = false,
  topicId,
}: {
  title: string;
  eyebrow: string;
  description: string;
  illustrated?: boolean;
  topicId?: string | null;
}) {
  return (
    <header className={`page-heading${illustrated ? " page-heading--illustrated" : ""}`}>
      <div className="page-heading__copy">
        <p className="eyebrow">
          <span className="eyebrow__line" />
          {eyebrow}
        </p>
        <h1>{title}</h1>
        <p className="page-heading__description">{description}</p>
      </div>
      {illustrated ? (
        <div className="page-heading__illustration" aria-hidden="true">
          <StudyIllustration topicId={topicId} />
        </div>
      ) : (
        <div className="page-heading__art" aria-hidden="true">
          <span className="orbit orbit--outer" />
          <span className="orbit orbit--inner" />
          <span className="orbit__center">
            <Icon name="leaf" />
          </span>
          <span className="orbit__point orbit__point--one" />
          <span className="orbit__point orbit__point--two" />
          <span className="orbit__point orbit__point--three" />
          <span className="orbit__plus">+</span>
        </div>
      )}
    </header>
  );
}
