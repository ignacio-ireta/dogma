import { Link } from "react-router";
import { FIRST_PROBLEM_TOUR_PATH, FIRST_STEPS } from "../../lib/welcomeContent";

/** The tutorial: four steps on one real problem, the first on the core path. */
export function FirstSteps() {
  return (
    <section className="welcome__section" aria-labelledby="welcome-first-steps">
      <h2 id="welcome-first-steps">Your first 20 minutes</h2>
      <p>
        Start with vector projection, the first problem on the path. It's short, and these steps
        work for every problem.
      </p>
      <ol className="welcome__steps">
        {FIRST_STEPS.map(([title, text]) => (
          <li key={title}>
            <strong>{title}</strong>
            <span>{text}</span>
          </li>
        ))}
      </ol>
      <p className="welcome__actions">
        <Link className="button button--primary" to={FIRST_PROBLEM_TOUR_PATH}>
          Try the first problem
        </Link>
      </p>
    </section>
  );
}
