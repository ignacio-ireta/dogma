import { MinutesBar } from "../../components/MinutesBar";
import { SESSION_BLOCKS } from "../../lib/welcomeContent";

/** The five blocks of a Learn session, in order, so the shape of the work is known up front. */
export function SessionShape() {
  return (
    <section className="welcome__section" aria-labelledby="welcome-session">
      <h2 id="welcome-session">What a session looks like</h2>
      <p>A session is 45 minutes, and every Learn session has the same five blocks.</p>
      <MinutesBar blocks={SESSION_BLOCKS.map(([name, minutes]) => ({ name, minutes }))} />
      <ol className="welcome__blocks">
        {SESSION_BLOCKS.map(([name, minutes, text]) => (
          <li key={name} className="welcome__block">
            <strong>{name}</strong>
            <span className="welcome__minutes">{minutes} min</span>
            <span>{text}</span>
          </li>
        ))}
      </ol>
      <p className="muted">
        Today also plans Consolidate sessions, for review and re-solves, and Interview sessions, for
        mocks. On a harder day, choose Low energy on Today for a 20-minute session with nothing new.
      </p>
    </section>
  );
}
