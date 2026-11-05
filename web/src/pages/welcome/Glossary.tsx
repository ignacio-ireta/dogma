import { GLOSSARY } from "../../lib/welcomeContent";

/** The words dogma uses that a newcomer won't know yet. */
export function Glossary() {
  return (
    <section className="welcome__section" aria-labelledby="welcome-glossary">
      <h2 id="welcome-glossary">Words you'll see</h2>
      <dl className="welcome__glossary">
        {GLOSSARY.map(([term, meaning]) => (
          <div key={term}>
            <dt>{term}</dt>
            <dd>{meaning}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
