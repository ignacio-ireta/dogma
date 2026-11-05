import { useSearchParams } from "react-router";
import { FIRST_STEPS } from "../lib/welcomeContent";
import "../styles/welcome.css";

/** The tutorial's steps, as one short row so the work stays on screen. `?tour=1` shows it. */
export function TourStrip() {
  const [params, setParams] = useSearchParams();
  if (params.get("tour") !== "1") {
    return null;
  }
  const hide = () =>
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        next.delete("tour");
        return next;
      },
      { replace: true },
    );
  return (
    <section className="tour-strip" aria-labelledby="tour-strip-title">
      <div className="tour-strip__heading">
        <h2 id="tour-strip-title">Your first problem</h2>
        <button className="button button--quiet" type="button" onClick={hide}>
          Hide this guide
        </button>
      </div>
      <ol>
        {FIRST_STEPS.map(([title]) => (
          <li key={title}>{title}</li>
        ))}
      </ol>
    </section>
  );
}
