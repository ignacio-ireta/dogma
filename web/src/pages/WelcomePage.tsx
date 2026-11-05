// The introduction: what dogma is, how a session runs, where things live and what to do first.
import { useEffect } from "react";
import { Link } from "react-router";
import { PageHeading } from "../components/PageHeading";
import { markWelcomeSeen } from "../lib/welcome";
import "../styles/welcome.css";
import { FirstSteps } from "./welcome/FirstSteps";
import { Glossary } from "./welcome/Glossary";
import { PlaceCards } from "./welcome/PlaceCards";
import { SessionShape } from "./welcome/SessionShape";

export function WelcomePage() {
  useEffect(() => markWelcomeSeen(), []);
  return (
    <div className="page welcome">
      <PageHeading
        title="Welcome to dogma"
        eyebrow="Start here"
        description="A short tour: what this is, how a session runs, and what to do first."
        illustrated
      />
      <section className="welcome__section" aria-labelledby="welcome-what">
        <h2 id="welcome-what">What dogma is</h2>
        <p>
          dogma is a practice space for data science and AI engineering interviews. You learn one
          idea at a time, solve a real problem on it, and come back later so that it sticks.
        </p>
        <p>Everything runs on this computer. Your code, your notes and your progress stay here.</p>
        <p className="welcome__actions">
          <Link className="button" to="/today">
            Skip to Today
          </Link>
        </p>
      </section>
      <SessionShape />
      <PlaceCards />
      <FirstSteps />
      <Glossary />
      <section className="welcome__section" aria-labelledby="welcome-next">
        <h2 id="welcome-next">When you're ready</h2>
        <p>Today picks your next step from here on. This page stays in the menu as Start here.</p>
        <p className="welcome__actions">
          <Link className="button button--primary" to="/today">
            Go to Today
          </Link>
        </p>
      </section>
    </div>
  );
}
