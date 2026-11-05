import { Link, Outlet } from "react-router";
import { Icon } from "./Icon";
import { NavRail } from "./NavRail";
import { StudyWeekBadge } from "./StudyWeekBadge";

export function AppShell() {
  return (
    <div className="shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="shell__header">
        <Link className="brand" to="/today" aria-label="dogma home">
          <span className="brand__mark">
            <Icon name="mark" />
          </span>
          <span>
            dogma<span className="brand__dot">.</span>
          </span>
        </Link>
        <div className="shell__context">
          <span className="shell__context-label">Your learning space</span>
          <StudyWeekBadge />
        </div>
        <span className="shell__mode">
          <Icon name="sun" />
          At your own pace
        </span>
      </header>
      <NavRail />
      <main id="main" className="shell__main" tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  );
}
