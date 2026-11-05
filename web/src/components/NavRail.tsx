import { NavLink } from "react-router";
import { Icon, type IconName } from "./Icon";

export const NAV_ITEMS = [
  { to: "/today", label: "Today" },
  { to: "/problem", label: "Problem" },
  { to: "/map", label: "Map" },
  { to: "/review", label: "Review" },
  { to: "/mock", label: "Mock" },
  { to: "/progress", label: "Progress" },
  { to: "/welcome", label: "Start here" },
] as const;

export function NavRail() {
  return (
    <nav aria-label="Main" className="nav">
      <p className="nav__eyebrow">Your practice</p>
      <ul className="nav__list">
        {NAV_ITEMS.map((item) => (
          <li key={item.to}>
            <NavLink to={item.to} className="nav__link">
              <Icon name={item.to.slice(1) as IconName} />
              <span>{item.label}</span>
              <span className="nav__active-dot" aria-hidden="true" />
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="nav__note">
        <span className="nav__note-icon">
          <Icon name="leaf" />
        </span>
        <p>
          Small steps.
          <br />
          Deep understanding.
        </p>
        <span>Make a little room to learn.</span>
      </div>
      <div className="nav__footer">
        <span className="status-dot" />
        Your work stays local
      </div>
    </nav>
  );
}
