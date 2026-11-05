import { Link } from "react-router";
import { Icon } from "../../components/Icon";
import { PLACES } from "../../lib/welcomeContent";

/** One card for each part of the app, in the order of the nav. */
export function PlaceCards() {
  return (
    <section className="welcome__section" aria-labelledby="welcome-places">
      <h2 id="welcome-places">Where things live</h2>
      <ul className="welcome__places">
        {PLACES.map((place) => (
          <li key={place.to}>
            <Link className="welcome__place" to={place.to}>
              <span className="panel-icon">
                <Icon name={place.icon} />
              </span>
              <h3>{place.label}</h3>
              <p>{place.text}</p>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
