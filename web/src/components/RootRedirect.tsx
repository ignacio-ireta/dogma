import { Navigate } from "react-router";
import { hasSeenWelcome } from "../lib/welcome";

/** The root: the introduction on a first visit, Today after that. Deep links never come here. */
export function RootRedirect() {
  return <Navigate to={hasSeenWelcome() ? "/today" : "/welcome"} replace />;
}
