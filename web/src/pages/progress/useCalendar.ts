import { type QueryClient, useQuery } from "@tanstack/react-query";
import { getCalendar } from "../../api/calendar";

export function useCalendar() {
  return useQuery({ queryKey: ["calendar"], queryFn: getCalendar, retry: false });
}

/** After a move or a reset, the calendar, the progress views and the Today page all change. */
export function refreshCalendarData(client: QueryClient): Promise<unknown> {
  return Promise.all([
    client.invalidateQueries({ queryKey: ["calendar"] }),
    client.invalidateQueries({ queryKey: ["progress"] }),
    client.invalidateQueries({ queryKey: ["today"] }),
  ]);
}
