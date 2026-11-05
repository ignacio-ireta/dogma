# RASS mini-case, debrief

## Reframe

Which stations should a truck visit next, to avoid the most empty station-hours in the next two
hours? Success is lost trips avoided per truck-hour, measured on held-out evenings.

## Ask

Ask what a truck visit costs compared with a lost trip, how far ahead the dispatcher must know,
whether events are known in advance, and whether the dispatcher can act on a ranked list.

## Structure

Forecast each station's chance of being empty in two hours, from recent departures, time of day and
weather. Rank stations by expected lost trips and give the top stations to the trucks. Compare with a
baseline that uses the last hour's empty rate. Keep the model simple enough to explain to a dispatcher,
and evaluate on evenings it has not seen.

## Synthesize

About one station in six is empty, so roughly 150 stations are affected at a time. If the ranking
avoids a quarter of those station-hours, that is about 37 stations and 1,500 trips an hour, or about
3,000 dollars of margin per peak hour before truck costs. Pilot the ranking in two districts for two
weeks against a holdout, then scale if the avoided empty hours hold up.
