# Card fraud, debrief

## Reframe

Which transactions should be declined or sent to review, so that more fraud is caught while the review
queue stays within about 30,000 cases a month? Success is fraud value caught at that queue size, within
the 200-millisecond decision window.

## Ask

Ask for the latency budget per stage, the cost of each error type, the class balance, the review
capacity, and what a decline explanation must contain for customers and regulators. The explanation
requirement decides whether the model must be interpretable or only explainable.

## Structure

Use two stages. A fast rule filter decides the obvious cases within the latency budget. A scored model
handles the borderline cases. Handle the 0.05 percent rate with class weights, and set the threshold for
the review queue. Evaluate with precision at the review budget and recall at a fixed false-alarm rate.
Features: velocity over five minutes, deviation from the spending pattern, a new device, merchant risk and
time of day. Attach per-decision attributions to every decline, or use an interpretable model if the
attributions are not accepted.

## Synthesize

Moving from 60 to 85 percent recall on about 25,000 frauds a month, at 640 dollars each, prevents about
4 million dollars a month, before model running costs. Pilot the system in shadow mode on one card product
for four weeks, comparing alerts and caught fraud with the current rules, then scale when the review queue
and the explanations hold up.
