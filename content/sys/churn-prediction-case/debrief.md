# Mobile churn, debrief

## Reframe

Which at-risk subscribers should get a targeted retention offer, so that monthly churn returns below
the industry average without blanket discounts? Success is saves per offer dollar, against a holdout.

## Ask

Ask how churn is defined (a cancelled line, or a lapsed prepaid top-up), what an offer costs against
a subscriber's lifetime value, how fresh the data is, and what the retention team can act on in a
week. The answers decide the tiers and the threshold.

## Structure

Segment by plan first. Build features from the three-month usage trend, changes in payment behaviour
and support-contact spikes. Use gradient-boosted trees for mixed, imbalanced tabular data, and compare
them with logistic regression. Set the threshold for the number of offers the team can make, and
explain individual scores with attributions so the retention team trusts them.

## Synthesize

Tier by lifetime value times risk. High value and high risk gets a personal call. Medium value gets an
automated offer. Low value is left alone, because the offer would cost more than the subscriber is
worth. At 12 million subscribers, a churn rate of 3.4 percent is about 408,000 departures a month.
Reaching 2.5 percent avoids about 108,000, worth roughly 13 million dollars a month in avoided
acquisition cost. Pilot the tiers in one region against a holdout for one billing cycle, then scale.
