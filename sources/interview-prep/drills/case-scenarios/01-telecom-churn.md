# Case Scenario 1: ConnectMobile — Telecom Churn

## Setup (Read Aloud as Interviewer)

Our client is **ConnectMobile**, a mid-sized telecom operator in Latin America with 12 million subscribers across prepaid and postpaid plans. Over the past 18 months, their monthly churn rate has climbed from 2.1% to 3.4%, well above the industry average of 2.5%.

The CEO is concerned because acquiring a new customer costs roughly $120, while retaining one costs about $30. They've asked a consulting firm to help them understand WHY churn is increasing and build a solution to identify and retain at-risk customers before they leave.

They have 3 years of customer data including demographics, usage patterns, billing history, and customer service interactions.

---

## Information to Reveal When Asked

**If asked about current retention efforts:**
"They have a basic rule-based system — if a customer calls to cancel, they offer a 20% discount. No proactive outreach. About 40% of customers who call to cancel accept the offer."

**If asked about data specifics:**
- Demographics: age, gender, location, plan type
- Usage: monthly minutes, data GB, SMS count, app usage
- Billing: payment history, late payments, plan changes, average bill
- Service: number of support calls, complaint types, NPS scores
- Contract: tenure, contract type (1yr/2yr/month-to-month), handset financing

**If asked about class balance:**
"About 3.4% churn monthly, so ~96.6% stay. Heavily imbalanced."

**If asked about segments:**
"Prepaid is 60% of subscribers but 75% of churners. Postpaid is more stable."

**If asked about competition:**
"Two major competitors launched aggressive pricing in the last year — unlimited data plans at 15-20% below ConnectMobile's rates."

---

## The Twist

After the candidate structures their approach, reveal:
"One thing I should mention — the marketing team already tried a blanket 15% discount campaign last quarter targeting all customers. Churn actually increased slightly during that period. They're not sure why."

*Expected insight: Untargeted discounts may signal weakness / attract price-sensitive switchers / miss the real drivers. Need targeted, differentiated approach.*

---

## What a Strong Answer Covers

1. **Reframe:** "How do we proactively identify at-risk subscribers and deploy targeted retention — not blanket discounts — to reduce monthly churn back below industry average?"
2. **Segmentation first:** Separate prepaid vs postpaid strategy (very different dynamics)
3. **Feature engineering:** Usage trend (declining usage over 3 months = strong signal), payment behavior changes, support call frequency spikes
4. **Model choice:** Gradient boosting with justification (imbalanced data, tabular, mixed features). Optimize for recall with threshold tuning.
5. **Why blanket discounts failed:** No targeting → wasted on loyal customers + may have attracted competitors' attention. Need CLV-based tiering.
6. **Business recommendation:** High-CLV + high-risk → personal call from retention team + customized plan. Medium-CLV → automated targeted offer. Low-CLV → let go (cost of retention > value).
7. **Quantitative impact:** 12M × 3.4% = 408K churners/month. Even reducing to 2.5% = 108K fewer churners/month × $120 acquisition savings = $13M/month potential value.
