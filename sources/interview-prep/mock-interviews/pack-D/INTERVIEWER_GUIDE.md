# Pack D — FinServe: Credit Scoring for the Unbanked

---

## Overview

| Item | Detail |
|------|--------|
| Domain | Risk / Fintech |
| Problem Type | Binary Classification (default prediction) |
| Client | FinServe — digital lending platform in Latin America |
| Core Issue | 60% of applicants have no traditional credit history; current model rejects them all |
| Twist | Regulators require the model to be fair across gender and region — what do you do? |

---

## Opening Script (Read Aloud)

> "Our client is **FinServe**, a digital lending platform operating in several Latin American countries. They offer microloans of $200-$2,000 through a mobile app, targeting underserved populations — small shop owners, gig workers, and people without formal banking relationships.
>
> Their problem: 60% of their loan applicants have no traditional credit history. The current scoring model — built on credit bureau data — simply rejects anyone without a credit score. That means they're turning away a massive market, including many creditworthy borrowers.
>
> FinServe has partnered with telecom providers and utility companies to access alternative data — mobile top-up patterns, utility payment history, mobile money transactions. They want to build a credit scoring model that can assess these 'thin-file' and 'no-file' applicants.
>
> Their goal: approve 40% of currently rejected applicants while keeping the default rate below 8%."

---

## Information to Reveal ONLY When Asked

### If asked about current default rates:
**Show Exhibit 1** (default rates by data segment).

"For applicants with full credit bureau data, default rate is 4.2%. For those with only alternative data, it jumps to 12.5%. For minimal data applicants, 22%. The challenge is building a model accurate enough for the thin-file segments."

### If asked about available alternative data:
**Show Exhibit 2** (feature importance from pilot).

"They ran a small pilot with 5,000 applicants. Mobile top-up frequency was the strongest predictor, followed by utility payment history and mobile money transactions. Social media activity was nearly useless. Phone model tier was surprisingly predictive — it proxies for income level."

### If asked about loan economics:
"Average loan is $800. Interest rate is 3.5% per month. If a borrower defaults, they lose the entire principal — no collateral. So a $800 loan generates about $28/month in interest, but a default costs $800. Break-even: a borrower needs to stay current for ~28 months. Most loans are 6-12 months."

### If asked about current model:
"Simple logistic regression on credit bureau features. Binary accept/reject with a hard threshold. No granularity — a 61% probability of repayment gets the same 'reject' as a 20%."

### If asked about data volume:
"2 years of loan data — about 200,000 approved loans (from the 40% with credit history). For the alternative data pilot, only 5,000 observations with labeled outcomes."

---

## Challenges to Pose

1. "You have 200K labeled observations with credit bureau data and only 5K with alternative data. How do you train a reliable model for the alternative data segment?"

2. "Mobile top-up frequency is predictive — but what if someone loses their job and stops topping up? How robust is this signal over time?"

3. "FinServe wants to approve 40% of the currently rejected pool. How do you set the threshold? What's the tradeoff?"

4. "You're scoring people who've NEVER had a loan. How do you validate the model without historical default data for this population?"

---

## The Twist (Reveal After Initial Structure)

> "One more thing — the financial regulator has announced new rules requiring all credit models to demonstrate fairness. Specifically: the approval rate for women must be within 5 percentage points of the approval rate for men, and rural applicants must not be systematically disadvantaged. How does this affect your model?"

**Expected insight:** Model fairness is a real constraint. Options: (1) Fairness-aware training (equalized odds, demographic parity). (2) Post-processing: adjust thresholds by group. (3) Remove protected features — but proxies (region, phone model) might encode them. (4) Regular fairness audits with monitoring. Key tension: fairness constraints might INCREASE default rates. Need to find the balance. Understanding that fairness ≠ just removing gender from features.

---

## What a Strong Answer Covers

1. **Reframe:** "How do we build a credit model that can reliably score applicants with no traditional credit history, approve 40% of them, keep defaults under 8%, and satisfy regulatory fairness requirements?"
2. **Two-model strategy or feature augmentation:** Use credit bureau model for those who have it, alternative data model for thin-file applicants. Or a unified model that uses all available features.
3. **Small sample problem:** Only 5K alternative data observations. Transfer learning from the 200K traditional model? Semi-supervised? Active learning to build the dataset over time?
4. **Calibration matters:** In lending, you need accurate probabilities, not just rankings. A well-calibrated model lets you set precise thresholds.
5. **Risk-tiered pricing:** Don't just accept/reject — offer different loan terms (amount, rate, term length) based on risk score.
6. **Fairness approach:** Not just "remove gender" — address proxies, use fairness metrics, monitor continuously.
7. **Quantifies:** 60% of applicants rejected × assume 500K annual applicants = 300K rejected. Approve 40% = 120K new loans × $800 = $96M in new lending. At 8% default rate: $7.7M in losses. At 3.5% monthly interest over 9 months average: ~$30M in interest revenue. Net: very positive.

---

## Timing Guide

| Phase | Target Time |
|-------|-------------|
| Read scenario | 0:00 - 2:00 |
| Reframe + questions | 2:00 - 7:00 |
| Structure | 7:00 - 20:00 |
| Drop twist (fairness) | ~18:00 |
| Synthesis | 22:00 - 25:00 |
