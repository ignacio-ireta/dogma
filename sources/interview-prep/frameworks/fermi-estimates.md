# Fermi Estimates & Quantitative Reasoning

*A consulting firm expects "quantitatively sound answers." That means estimates, bounds, and back-of-envelope math — not "a lot" or "significant."*

---

## The Method

Every Fermi estimate follows the same pattern:

1. **Break the question into components** you can estimate independently
2. **Estimate each component** — round aggressively, use nice numbers
3. **Multiply / add** to get the answer
4. **Sanity check** — does this number make sense in the real world?
5. **Bound it** — give a range, not a point estimate

### Example: "How much revenue is at risk from SME churn?"

| Component | Estimate | Reasoning |
|-----------|----------|-----------|
| Total SME customers | 100,000 | Major European utility, SME segment |
| Annual churn rate | 15% | "Above industry average" → industry ~10%, so ~15% |
| Churning customers/year | 15,000 | 100K × 15% |
| Average SME annual revenue | €5,000 | Small-medium businesses, gas + electricity |
| Revenue at risk | €75M/year | 15,000 × €5,000 |
| Realistic retention rate | 20-30% | Can't save everyone; targeted campaigns |
| Saveable revenue | €15-22M/year | €75M × 20-30% |

> "Rough estimate: we're looking at €75M in annual revenue at risk from SME churn. If a targeted retention program saves even 20-30% of those customers, that's €15-22M preserved. Against a campaign cost of maybe €2-3M, the ROI is compelling."

**That's the level a consulting firm wants.** Not precise. Not vague. Structured, bounded, and grounded.

---

## Rules of Thumb

### Rounding
- 47% → 50%. 1,237 → 1,200. 8.7 million → ~9 million.
- Nice numbers make mental math possible and show you're focused on magnitude, not precision.

### Anchoring to Known Numbers
Keep a few reference points in your head:

| Fact | Number |
|------|--------|
| World population | ~8 billion |
| Brazil population | ~215 million |
| Denmark population | ~6 million |
| EU population | ~450 million |
| Average household size (developed) | ~2.5 people |
| Working-age adults | ~60% of population |
| Smartphone penetration (developed) | ~85% |
| E-commerce share of retail | ~20% |
| Typical corporate churn rate | 5-10%/year |
| Typical consumer subscription churn | 3-7%/month |

### Bounding
Always give a range:
- "I estimate between X and Y, with the most likely value around Z"
- The range shows you understand uncertainty
- If the range is too wide, narrow by identifying the most uncertain component and pressure-testing it

---

## Practice Problems

### 1. Campaign ROI
*"We're considering a €2M email campaign to reduce churn. Is it worth it?"*

Break it down:
- How many customers does the campaign reach?
- What's the expected response rate?
- What's the value of a retained customer?
- What's €retained minus €campaign cost?

### 2. Fraud Impact
*"Our client processes 10M transactions/month. 0.1% are fraudulent, averaging $500 each. How much does fraud cost annually?"*

Quick math:
- 10M × 0.001 × $500 × 12 = $60M/year
- But factor in: investigation costs per case (~$50), false positive costs, reputational damage
- Total impact probably $80-100M when you include indirect costs

### 3. Model Value
*"Your churn model has an AUC of 0.85. What's that worth to the business?"*

Think about it as:
- Without model: random targeting = waste most of budget on non-churners
- With model: concentrate on top-risk decile
- If top decile contains 60% of actual churners → 6x more efficient than random
- Same budget, 6x more impact
- Translate to: "If we're spending €3M on retention, the model makes that €3M work like €18M of random effort"

### 4. Market Sizing
*"How big is the potential market for a predictive maintenance product in European manufacturing?"*

Components:
- European manufacturing companies: ~2M (rough)
- Companies large enough to afford ML solutions: top 5% = ~100K
- Willing to adopt ML in next 3 years: ~20% = ~20K
- Average contract value: ~€50K/year
- Addressable market: ~€1B/year
- Realistic capture: 1-2% = €10-20M

### 5. Retention Economics
*"A customer costs $200 to acquire and generates $50/month. How long before they're profitable? If they churn at month 8, what did we lose?"*

- Break-even: $200 / $50 = 4 months
- At month 8: $50 × 8 = $400 revenue, $200 acquisition = $200 profit
- But: if they DON'T churn and stay 24 months = $1,000 profit
- Lost potential: $1,000 - $200 = $800 in unrealized value
- Retention offer of $100 at month 6 → if it extends them to month 24, ROI = $800/$100 = 8x

---

## In the Interview

### When to Estimate
- When the interviewer says "How much impact would that have?"
- When you propose a solution and need to justify the investment
- When comparing two approaches (which one has better ROI?)
- When defining "success" — what specific numbers are we targeting?

### How to Present
> "Let me do a quick back-of-envelope calculation..."
> [State your assumptions clearly]
> [Do the math visibly — say the numbers out loud]
> "So we're looking at roughly €X, give or take 30%."

### If Your Math Is Wrong
It's okay. The interviewer cares about your APPROACH more than the exact number. If you realize you made an error:
> "Wait, let me double-check that... [correct it]. So it's actually closer to Y."

That's better than hiding the error. It shows rigor.

---

*Numbers > adjectives. Always.* 🏔️
