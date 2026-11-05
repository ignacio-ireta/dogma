# RASS Framework — Quick Reference

---

## R — REFRAME (30 seconds)

Restate the problem. Quantify it. Frame as a question.

> "So if I understand correctly — [restate problem]. The core question is: [frame as question]. Success looks like [define metric/KPI]. Does that capture it?"

**What this does:** Shows you listened. Anchors the conversation. Invites correction early.

**Example (churn case):**
> "The client is losing customers at an accelerating rate — monthly churn went from 1.5% to 2.8%, which on 8,000 customers at $15K/year is roughly $1.6M in additional annual revenue loss. The core question is: can we predict which customers will churn and intervene before they leave?"

---

## A — ASK (2-3 minutes)

Two tracks. State WHY for each question.

**Business questions:**
- What's the current intervention? (baseline you're beating)
- What actions can the client take? (defines useful model output)
- Who's the end user of the model? (determines interpretability needs)
- What's the cost of a wrong prediction? (asymmetric cost → metric choice)

**Technical questions:**
- How is the target defined? (what counts as "churn"?)
- How much labeled data exists? (model complexity ceiling)
- What's the positive rate? (class imbalance severity)
- Data freshness? Quality? Access?

**The uncertainty move:** You won't get answers to everything. That's intentional. Say: "Given what we know, here's how I'd proceed — and I'd validate these assumptions with the client."

---

## S — STRUCTURE (5-7 minutes)

Tell a story, don't list steps.

**The flow:**
1. Define the problem type (classification, regression, etc.)
2. Feature engineering approach (what raw data → what features)
3. Model choice with justification (WHY this, WHY NOT that)
4. Evaluation strategy (metric + validation approach)
5. Deployment consideration (who uses it, how)

**Model justification template:**
> "I'd use [model] because ① [fits problem type], ② [matches constraints], ③ [handles data characteristics]. I considered [alternative] but [specific reason]. If performance is insufficient, I'd escalate to [fallback]."

**Handle pushback here:**
- "Why not neural nets?" → "GBDTs dominate tabular data with less tuning burden."
- "Why not simpler?" → "LogReg is my baseline. If it's sufficient, we ship that."
- "How do you explain it?" → "SHAP values give feature-level explanations for any model."

---

## S — SYNTHESIZE (1-2 minutes)

Land on BUSINESS IMPACT. Not model metrics.

> "If we deploy this model and [intervention], we estimate [quantified impact]. I'd recommend a pilot with [scope] over [timeframe], measure [KPI], then scale if results hold."

**Revenue > F1. Always.**

**The formula:**
Problem size × expected improvement × intervention rate = business value

Example: 8,000 customers × $15K × (2.8%-1.5%) reduction × 60% intervention success = ~$940K annual value

---

## Transition Markers (Say Out Loud)

- "Let me **REFRAME** what I've heard..."
- "I'd like to **ASK** a few clarifying questions..."
- "Let me **STRUCTURE** my approach..."
- "To **SYNTHESIZE** my recommendation..."

These serve two purposes:
1. Signal organized thinking to the interviewer
2. Self-generated cues that survive stress (when PFC goes offline, habitual markers still fire)

---

*RASS is not a framework to recite. It's a HABIT to deploy. Say it until it's automatic.* 🏔️
