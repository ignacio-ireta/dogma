# Pack F — TalentMatch: Employee Attrition Prediction

---

## Overview

| Item | Detail |
|------|--------|
| Domain | Enterprise / HR Analytics |
| Problem Type | Binary Classification + Strategic Recommendation |
| Client | TalentMatch — 8,000-employee tech company |
| Core Issue | Top performers leave at 2x the rate of average employees |
| Twist | The #1 reason top performers leave is "lack of career growth" — but the company just froze promotions for 6 months due to cost cutting |

---

## Opening Script (Read Aloud)

> "Our client is **TalentMatch**, a tech company with 8,000 employees across engineering, product, data science, marketing, sales, and operations. They're growing fast but facing a talent retention crisis.
>
> Overall attrition is 11% annually — slightly above industry average. But the real problem is WHERE the attrition is concentrated: their top performers — the top 15% by performance rating — are leaving at 18.2% per year. That's nearly double the average performer attrition of 8.3%.
>
> Replacing a top performer costs an estimated 2x their annual salary when you factor in recruiting, onboarding, lost productivity, and knowledge loss. For an average senior engineer at $150K, that's $300K per departure.
>
> TalentMatch wants to predict which employees are at risk of leaving in the next 6 months so they can intervene proactively. But they also want to understand WHY top performers are leaving at higher rates."

---

## Information to Reveal ONLY When Asked

### If asked about attrition patterns:
**Show Exhibit 1** (attrition by performance tier).

"It's inverted from what you'd expect. Top performers at 18.2%, above average at 11.5%, average at 8.3%, below average at 6.1%. The best people have the most options."

### If asked about exit survey data:
**Show Exhibit 2** (exit survey themes).

"We have exit survey data. The pattern is striking: top performers cite 'career growth opportunities' (52%) and 'burnout' (48%) as top reasons. Average performers cite 'compensation' (42%) and 'work-life balance' (35%). They're leaving for DIFFERENT reasons."

### If asked about available data:
"HR system has: tenure, department, performance scores (annual, 1-5), salary, last promotion date, overtime hours, satisfaction survey scores (quarterly), number of projects, manager ratings, remote work percentage, stock options. Two years of history."

### If asked about interventions available:
"HR has a few levers: retention bonuses ($10-50K), accelerated promotion, role rotation to new projects, reduced workload, manager coaching, flexible work arrangements. But each has a cost and a limit — they can't offer retention bonuses to everyone."

### If asked about the prediction window:
"They want 6-month forward prediction. Enough time to intervene meaningfully, but recent enough to be actionable."

### If asked about model usage:
"The CHRO wants a dashboard that flags at-risk employees with a risk score and the TOP 3 reasons for each person. Managers would get alerts for their direct reports."

---

## Challenges to Pose

1. "Satisfaction surveys are quarterly — by the time someone shows low satisfaction, they might already be interviewing. What leading indicators would you look for that appear BEFORE satisfaction drops?"

2. "You're building a model that predicts who will quit. If managers get an alert saying 'this person is likely to leave,' how might that change the manager's behavior — and could it backfire?"

3. "Top performers leave for different reasons than average performers. Should you build one model or two? What are the tradeoffs?"

4. "How do you handle the fact that you only observe who LEFT — you don't observe who THOUGHT about leaving but stayed? Your negative class includes both happy employees and unhappy ones who haven't left yet."

---

## The Twist (Reveal After Initial Structure)

> "There's a complication: the CFO just announced a 6-month promotion freeze as part of cost reduction. No one can be promoted until Q3. But 'lack of career growth' is the #1 reason top performers are leaving. You can't use the most obvious lever. What do you recommend?"

**Expected insight:** Creative alternatives to formal promotion: (1) Title changes without comp increase, (2) High-visibility project assignments, (3) Learning and development budget (conferences, courses), (4) Mentorship of junior staff (top performers often value impact), (5) Public recognition programs, (6) Explicit career path conversations ("we can't promote now, but here's the plan for Q3"), (7) Stock option acceleration. Key: the model's feature importance tells you WHICH lever matters for EACH person. Personalized retention strategy > blanket policy.

---

## What a Strong Answer Covers

1. **Reframe:** "How do we predict which employees — especially top performers — are at risk of leaving in the next 6 months, understand their individual reasons, and design targeted retention strategies that work within the promotion freeze?"
2. **Recognizes the different dynamics:** Top performers leave for growth/burnout; average performers leave for compensation/balance. One-size-fits-all doesn't work.
3. **Feature engineering:** Promotion gap (time since last promotion vs. peers), workload trend (is overtime increasing?), satisfaction trajectory (declining vs. stable), manager change, peer departures (attrition is contagious), market salary delta.
4. **Model with explainability:** Gradient boosting with SHAP for individual-level feature attribution. Each flagged employee gets "top 3 reasons."
5. **Addresses the self-fulfilling prophecy risk:** If a manager treats a flagged employee differently, it could push them out faster. Need careful deployment — flag to HR first, not directly to managers?
6. **Handles the promotion freeze creatively:** Non-monetary retention levers tailored to individual drivers.
7. **Quantifies:** 8,000 employees × 15% top performers = 1,200 top performers. At 18.2% attrition = ~218 departures/year. At $300K replacement cost = ~$65M/year in turnover costs. Reducing top performer attrition to 12% = ~74 fewer departures = ~$22M saved.

---

## Timing Guide

| Phase | Target Time |
|-------|-------------|
| Read scenario | 0:00 - 2:00 |
| Reframe + questions | 2:00 - 7:00 |
| Structure | 7:00 - 20:00 |
| Drop twist (promotion freeze) | ~18:00 |
| Synthesis | 22:00 - 25:00 |
