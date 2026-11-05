# Thinking Tools for Case Interviews

*These are tools to organize YOUR thinking — not scripts to perform. The interview evaluates 4 things: your **Experiences**, your **Analytical** thinking, your **Creativity**, and your **Presence**. These frameworks help you deliver all four naturally.*

---

## MECE — The Foundation of Everything

**Mutually Exclusive, Collectively Exhaustive.**

Every time you break a problem into parts, those parts should:
- **Not overlap** (mutually exclusive)
- **Cover everything** (collectively exhaustive)

### Why This Matters
When you break a problem into clean buckets, you're demonstrating **Analytics** — "the ability to structure an unstructured problem." It also shows the interviewer your thinking is transparent and logical.

### In Practice

**Bad (not MECE):** "I'd look at pricing, customer satisfaction, and complaints."
→ Satisfaction and complaints overlap. And where's competition? Product quality? Not exhaustive.

**Good (MECE):** "I'd break the churn drivers into three buckets: (1) price-related — are we competitive? (2) service-related — are customers happy with what they get? (3) external — are competitors pulling them away? These cover the universe of reasons a customer leaves."

### Common MECE Breakdowns
- **Revenue:** Price × Volume × Mix
- **Customers:** Acquisition vs. Retention vs. Monetization
- **Churn drivers:** Internal (controllable) vs. External (market forces)
- **Model pipeline:** Data → Features → Model → Evaluation → Deployment
- **Impact:** Short-term (quick wins) vs. Long-term (structural changes)

### The Rule
Before presenting ANY analysis structure in a case, mentally check: "Do these buckets overlap? Do they cover everything?" If not, restructure.

---

## Hypothesis-Driven Thinking — Lead, Don't Follow

Most people analyze like this: Collect data → Look for patterns → Draw conclusions.

A consulting firm thinks like this: **Form a hypothesis → Design tests → Confirm or reject → Refine.**

### Why This Matters
This is where **Creativity** meets **Analytics**. Starting with a hypothesis shows you can draw conclusions from incomplete information and make assumptions — two things the interview explicitly evaluates. It also demonstrates you can drive a problem forward rather than waiting for someone to tell you what to do (**Experiences** — "drive").

### The Pattern

1. **State your hypothesis:** "I believe SME churn at PowerCo is primarily driven by price sensitivity after market liberalization."
2. **Design the test:** "To validate this, I'd compare our pricing vs. competitors for churned vs. retained customers."
3. **Anticipate outcomes:** "If price is the driver, I'd expect churned customers to be in regions with the most competitive alternatives."
4. **Have a backup:** "If pricing doesn't explain it, my second hypothesis is service quality — I'd look at complaint rates and resolution times."

### In a Case Interview
- After RASS Step 1 (Reframe), immediately form 2-3 hypotheses
- Rank them by likelihood
- Structure your analysis to test the top hypothesis first
- If the interviewer adds information that contradicts your hypothesis, **pivot gracefully** — "That changes my thinking. Given that pricing is competitive, the driver is more likely service quality. Let me restructure..."
- Pivoting is NOT failure. As one interviewer put it: "It's about how you recover."

---

## Issue Trees — Visualizing the Problem

An issue tree breaks the big question into smaller, answerable questions. It's MECE applied as a tree structure.

### Structure

```
How do we reduce SME churn at PowerCo?
├── Why are customers leaving?
│   ├── Price: Are we competitive?
│   ├── Service: Are we meeting expectations?
│   └── Market: Are competitors offering something we don't?
├── Which customers matter most?
│   ├── High-value: What's their churn rate vs. average?
│   └── Growth potential: Who are we losing before they scale?
└── What interventions are available?
    ├── Reactive: Retention offers for at-risk customers
    ├── Proactive: Improve service to prevent dissatisfaction
    └── Structural: Change pricing model or contracts
```

### How to Use in an Interview
- You won't draw a formal tree, but you WILL walk the interviewer through this structure verbally
- "I see three main questions here: Why are they leaving? Who matters most? And what can we do about it? Let me start with the first..."
- This shows structured thinking without being robotic

---

## The 80/20 Principle — Simple Wins

**20% of the effort produces 80% of the value.**

The webinar was clear: find the approach that's "most comfortable and repeatable for you." Nobody's impressed by complexity for its own sake. The best answer is the one you can explain clearly and defend confidently — that's **Presence**.

### What This Means in Your Interview

- **Model choice:** Logistic regression that's interpretable and ships in a week > deep learning that takes 3 months
- **Feature engineering:** 10 well-chosen features > 200 features thrown at XGBoost
- **Recommendation:** One clear, actionable strategy > five vague suggestions
- **Communication:** One sharp sentence > three hedging paragraphs

### The Mindset
"What's the simplest model that solves this business problem?"

If you propose a complex solution, be ready to answer: "Could we get 80% of the benefit with something simpler?" That self-challenge is both **Analytics** (transparent logic) and **Creativity** (seeing the bigger picture).

---

## Profitability Framework — When Money Enters the Chat

Many consulting cases have a financial component. Simple but powerful:

```
Profit = Revenue - Costs
Revenue = Price × Volume × Mix
Costs = Fixed + Variable
```

### The Pattern (from a live demo)

In the webinar, the live case demonstration showed this exact sequence:

1. **State the framework:** "Profitability is revenue minus cost."
2. **Hypothesis on which lever:** "The client thinks costs aren't the main driver, so my hypothesis is revenue is where we should look. But let's keep costs as a backup lever."
3. **Drill into the priority:** "The lowest hanging fruit on revenue is price. Do we have data on price changes and their effect on volume?"
4. **Quantify with simple math:** "At $9.99 with $3.49 variable cost, that's $6.50 margin × 1M units = $6.5M variable profit. That's our starting point."
5. **Analyze scenarios:** Work through 2-3 alternatives, compare impact
6. **Connect to broader context:** "This also makes me think about brand reputation — how is the product perceived? Decreasing price could open new segments but also signal lower quality."

**The pattern:** Framework → Hypothesis → Data request → Simple math → Scenarios → Broader business implications.

### In a DS Case
- **Churn case:** "Each retained customer is worth €X/year. If our model identifies 1,000 at-risk customers and our retention campaign saves 30% of them, that's 300 × €X = €Y in preserved revenue."
- **Fraud case:** "Each false negative costs $Z. Each false positive costs $W in investigation time. We should optimize our threshold to minimize total cost, not just maximize accuracy."
- **Recommendation case:** "A 2% improvement in conversion rate across 1M monthly users at $50 average order = $1M/month incremental revenue."

### The Key Insight
Consulting firms don't sell models. They sell **business impact measured in money.** Always translate your model output to dollars/euros/pesos. This is what turns a technical answer into an **actionable recommendation** (Analytics pillar).

---

## Issue-Based Consulting (IBC)

IBC is how a consulting firm structures client engagements. You don't need deep methodology knowledge, but understanding the mindset helps:

1. **Define the issue** — What specific question are we answering? (Not "improve things" but "how do we reduce SME churn by 15% in 6 months?")
2. **Structure the analysis** — Break the issue into MECE components (issue tree)
3. **Prioritize** — Which branches of the tree have the highest impact? Start there. (80/20)
4. **Gather evidence** — For a DS case, this is your data + model
5. **Synthesize** — So what? What do we DO?

### How It Maps to RASS
The RASS framework IS IBC applied to a case interview:
- **Reframe** = Define the issue
- **Ask** = Structure + Prioritize
- **Structure** = Gather evidence (design your analysis)
- **Synthesize** = So what?

---

## Quantitative Rigor

The firm's slides: "We might ask you to estimate numbers or give lower and upper bounds."

The firm's FAQ: "While we won't ask you to calculate all numbers out exactly, we are expecting quantitatively sound answers."

### Rules
- Never give a qualitative answer where a quantitative one is possible
- "Many customers churn" ❌ → "If churn rate is 15% on a base of 100K SMEs, that's 15,000 lost customers per year" ✅
- Bound your estimates: "I'd expect the impact to be between €2M and €5M annually"
- Simple math > complex math. Round aggressively. 47% ≈ 50%. 1,237 ≈ 1,200.

---

## Putting It All Together — The 4 Pillars

The interview evaluates exactly four things. Everything above is just a tool to help you deliver them naturally.

### 🏗️ Experiences
*"Lead teams, collaborate, drive, persevere under challenging circumstances."*


### 📐 Analytics
*"Structure an unstructured problem, break into components, transparent logic, actionable recommendations."*

This is where MECE, issue trees, and hypothesis-driven thinking help. Not as a performance — as a way to organize your natural problem-solving so it's visible to the interviewer.

### 💡 Creativity
*"Unique perspective, big picture, draw conclusions from partial info, make assumptions and hypotheses."*

This is where you go beyond the textbook answer. Connect to your real experience. Propose creative interventions. Work confidently with incomplete information. See patterns others miss.

### 🎯 Presence
*"Confidence, communicate effectively, demonstrate enthusiasm about the types of issues consultants face."*

Think out loud. Be conversational. Show that you genuinely enjoy solving messy business problems — because you do. Confidence comes from evidence: you've already done this at production scale.

---

From the webinar:

> **"Be your authentic self. Don't try to guess what you think the interviewer wants. We want you to be your authentic self."**

The frameworks are scaffolding. You're the building. Walk in as yourself.

---

*"The most dangerous phrase in consulting is: 'We've always done it this way.'"*
