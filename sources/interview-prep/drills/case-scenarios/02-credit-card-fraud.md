# Case Scenario 2: SecureBank — Credit Card Fraud Detection

## Setup (Read Aloud as Interviewer)

Our client is **SecureBank**, a top-5 bank in Latin America processing 50 million credit card transactions per month. Fraud losses have doubled over the past year — from $8M to $16M monthly — driven by a surge in online transaction fraud.

Their current rule-based fraud detection system catches about 60% of fraudulent transactions but generates so many false positives that the investigation team can only review 30% of flagged cases. The rest get auto-approved.

SecureBank wants a consulting firm to design a smarter detection system that catches more fraud while reducing the investigation burden.

---

## Information to Reveal When Asked

**If asked about fraud rate:**
"Roughly 0.05% of transactions are fraudulent — about 25,000 per month. Average fraud amount is $640. Legitimate transactions average $85."

**If asked about data:**
- Transaction: amount, merchant category, location, time, online vs. in-person, device fingerprint
- Customer: account age, credit limit, average monthly spend, number of cards
- Historical: previous fraud flags, disputed transactions, velocity patterns
- Merchant: merchant risk score, country, category

**If asked about latency requirements:**
"Ideally real-time — the system needs to approve or flag a transaction within 200ms. But we could also accept a two-stage approach: fast filter first, then deeper analysis."

**If asked about cost of errors:**
- False negative (missed fraud): average $640 per incident + reputational cost
- False positive (blocking legitimate transaction): customer frustration, estimated $15 per false positive in lost commerce + support cost

**If asked about current investigation capacity:**
"20 analysts, each can review ~50 cases per day = 1,000 reviews per day = ~30,000 per month."

---

## The Twist

After the candidate proposes a model: "The bank's chief risk officer insists that any flagged transaction must come with an explanation — they can't just say 'the AI said no' to customers or regulators. How does that affect your approach?"

*Expected insight: Interpretability constraint limits model choice. Could use interpretable model, or use complex model with SHAP/LIME explanations. Feature importance per prediction.*

---

## What a Strong Answer Covers

1. **Reframe:** "How do we build a fraud detection system that catches >85% of fraud while keeping false positives manageable enough for the 20-person investigation team?"
2. **Two-stage architecture:** Fast rule-based filter (< 200ms) + ML scoring for borderline cases
3. **Class imbalance strategy:** 0.05% fraud rate is extreme. Discuss SMOTE, class weights, anomaly detection approach, or cost-sensitive learning
4. **Precision-recall tradeoff:** With 50M transactions, even 0.1% FPR = 50K false positives/month. Need to tune threshold for investigation capacity (30K/month).
5. **Feature engineering:** Transaction velocity (3 transactions in 5 minutes), deviation from spending pattern, geographic anomaly, time-of-day patterns
6. **Interpretability solution:** SHAP values per prediction — "This transaction was flagged because: unusual merchant category (40%), amount 8x above average (35%), new device (25%)"
7. **Quantitative impact:** Catching 85% vs 60% of fraud = 25,000 × 25% × $640 = $4M/month additional fraud prevented. Minus model operating costs. Clear ROI.
