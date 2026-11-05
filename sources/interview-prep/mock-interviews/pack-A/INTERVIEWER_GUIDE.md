# Pack A — MediCare Plus: Hospital Readmission Prediction

## ⚠️ INTERVIEWER ONLY — do not read this file before the exercise.

---

## Overview

| Item | Detail |
|------|--------|
| Domain | Healthcare / Risk |
| Problem Type | Binary Classification |
| Client | MediCare Plus — 15-hospital network in Latin America |
| Core Issue | 30-day readmission rate of 18% vs 12% national benchmark |
| Twist | Data completeness varies wildly — social determinants at 41%, clinical notes unstructured |

---

## Opening Script (Read Aloud — ~2 min)

> "Our client is **MediCare Plus**, a network of 15 hospitals across several Latin American regions with about 120,000 admissions per year. They're facing a problem with patient readmissions — patients who are discharged but return to the hospital within 30 days.
>
> Their current 30-day readmission rate is 18%, well above the national benchmark of 12%. Each avoidable readmission costs the hospital an estimated $8,500 in unreimbursed care, and regulators are starting to tie accreditation scores to readmission performance.
>
> MediCare Plus has asked a consulting team to help them predict which patients are at risk of readmission so they can intervene before discharge. They want a solution they can deploy across all 15 hospitals within 6 months."

---

## Information to Reveal ONLY When Asked

### If asked about current discharge process:
"Currently, doctors make discharge decisions based on clinical judgment. There's no standardized risk scoring. Some departments have informal follow-up calls, but it's inconsistent."

### If asked about data / what's available:
**Show Exhibit 2** (feature availability table). Point out the varying completeness rates.

"They have an EMR system, but data completeness varies. Demographics and clinical history are solid. Social determinants — things like income, transportation access, whether the patient has a caregiver — are only captured for 41% of patients. And about 60% of patients have unstructured clinical notes that aren't in any structured database."

### If asked about readmission patterns:
**Show Exhibit 1** (readmission rates by department).

"Oncology and Cardiology are the worst — both above 20%. Orthopedics and General Surgery are closer to the benchmark. There seems to be a departmental pattern but nobody's investigated why."

### If asked about class balance:
"18% readmission means roughly 82/18 split. Not as extreme as fraud, but still imbalanced enough to think about."

### If asked about interpretability:
"The chief medical officer insists that clinicians need to understand WHY a patient is flagged. They won't change their discharge decisions based on a black box."

### If asked about intervention budget:
"They can afford enhanced discharge planning for about 15% of patients — things like follow-up calls, home health visits, medication reconciliation. It costs about $500 per patient. So targeting matters."

### If asked about success metrics:
"They want to reduce the readmission rate from 18% to below 14% within the first year. That would save roughly $5M in unreimbursed costs."

---

## Challenges to Pose (pick 2-3 during Structure)

1. "You mentioned [model]. How would you handle the fact that 41% of patients are missing social determinant data? Do you drop those features? Impute? Something else?"

2. "The CMO wants interpretability. You proposed [model]. How would you explain a specific patient's risk score to a cardiologist who has 5 minutes between rounds?"

3. "Oncology has the highest readmission rate. But is that because oncology patients are sicker, or because the oncology department has a discharge problem? How would you disentangle that?"

4. "You're optimizing for recall to catch at-risk patients. But if you flag too many, the intervention team gets overwhelmed and starts ignoring the alerts. How do you balance this?"

---

## The Twist (Reveal After Initial Structure)

> "One thing I should mention — the hospital recently started a pilot with a partner to digitize clinical notes using NLP. They've processed about 10,000 notes so far and found that physician sentiment in discharge notes (phrases like 'patient is not confident in self-care' or 'limited family support') is actually one of the strongest predictors of readmission. But it's only available for a small subset. How does this change your approach?"

**Expected insight:** Two-model approach or staged rollout. Start with structured data (available for all patients) for the base model. Use NLP features as a booster for patients where notes are available. As the digitization scales, the NLP model improves. Don't wait for perfect data to deploy — ship what works now, improve iteratively.

---

## What a Strong Answer Covers

1. **Reframe:** "How do we identify high-risk patients before discharge so MediCare Plus can target interventions and reduce readmissions from 18% to below 14%?"
2. **Asks about data completeness** (the key challenge in this case)
3. **Addresses the 82/18 class imbalance** — class weights, threshold tuning, or SMOTE
4. **Chooses an interpretable model** given the CMO requirement — gradient boosted trees with SHAP, or logistic regression
5. **Has a strategy for missing data** — doesn't just say "impute" but discusses which imputation method and why, or uses models that handle missingness natively
6. **Addresses the NLP twist** with a pragmatic staged approach
7. **Quantifies impact:** 120K admissions × 18% = 21,600 readmissions. Reducing to 14% = 16,800. Saving 4,800 readmissions × $8,500 = ~$41M, minus intervention costs of 18K × $500 = $9M. Net savings: ~$32M.
8. **Addresses implementation:** Integration with EMR, clinician training, feedback loop for model monitoring

---

## Timing Guide

| Phase | Target Time | Notes |
|-------|-------------|-------|
| Read scenario | 0:00 - 2:00 | |
| His reframe + questions | 2:00 - 7:00 | Show exhibits when relevant |
| His structure | 7:00 - 20:00 | Challenge at ~12:00 and ~16:00 |
| Drop the twist | ~18:00 | Watch how he adapts |
| Ask for synthesis | 22:00 | "Can you wrap up with a recommendation?" |
| Synthesis | 22:00 - 25:00 | |
