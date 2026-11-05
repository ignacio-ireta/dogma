# Code Comprehension — What Does This Code Do?

**Format:** Show this code. Ask the questions below. Conversational.

```python
def retention_roi(df, intervention_cost, success_rate=0.3):
    df = df.copy()

    df['replacement_cost'] = df['salary_usd'] * 2
    df['risk_score'] = df['predicted_attrition_prob']
    df['expected_loss'] = df['risk_score'] * df['replacement_cost']
    df['expected_save'] = df['expected_loss'] * success_rate - intervention_cost

    candidates = df[df['expected_save'] > 0].sort_values('expected_save', ascending=False)

    total_investment = len(candidates) * intervention_cost
    total_expected_savings = candidates['expected_save'].sum()
    roi = (total_expected_savings / total_investment) * 100

    return candidates[['employee_id', 'department', 'risk_score', 'expected_save']], roi
```

## Questions to Ask

1. **"Walk me through the ROI calculation."**
   - Expected: For each employee, it calculates the expected loss if they leave (attrition probability × 2x salary replacement cost). Then it calculates the expected savings from intervention (expected loss × 30% success rate minus the intervention cost). It only recommends intervention for employees where expected savings is positive, and calculates overall ROI.

2. **"What assumption is problematic here?"**
   - Expected: The 30% success rate is applied uniformly. In reality, some interventions (pay raise for compensation-driven attrition) are more effective than others (career path conversation for growth-driven attrition). A better model would estimate success rate based on the REASON for potential departure, not a flat rate. Also, replacement_cost as 2x salary is a rough heuristic — some roles are harder to fill than others.

3. **"An employee has a 90% attrition probability and earns $200K. The model says to intervene. Should you?"**
   - Expected: Maybe not. At 90% probability, they may have already accepted another offer — intervention success rate is likely much lower than 30%. There's a "window of intervention" problem: the model should factor in not just probability but also how far along the departure process is. Early-stage at-risk employees are more saveable than late-stage ones.
