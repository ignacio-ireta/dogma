# Mock Assessment v2 — Answer Key
**DO NOT READ until you've attempted all questions.**

---

## Part 1: Pandas

### Q1.1 — Missing values
```python
missing = customers.isnull().sum()
missing_pct = (missing / len(customers) * 100).round(2)
missing_df = pd.DataFrame({'count': missing, 'pct': missing_pct})
missing_df[missing_df['count'] > 0].sort_values('pct', ascending=False)
```

### Q1.2 — Customer segments + churn rate
```python
bins = [-np.inf, 25000, 60000, np.inf]
labels = ['low', 'mid', 'high']
customers['customer_segment'] = pd.cut(customers['annual_income'], bins=bins, labels=labels)

customers.groupby('customer_segment', observed=True)['churned'].mean()
```

Alternative (without pd.cut):
```python
def segment(income):
    if pd.isna(income):
        return np.nan
    elif income < 25000:
        return 'low'
    elif income < 60000:
        return 'mid'
    else:
        return 'high'

customers['customer_segment'] = customers['annual_income'].apply(segment)
customers.groupby('customer_segment')['churned'].mean()
```

### Q1.3 — Merge + aggregate
```python
order_agg = orders.groupby('customer_id').agg(
    total_spend=('amount', 'sum'),
    order_count=('order_id', 'count')
).reset_index()

customer_orders = customers.merge(order_agg, on='customer_id', how='left')
customer_orders[['total_spend', 'order_count']] = customer_orders[['total_spend', 'order_count']].fillna(0)
customer_orders.nlargest(5, 'total_spend')
```

Key points:
- `how='left'` keeps all customers (even those with no orders)
- `fillna(0)` for customers with no orders
- Aggregate BEFORE merging (more efficient than merge-then-groupby)

---

## Part 2: SQL

### Q2.1 — Top cities by avg spend
```sql
SELECT city, AVG(monthly_spend) AS avg_spend, COUNT(*) AS n_customers
FROM customers
GROUP BY city
HAVING COUNT(*) >= 100
ORDER BY avg_spend DESC
LIMIT 3;
```

### Q2.2 — Most recent order + total per customer
```sql
SELECT c.customer_id,
       MAX(o.order_date) AS last_order_date,
       SUM(o.amount) AS total_amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
ORDER BY total_amount DESC;
```

Key: LEFT JOIN to include customers with no orders.

### Q2.3 — Unsatisfied churned customers
```sql
SELECT c.customer_id, c.city, s.satisfaction_score, s.nps_score
FROM customers c
INNER JOIN surveys s ON c.customer_id = s.customer_id
WHERE s.satisfaction_score < 3
  AND c.churned = 1;
```

---

## Part 3: sklearn

### Q3.1 — Data prep
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = ['age', 'annual_income', 'tenure_months', 'support_tickets', 'monthly_spend']
df = customers[features + ['churned']].dropna()

X = df[features]
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Critical: `.fit_transform()` on train, `.transform()` on test. Never fit on test data.

### Q3.2 — Logistic regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Recall matters more for churn prediction because:
# Missing a churner (false negative) costs more than flagging a non-churner (false positive).
# A missed churner = lost customer = lost revenue.
# A false positive = maybe you send them a retention offer they didn't need (low cost).
```

### Q3.3 — Confusion matrix + ROC-AUC
```python
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.4f}")

disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
disp.plot(cmap='Blues')
plt.title('Churn Prediction — Confusion Matrix')
plt.show()
```
