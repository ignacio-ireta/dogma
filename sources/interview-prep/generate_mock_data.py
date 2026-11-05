"""
Generate mock datasets for SQL and pandas interview practice.
Scenario: E-commerce customer churn prediction.
Deliberately messy — nulls, duplicates, mixed types, date issues.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
N = 5000

# --- customers.csv ---
customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, N + 1)]

# Introduce 50 duplicate customer IDs
dup_indices = np.random.choice(range(N), size=50, replace=False)
for idx in dup_indices:
    customer_ids[idx] = customer_ids[max(0, idx - 1)]

cities = np.random.choice(
    ["Chihuahua", "Guadalajara", "Monterrey", "Puebla", "Tijuana",
     "León", "Cancún", "Mérida", "Querétaro", None],
    size=N, p=[0.25, 0.15, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.05]
)

genders = np.random.choice(["M", "F", "Other", "m", "f", " ", None], size=N,
                            p=[0.35, 0.35, 0.05, 0.05, 0.05, 0.02, 0.13])

ages = np.random.normal(35, 12, N).astype(int)
ages = np.clip(ages, 18, 80)
# Inject some bad ages
bad_age_idx = np.random.choice(range(N), size=30, replace=False)
ages = ages.astype(float)
ages[bad_age_idx[:15]] = np.nan
ages[bad_age_idx[15:20]] = -1
ages[bad_age_idx[20:25]] = 0
ages[bad_age_idx[25:]] = 150

income = np.random.lognormal(mean=10.5, sigma=0.8, size=N).astype(int)
null_income = np.random.choice(range(N), size=200, replace=False)
income = income.astype(float)
income[null_income] = np.nan

signup_dates = []
for _ in range(N):
    base = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1800))
    signup_dates.append(base.strftime("%Y-%m-%d"))
# Inject some bad dates
for i in np.random.choice(range(N), size=20, replace=False):
    signup_dates[i] = np.random.choice(["not_a_date", "2025/13/01", "", "N/A"])

plan_types = np.random.choice(["Basic", "Premium", "Enterprise", "basic", "PREMIUM", None],
                               size=N, p=[0.3, 0.25, 0.15, 0.1, 0.1, 0.1])

tenure_months = np.random.exponential(scale=18, size=N).astype(int)
tenure_months = np.clip(tenure_months, 0, 60)

support_tickets = np.random.poisson(lam=2, size=N)

monthly_spend = np.random.lognormal(mean=4.5, sigma=0.6, size=N).round(2)
null_spend = np.random.choice(range(N), size=80, replace=False)
monthly_spend[null_spend] = np.nan

# Target: churn (imbalanced ~15%)
churn_prob = 1 / (1 + np.exp(-(
    -2.0
    + 0.3 * (support_tickets - 2)
    - 0.05 * tenure_months
    + 0.5 * (np.array([1 if p in ["Basic", "basic"] else 0 for p in plan_types]))
    - 0.00001 * np.nan_to_num(income, nan=40000)
    + np.random.normal(0, 0.3, N)
)))
churned = (np.random.random(N) < churn_prob).astype(int)

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "city": cities,
    "gender": genders,
    "age": ages,
    "annual_income": income,
    "signup_date": signup_dates,
    "plan_type": plan_types,
    "tenure_months": tenure_months,
    "support_tickets": support_tickets,
    "monthly_spend": monthly_spend,
    "churned": churned,
})

customers.to_csv("data/customers.csv", index=False)
print(f"customers.csv: {len(customers)} rows, {customers.columns.tolist()}")
print(f"  Nulls:\n{customers.isnull().sum()}")
print(f"  Churn rate: {churned.mean():.1%}")
print(f"  Duplicate customer_ids: {customers.customer_id.duplicated().sum()}")

# --- orders.csv ---
n_orders = 15000
order_customer_ids = np.random.choice(customer_ids[:N], size=n_orders)
order_dates = [(datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 730))).strftime("%Y-%m-%d")
               for _ in range(n_orders)]
order_amounts = np.random.lognormal(mean=3.5, sigma=1.0, size=n_orders).round(2)
categories = np.random.choice(
    ["Electronics", "Clothing", "Home", "Food", "Sports", "Books"],
    size=n_orders, p=[0.2, 0.25, 0.15, 0.2, 0.1, 0.1]
)
# Some orders have no amount (returns/cancellations)
null_orders = np.random.choice(range(n_orders), size=300, replace=False)
order_amounts[null_orders] = np.nan

orders = pd.DataFrame({
    "order_id": [f"O{str(i).zfill(6)}" for i in range(1, n_orders + 1)],
    "customer_id": order_customer_ids,
    "order_date": order_dates,
    "amount": order_amounts,
    "category": categories,
})

orders.to_csv("data/orders.csv", index=False)
print(f"\norders.csv: {len(orders)} rows, {orders.columns.tolist()}")
print(f"  Null amounts: {orders.amount.isnull().sum()}")
print(f"  Unique customers with orders: {orders.customer_id.nunique()}")

# --- satisfaction_surveys.csv ---
n_surveys = 3000
survey_customer_ids = np.random.choice(customer_ids[:N], size=n_surveys, replace=False)
satisfaction_scores = np.random.choice([1, 2, 3, 4, 5, None], size=n_surveys,
                                        p=[0.05, 0.1, 0.25, 0.35, 0.2, 0.05])
nps = np.random.randint(0, 11, size=n_surveys).astype(float)
nps[np.random.choice(range(n_surveys), size=100, replace=False)] = np.nan
response_dates = [(datetime(2024, 6, 1) + timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d")
                  for _ in range(n_surveys)]

surveys = pd.DataFrame({
    "customer_id": survey_customer_ids,
    "satisfaction_score": satisfaction_scores,
    "nps_score": nps,
    "survey_date": response_dates,
})

surveys.to_csv("data/surveys.csv", index=False)
print(f"\nsurveys.csv: {len(surveys)} rows, {surveys.columns.tolist()}")
print(f"  Null satisfaction: {surveys.satisfaction_score.isnull().sum()}")
print(f"  Null NPS: {surveys.nps_score.isnull().sum()}")

print("\n✅ All datasets generated in data/")
