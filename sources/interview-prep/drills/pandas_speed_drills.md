# Pandas Speed Drills

*Set a timer. 15 problems. Target: under 20 minutes total.*
*Use `customers.csv` and `orders.csv` from this folder.*

---

## Setup
```python
import pandas as pd
import numpy as np
customers = pd.read_csv("customers.csv")
orders = pd.read_csv("orders.csv")
```

---

## Drill 1: Basic Inspection (30 sec each)
```python
# 1. How many rows and columns in customers?

# 2. What are the dtypes of each column?

# 3. How many nulls per column?

# 4. How many unique values in the 'city' column?

# 5. What's the mean, median, and std of 'annual_income'?
```

## Drill 2: Filtering (1 min each)
```python
# 6. Select all customers from "Monterrey" with annual_income > 50000

# 7. Select all customers who have NULL plan_type

# 8. Select customers with age between 25 and 35 (inclusive)

# 9. Select customers whose gender is either "M" or "F" (case-sensitive)

# 10. Count how many customers have support_tickets > 5
```

## Drill 3: GroupBy & Aggregation (2 min each)
```python
# 11. Average monthly_spend per city, sorted descending

# 12. For each plan_type: count of customers, mean tenure_months, churn rate

# 13. From orders: total revenue per category, sorted descending

# 14. From orders: for each customer_id, their order count and most recent order_date

# 15. Merge customers with order counts from #14. Customers with no orders should show 0.
```

---

## Solutions

```python
# 1
customers.shape

# 2
customers.dtypes

# 3
customers.isnull().sum()

# 4
customers["city"].nunique()

# 5
customers["annual_income"].agg(["mean", "median", "std"])

# 6
customers[(customers["city"] == "Monterrey") & (customers["annual_income"] > 50000)]

# 7
customers[customers["plan_type"].isnull()]

# 8
customers[customers["age"].between(25, 35)]

# 9
customers[customers["gender"].isin(["M", "F"])]

# 10
(customers["support_tickets"] > 5).sum()

# 11
customers.groupby("city")["monthly_spend"].mean().sort_values(ascending=False)

# 12
customers.groupby("plan_type").agg(
    count=("customer_id", "count"),
    avg_tenure=("tenure_months", "mean"),
    churn_rate=("churned", "mean"),
)

# 13
orders.groupby("category")["amount"].sum().sort_values(ascending=False)

# 14
order_stats = orders.groupby("customer_id").agg(
    order_count=("order_id", "count"),
    last_order=("order_date", "max"),
).reset_index()

# 15
merged = customers.merge(order_stats, on="customer_id", how="left")
merged["order_count"] = merged["order_count"].fillna(0).astype(int)
```

---

## Bonus Round: Time Pressure Drill

*Set timer for 5 minutes. Complete as many as possible:*

```python
# B1. One-hot encode the 'city' column (drop_first=True)

# B2. Replace all negative ages with NaN, then fill with median

# B3. Create a new column 'income_bracket': "low" if < 30000, "mid" if 30000-80000, "high" if > 80000

# B4. Convert signup_date to datetime, extract the year into a new column

# B5. Find the top 5 customers by total order amount (need to merge + aggregate)
```

### Bonus Solutions
```python
# B1
pd.get_dummies(customers, columns=["city"], drop_first=True)

# B2
customers.loc[customers["age"] < 0, "age"] = np.nan
customers["age"] = customers["age"].fillna(customers["age"].median())

# B3
customers["income_bracket"] = pd.cut(
    customers["annual_income"],
    bins=[0, 30000, 80000, float("inf")],
    labels=["low", "mid", "high"],
)
# OR
customers["income_bracket"] = np.where(
    customers["annual_income"] < 30000, "low",
    np.where(customers["annual_income"] <= 80000, "mid", "high")
)

# B4
customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
customers["signup_year"] = customers["signup_date"].dt.year

# B5
top5 = (
    orders.groupby("customer_id")["amount"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
```
