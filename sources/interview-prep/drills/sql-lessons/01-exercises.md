# SQL Exercises — Lessons 1 & 2

## Schema Reference

```sql
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    signup_date DATE,
    plan_type VARCHAR(20),  -- 'basic', 'premium', 'enterprise'
    region VARCHAR(50),
    monthly_revenue DECIMAL(10,2)
);

CREATE TABLE usage_logs (
    log_id INT PRIMARY KEY,
    customer_id INT,
    log_date DATE,
    data_gb DECIMAL(5,2),
    support_tickets INT
);

CREATE TABLE churns (
    customer_id INT PRIMARY KEY,
    churn_date DATE,
    reason VARCHAR(100)
);
```

---

## Lesson 1 Exercises

### Basics (SELECT, WHERE, ORDER BY, GROUP BY)

1. Get all premium customers sorted by revenue (highest first)
2. Count how many customers are in each region
3. Find the average monthly revenue per plan type, only show plans where avg > 50
4. Get the 3 most recent signups

### Key mistakes made:
- ❌ Forgot to SELECT what you GROUP BY (mystery numbers)
- ❌ Forgot LIMIT
- ⚠️ Using aliases in HAVING (works MySQL, fails PostgreSQL)

---

## Lesson 2 Exercises

### JOINs

1. Get each customer's ID, plan type, and total data usage. Include customers with no usage logs.
2. Find all customers who churned on the 'basic' plan. Show customer_id, churn_date, reason, monthly_revenue.
3. Calculate churn rate per region: total customers, churned count, percentage.

### Key mistakes made:
- ❌ `customer_id.c` → should be `c.customer_id` (table.column, like Python object.attribute)
- ❌ WHERE before JOIN → JOIN comes first (FROM → JOIN → WHERE)
- ❌ Missing GROUP BY when using aggregates
- ⚠️ Division flipped (total/churned instead of churned/total)
- ✅ Used COUNT(DISTINCT ch.customer_id) correctly — COUNT skips NULLs

---

## Speed Round

1. Customers per plan type, ordered by count DESC — ✅
2. Average monthly revenue of churned customers — ✅
3. Regions with >50 premium customers — ⚠️ forgot `WHERE plan_type = 'premium'`
4. Customer ID + churn reason (show 'active' if no churn) — ✅ (CASE WHEN or COALESCE)
5. Total support tickets per plan type — ⚠️ used COUNT instead of SUM

### Key lessons:
- **WHERE vs HAVING:** WHERE filters rows BEFORE grouping, HAVING filters AFTER
- **COUNT vs SUM:** COUNT counts rows, SUM adds values. `support_tickets = 5` → COUNT says 1, SUM says 5
- **COALESCE(x, 'default')** = return first non-NULL value (shorthand for CASE WHEN x IS NOT NULL)

---

## Rules Learned

1. Always SELECT what you GROUP BY
2. GROUP BY + aggregation always go together
3. Multiple GROUP BY = unique combos (not hierarchy, order irrelevant)
4. Multiple ORDER BY = cascading sort (order matters!)
5. COUNT(*) includes NULLs, COUNT(column) skips NULLs
6. Use 100.0 for percentages (avoid integer division)
7. Don't use aliases in HAVING — repeat the function
8. table.column, not column.table
9. WHERE before GROUP BY for pre-filtering, HAVING for post-filtering
10. COUNT counts rows, SUM adds values

---

*Day 1 of SQL: Mar 5, 2026*
