# SQL Lesson 2: JOINs — Combining Tables

In pandas, you do `pd.merge()`. In SQL, you do `JOIN`. Same concept.

**Why JOINs?** Data lives in separate tables. Customers in one, their usage in another, their churns in another. JOINs connect them.

---

## Schema Reference

```sql
-- customers
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    signup_date DATE,
    plan_type VARCHAR(20),  -- 'basic', 'premium', 'enterprise'
    region VARCHAR(50),
    monthly_revenue DECIMAL(10,2)
);

-- usage_logs
CREATE TABLE usage_logs (
    log_id INT PRIMARY KEY,
    customer_id INT,
    log_date DATE,
    data_gb DECIMAL(5,2),
    support_tickets INT
);

-- churns
CREATE TABLE churns (
    customer_id INT PRIMARY KEY,
    churn_date DATE,
    reason VARCHAR(100)
);
```

---

## The 4 JOIN Types (you really only need 3)

```
INNER JOIN  — only rows that match in BOTH tables
LEFT JOIN   — all rows from left table + matches from right (NULLs if no match)
RIGHT JOIN  — all rows from right table + matches from left (rarely used)
FULL JOIN   — all rows from both (NULLs where no match)
```

**Visual:**

```
Table A: [1, 2, 3, 4]
Table B: [3, 4, 5, 6]

INNER JOIN → [3, 4]              (intersection)
LEFT JOIN  → [1, 2, 3, 4]        (all of A, matches from B)
FULL JOIN  → [1, 2, 3, 4, 5, 6]  (everything)
```

---

## Basic Syntax

```sql
SELECT 
    c.customer_id,
    c.plan_type,
    u.log_date,
    u.data_gb
FROM customers c
INNER JOIN usage_logs u ON c.customer_id = u.customer_id;
```

Breaking it down:
- `FROM customers c` — "c" is a short alias for customers
- `INNER JOIN usage_logs u` — joining with usage_logs (aliased "u")
- `ON c.customer_id = u.customer_id` — the key they share

---

## LEFT JOIN — The Most Important One

```sql
-- All customers + their churn info (if they churned)
SELECT 
    c.customer_id,
    c.plan_type,
    ch.churn_date,
    ch.reason
FROM customers c
LEFT JOIN churns ch ON c.customer_id = ch.customer_id;
```

Customers who DIDN'T churn will have `NULL` for churn_date and reason. That's the point — you keep everyone.

---

## Finding Non-Matches (LEFT JOIN + IS NULL)

```sql
-- Customers who have NEVER churned
SELECT c.customer_id, c.plan_type
FROM customers c
LEFT JOIN churns ch ON c.customer_id = ch.customer_id
WHERE ch.customer_id IS NULL;
```

This pattern comes up constantly. Memorize it.

---

## Joining Multiple Tables

```sql
-- Customer info + usage + churn status, all in one
SELECT 
    c.customer_id,
    c.plan_type,
    c.monthly_revenue,
    AVG(u.data_gb) AS avg_data_usage,
    AVG(u.support_tickets) AS avg_tickets,
    CASE WHEN ch.customer_id IS NOT NULL THEN 'churned' ELSE 'active' END AS status
FROM customers c
LEFT JOIN usage_logs u ON c.customer_id = u.customer_id
LEFT JOIN churns ch ON c.customer_id = ch.customer_id
GROUP BY c.customer_id, c.plan_type, c.monthly_revenue, ch.customer_id;
```

The `CASE WHEN` — SQL's version of `if/else`. Super useful.

---

## ⚠️ Common JOIN Traps

1. **Duplicating rows:** If a customer has 12 usage_logs, joining gives you 12 rows per customer. Aggregate (GROUP BY) or be aware.

2. **Wrong JOIN type:** Using INNER when you want LEFT = silently dropping rows with no match. Always ask: "do I want to KEEP rows with no match?"

3. **Forgetting the ON clause:** SQL won't stop you from doing a cross join (every row × every row). Millions of rows fast.

4. **Assuming columns exist:** Always check the schema before writing queries! Don't assume there's a `name` column just because it seems logical.

---

## 📝 Exercises

1. Get each customer's ID, plan type, and their **total data usage** (sum of data_gb). Include customers even if they have no usage logs. (Hint: what JOIN type preserves all customers?)

2. Find all customers who churned on the 'basic' plan. Show their customer_id, churn_date, reason, and monthly_revenue.

3. **The thinker:** Calculate the churn rate per region. For each region show: total customers, number who churned, and the churn percentage. (Hint: LEFT JOIN + CASE WHEN + GROUP BY)

---

*Next lesson: Subqueries & CTEs*
