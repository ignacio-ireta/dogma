# SQL Lesson 1: The Basics

## The Core Statement

```sql
SELECT columns
FROM table
WHERE condition
ORDER BY column
LIMIT n;
```

---

## 1. SELECT — What do you want to see?

```sql
-- Everything
SELECT * FROM customers;

-- Specific columns
SELECT customer_id, plan_type, monthly_revenue
FROM customers;

-- Rename columns (aliases)
SELECT 
    customer_id AS id,
    monthly_revenue AS revenue
FROM customers;

-- DISTINCT — remove duplicate rows
SELECT DISTINCT plan_type FROM customers;
-- Returns: 'basic', 'premium', 'enterprise' (one of each)

-- COUNT(DISTINCT ...) — count unique values
SELECT COUNT(DISTINCT region) AS num_regions FROM customers;
```

## 2. WHERE — Filter rows

```sql
-- Exact match
SELECT * FROM customers WHERE plan_type = 'premium';

-- Comparison
SELECT * FROM customers WHERE monthly_revenue > 100;

-- Multiple conditions (AND)
SELECT * FROM customers 
WHERE plan_type = 'premium' 
  AND monthly_revenue > 100;

-- OR
SELECT * FROM customers 
WHERE plan_type = 'premium' 
   OR plan_type = 'enterprise';

-- IN (cleaner multiple OR)
SELECT * FROM customers 
WHERE plan_type IN ('premium', 'enterprise');

-- Date range
SELECT * FROM customers 
WHERE signup_date BETWEEN '2024-01-01' AND '2024-12-31';

-- NULL checks (special syntax!)
SELECT * FROM churns WHERE reason IS NULL;
-- WRONG: WHERE reason = NULL  ← NEVER works!
```

## 3. ORDER BY — Sort results

```sql
-- Ascending (default)
SELECT * FROM customers ORDER BY monthly_revenue;

-- Descending
SELECT * FROM customers ORDER BY monthly_revenue DESC;

-- Multiple columns — cascading priority!
SELECT * FROM customers 
ORDER BY plan_type, monthly_revenue DESC;
```

**How multiple ORDER BY works:**
It sorts by the FIRST column, then breaks ties with the SECOND column:

```
plan_type  | monthly_revenue
-----------+----------------
basic      | 200            ← basics sorted by revenue DESC
basic      | 150
basic      | 80
enterprise | 500            ← enterprises sorted by revenue DESC
enterprise | 300
premium    | 400            ← premiums sorted by revenue DESC
premium    | 120
```

Think of it like sorting a spreadsheet: click "plan_type" first, then "revenue" within each group. No hierarchy, just tie-breaking.

## 4. LIMIT — Cap results

```sql
SELECT * FROM customers 
ORDER BY monthly_revenue DESC 
LIMIT 10;  -- Top 10 highest paying
```

## 5. Aggregations — Summarize data

```sql
-- Count rows
SELECT COUNT(*) FROM customers;

-- Count non-NULL values only
SELECT COUNT(reason) FROM churns;  -- ignores NULLs!

-- Count unique values
SELECT COUNT(DISTINCT region) FROM customers;

-- Sum, Average, Min, Max
SELECT 
    SUM(monthly_revenue) AS total_rev,
    AVG(monthly_revenue) AS avg_rev,
    MIN(monthly_revenue) AS min_rev,
    MAX(monthly_revenue) AS max_rev
FROM customers;
```

### COUNT behavior with NULLs (important!):
- `COUNT(*)` — counts ALL rows, including NULLs
- `COUNT(column)` — counts only non-NULL values in that column
- `COUNT(DISTINCT column)` — counts unique non-NULL values

This is why `COUNT(DISTINCT ch.customer_id)` works for counting churned customers in a LEFT JOIN — non-churned customers have NULL, and COUNT skips them.

## 6. GROUP BY — Aggregate per category

The pandas `.groupby()` equivalent.

```sql
-- Revenue by plan type
SELECT 
    plan_type,
    COUNT(*) AS num_customers,
    AVG(monthly_revenue) AS avg_revenue
FROM customers
GROUP BY plan_type;
```

### 🔑 The Golden Rule:
**If you use an aggregate (SUM, COUNT, AVG, MIN, MAX), every non-aggregated column in SELECT MUST be in GROUP BY.**

```sql
-- ❌ WRONG: region is not aggregated and not in GROUP BY
SELECT region, plan_type, COUNT(*) 
FROM customers 
GROUP BY plan_type;

-- ✅ RIGHT: both non-aggregated columns in GROUP BY
SELECT region, plan_type, COUNT(*) 
FROM customers 
GROUP BY region, plan_type;
```

### How multiple GROUP BY works:

It creates one group for each **unique combination** of values:

```sql
SELECT region, plan_type, COUNT(*) AS n
FROM customers
GROUP BY region, plan_type;
```

```
region    | plan_type  | n
----------+------------+----
North     | basic      | 45    ← one group per combo
North     | premium    | 30
North     | enterprise | 12
South     | basic      | 60
South     | premium    | 25
South     | enterprise | 8
```

**It's NOT a hierarchy or multi-index.** The order doesn't matter — `GROUP BY region, plan_type` gives the same groups as `GROUP BY plan_type, region`. Just a flat table of unique combos with their aggregates.

Think of it like pandas: `df.groupby(['region', 'plan_type']).count()` — same thing, flat result.

## 7. HAVING — Filter AFTER grouping

`WHERE` filters rows *before* grouping. `HAVING` filters *after*.

```sql
SELECT 
    plan_type,
    COUNT(*) AS num_customers
FROM customers
GROUP BY plan_type
HAVING COUNT(*) > 100;
```

**⚠️ Gotcha:** Don't use aliases in HAVING! Works in MySQL, fails in PostgreSQL:
```sql
-- ❌ Risky: HAVING num_customers > 100
-- ✅ Safe:  HAVING COUNT(*) > 100
```

## 8. Column Operations

You can do math and logic directly in SELECT:

```sql
-- Math
SELECT 
    monthly_revenue * 12 AS annual_revenue,
    monthly_revenue * 1.16 AS revenue_with_tax
FROM customers;

-- Conditional logic: CASE WHEN (SQL's if/else)
SELECT 
    customer_id,
    CASE 
        WHEN monthly_revenue > 200 THEN 'high'
        WHEN monthly_revenue > 50  THEN 'medium'
        ELSE 'low'
    END AS revenue_tier
FROM customers;

-- Conditional counting (super useful pattern!)
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN plan_type = 'premium' THEN 1 ELSE 0 END) AS premium_count,
    SUM(CASE WHEN plan_type = 'premium' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS premium_pct
FROM customers;
```

**💡 Use `100.0` not `100`** to force decimal division. `5/3 = 1` but `5/3.0 = 1.667`.

## 🧠 Execution Order

SQL doesn't run top-to-bottom:

```
1. FROM     — pick the table
2. JOIN     — combine tables
3. WHERE    — filter rows
4. GROUP BY — group them
5. HAVING   — filter groups
6. SELECT   — pick columns
7. ORDER BY — sort
8. LIMIT    — cap results
```

That's why you can't use a SELECT alias in WHERE — WHERE runs *before* SELECT.

### 🇲🇽 Mnemonic

**Write order:** **S**elecciona **D**e dónde **J**untamos lo que **Q**ueremos, **A**grupamos **H**abiendo **O**rdenado por **L**ímite

**Execution order:** ¿**D**e dónde **J**untamos lo que **Q**ueremos? Los **A**grupamos **H**abiendo **S**eleccionado el **O**rden por **L**ímite

```
FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

## 📝 Key Rules Summary

1. **Always SELECT what you GROUP BY** — otherwise you get mystery numbers
2. **GROUP BY + aggregation always go together** — if you SUM/COUNT/AVG, you need GROUP BY
3. **Multiple GROUP BY = unique combos, not hierarchy** — order doesn't matter
4. **Multiple ORDER BY = cascading sort** — first column primary, rest break ties
5. **COUNT(\*) includes NULLs, COUNT(column) skips NULLs** — use this to your advantage
6. **Use 100.0 for percentages** — avoid integer division
7. **Don't use aliases in HAVING** — repeat the aggregate function
8. **Table.column, not column.table** — think Python: `object.attribute`

---

*Next lesson: JOINs — combining tables → 02-joins.md*
