# 📐 Calculus Review — Day 3: Multivariable Basics
*Date: February 5, 2026*
*Phase 0: Prerequisites Refresh (Week 1)*

---

## 🎯 Today's Focus

**Partial derivatives and gradients** — needed for:
- Exact differential equations (DiffEq)
- Multivariate statistics (gradients everywhere in optimization)

---

## 📚 Concepts

### 1. Partial Derivatives

For a function of two variables f(x, y):

**∂f/∂x** = derivative treating y as a constant
**∂f/∂y** = derivative treating x as a constant

**Example:** f(x, y) = x²y + 3xy² - 2y

∂f/∂x = 2xy + 3y² (treat y as constant)
∂f/∂y = x² + 6xy - 2 (treat x as constant)

### 2. The Gradient

The **gradient** ∇f is a vector of all partial derivatives:

∇f = (∂f/∂x, ∂f/∂y)

**Geometric meaning:** The gradient points in the direction of steepest increase.

**Example:** f(x, y) = x² + y²

∇f = (2x, 2y)

At point (1, 2): ∇f = (2, 4) — steepest ascent direction

### 3. Second Partial Derivatives

You can take derivatives again:

- ∂²f/∂x² = take ∂/∂x of ∂f/∂x
- ∂²f/∂y² = take ∂/∂y of ∂f/∂y
- ∂²f/∂x∂y = take ∂/∂x of ∂f/∂y (mixed partial)
- ∂²f/∂y∂x = take ∂/∂y of ∂f/∂x (mixed partial)

**Clairaut's Theorem:** If f is "nice" (continuous second partials), then:
∂²f/∂x∂y = ∂²f/∂y∂x

**Why this matters for DiffEq:** Exact equations require checking if ∂M/∂y = ∂N/∂x

### 4. The Chain Rule (Multivariable)

If z = f(x, y) and x = x(t), y = y(t):

dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)

**This is the multivariable chain rule.** You'll use it constantly.

---

## 📝 Practice Problems

### Problem 1: Basic Partials
Find ∂f/∂x and ∂f/∂y for:

a) f(x, y) = x³ - 2x²y + y³
b) f(x, y) = eˣʸ
c) f(x, y) = sin(x²y)
d) f(x, y) = ln(x² + y²)

### Problem 2: Gradients
Find the gradient ∇f and evaluate at the given point:

a) f(x, y) = x²y - y³ at (2, 1)
b) f(x, y) = eˣcos(y) at (0, π/2)
c) f(x, y) = √(x² + y²) at (3, 4)

### Problem 3: Mixed Partials (Exactness Check)
For each pair M(x,y) and N(x,y), check if ∂M/∂y = ∂N/∂x:

a) M = 2xy + 3, N = x² + 4y
b) M = yeˣʸ, N = xeˣʸ
c) M = y²cos(x), N = 2ysin(x)

*If equal → the equation M dx + N dy = 0 is exact (DiffEq preview!)*

### Problem 4: Chain Rule
Given z = x²y, where x = t² and y = eᵗ, find dz/dt using the chain rule.

### Problem 5: Second Partials
For f(x, y) = x³y² - 2xy³:
a) Find all four second partial derivatives
b) Verify Clairaut's theorem (∂²f/∂x∂y = ∂²f/∂y∂x)

---

## ✅ Solutions

### Problem 1
a) ∂f/∂x = 3x² - 4xy, ∂f/∂y = -2x² + 3y²
b) ∂f/∂x = yeˣʸ, ∂f/∂y = xeˣʸ
c) ∂f/∂x = 2xy·cos(x²y), ∂f/∂y = x²·cos(x²y)
d) ∂f/∂x = 2x/(x² + y²), ∂f/∂y = 2y/(x² + y²)

### Problem 2
a) ∇f = (2xy, x² - 3y²); at (2,1): ∇f = (4, 1)
b) ∇f = (eˣcos(y), -eˣsin(y)); at (0, π/2): ∇f = (0, -1)
c) ∇f = (x/√(x²+y²), y/√(x²+y²)); at (3,4): ∇f = (3/5, 4/5)

### Problem 3
a) ∂M/∂y = 2x, ∂N/∂x = 2x ✅ EXACT
b) ∂M/∂y = eˣʸ + xyeˣʸ, ∂N/∂x = eˣʸ + xyeˣʸ ✅ EXACT
c) ∂M/∂y = 2ycos(x), ∂N/∂x = 2ycos(x) ✅ EXACT

### Problem 4
x = t², y = eᵗ
∂z/∂x = 2xy = 2t²eᵗ
∂z/∂y = x² = t⁴
dx/dt = 2t
dy/dt = eᵗ

dz/dt = (2t²eᵗ)(2t) + (t⁴)(eᵗ) = 4t³eᵗ + t⁴eᵗ = eᵗt³(4 + t)

### Problem 5
f(x, y) = x³y² - 2xy³

a) 
- ∂f/∂x = 3x²y² - 2y³
- ∂f/∂y = 2x³y - 6xy²
- ∂²f/∂x² = 6xy²
- ∂²f/∂y² = 2x³ - 12xy
- ∂²f/∂x∂y = 6x²y - 6y²
- ∂²f/∂y∂x = 6x²y - 6y²

b) ∂²f/∂x∂y = ∂²f/∂y∂x = 6x²y - 6y² ✅ Clairaut verified

---

## 🔗 Connection to DiffEq

When you see an equation like:
(2xy + 3)dx + (x² + 4y)dy = 0

You'll check: Is ∂M/∂y = ∂N/∂x?
- M = 2xy + 3, N = x² + 4y
- ∂M/∂y = 2x, ∂N/∂x = 2x ✅

If yes → **exact equation** → can solve by finding potential function F where ∂F/∂x = M and ∂F/∂y = N.

*You just practiced the exactness check in Problem 3!*

---

## 📌 Key Formulas to Memorize

| Concept | Formula |
|---------|---------|
| Partial derivative | ∂f/∂x = derivative holding other variables constant |
| Gradient | ∇f = (∂f/∂x, ∂f/∂y, ...) |
| Chain rule | dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt) |
| Clairaut's Theorem | ∂²f/∂x∂y = ∂²f/∂y∂x (if continuous) |
| Exactness condition | M dx + N dy = 0 is exact iff ∂M/∂y = ∂N/∂x |

---

## 🏠 Homework

Complete all 5 problems **by hand** before checking solutions. Write out every step.

For extra practice, try Paul's Online Math Notes, Calculus III, "Partial Derivatives" section.

---

*Prerequisites are the foundation. Build them solid.* 🏔️
