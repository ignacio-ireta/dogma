# 📐 Calculus Review — Day 4: Derivatives & Key Theorems
*Date: February 9, 2026*
*Phase 0: Prerequisites Refresh (Week 2)*

---

## 🎯 Today's Focus

**Derivative theorems the instructor actually uses on exams:**
- Rolle's Theorem
- Mean Value Theorem (Lagrange)
- L'Hôpital's Rule

From his exams: "Use Lagrange's theorem to prove..." appears constantly. You need to know these cold.

---

## 📚 Review: Derivative Rules (Quick Refresh)

Make sure these are automatic:

| Rule | Formula | Example |
|------|---------|---------|
| Power | d/dx[xⁿ] = nxⁿ⁻¹ | d/dx[x³] = 3x² |
| Product | d/dx[f·g] = f'g + fg' | d/dx[x·eˣ] = eˣ + xeˣ |
| Quotient | d/dx[f/g] = (f'g - fg')/g² | — |
| Chain | d/dx[f(g(x))] = f'(g(x))·g'(x) | d/dx[sin(x²)] = cos(x²)·2x |
| Exponential | d/dx[eˣ] = eˣ | — |
| Logarithm | d/dx[ln(x)] = 1/x | — |
| Trig | d/dx[sin(x)] = cos(x) | d/dx[cos(x)] = -sin(x) |

---

## 🔧 Rolle's Theorem

**Statement:** If f is:
1. Continuous on [a, b]
2. Differentiable on (a, b)
3. f(a) = f(b)

Then there exists at least one c ∈ (a, b) where **f'(c) = 0**

**Geometric meaning:** If a smooth curve starts and ends at the same height, somewhere in between it must have a horizontal tangent.

**Picture:**
```
    f(a) ●───────●─────────● f(b)
         │       ╱╲        │
         │      ╱  ╲       │
         │     ╱    ╲      │
         │    ╱      ╲     │
         └───●────────●────┘
             a    c    b
                 ↑
            f'(c) = 0
```

### Example:
f(x) = x² - 4x on [0, 4]

Check conditions:
- Continuous? ✅ (polynomial)
- Differentiable? ✅ (polynomial)
- f(0) = f(4)? ✅ (both = 0)

Find c: f'(x) = 2x - 4 = 0 → x = 2

So c = 2, and indeed f'(2) = 0. ✅

---

## 🔧 Mean Value Theorem (Lagrange)

**Statement:** If f is:
1. Continuous on [a, b]
2. Differentiable on (a, b)

Then there exists at least one c ∈ (a, b) where:

$$f'(c) = \frac{f(b) - f(a)}{b - a}$$

**Geometric meaning:** There's a point where the tangent line is parallel to the secant line connecting the endpoints.

**Picture:**
```
         f(b)●
            ╱
           ╱  ← tangent at c (parallel to secant)
          ╱
      ●──╱────────●
    f(a)          
      a     c     b
```

**Why the instructor loves it:** It connects the derivative (local behavior) to the overall change (global behavior). Perfect for proofs.

### Example:
f(x) = x³ on [0, 2]

Find c such that: f'(c) = (f(2) - f(0))/(2 - 0) = (8 - 0)/2 = 4

f'(x) = 3x² = 4 → x = √(4/3) = 2/√3 ≈ 1.15

So c = 2/√3 ∈ (0, 2). ✅

---

## 🔧 L'Hôpital's Rule

**When to use:** For indeterminate forms **0/0** or **∞/∞**

**Statement:** If lim(x→a) f(x)/g(x) gives 0/0 or ∞/∞, then:

$$\lim_{x→a} \frac{f(x)}{g(x)} = \lim_{x→a} \frac{f'(x)}{g'(x)}$$

(if the right side exists)

### Examples:

**Example 1:** lim(x→0) sin(x)/x

Check: sin(0)/0 = 0/0 ✅ (indeterminate)

Apply L'Hôpital: lim(x→0) cos(x)/1 = 1

**Example 2:** lim(x→∞) x²/eˣ

Check: ∞/∞ ✅ (indeterminate)

Apply L'Hôpital: lim(x→∞) 2x/eˣ = still ∞/∞

Apply again: lim(x→∞) 2/eˣ = 0

**Example 3:** lim(x→0) (eˣ - 1)/x

Check: (1-1)/0 = 0/0 ✅

Apply L'Hôpital: lim(x→0) eˣ/1 = 1

---

## ⚠️ Common Mistakes

1. **Using L'Hôpital when not 0/0 or ∞/∞**
   - Wrong: lim(x→1) x²/x = lim 2x/1 = 2 ❌
   - Right: lim(x→1) x²/x = lim x = 1 ✅

2. **Forgetting to check MVT conditions**
   - f(x) = |x| on [-1, 1] — NOT differentiable at 0!

3. **Rolle vs MVT confusion**
   - Rolle: f(a) = f(b) required, conclusion is f'(c) = 0
   - MVT: no f(a) = f(b) needed, conclusion is f'(c) = slope

---

## 📝 Practice Problems

### Problem 1: Rolle's Theorem
For each function, verify Rolle's theorem applies and find c:

a) f(x) = x³ - 3x on [-√3, √3]
b) f(x) = sin(x) on [0, π]
c) f(x) = x² - 2x - 3 on [-1, 3]

### Problem 2: Mean Value Theorem
Find the value of c guaranteed by MVT:

a) f(x) = x² + 1 on [0, 2]
b) f(x) = √x on [0, 4]
c) f(x) = 1/x on [1, 2]
d) f(x) = eˣ on [0, 1]

### Problem 3: L'Hôpital's Rule
Evaluate these limits:

a) lim(x→0) (1 - cos(x))/x²
b) lim(x→0) (eˣ - 1 - x)/x²
c) lim(x→∞) ln(x)/x
d) lim(x→0⁺) x·ln(x) [Hint: rewrite as ln(x)/(1/x)]
e) lim(x→0) tan(x)/x

### Problem 4: Exam-Style Proof
**Use Lagrange's theorem to prove:**

If f is differentiable on [a, b] and f'(x) > 0 for all x ∈ (a, b), then f is strictly increasing on [a, b].

*Hint: Take any x₁ < x₂ in [a, b]. Apply MVT to show f(x₂) > f(x₁).*

### Problem 5: Exam-Style Proof
**Use Rolle's theorem to prove:**

The equation x³ - 3x + 1 = 0 has at most 3 real roots.

*Hint: If it had 4 roots, what would Rolle's theorem say about f'(x)?*

---

## 📌 Summary: What Exams Test

| Theorem | When to Use | Key Formula |
|---------|-------------|-------------|
| **Rolle** | f(a) = f(b), need to find critical point | f'(c) = 0 |
| **MVT (Lagrange)** | Connect local to global, proofs about monotonicity | f'(c) = (f(b)-f(a))/(b-a) |
| **L'Hôpital** | 0/0 or ∞/∞ limits | lim f/g = lim f'/g' |

**On the instructor's exams, you'll see:**
- "Use Lagrange's theorem to prove that..."
- "Show using Rolle's theorem that..."
- Analysis of ODEs using MVT

---

## 🏠 Homework

1. Complete all practice problems above
2. For Problem 4 and 5: write out FULL formal proofs
3. Review: Can you state all three theorems from memory?

---

*These theorems are your proof toolbox. Master them.* 🔧
