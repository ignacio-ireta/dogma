# 🧮 Day 2 — Integration Review
*Week 1: Calculus Core | February 4, 2026*

---

## 🎯 Why This Matters for DiffEq

Integration is **the engine** of differential equations. You'll use it to:
- **Solve separable equations:** $\int \frac{dy}{f(y)} = \int g(x) dx$
- **Find particular solutions:** from general form + initial conditions
- **Verify solutions:** check if $y$ satisfies the original equation
- **Apply existence theorems:** many proofs require integral bounds

The instructor's exams won't just ask "find $\int x dx$" — they'll embed integration in existence proofs, solution verification, and boundary value problems. You need these patterns **automatic**.

---

## 📐 The Fundamental Tools

### Basic Antiderivatives
$$\int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)$$

**Examples:**
- $\int x^5 \, dx = \frac{x^6}{6} + C$
- $\int x^{-3} \, dx = \frac{x^{-2}}{-2} + C = -\frac{1}{2x^2} + C$
- $\int \sqrt{x} \, dx = \int x^{1/2} \, dx = \frac{x^{3/2}}{3/2} + C = \frac{2x\sqrt{x}}{3} + C$

**Special case:** $\int \frac{1}{x} \, dx = \ln|x| + C$

### Exponential and Logarithmic
- $\int e^x \, dx = e^x + C$
- $\int e^{kx} \, dx = \frac{1}{k}e^{kx} + C$
- $\int a^x \, dx = \frac{a^x}{\ln(a)} + C$
- $\int \frac{1}{x} \, dx = \ln|x| + C$

### Trigonometric Functions
- $\int \sin(x) \, dx = -\cos(x) + C$
- $\int \cos(x) \, dx = \sin(x) + C$
- $\int \sec^2(x) \, dx = \tan(x) + C$
- $\int \csc^2(x) \, dx = -\cot(x) + C$
- $\int \sec(x)\tan(x) \, dx = \sec(x) + C$

---

## ⛓️ Substitution — The Chain Rule in Reverse

**Pattern recognition:** If you see $f'(x) \cdot g(f(x))$, use $u = f(x)$.

### Method:
1. **Identify inner function** $u$ and find $du$
2. **Rewrite integral** entirely in terms of $u$
3. **Integrate** using basic rules
4. **Substitute back** to original variable

### Examples — Build Pattern Recognition

**Example 1:** $\int (3x+1)^5 \, dx$
- Let $u = 3x+1$, then $du = 3 \, dx$, so $dx = \frac{du}{3}$
- $\int (3x+1)^5 \, dx = \int u^5 \cdot \frac{du}{3} = \frac{1}{3} \int u^5 \, du = \frac{1}{3} \cdot \frac{u^6}{6} + C$
- $= \frac{u^6}{18} + C = \frac{(3x+1)^6}{18} + C$

**Example 2:** $\int x \cos(x^2) \, dx$
- Let $u = x^2$, then $du = 2x \, dx$, so $x \, dx = \frac{du}{2}$
- $\int x \cos(x^2) \, dx = \int \cos(u) \cdot \frac{du}{2} = \frac{1}{2} \sin(u) + C$
- $= \frac{1}{2} \sin(x^2) + C$

**Example 3:** $\int \frac{2x}{x^2+1} \, dx$
- Let $u = x^2+1$, then $du = 2x \, dx$
- $\int \frac{2x}{x^2+1} \, dx = \int \frac{1}{u} \, du = \ln|u| + C$
- $= \ln|x^2+1| + C = \ln(x^2+1) + C$ (since $x^2+1 > 0$ always)

**Example 4:** $\int e^{3x} \, dx$
- Let $u = 3x$, then $du = 3 \, dx$, so $dx = \frac{du}{3}$
- $\int e^{3x} \, dx = \int e^u \cdot \frac{du}{3} = \frac{1}{3} e^u + C = \frac{1}{3} e^{3x} + C$

---

## 🔄 Integration by Parts — When Products Attack

**Formula:** $\int u \, dv = uv - \int v \, du$

**LIATE Priority:** Choose $u$ by this order:
- **L**ogarithmic ($\ln x$)
- **I**nverse trig ($\arcsin x$, $\arctan x$)
- **A**lgebraic ($x$, $x^2$, polynomials)
- **T**rigonometric ($\sin x$, $\cos x$)
- **E**xponential ($e^x$)

### Examples

**Example 1:** $\int x e^x \, dx$
- $u = x$ (algebraic), $dv = e^x \, dx$
- $du = dx$, $v = e^x$
- $\int x e^x \, dx = xe^x - \int e^x \, dx = xe^x - e^x + C = e^x(x-1) + C$

**Example 2:** $\int x \cos x \, dx$
- $u = x$ (algebraic), $dv = \cos x \, dx$
- $du = dx$, $v = \sin x$
- $\int x \cos x \, dx = x \sin x - \int \sin x \, dx = x \sin x + \cos x + C$

**Example 3:** $\int \ln x \, dx$
- $u = \ln x$ (logarithmic), $dv = dx$
- $du = \frac{1}{x} dx$, $v = x$
- $\int \ln x \, dx = x \ln x - \int x \cdot \frac{1}{x} \, dx = x \ln x - \int 1 \, dx = x \ln x - x + C$

---

## 🌀 Partial Fractions — When Rational Functions Resist

For $\frac{P(x)}{Q(x)}$ where degree of $P < $ degree of $Q$:

**Step 1:** Factor $Q(x)$
**Step 2:** Decompose into simpler fractions
**Step 3:** Solve for constants
**Step 4:** Integrate each piece

### Example: $\int \frac{1}{x^2-1} \, dx$

**Factor:** $x^2 - 1 = (x-1)(x+1)$

**Decompose:** $\frac{1}{(x-1)(x+1)} = \frac{A}{x-1} + \frac{B}{x+1}$

**Solve:** $1 = A(x+1) + B(x-1)$
- Let $x = 1$: $1 = A(2) \Rightarrow A = \frac{1}{2}$
- Let $x = -1$: $1 = B(-2) \Rightarrow B = -\frac{1}{2}$

**Integrate:** $\int \frac{1}{x^2-1} \, dx = \int \frac{1/2}{x-1} \, dx + \int \frac{-1/2}{x+1} \, dx$
$= \frac{1}{2}\ln|x-1| - \frac{1}{2}\ln|x+1| + C = \frac{1}{2}\ln\left|\frac{x-1}{x+1}\right| + C$

---

## 🧠 Standard Forms to Memorize

| Integral | Result |
|----------|--------|
| $\int \frac{1}{x^2+a^2} \, dx$ | $\frac{1}{a}\arctan\left(\frac{x}{a}\right) + C$ |
| $\int \frac{1}{\sqrt{a^2-x^2}} \, dx$ | $\arcsin\left(\frac{x}{a}\right) + C$ |
| $\int \frac{1}{x\sqrt{x^2-a^2}} \, dx$ | $\frac{1}{a}\operatorname{arcsec}\left(\frac{x}{a}\right) + C$ |
| $\int \sqrt{a^2-x^2} \, dx$ | $\frac{x}{2}\sqrt{a^2-x^2} + \frac{a^2}{2}\arcsin\left(\frac{x}{a}\right) + C$ |
| $\int xe^{ax} \, dx$ | $\frac{e^{ax}}{a^2}(ax-1) + C$ |

---

## ✍️ Practice Problems

**Basic antiderivatives:**
1. $\int (4x^3 - 2x + 5) \, dx$
2. $\int \left(e^{2x} + \frac{3}{x}\right) \, dx$
3. $\int (\sin x - \cos x) \, dx$

**Substitution:**
4. $\int (2x-3)^4 \, dx$
5. $\int \frac{x}{\sqrt{x^2+1}} \, dx$
6. $\int x e^{x^2} \, dx$
7. $\int \frac{\ln x}{x} \, dx$

**Integration by parts:**
8. $\int x \sin x \, dx$
9. $\int x^2 e^x \, dx$
10. $\int x \ln x \, dx$

**DiffEq-style problems:**
11. Solve: $\frac{dy}{dx} = \frac{x}{y}$ with $y(0) = 2$
12. Verify that $y = Ce^{-x^2/2}$ satisfies $y' + xy = 0$

---

## 🔑 Answers

**Basic:**
1. $x^4 - x^2 + 5x + C$
2. $\frac{1}{2}e^{2x} + 3\ln|x| + C$
3. $-\cos x - \sin x + C$

**Substitution:**
4. $\frac{(2x-3)^5}{10} + C$ (let $u = 2x-3$)
5. $\sqrt{x^2+1} + C$ (let $u = x^2+1$)
6. $\frac{1}{2}e^{x^2} + C$ (let $u = x^2$)
7. $\frac{(\ln x)^2}{2} + C$ (let $u = \ln x$)

**By parts:**
8. $-x\cos x + \sin x + C$
9. $e^x(x^2 - 2x + 2) + C$ (use parts twice)
10. $\frac{x^2\ln x}{2} - \frac{x^2}{4} + C$

**DiffEq-style:**
11. Separate: $y \, dy = x \, dx$ → $\frac{y^2}{2} = \frac{x^2}{2} + C_1$ → $y^2 = x^2 + C$
    Use $y(0) = 2$: $4 = 0 + C \Rightarrow C = 4$
    **Answer:** $y = \pm\sqrt{x^2 + 4}$ (choose $+$ since $y(0) = 2 > 0$)

12. Compute $y' = C \cdot e^{-x^2/2} \cdot (-x) = -Cxe^{-x^2/2}$
    Check: $y' + xy = -Cxe^{-x^2/2} + x(Ce^{-x^2/2}) = -Cxe^{-x^2/2} + Cxe^{-x^2/2} = 0$ ✓

---
