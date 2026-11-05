# 🧮 Day 1 — Derivatives Refresher
*Week 1: Calculus Core | February 3, 2026*

---

## 🎯 Why This Matters for DiffEq

Every differential equation is about **derivatives**. If your derivative skills aren't automatic, you'll waste brainpower on mechanics instead of understanding the actual problem.

The instructor won't ask "find dy/dx" — he'll ask you to prove existence, study monotonicity, find extrema. But ALL of that requires derivatives to be second nature.

**Today's goal:** Make these rules feel like breathing.

---

## 📐 The Core Rules (You Know These)

### Power Rule
$$\frac{d}{dx}[x^n] = nx^{n-1}$$

**Examples:**
- $\frac{d}{dx}[x^5] = 5x^4$
- $\frac{d}{dx}[x^{-2}] = -2x^{-3} = \frac{-2}{x^3}$
- $\frac{d}{dx}[\sqrt{x}] = \frac{d}{dx}[x^{1/2}] = \frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}$

### Sum/Difference Rule
$$\frac{d}{dx}[f(x) \pm g(x)] = f'(x) \pm g'(x)$$

Just differentiate term by term.

### Constant Multiple Rule
$$\frac{d}{dx}[c \cdot f(x)] = c \cdot f'(x)$$

Constants slide out.

### Product Rule
$$\frac{d}{dx}[f(x) \cdot g(x)] = f'(x)g(x) + f(x)g'(x)$$

**Mnemonic:** "First times derivative of second, plus second times derivative of first"

**Example:**
$\frac{d}{dx}[x^2 \cdot \sin(x)] = 2x \cdot \sin(x) + x^2 \cdot \cos(x)$

### Quotient Rule
$$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$

**Mnemonic:** "Low d-high minus high d-low, over low squared"

**Example:**
$\frac{d}{dx}\left[\frac{x^2}{e^x}\right] = \frac{2x \cdot e^x - x^2 \cdot e^x}{e^{2x}} = \frac{e^x(2x - x^2)}{e^{2x}} = \frac{2x - x^2}{e^x}$

---

## ⛓️ Chain Rule — The One That Matters Most

$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

**In words:** Derivative of the outside (keeping inside unchanged) × derivative of the inside.

### Examples — Build Intuition

**Example 1:** $\frac{d}{dx}[(3x+1)^5]$
- Outside: $(\cdot)^5$ → derivative: $5(\cdot)^4$
- Inside: $3x+1$ → derivative: $3$
- **Answer:** $5(3x+1)^4 \cdot 3 = 15(3x+1)^4$

**Example 2:** $\frac{d}{dx}[\sin(x^2)]$
- Outside: $\sin(\cdot)$ → derivative: $\cos(\cdot)$
- Inside: $x^2$ → derivative: $2x$
- **Answer:** $\cos(x^2) \cdot 2x = 2x\cos(x^2)$

**Example 3:** $\frac{d}{dx}[e^{3x^2}]$
- Outside: $e^{(\cdot)}$ → derivative: $e^{(\cdot)}$
- Inside: $3x^2$ → derivative: $6x$
- **Answer:** $e^{3x^2} \cdot 6x = 6xe^{3x^2}$

**Example 4 (nested):** $\frac{d}{dx}[\sqrt{\sin(x)}]$
- Outer: $\sqrt{\cdot}$ → derivative: $\frac{1}{2\sqrt{\cdot}}$
- Inner: $\sin(x)$ → derivative: $\cos(x)$
- **Answer:** $\frac{1}{2\sqrt{\sin(x)}} \cdot \cos(x) = \frac{\cos(x)}{2\sqrt{\sin(x)}}$

---

## 🔄 Implicit Differentiation — When y is Trapped

When you can't solve for $y$ explicitly, differentiate both sides and solve for $\frac{dy}{dx}$.

**Key insight:** Every time you differentiate a $y$ term, multiply by $\frac{dy}{dx}$ (chain rule!).

### Example 1: Circle
$x^2 + y^2 = 25$

Differentiate both sides:
$2x + 2y\frac{dy}{dx} = 0$

Solve for $\frac{dy}{dx}$:
$\frac{dy}{dx} = -\frac{x}{y}$

### Example 2: Product with y
$xy = 1$

Differentiate (product rule on left):
$y + x\frac{dy}{dx} = 0$

Solve:
$\frac{dy}{dx} = -\frac{y}{x}$

### Example 3: Mixed terms
$x^2 + xy + y^2 = 7$

Differentiate:
$2x + (y + x\frac{dy}{dx}) + 2y\frac{dy}{dx} = 0$

Collect $\frac{dy}{dx}$ terms:
$2x + y + \frac{dy}{dx}(x + 2y) = 0$

Solve:
$\frac{dy}{dx} = -\frac{2x + y}{x + 2y}$

---

## 🧠 Derivatives You Should Know Cold

| Function | Derivative |
|----------|------------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $\frac{1}{x}$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |
| $\tan(x)$ | $\sec^2(x)$ |
| $e^{kx}$ | $ke^{kx}$ |
| $\ln(kx)$ | $\frac{1}{x}$ |
| $a^x$ | $a^x \ln(a)$ |

---

## ✍️ Practice Problems

**Basic (warm-up):**
1. $\frac{d}{dx}[3x^4 - 2x^2 + 7]$
2. $\frac{d}{dx}[x^3 \cdot e^x]$
3. $\frac{d}{dx}\left[\frac{x^2 + 1}{x}\right]$

**Chain rule:**
4. $\frac{d}{dx}[(2x-5)^7]$
5. $\frac{d}{dx}[\cos(3x)]$
6. $\frac{d}{dx}[e^{-x^2}]$
7. $\frac{d}{dx}[\ln(x^2 + 1)]$

**Implicit differentiation:**
8. Find $\frac{dy}{dx}$ for: $x^2 - y^2 = 16$
9. Find $\frac{dy}{dx}$ for: $x^3 + y^3 = 6xy$
10. Find $\frac{dy}{dx}$ for: $\sin(y) = x$

---

## 🔑 Answers

**Basic:**
1. $12x^3 - 4x$
2. $3x^2 e^x + x^3 e^x = e^x(3x^2 + x^3) = x^2 e^x(3 + x)$
3. Simplify first: $x + \frac{1}{x}$ → derivative: $1 - \frac{1}{x^2}$

**Chain rule:**
4. $7(2x-5)^6 \cdot 2 = 14(2x-5)^6$
5. $-\sin(3x) \cdot 3 = -3\sin(3x)$
6. $e^{-x^2} \cdot (-2x) = -2xe^{-x^2}$
7. $\frac{1}{x^2+1} \cdot 2x = \frac{2x}{x^2+1}$

**Implicit:**
8. $2x - 2y\frac{dy}{dx} = 0$ → $\frac{dy}{dx} = \frac{x}{y}$
9. $3x^2 + 3y^2\frac{dy}{dx} = 6y + 6x\frac{dy}{dx}$
   → $3y^2\frac{dy}{dx} - 6x\frac{dy}{dx} = 6y - 3x^2$
   → $\frac{dy}{dx} = \frac{6y - 3x^2}{3y^2 - 6x} = \frac{2y - x^2}{y^2 - 2x}$
10. $\cos(y)\frac{dy}{dx} = 1$ → $\frac{dy}{dx} = \frac{1}{\cos(y)} = \sec(y)$

---
