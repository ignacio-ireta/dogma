# Probability From Zero — Ground-Up Lecture

*Because "refresher" implies you had it once. Let's just build it right.*

---

## Part 0: What Is Probability?

Probability = a number between 0 and 1 that measures how likely something is to happen.

- **0** = impossible (you will not roll a 7 on a standard die)
- **1** = certain (the sun will rise tomorrow, probably)
- **0.5** = coin flip

That's it. Everything else is just techniques for *calculating* that number in different situations.

**The fundamental formula:**

```
P(something) = (ways it can happen) / (total possible outcomes)
```

**Example:** What's the probability of rolling a 4 on a fair die?
- Ways to roll a 4: **1** (there's one face with a 4)
- Total outcomes: **6** (faces 1-6)
- P(rolling 4) = 1/6 ≈ 0.167

**Example:** What's the probability of drawing a Heart from a standard deck?
- Hearts in a deck: **13**
- Total cards: **52**
- P(Heart) = 13/52 = 1/4 = 0.250

That's your foundation. Everything below builds on this.

---

## Part 1: The Complement — "At Least One" Problems

The **complement** of an event is "everything else." If something has a 30% chance of happening, it has a 70% chance of NOT happening.

```
P(NOT A) = 1 - P(A)
```

**Why this matters:** When a problem asks "what's the probability of *at least one*..." — don't try to count all the ways it happens. Count the ways it DOESN'T happen and subtract from 1.

**Example:** You flip a coin 3 times. P(at least one heads)?

The hard way: count HHH, HHT, HTH, THH, HTT, THT, TTH... nightmare.

The easy way:
- P(NO heads in 3 flips) = P(all tails) = (1/2)³ = 1/8 = 0.125
- P(at least one heads) = 1 - 0.125 = **0.875**

**🔑 Rule:** Whenever you see "at least one" or "at least k", think complement first.

---

## Part 2: AND — Two Things Both Happening

When you want BOTH event A AND event B to happen, you multiply.

### Case 1: Independent events (one doesn't affect the other)

```
P(A AND B) = P(A) × P(B)
```

**Example:** You flip a coin AND roll a die. P(heads AND a 6)?
- P(heads) = 1/2
- P(6) = 1/6
- P(heads AND 6) = 1/2 × 1/6 = 1/12 ≈ 0.083

The coin doesn't care what the die does. They're independent.

### Case 2: Dependent events (one DOES affect the other)

```
P(A AND B) = P(A) × P(B | A)
```

That `P(B | A)` reads as "probability of B *given that* A happened."

**Example:** Draw 2 cards from a deck WITHOUT replacement. P(both are Aces)?
- P(1st card is Ace) = 4/52
- P(2nd card is Ace | 1st was Ace) = 3/51 ← deck changed!
- P(both Aces) = (4/52) × (3/51) = 12/2652 ≈ **0.005**

**⚠️ CodeSignal loves this.** Card problems without replacement are confirmed question types. The key: after each draw, the denominator drops by 1, and the numerator drops by 1 IF you drew from that category.

More examples (drill these):
- P(Ace then King, no replacement) = (4/52) × (4/51) = 16/2652 ≈ **0.006**
  - Why 4/51 for King? You drew an Ace, not a King. Still 4 Kings left. But only 51 cards.
- P(Heart then Heart, no replacement) = (13/52) × (12/51) = 156/2652 ≈ **0.059**
  - Numerator drops because you already drew one Heart.

---

## Part 3: OR — Either Thing Happening

When you want event A OR event B (or both):

```
P(A OR B) = P(A) + P(B) - P(A AND B)
```

You subtract `P(A AND B)` because otherwise you'd count the overlap twice.

**Example:** Draw one card. P(Heart OR King)?
- P(Heart) = 13/52
- P(King) = 4/52
- P(Heart AND King) = 1/52 (the King of Hearts)
- P(Heart OR King) = 13/52 + 4/52 - 1/52 = 16/52 ≈ **0.308**

**Special case — mutually exclusive events** (can't both happen):
If A and B can't both occur, then P(A AND B) = 0, so:

```
P(A OR B) = P(A) + P(B)
```

**Example:** P(rolling a 2 OR a 5) = 1/6 + 1/6 = 2/6 ≈ 0.333
(You can't roll both a 2 and a 5 on one die.)

---

## Part 4: Conditional Probability — Given That...

This is where it gets interesting. Conditional probability answers: "If I already know something happened, how does that change the odds?"

```
P(A | B) = P(A AND B) / P(B)
```

Reads as: "Probability of A, given B."

**Example:** In a class of 30 students:
- 18 passed the exam
- 12 are female
- 8 are female AND passed

P(passed | female) = P(passed AND female) / P(female) = 8/30 ÷ 12/30 = 8/12 = **0.667**

Translation: if you pick a random female student, there's a 66.7% chance she passed.

Notice this is DIFFERENT from P(female | passed):
P(female | passed) = 8/30 ÷ 18/30 = 8/18 = **0.444**

**🔑 The order matters.** P(A|B) ≠ P(B|A). This is the #1 mistake people make in probability.

---

## Part 5: Bayes' Theorem — Flipping the Condition

Bayes is just a formula for when you know P(B|A) but NEED P(A|B). It lets you "flip" the condition.

```
P(A|B) = P(B|A) × P(A) / P(B)
```

But the tricky part is usually calculating P(B) — the total probability. You expand it:

```
P(B) = P(B|A) × P(A) + P(B|not A) × P(not A)
```

### Let me walk through this with a real example, step by step.

**The Factory Problem (this EXACT pattern shows up on CodeSignal):**

A factory has two machines:
- Machine A makes **60%** of all products, with a **2%** defect rate
- Machine B makes **40%** of all products, with a **3%** defect rate

You pick a random product. It's defective. What's the probability it came from Machine B?

**Step 1: Write down what you know.**
- P(A) = 0.60 — probability product came from Machine A
- P(B) = 0.40 — probability product came from Machine B
- P(defective | A) = 0.02 — defect rate if it came from A
- P(defective | B) = 0.03 — defect rate if it came from B

**Step 2: What are we solving for?**
- P(B | defective) — "it's defective, what's the chance it came from B?"

**Step 3: Find P(defective) — the total defect rate across the whole factory.**
```
P(defective) = P(def|A) × P(A) + P(def|B) × P(B)
P(defective) = 0.02 × 0.60 + 0.03 × 0.40
P(defective) = 0.012 + 0.012
P(defective) = 0.024
```

**Step 4: Apply Bayes.**
```
P(B | defective) = P(def|B) × P(B) / P(defective)
P(B | defective) = 0.03 × 0.40 / 0.024
P(B | defective) = 0.012 / 0.024
P(B | defective) = 0.500
```

**Answer: 0.500**

Despite Machine B having a higher defect rate, it only makes 40% of products. So a defective product is equally likely to come from either machine.

### The Disease Test (another classic Bayes pattern):

- 1% of people have a disease: P(disease) = 0.01
- Test catches 99% of sick people: P(positive | disease) = 0.99
- Test has 5% false positive rate: P(positive | healthy) = 0.05
- You test positive. What's P(disease | positive)?

**Step 1: P(positive) total**
```
P(+) = P(+|disease) × P(disease) + P(+|healthy) × P(healthy)
P(+) = 0.99 × 0.01 + 0.05 × 0.99
P(+) = 0.0099 + 0.0495
P(+) = 0.0594
```

**Step 2: Bayes**
```
P(disease | +) = P(+|disease) × P(disease) / P(+)
P(disease | +) = 0.99 × 0.01 / 0.0594
P(disease | +) = 0.0099 / 0.0594
P(disease | +) ≈ 0.167
```

**Only 16.7%!** Even with a 99% accurate test, if the disease is rare (1%), most positives are false alarms. This is called **base rate neglect** — people forget how rare the condition is.

### Bayes Recipe (memorize this):

1. **List what you know**: priors P(A), P(not A), likelihoods P(B|A), P(B|not A)
2. **Calculate total P(B)**: multiply each likelihood by its prior, add them up
3. **Apply the formula**: P(A|B) = P(B|A) × P(A) / P(B)
4. **Round to 3 decimals**

---

## Part 6: Expected Value — The Long-Run Average

Expected value = the average outcome if you repeated something infinite times.

```
E[X] = Σ (outcome × probability of that outcome)
```

Just multiply each possible outcome by its probability, then add them all up.

**Example:** Roll a fair die. What's the expected value?
```
E[X] = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
E[X] = (1+2+3+4+5+6) / 6
E[X] = 21/6 = 3.5
```

No, you can't actually roll a 3.5. That's fine. It's the average over many rolls.

**Example (CodeSignal style):** A game costs $5 to play. You flip a coin 3 times. You win $2 per heads. Expected profit?

- Expected heads in 3 flips = 3 × 0.5 = 1.5
- Expected winnings = 1.5 × $2 = $3
- Expected profit = $3 - $5 = **-$2.000**

Negative expected value = you lose money on average. Don't play this game.

**Example with a table:**

| Outcome | Probability | Value |
|---------|------------|-------|
| Win big | 0.05 | $100 |
| Win small | 0.20 | $10 |
| Lose | 0.75 | -$5 |

```
E[X] = 0.05×100 + 0.20×10 + 0.75×(-5)
E[X] = 5 + 2 - 3.75
E[X] = 3.25
```

---

## Part 7: Binomial Distribution — Exactly k Successes in n Tries

This is the big one for CodeSignal. It answers: "If I do something n times, each time with probability p of success, what's the chance of getting EXACTLY k successes?"

**Requirements (all must be true):**
1. Fixed number of trials (n)
2. Each trial is independent (one doesn't affect the next)
3. Each trial has the same probability of success (p)
4. You're counting exact number of successes (k)

**Formula:**
```
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
```

Three pieces multiplied together:
1. **C(n,k)** = how many ways to arrange k successes among n trials
2. **p^k** = probability of k successes
3. **(1-p)^(n-k)** = probability of the remaining trials being failures

### Combinations: C(n,k) = "n choose k"

```
C(n,k) = n! / (k! × (n-k)!)
```

**What's n! (factorial)?** Just multiply all numbers from 1 to n:
- 5! = 5 × 4 × 3 × 2 × 1 = 120
- 3! = 3 × 2 × 1 = 6
- 0! = 1 (by definition, just memorize it)

**Example:** C(10, 3) = 10! / (3! × 7!)

Shortcut — you don't need to compute all of 10!. Cancel out:
```
C(10,3) = (10 × 9 × 8) / (3 × 2 × 1) = 720/6 = 120
```

The trick: in the numerator, write k numbers counting down from n. In the denominator, write k!.

More examples:
- C(5,2) = (5×4) / (2×1) = 10
- C(8,3) = (8×7×6) / (3×2×1) = 336/6 = 56
- C(6,1) = 6/1 = 6
- C(n,0) = 1 always (there's exactly one way to choose nothing)
- C(n,n) = 1 always (there's exactly one way to choose everything)

### Full Binomial Example (CONFIRMED CodeSignal question type):

**"A fair die is rolled 10 times. P(exactly 3 sixes)?"**

Extract the variables:
- n = 10 (number of rolls)
- k = 3 (we want exactly 3 sixes)
- p = 1/6 (probability of a six on any single roll)

**Step 1: C(10,3)**
```
C(10,3) = (10 × 9 × 8) / (3 × 2 × 1) = 720/6 = 120
```

**Step 2: p^k**
```
(1/6)^3 = 1/216 ≈ 0.004630
```

**Step 3: (1-p)^(n-k)**
```
(5/6)^7 ≈ 0.27908
```

How to compute (5/6)^7 by hand:
- (5/6)^1 = 0.8333
- (5/6)^2 = 0.6944
- (5/6)^3 = 0.5787
- (5/6)^4 = 0.4823
- (5/6)^5 = 0.4019
- (5/6)^6 = 0.3349
- (5/6)^7 = 0.2791

(Each step: multiply previous by 0.8333)

**Step 4: Multiply all three**
```
P(X=3) = 120 × 0.004630 × 0.2791
P(X=3) = 120 × 0.001292
P(X=3) ≈ 0.155
```

**Answer: 0.155**

### "At Least" and "At Most" with Binomial

- **P(at most k)** = P(X ≤ k) = P(X=0) + P(X=1) + ... + P(X=k)
- **P(at least k)** = P(X ≥ k) = 1 - P(X ≤ k-1)

**⚠️ Watch the boundary!** "At least 4" = 1 - P(X ≤ 3). The k-1 matters. Miss it and you're off by one term.

**Example:** 8 questions, 4 choices each, random guessing. P(at least 4 correct)?
- n=8, p=0.25, want P(X ≥ 4) = 1 - P(X ≤ 3)

You'd need P(X=0) + P(X=1) + P(X=2) + P(X=3) and subtract from 1.

This is tedious by hand but doable. On the CodeSignal IDE, you CAN write:
```python
from scipy.stats import binom
result = 1 - binom.cdf(3, n=8, p=0.25)
# ≈ 0.114
```

But Module 1 is fill-in-the-blank, not a coding environment. So for M1 you need to do it by hand or recognize common patterns. For coding modules, scipy is available.

**Hand calculation tip:** For "at least" problems on M1, if the numbers are small enough, sometimes it's faster to calculate P(X ≥ k) directly by adding up terms from k to n, rather than using the complement. Depends on which side has fewer terms.

---

## Part 8: Arithmetic Survival Guide

You can't use a calculator on M1. Here are the tricks:

### Fraction → Decimal conversions (memorize these):
- 1/2 = 0.500
- 1/3 = 0.333
- 1/4 = 0.250
- 1/5 = 0.200
- 1/6 = 0.167
- 1/7 = 0.143
- 1/8 = 0.125
- 1/10 = 0.100
- 1/12 = 0.083
- 1/20 = 0.050
- 1/50 = 0.020
- 1/100 = 0.010

### Powers you'll need:
- (1/2)^3 = 0.125
- (1/2)^4 = 0.0625
- (1/6)^2 = 0.02778
- (1/6)^3 = 0.00463
- (5/6)^2 = 0.6944
- (5/6)^3 = 0.5787
- (0.5)^5 = 0.03125
- (0.7)^3 = 0.343
- (0.3)^3 = 0.027

### Rounding rules (3 decimal places):
- Look at the 4th decimal digit
- If it's 5 or more → round up the 3rd digit
- If it's 4 or less → keep the 3rd digit as is
- 0.15529 → **0.155**
- 0.16667 → **0.167**
- 0.00603 → **0.006**

### Don't round intermediate steps!
Keep at least 5 decimal places through your calculations. Only round the FINAL answer. Rounding early compounds errors.

---

## Part 9: The 4 Question Types You'll See Wednesday

Based on confirmed candidate reports, Module 1 will have 2 questions from these types:

### Type 1: Binomial ("exactly k" or "at least k" successes)
**Pattern:** "n trials, each with probability p. What's P(exactly/at least k successes)?"
**Formula:** P(X=k) = C(n,k) × p^k × (1-p)^(n-k)
**Speed tip:** Compute C(n,k) first, then the powers, then multiply.

### Type 2: Bayes' Theorem (factory/disease/filter)
**Pattern:** You know the "forward" probability, they want the "reverse."
**Formula:** P(A|B) = P(B|A)×P(A) / [P(B|A)×P(A) + P(B|¬A)×P(¬A)]
**Speed tip:** Draw a tree diagram. Left branches = priors, right branches = likelihoods.

### Type 3: Dependent probability (cards without replacement)
**Pattern:** Drawing multiple items from a finite set without putting them back.
**Formula:** P(1st AND 2nd) = P(1st) × P(2nd | 1st happened)
**Speed tip:** Denominator shrinks by 1 each draw. Numerator shrinks only if you drew from that group.

### Type 4: Expected value
**Pattern:** Multiple outcomes with different probabilities. What's the average?
**Formula:** E[X] = Σ (outcome × probability)
**Speed tip:** If there's a cost, subtract it at the end.

---

## Part 10: Practice Problems — Do These BY HAND

**Instructions:** No code. No calculator. Paper and pen. Round final answers to 3 decimal places. Time yourself — aim for under 5 minutes each.

---

### Problem 1 (Binomial — "exactly")
A quality inspector checks 8 items from a production line where 10% are defective.
What is the probability that exactly 2 items are defective?

**Your workspace:**
```
n = ?    k = ?    p = ?

C(n,k) = 

p^k = 

(1-p)^(n-k) = 

P(X=k) = C(n,k) × p^k × (1-p)^(n-k) = 

Answer (3 decimals): _______
```

---

### Problem 2 (Bayes — factory pattern)
A company receives emails from two servers:
- Server X handles 70% of traffic with a 4% spam rate
- Server Y handles 30% of traffic with a 10% spam rate

An email is spam. What is the probability it came from Server Y?

**Your workspace:**
```
P(X) = ?    P(Y) = ?
P(spam|X) = ?    P(spam|Y) = ?

P(spam) = P(spam|X)×P(X) + P(spam|Y)×P(Y) = 

P(Y|spam) = P(spam|Y)×P(Y) / P(spam) = 

Answer (3 decimals): _______
```

---

### Problem 3 (Cards — dependent)
You draw 3 cards from a standard 52-card deck without replacement.
What is the probability that all 3 are Spades?

**Your workspace:**
```
P(1st Spade) = 

P(2nd Spade | 1st Spade) = 

P(3rd Spade | first 2 Spades) = 

P(all 3 Spades) = 

Answer (3 decimals): _______
```

---

### Problem 4 (Expected Value)
A game costs $8 to play. You roll two dice and sum them.
- If the sum is 12: you win $50
- If the sum is 7: you win $15
- Otherwise: you win $0

What is your expected profit per game?

**Your workspace:**
```
P(sum=12) = ?/36 = 
P(sum=7) = ?/36 = 
P(other) = 

E[winnings] = 50×P(12) + 15×P(7) + 0×P(other) = 

E[profit] = E[winnings] - 8 = 

Answer (3 decimals): _______
```

---

### Problem 5 (Binomial — "at least")
A basketball player makes 80% of free throws. She shoots 5 free throws.
What is the probability she makes at least 4?

**Your workspace:**
```
P(X ≥ 4) = P(X=4) + P(X=5)

P(X=4):
  C(5,4) = 
  (0.8)^4 = 
  (0.2)^1 = 
  P(X=4) = 

P(X=5):
  C(5,5) = 
  (0.8)^5 = 
  (0.2)^0 = 
  P(X=5) = 

P(X ≥ 4) = P(X=4) + P(X=5) = 

Answer (3 decimals): _______
```

---

### Problem 6 (Bayes — disease/test pattern)
A factory produces light bulbs. 5% are defective.
- A test correctly identifies 90% of defective bulbs (true positive)
- The test incorrectly flags 8% of good bulbs (false positive)

A bulb is flagged by the test. What's the probability it's actually defective?

**Your workspace:**
```
P(def) = ?    P(good) = ?
P(flag|def) = ?    P(flag|good) = ?

P(flag) = P(flag|def)×P(def) + P(flag|good)×P(good) = 

P(def|flag) = P(flag|def)×P(def) / P(flag) = 

Answer (3 decimals): _______
```

---

### Problem 7 (Combined — reading comprehension test)
A consulting firm is analyzing client churn for a telecom company. Historical data shows that 15% of customers on the "Basic" plan churn each month. The firm randomly selects 6 Basic plan customers for a deep-dive study. What is the probability that at most 1 of the selected customers will churn this month?

**Your workspace:**
```
n = ?    p = ?
P(X ≤ 1) = P(X=0) + P(X=1)

P(X=0):
  C(?,0) = 
  p^0 = 
  (1-p)^? = 
  P(X=0) = 

P(X=1):
  C(?,1) = 
  p^1 = 
  (1-p)^? = 
  P(X=1) = 

P(X ≤ 1) = 

Answer (3 decimals): _______
```

---

## Solutions (DON'T PEEK UNTIL YOU'VE TRIED)

**Scroll down...**

.

.

.

.

.

.

.

.

.

.

**Problem 1:** n=8, k=2, p=0.10
- C(8,2) = (8×7)/(2×1) = 28
- (0.10)^2 = 0.01
- (0.90)^6 = 0.531441
- P = 28 × 0.01 × 0.531441 = 0.14880
- **Answer: 0.149**

**Problem 2:**
- P(spam) = 0.04×0.70 + 0.10×0.30 = 0.028 + 0.030 = 0.058
- P(Y|spam) = 0.030 / 0.058 = 0.51724
- **Answer: 0.517**

**Problem 3:**
- P = (13/52) × (12/51) × (11/50)
- P = 0.2500 × 0.23529 × 0.22000
- P = 0.01294
- **Answer: 0.013**

**Problem 4:**
- P(sum=12) = 1/36 ≈ 0.02778 (only 6+6)
- P(sum=7) = 6/36 ≈ 0.16667 (1+6, 2+5, 3+4, 4+3, 5+2, 6+1)
- E[winnings] = 50×(1/36) + 15×(6/36) = 1.3889 + 2.5000 = 3.8889
- E[profit] = 3.8889 - 8 = -4.1111
- **Answer: -4.111**

**Problem 5:**
- P(X=4) = C(5,4) × (0.8)^4 × (0.2)^1 = 5 × 0.4096 × 0.2 = 0.4096
- P(X=5) = C(5,5) × (0.8)^5 × (0.2)^0 = 1 × 0.32768 × 1 = 0.32768
- P(X≥4) = 0.4096 + 0.32768 = 0.73728
- **Answer: 0.737**

**Problem 6:**
- P(flag) = 0.90×0.05 + 0.08×0.95 = 0.045 + 0.076 = 0.121
- P(def|flag) = 0.045 / 0.121 = 0.37190
- **Answer: 0.372**

**Problem 7:** n=6, p=0.15
- P(X=0) = C(6,0) × (0.15)^0 × (0.85)^6 = 1 × 1 × 0.37715 = 0.37715
- P(X=1) = C(6,1) × (0.15)^1 × (0.85)^5 = 6 × 0.15 × 0.44371 = 0.39934
- P(X≤1) = 0.37715 + 0.39934 = 0.77649
- **Answer: 0.776**

---

## Quick Reference Card (tear this off mentally)

```
COMPLEMENT:    P(not A) = 1 - P(A)
AND (indep):   P(A and B) = P(A) × P(B)  
AND (depend):  P(A and B) = P(A) × P(B|A)
OR:            P(A or B) = P(A) + P(B) - P(A and B)
CONDITIONAL:   P(A|B) = P(A and B) / P(B)
BAYES:         P(A|B) = P(B|A)×P(A) / P(B)
TOTAL PROB:    P(B) = P(B|A)×P(A) + P(B|¬A)×P(¬A)
EXPECTED:      E[X] = Σ outcome × P(outcome)
BINOMIAL:      P(X=k) = C(n,k) × p^k × (1-p)^(n-k)
COMBINATION:   C(n,k) = n! / (k!(n-k)!)
AT LEAST k:    P(X≥k) = 1 - P(X≤k-1)
```

---

*Now go do the practice problems by hand. The formula means nothing until your pencil has written it 7 times.*
