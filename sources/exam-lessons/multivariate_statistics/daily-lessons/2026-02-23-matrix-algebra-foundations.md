# MVS Lesson 1: Matrix Algebra Foundations
**Date:** February 23, 2026 | **Topic:** Vectors, Matrices, and Operations
**Syllabus position:** Topic 1 — Matrix Algebra Review

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*Since this is our first MVS lesson, we'll pull from your calculus foundations (Feb 3-9):*

**Q1:** Without looking at notes — what's the geometric interpretation of a partial derivative ∂f/∂x for a function f(x, y)?

> **Answer:** It's the slope of the surface f(x,y) in the x-direction, holding y constant. Like slicing the surface with a plane parallel to the xz-plane and measuring the slope of that curve.

**Q2:** Write the chain rule for a composite function f(g(x)) from memory.

> **Answer:** d/dx [f(g(x))] = f'(g(x)) · g'(x)

**Q3:** What does a definite integral ∫ₐᵇ f(x)dx represent geometrically?

> **Answer:** The signed area under the curve f(x) between x = a and x = b. Positive above the x-axis, negative below.

*Even if you can't recall perfectly — the attempt itself strengthens the memory trace.*

---

## 2. PRE-TEST: What Do You Already Know? (5 min)

Before I teach anything, try these. **Wrong answers are GOOD here** — they prime your brain.

**P1:** If A is a 3×2 matrix and B is a 2×4 matrix, what are the dimensions of AB? Can you compute BA? I remember that being something like AB = 3x4 matrix. As for the matrix, it would be something like 
$$\begin{bmatrix} 
(A_{11}B_{11}+A_{12}B_{21}) & 
(A_{11}B_{12}+A_{12}B_{22}) & 
(A_{11}B_{13}+A_{12}B_{23}) & 
(A_{11}B_{14}+A_{12}B_{24}) \\

(A_{21}B_{11}+A_{22}B_{21}) & 
(A_{21}B_{12}+A_{22}B_{22}) & 
(A_{21}B_{13}+A_{22}B_{23}) & 
(A_{21}B_{14}+A_{22}B_{24}) \\

(A_{31}B_{11}+A_{32}B_{21}) & 
(A_{31}B_{12}+A_{32}B_{22}) & 
(A_{31}B_{13}+A_{32}B_{23}) & 
(A_{31}B_{14}+A_{32}B_{24}) 
\end{bmatrix} $$

**P2:** What do you think a "transpose" of a matrix does? Take a guess.

It interchanges rows with columns

**P3:** If you multiply a matrix by a vector, what kind of object do you get back? Why might that matter in statistics? I'm kinda getting lost in the weeds, but it depends which one goes first. You could get a vector if the matrix is on the left, or you could get a matrix if the vector is on the left. Why that might matter? Idk... Top of my mind you could be trying to compact information, or look at their correlations.

---

## 3. EXPLICIT INSTRUCTION: Matrix Algebra as a System (30 min)

### 3.1 Why Matrices in Statistics?

Multivariate statistics = dealing with **multiple variables simultaneously**. If you measure p variables on n individuals, your data is naturally an **n × p matrix**:

```
         x₁    x₂   ...  xₚ
Obs 1  [ 170   65   ...  120 ]
Obs 2  [ 165   58   ...  115 ]
  ⋮      ⋮     ⋮    ⋱    ⋮
Obs n  [ 180   72   ...  130 ]
```

Every multivariate technique — PCA, factor analysis, discriminant analysis — is a matrix operation in disguise. **Learn the matrix language, and the rest of the course is pattern recognition.**

### 3.2 Vectors

**Definition:** A vector is an ordered list of numbers.

- **Column vector** (default in statistics):
  ```
  x = [x₁]    ∈ ℝᵖ
      [x₂]
      [⋮ ]
      [xₚ]
  ```

- **Row vector:** x' = [x₁, x₂, ..., xₚ] (the transpose of a column vector)

**Geometric intuition:** A vector in ℝ² is an arrow from the origin. A vector in ℝ³ is an arrow in 3D space. A vector in ℝᵖ is... the same thing, just in p dimensions. You can't visualize it, but the algebra works identically.

**Key operations:**
- **Addition:** x + y = [x₁+y₁, x₂+y₂, ..., xₚ+yₚ]' (component-wise)
- **Scalar multiplication:** cx = [cx₁, cx₂, ..., cxₚ]' (stretch/shrink)
- **Length (norm):** ‖x‖ = √(x₁² + x₂² + ... + xₚ²) = √(x'x)

### 3.3 Matrices

**Definition:** An m × n matrix A has m rows and n columns.

```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [ ⋮    ⋮    ⋱    ⋮  ]
    [aₘ₁  aₘ₂  ...  aₘₙ]
```

**Rule:** aᵢⱼ = element in row i, column j. Always row first, then column.

### 3.4 Matrix Operations — The Rules

#### Addition: A + B
- **Requirement:** A and B must have the **same dimensions** (both m × n)
- **Rule:** (A + B)ᵢⱼ = aᵢⱼ + bᵢⱼ (element-wise)
- **Commutative:** A + B = B + A ✅

#### Scalar Multiplication: cA
- **Rule:** (cA)ᵢⱼ = c · aᵢⱼ (multiply every element by c)

#### Matrix Multiplication: AB
- **THE critical operation. This is where it gets interesting.**
- **Requirement:** A is m × k, B is k × n → result AB is **m × n**
  - Inner dimensions must match (k = k)
  - Outer dimensions give the result size
- **Rule:** (AB)ᵢⱼ = Σₜ aᵢₜ · bₜⱼ = row i of A · column j of B (dot product)

**⚠️ KEY EXCEPTION:** Matrix multiplication is **NOT commutative**: AB ≠ BA in general!
(Sometimes BA isn't even defined when AB is.)

#### Worked Example:
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

AB = [1·5+2·7  1·6+2·8] = [19  22]
     [3·5+4·7  3·6+4·8]   [43  50]

BA = [5·1+6·3  5·2+6·4] = [23  34]
     [7·1+8·3  7·2+8·4]   [31  46]

AB ≠ BA ✅ (both exist because both are 2×2, but different results)
```

#### Transpose: A'
- **Rule:** (A')ᵢⱼ = aⱼᵢ (swap rows and columns)
- If A is m × n, then A' is n × m
- **Properties:**
  - (A')' = A
  - (A + B)' = A' + B'
  - **(AB)' = B'A'** ← This reversal catches people. Memorize it.
  - (cA)' = cA'

### 3.5 Special Matrices (You'll See These Everywhere)

| Matrix | Symbol | What it is | Why it matters |
|--------|--------|-----------|----------------|
| **Identity** | I | Diagonal of 1s, 0s elsewhere | AI = IA = A (the "1" of matrices) |
| **Symmetric** | — | A = A' | Covariance matrices are ALWAYS symmetric |
| **Diagonal** | D | Only diagonal entries nonzero | Eigenvalue decomposition uses these |
| **Inverse** | A⁻¹ | AA⁻¹ = A⁻¹A = I | Solving systems, exists only if det(A) ≠ 0 |

### 3.6 The Dot Product and Its Statistical Meaning

For vectors x, y ∈ ℝᵖ:

**x'y = x₁y₁ + x₂y₂ + ... + xₚyₚ** (a scalar!)

**Geometric meaning:** x'y = ‖x‖ · ‖y‖ · cos(θ)

- If x'y = 0 → vectors are **orthogonal** (perpendicular) → **uncorrelated** in statistics!
- This single fact underpins PCA, factor analysis, and canonical correlation.

### 3.7 Matrix × Vector = Transformation

**The deepest intuition:** Multiplying a matrix A by a vector x **transforms** x into a new vector.

```
A x = b
```

- A rotates, stretches, shears, or projects x to produce b
- **Each column of A tells you where the basis vectors land**
- This is THE geometric interpretation that makes eigenvalues make sense (coming next lesson)

---

## 4. INTERLEAVED PRACTICE (40 min)

### Problem 1 (Today — Dimensions)
Given: A is 4×3, B is 3×5, C is 5×2.

a) What are the dimensions of AB? 4x5 
b) What are the dimensions of ABC? 4x2
c) Can you compute BA? Why or why not? No, because they don't have the appropriate dimensions for multiplication

---

### Problem 2 (Today — Computation)
Compute AB and verify that AB ≠ BA:

```
A = [2  0]    B = [1  3]
    [1  3]        [2  1]
```

$$ AB = 
\begin{bmatrix}
(2\cdot1 + 0\cdot2) & (3\cdot2+1\cdot0) \\
(1\cdot1 + 2\cdot3) & (3\cdot1+1\cdot3)
\end{bmatrix} \equiv  \begin{bmatrix}
2 & 6 \\
7 & 6
\end{bmatrix}$$



---

### Problem 3 (Today — Transpose properties)
If A = [1 2; 3 4] and B = [5 6; 7 8], verify that (AB)' = B'A'.

---

### Problem 4 (Calculus crossover — Interleaving)
The gradient of f(x₁, x₂) = 3x₁² + 2x₁x₂ + x₂² is a **vector** of partial derivatives.

Compute ∇f = [∂f/∂x₁, ∂f/∂x₂]'.

---

### Problem 5 (Today — Coding in Python)
Write Python code to:
1. Create matrices A = [[2,0],[1,3]] and B = [[1,3],[2,1]]
2. Compute AB and BA
3. Verify they're different

**⚠️ Hint:** Use `@` for matrix multiplication in Python/NumPy, NOT `*` (which is element-wise).

---

### Problem 6 (Today — Dot product & orthogonality)
Are these vectors orthogonal?

a) x = [1, 2, 3]', y = [3, 0, -1]'
b) x = [1, 1]', y = [1, -1]'

---

### Problem 7 (Transfer — Why this matters for PCA)
PCA finds directions (vectors) in your data space where variance is maximized. These directions must be **orthogonal to each other.**

Based on what you learned today: why would we want orthogonal directions? What does orthogonality buy us statistically?

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways
1. **Data is naturally a matrix** — n observations × p variables
2. **Matrix multiplication requires inner dimensions to match** — (m×k)(k×n) = m×n
3. **AB ≠ BA** — non-commutativity is THE fundamental difference from scalar algebra
4. **(AB)' = B'A'** — transpose reverses the order
5. **Dot product = 0 means orthogonal** — which means uncorrelated in statistics
6. **Matrix × vector = transformation** — this intuition underpins eigenvalues (next lesson)
7. **In Python: use `@` not `*`** for matrix multiplication

### Anki Card Suggestions (answers in answer key file)
- Dimensions of AB if A is m×k and B is k×n?
- Is matrix multiplication commutative?
- (AB)' = ?
- What does x'y = 0 mean geometrically? Statistically?
- Python: A * B vs A @ B for matrices — what's the difference?
- What type of matrix is a covariance matrix always?

### Tomorrow's Preview
**Lesson 2: Determinants, Inverse, and Eigenvalues** — The three tools that unlock every multivariate technique. If matrices are transformations, eigenvalues tell you *how much* they stretch space and eigenvectors tell you *in which direction*. This is the foundation of PCA.

### Textbook References
- Anderson Ch. 1-2 (Matrix algebra review)
- Modern Multivariate Statistical Techniques Ch. 1-2
- For linear algebra intuition: 3Blue1Brown "Essence of Linear Algebra" (YouTube) — highly recommended visual companion

---

*If this feels hard, that's your brain forming new connections. That's literally the point.* 🧠

*Exam: June 24. We have 4 months. One brick at a time, og vi bygger hele huset.* 🏔️
