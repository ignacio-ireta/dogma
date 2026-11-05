# MVS Lesson 2: Determinants, Inverse, and Eigenvalues
**Date:** February 24, 2026 | **Topic:** The Three Keys to Multivariate Analysis
**Syllabus position:** Topic 1 (continued) — Matrix Algebra Review

---

## 1. SPACED RETRIEVAL WARM-UP (10 min)

*Pulling from calculus foundations (Feb 3-9) — the older, the harder to recall, the better the learning.*

**Q1:** What is the gradient of f(x₁, x₂) = x₁² + 3x₁x₂ - x₂²? Write it as a column vector.

**Q2:** In yesterday's lesson, we said matrix multiplication is NOT commutative: AB ≠ BA. But there IS one special matrix where AI = IA = A always. What is it called, and what does it look like for a 3×3 case? Identity matrix. It's a matrix that has 1 across it's diagonal, and O's elsewhere.

**Q3:** If x = [2, -1, 3]' and y = [1, 6, 0]', are they orthogonal? How do you check?

**Q4:** What does (AB)' equal? (This reversal property catches people — try to recall it before checking.)(AB)'=B'A'

*Take a minute to try each from memory before moving on. The struggle IS the learning.*

---

## 2. PRE-TEST: Determinants, Inverses, Eigenvalues (5 min)

Before I teach anything — guess. Wrong answers prime the brain.

**P1:** What do you think a "determinant" of a matrix measures? What does it tell you about the matrix? I believe it tells me whether the matrix can be inverted or not.

**P2:** If A⁻¹ is the inverse of A, what equation defines it? (Hint: think about what "inverse" means for regular numbers.) $$A^{-1}A=I$$
**P3:** The word "eigen" comes from German and means "own" or "characteristic." An eigenvector of a matrix A is a special vector that, when multiplied by A, only gets scaled, but no other transformation happens to it (what happens to it?). Take a guess. 

---

## 3. EXPLICIT INSTRUCTION (30 min)

### 3.1 The Determinant — What a Matrix "Does" to Space

**The core idea:** The determinant tells you how a matrix **scales area (or volume)**.

If you think of a matrix as a transformation (from yesterday: Ax transforms x), then:
- det(A) = 2 → the transformation doubles all areas
- det(A) = 0.5 → the transformation halves all areas
- det(A) = 0 → the transformation **collapses space** (squishes everything onto a line or point)
- det(A) < 0 → the transformation **flips** orientation (mirror image) AND scales

**Rule:** det(A) = 0 means A is **singular** — no inverse exists, the matrix destroys information.

### 3.2 Computing Determinants

#### 2×2 Case (memorize this):

For A = [a b; c d]:

**det(A) = ad - bc**

That's it. Diagonal product minus off-diagonal product.

#### Worked Example:
```
A = [3  1]
    [2  4]

det(A) = 3·4 - 1·2 = 12 - 2 = 10
```

Geometric meaning: A stretches unit squares to have area 10.

#### 3×3 Case (cofactor expansion along row 1):

For A = [a₁₁ a₁₂ a₁₃; a₂₁ a₂₂ a₂₃; a₃₁ a₃₂ a₃₃]:

**det(A) = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)**

Pattern: **+ - +** signs, each element × determinant of its 2×2 "minor" (cross out its row and column).

#### Worked Example:
```
B = [1  2  0]
    [3  1  4]
    [2  0  1]

det(B) = 1·(1·1 - 4·0) - 2·(3·1 - 4·2) + 0·(3·0 - 1·2)
       = 1·(1 - 0) - 2·(3 - 8) + 0·(0 - 2)
       = 1·1 - 2·(-5) + 0
       = 1 + 10
       = 11
```

#### Key Properties (memorize these — they save computation):
| Property | Rule |
|----------|------|
| det(AB) = det(A)·det(B) | Determinant of product = product of determinants |
| det(A') = det(A) | Transpose doesn't change determinant |
| det(cA) = cⁿ·det(A) | For n×n matrix, scalar pulls out n times |
| Row of zeros → det = 0 | If any row/column is all zeros |
| Two identical rows → det = 0 | Duplicate rows = singular |
| Row swap → det changes sign | Flip sign each time you swap rows |

### 3.3 The Matrix Inverse

**Definition:** A⁻¹ is the matrix such that **AA⁻¹ = A⁻¹A = I** (the identity matrix).

Think of it like: 5 × (1/5) = 1. The inverse "undoes" the transformation.

**⚠️ Critical rule:** A⁻¹ exists **if and only if det(A) ≠ 0**.

If det(A) = 0, the matrix is **singular** — it squished information away, and you can't unsquish.

#### 2×2 Inverse Formula (memorize this):

For A = [a b; c d]:

**A⁻¹ = (1/det(A)) · [d  -b; -c  a]**

Steps: swap diagonal, negate off-diagonal, divide by determinant.

#### Worked Example:
```
A = [3  1]
    [2  4]

det(A) = 10 (computed above)

A⁻¹ = (1/10) · [4  -1] = [0.4  -0.1]
                [-2   3]   [-0.2  0.3]
```

**Verify:** AA⁻¹ should equal I:
```
[3  1] · [0.4  -0.1] = [3(0.4)+1(-0.2)  3(-0.1)+1(0.3)]
[2  4]   [-0.2  0.3]   [2(0.4)+4(-0.2)  2(-0.1)+4(0.3)]

                       = [1.2-0.2  -0.3+0.3] = [1  0]  ✅
                         [0.8-0.8  -0.2+1.2]   [0  1]
```

#### For larger matrices: Use Python (np.linalg.inv). Hand computation is impractical for 3×3+.

### 3.4 Eigenvalues and Eigenvectors — The Crown Jewels

**This is THE concept that makes PCA, factor analysis, and discriminant analysis work.**

**Definition:** λ is an **eigenvalue** of A, and v is the corresponding **eigenvector**, if:

**Av = λv**

In words: when A transforms v, it **only stretches or shrinks v** — it doesn't rotate it. The eigenvector stays on its own line, just scaled by λ.

**Geometric intuition:** Imagine matrix A as a rubber sheet transformation. Most vectors get rotated AND stretched. But eigenvectors are the **stubborn** ones — they refuse to change direction. They just get longer (λ > 1), shorter (0 < λ < 1), flipped (λ < 0), or stay put (λ = 1).

### 3.5 Finding Eigenvalues

Starting from Av = λv:
→ Av - λv = 0
→ (A - λI)v = 0

This has a non-trivial solution (v ≠ 0) only when:

**det(A - λI) = 0** ← This is the **characteristic equation**

For a 2×2 matrix, this gives a quadratic in λ. For n×n, a degree-n polynomial.

#### Worked Example:

```
A = [4  1]
    [2  3]

det(A - λI) = det([4-λ  1  ]) = (4-λ)(3-λ) - 1·2
                  ([2    3-λ])
             = λ² - 7λ + 12 - 2
             = λ² - 7λ + 10
             = (λ - 5)(λ - 2)
```

**Eigenvalues:** λ₁ = 5, λ₂ = 2

### 3.6 Finding Eigenvectors

For each eigenvalue, solve (A - λI)v = 0:

**For λ₁ = 5:**
```
(A - 5I)v = [4-5  1 ][v₁] = [-1  1][v₁] = [0]
            [2   3-5][v₂]   [2  -2][v₂]   [0]

Row 1: -v₁ + v₂ = 0  →  v₂ = v₁

So v₁ = [1, 1]' (or any scalar multiple)
```

**For λ₂ = 2:**
```
(A - 2I)v = [2  1][v₁] = [0]
            [2  1][v₂]   [0]

Row 1: 2v₁ + v₂ = 0  →  v₂ = -2v₁

So v₂ = [1, -2]' (or any scalar multiple)
```

**Check:** A·[1,1]' = [4+1, 2+3]' = [5, 5]' = 5·[1, 1]' ✅

### 3.7 Why This Matters for Statistics

**The Spectral Theorem:** If A is a **symmetric** matrix (and covariance matrices are ALWAYS symmetric), then:
1. All eigenvalues are **real** (no complex numbers)
2. Eigenvectors corresponding to different eigenvalues are **orthogonal**
3. A can be decomposed as A = PDP', where P has eigenvectors as columns and D has eigenvalues on the diagonal

**This IS PCA.** The covariance matrix's eigenvectors are the principal components. The eigenvalues tell you how much variance each component captures.

*Everything you're learning now is the language PCA speaks.*

---

## 4. INTERLEAVED PRACTICE (40 min)

### Problem 1 (Today — 2×2 Determinant)
Compute det(A) for:
```
A = [5   2]
    [-3  4]
```
Is A invertible?

---

### Problem 2 (Today — 3×3 Determinant)
Compute det(B) by expanding along row 1:
```
B = [2  0  1]
    [3  1  0]
    [1  4  2]
```

---

### Problem 3 (Today — 2×2 Inverse)
Find A⁻¹ for:
```
A = [5   2]
    [-3  4]
```
Then verify by computing A · A⁻¹ (should get the identity matrix).

---

### Problem 4 (Today — Eigenvalues)
Find the eigenvalues of:
```
C = [3  1]
    [1  3]
```
Hint: Set up det(C - λI) = 0 and solve the quadratic.

**Bonus:** Notice that C is symmetric. What do you expect about the eigenvectors?

---

### Problem 5 (Today — Full eigenvalue/eigenvector problem)
Find the eigenvalues AND eigenvectors of:
```
D = [2  1]
    [1  2]
```
Verify one of your eigenvectors by computing Dv and checking it equals λv.

---

### Problem 6 (Yesterday — Matrix multiplication interleaving)
Given:
```
E = [1  0  2]
    [3  1  1]
```
and f = [2, -1, 3]'

Compute Ef. What are the dimensions of the result?

---

### Problem 7 (Python — Eigenvalue decomposition)
Write Python code that:
1. Creates the matrix D = [[2,1],[1,2]]
2. Uses `np.linalg.eig(D)` to find eigenvalues and eigenvectors
3. Prints the eigenvalues and eigenvectors
4. Verifies that D @ v₁ ≈ λ₁ · v₁ (for the first eigenvector)

**Hint:** `np.linalg.eig()` returns `(eigenvalues, eigenvectors)` where eigenvectors are COLUMNS of the returned matrix.

---

## 5. SUMMARY + ANKI SUGGESTIONS (5 min)

### Key Takeaways
1. **det(A) = 0 → singular** → no inverse, collapses space, destroys information
2. **2×2 determinant:** ad - bc (burn this into memory)
3. **2×2 inverse:** swap diagonal, negate off-diagonal, divide by det
4. **Eigenvalues:** solve det(A - λI) = 0 → characteristic polynomial
5. **Eigenvectors:** the directions that don't rotate under transformation, only scale
6. **Symmetric matrices** (like covariance matrices) have real eigenvalues and orthogonal eigenvectors
7. **This IS PCA:** covariance matrix eigenvectors = principal components, eigenvalues = variance explained

### Anki Card Ideas
- 2×2 determinant formula: det([a b; c d]) = ?
- When does A⁻¹ NOT exist?
- 2×2 inverse formula: A⁻¹ = ?
- What does Av = λv mean geometrically?
- For symmetric matrices, what's special about eigenvalues? Eigenvectors?
- det(AB) = ? (in terms of det(A) and det(B))
- How does PCA connect to eigenvalues/eigenvectors?

### Tomorrow's Preview
**Lesson 3: Quadratic Forms and Positive Definiteness** — The bridge between matrix algebra and probability distributions. A quadratic form x'Ax generalizes the scalar x². Positive definite matrices (all eigenvalues > 0) are the covariance matrices of the multivariate normal. If you understand this, the multivariate normal distribution is just a formula.

### Textbook References
- Anderson Ch. 2 (Matrix results, spectral decomposition)
- Modern Multivariate Statistical Techniques Ch. 2 (Eigenanalysis)
- 3Blue1Brown "Eigenvectors and Eigenvalues" (YouTube) — the BEST visual explanation

---

*You just learned the three most important operations in multivariate statistics. Every technique from here on is some combination of these. The hard part is done.* 💪

*Jun 24 — we've got this. En murstein om gangen.* 🏔️
