# MVS Lesson 1: Answer Key + Feedback
**Date:** February 23, 2026 | **Topic:** Matrix Algebra Foundations

---

## 1. SPACED RETRIEVAL — Expected Answers

**Q1 (Partial derivative):** The slope of the surface f(x,y) in the x-direction, holding y constant. Like slicing the surface with a plane parallel to the xz-plane.

**Q2 (Chain rule):** d/dx [f(g(x))] = f'(g(x)) · g'(x)

**Q3 (Definite integral):** The signed area under the curve f(x) between x = a and x = b. Positive above x-axis, negative below.

---

## 2. PRE-TEST — Expected Answers

**P1:** AB = 3×4 matrix. BA is NOT defined (inner dimensions 4≠3).

**P2:** Transpose swaps rows and columns. (A')ᵢⱼ = aⱼᵢ

**P3:** Matrix × vector = vector (always). A vector is a special case of a matrix (n×1). In statistics: transforming data (projections, rotations, combining variables). "Compacting information" is actually a great intuition — that's essentially what PCA does.

---

## 4. INTERLEAVED PRACTICE — Answer Key

### Problem 1 (Dimensions)
a) AB: 4×5 (inner 3=3 matches, outer 4,5)
b) ABC: 4×2 (AB is 4×5, C is 5×2, inner 5=5 matches)
c) No. B is 3×5 and A is 4×3 → inner dimensions 5≠4. Not defined.

### Problem 2 (Computation)
AB = [2·1+0·2  2·3+0·1] = [2  6]
     [1·1+3·2  1·3+3·1]   [7  6]

BA = [1·2+3·1  1·0+3·3] = [5   9]
     [2·2+1·1  2·0+1·3]   [5   3]

AB ≠ BA ✅

### Problem 3 (Transpose properties)
AB = [19 22; 43 50] → (AB)' = [19 43; 22 50]

A' = [1 3; 2 4], B' = [5 7; 6 8]
B'A' = [5·1+7·2  5·3+7·4] = [19  43]
       [6·1+8·2  6·3+8·4]   [22  50]

(AB)' = B'A' ✅

### Problem 4 (Gradient)
∂f/∂x₁ = 6x₁ + 2x₂
∂f/∂x₂ = 2x₁ + 2x₂
∇f = [6x₁ + 2x₂, 2x₁ + 2x₂]'

**Bonus:** f(x) = x'Ax where A = [3 1; 1 1]. The gradient is ∇f = 2Ax. Quadratic forms + matrices = the language of MVS.

### Problem 5 (Python)
```python
import numpy as np

A = np.array([[2, 0], [1, 3]])
B = np.array([[1, 3], [2, 1]])

AB = A @ B
BA = B @ A

print("AB =\n", AB)
print("BA =\n", BA)
print("AB == BA?", np.allclose(AB, BA))
```

Expected output:
AB = [[2 6] [7 6]]
BA = [[5 9] [5 3]]
AB == BA? False

### Problem 6 (Dot product & orthogonality)
a) x'y = 1·3 + 2·0 + 3·(-1) = 3 + 0 - 3 = 0 → **Orthogonal** ✅
b) x'y = 1·1 + 1·(-1) = 1 - 1 = 0 → **Orthogonal** ✅

In statistics: orthogonal observation vectors → zero correlation → independent information.

### Problem 7 (PCA transfer)
Orthogonal directions = uncorrelated components. Each principal component captures **unique** information that no other component has. If they weren't orthogonal, they'd share redundant information, defeating the purpose of dimensionality reduction.

---

## Answers + Feedback (Feb 23)

### What he answered:
- **P1 (Pre-test):** Said AB = 3×4 ✅ and wrote out the full matrix multiplication formula in LaTeX. Impressive — shows he remembers the mechanics, not just the dimension rule. Didn't address whether BA is computable.
- **P2 (Pre-test):** "It interchanges rows with columns" ✅ Clean and correct.
- **P3 (Pre-test):** Said "it depends which one goes first" — partially correct intuition. Said matrix on left → vector, vector on left → matrix. Actually: vector is just an n×1 matrix, so result is always a vector (or scalar). His "compact information / correlations" guess is surprisingly on-target for PCA.
- **Problem 1:** All three correct ✅✅✅
- **Problem 2:** Got AB = [2 6; 7 6] ✅ correct answer, but intermediate LaTeX work shows some element index confusion (wrote 3·2+1·0 for top-right instead of 2·3+0·1). Right answer, wrong path. Didn't compute BA.
- **Problems 3-7:** Not attempted (LaTeX fatigue)

### Overall: Strong foundations. Not rusty — just needs practice speed.
