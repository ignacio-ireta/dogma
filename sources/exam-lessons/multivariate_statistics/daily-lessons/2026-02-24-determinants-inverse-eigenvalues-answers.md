# MVS Lesson 2: Answer Key
**Date:** February 24, 2026 | **Topic:** Determinants, Inverse, Eigenvalues

---

## 1. SPACED RETRIEVAL — Expected Answers

**Q1 (Gradient):**
∂f/∂x₁ = 2x₁ + 3x₂
∂f/∂x₂ = 3x₁ - 2x₂
∇f = [2x₁ + 3x₂, 3x₁ - 2x₂]'

**Q2 (Identity matrix):** The identity matrix I. For 3×3:
```
I = [1 0 0]
    [0 1 0]
    [0 0 1]
```

**Q3 (Orthogonality check):**
x'y = 2·1 + (-1)·6 + 3·0 = 2 - 6 + 0 = -4
Not zero → NOT orthogonal.

**Q4 (Transpose of product):**
(AB)' = B'A' (order reverses)

---

## 2. PRE-TEST — Expected Answers

**P1:** The determinant measures how much the matrix scales area/volume. det = 0 means the matrix is singular (not invertible, collapses dimensions).

**P2:** AA⁻¹ = A⁻¹A = I (like 5 × 1/5 = 1)

**P3:** An eigenvector only gets stretched/shrunk — it stays on the same line. Av = λv.

---

## 4. INTERLEAVED PRACTICE — Answer Key

### Problem 1 (2×2 Determinant)
det(A) = 5·4 - 2·(-3) = 20 + 6 = **26**
Since det(A) = 26 ≠ 0, **yes, A is invertible**.

### Problem 2 (3×3 Determinant)
```
det(B) = 2·(1·2 - 0·4) - 0·(3·2 - 0·1) + 1·(3·4 - 1·1)
       = 2·(2 - 0) - 0·(6 - 0) + 1·(12 - 1)
       = 2·2 - 0 + 1·11
       = 4 + 11
       = 15
```
**det(B) = 15**

### Problem 3 (2×2 Inverse)
det(A) = 26 (from Problem 1)

A⁻¹ = (1/26) · [4  -2] = [4/26   -2/26] = [2/13   -1/13]
                [3   5]   [3/26    5/26]   [3/26    5/26]

Numerical: A⁻¹ ≈ [0.1538  -0.0769]
                  [0.1154   0.1923]

**Verification:** A · A⁻¹:
```
[5   2] · [2/13   -1/13] = [10/13+6/26    -5/13+10/26]
[-3  4]   [3/26    5/26]   [-6/13+12/26   3/13+20/26]

= [10/13+3/13    -5/13+5/13] = [1  0]  ✅
  [-6/13+6/13    3/13+10/13]   [0  1]
```

### Problem 4 (Eigenvalues of symmetric matrix)
```
det(C - λI) = det([3-λ  1  ]) = (3-λ)² - 1
                  ([1    3-λ])
             = 9 - 6λ + λ² - 1
             = λ² - 6λ + 8
             = (λ - 4)(λ - 2)
```

**Eigenvalues:** λ₁ = 4, λ₂ = 2

**Bonus:** Since C is symmetric, the eigenvectors must be orthogonal to each other.

For λ₁ = 4: (C - 4I)v = [-1 1; 1 -1]v = 0 → v₁ = [1, 1]'
For λ₂ = 2: (C - 2I)v = [1 1; 1 1]v = 0 → v₂ = [1, -1]'

Check orthogonality: v₁'v₂ = 1·1 + 1·(-1) = 0 ✅

### Problem 5 (Full eigenvalue/eigenvector)
```
det(D - λI) = det([2-λ  1  ]) = (2-λ)² - 1
                  ([1    2-λ])
             = 4 - 4λ + λ² - 1
             = λ² - 4λ + 3
             = (λ - 3)(λ - 1)
```

**Eigenvalues:** λ₁ = 3, λ₂ = 1

**Eigenvectors:**
For λ₁ = 3: (D - 3I)v = [-1 1; 1 -1]v = 0 → v₁ = [1, 1]'
For λ₂ = 1: (D - I)v = [1 1; 1 1]v = 0 → v₂ = [1, -1]'

**Verification:** D·v₁ = [2+1, 1+2]' = [3, 3]' = 3·[1, 1]' ✅

### Problem 6 (Matrix × vector — interleaving)
```
Ef = [1  0  2] · [2 ] = [1·2 + 0·(-1) + 2·3] = [8]
     [3  1  1]   [-1]   [3·2 + 1·(-1) + 1·3]   [8]
                 [3 ]
```

**Result:** [8, 8]' — a 2×1 vector (column vector with 2 entries).
E is 2×3, f is 3×1 → result is **2×1**.

### Problem 7 (Python)
```python
import numpy as np

D = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(D)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors (as columns):\n", eigenvectors)

# Verify: D @ v1 should equal lambda1 * v1
v1 = eigenvectors[:, 0]  # first column = first eigenvector
lambda1 = eigenvalues[0]

print("\nD @ v1 =", D @ v1)
print("lambda1 * v1 =", lambda1 * v1)
print("Match?", np.allclose(D @ v1, lambda1 * v1))
```

Expected output (NumPy normalizes eigenvectors to unit length):
```
Eigenvalues: [3. 1.]
Eigenvectors (as columns):
 [[ 0.70710678  0.70710678]
  [ 0.70710678 -0.70710678]]

D @ v1 = [2.12132034 2.12132034]
lambda1 * v1 = [2.12132034 2.12132034]
Match? True
```

Note: 0.70710678 ≈ 1/√2. NumPy normalizes eigenvectors so ‖v‖ = 1. The direction is the same as [1, 1]', just scaled.
