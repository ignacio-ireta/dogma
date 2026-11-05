# Hamilton — Quaternions (1843) and the Theory of Holomorphic Functions

#concept #paper #read #milestone

**Primary event:** William Rowan Hamilton discovers quaternions on October 16, 1843 (Broome Bridge, Dublin)
**arXiv survey paper:** Buchmann (2011), arXiv:1111.6088 — *A Brief History of Quaternions and of the Theory of Holomorphic Functions of Quaternionic Variables*
**PDF:** `papers/1111.6088.pdf`

---

## The Discovery

Hamilton spent a decade trying to extend complex numbers from 2D to 3D — a system with two imaginary units i and j where multiplication still works "nicely." He kept failing. The obstacle: in 3D, you can't have multiplicative inverses and be associative and work over ℝ. It's impossible.

On October 16, 1843, walking across Broome Bridge with his wife, it hit him: **you don't need three dimensions, you need four.** He carved the fundamental formula into the bridge:

```
i² = j² = k² = ijk = -1
```

Hamilton was so excited he carved it into the stone rather than find paper. The bridge still has a commemorative plaque.

---

## What Quaternions Are

ℍ (the quaternions) = { a + bi + cj + dk : a, b, c, d ∈ ℝ }

Rules:
- i² = j² = k² = -1
- ij = k, jk = i, ki = j
- **ji = -k, kj = -i, ik = -j** ← multiplication is *non-commutative*

This is a **skew field** (division ring): all the field axioms except commutativity of multiplication.

ℝ ⊂ ℂ ⊂ ℍ — each is a field extension of the previous, with dimensions 1, 2, and 4 over ℝ.

---

## Why Not 3D? (The Frobenius Theorem)

Hamilton wanted triplets (3D). It can't work. Buchmann's paper gives the elementary proof:

Suppose you have a 3D algebra over ℝ with units {1, i, j} where i² = j² = -1. What is ij?

It must be a + bi + cj for some a, b, c ∈ ℝ. Work through all cases (multiply on left and right by i and j, use i² = j² = -1) and you get contradictions in every case. There's no consistent assignment.

**Frobenius theorem (1877):** The only finite-dimensional associative division algebras over ℝ are:
- ℝ (dimension 1)
- ℂ (dimension 2)
- ℍ (dimension 4)

That's it. You can't do division in a nice algebraic sense in any other dimension.

**Broader theorem (Hurwitz, 1898):** If you allow non-associativity, dimension 8 also works — that's the **octonions** (ℕ). But lose associativity and most algebraic machinery breaks. ℍ is the last "nice" system before things go wrong.

---

## What Hamilton Was Actually After: Rotations

Hamilton didn't frame it this way initially, but quaternions are the natural language for 3D rotations. A unit quaternion (|q| = 1) represents a rotation in 3D space:

```
Rotate vector v by quaternion q: v' = q · v · q⁻¹
```

This is cleaner than rotation matrices (no gimbal lock), faster to compose, and numerically more stable. **This is why quaternions are in every game engine and robotics system today.** They're the standard representation for orientation in 3D space.

Hamilton wanted to describe physics in 3D. He got rotations in 3D instead — but via a 4D algebra. The universe made him go one dimension higher.

---

## The Holomorphicity Problem

When you go from ℝ to ℂ, you get **holomorphic functions** — a beautiful theory with:
1. A limit-based differentiability definition
2. Power series representations
3. Cauchy-Riemann equations (∂f/∂x = i · ∂f/∂y)

All three characterizations are **equivalent** for complex functions. Can you generalize this to ℍ?

Turns out: it's hard, and each approach gives different answers.

| Approach | Quaternionic result | Problem |
|----------|--------------------|---------| 
| Limit differentiability | f(q) = a + qb only | Too restrictive — only linear functions |
| Power series of monomials | Real-analytic functions | Nothing new |
| Cauchy-Fueter equations (1935) | Rich theory, but... | f(q) = q is **not** regular! |
| Slice-regularity (Gentili & Struppa 2007) | Polynomials, power series, identity all regular | Newest approach; requires holomorphicity on each complex slice |

The cleanest modern approach is **slice-regularity**: for each "complex slice" L_I = ℝ + ℝI (where I is a unit pure quaternion, I² = -1), restrict f to that slice and require it to be holomorphic there. This recovers all polynomials and convergent power series as regular, and has a Cauchy-type integral formula.

---

## Why This Is in the Library

This paper sits at the intersection of two threads:

**Mathematics thread:** Hamilton's quaternions (1843) come between the algebraic revolution (Galois, Abel) and the emergence of abstract algebra. They're one of the first "number systems" that violate a familiar axiom (commutativity). This expansion of what algebra can be — from "numbers that happen to multiply" to "algebraic structures defined by axioms" — is the conceptual step toward groups, rings, fields, and eventually category theory.

**AI/ML thread (indirect):** Quaternion-based neural networks and rotation representations appear in modern graphics and robotics. More deeply, the question "can we generalize calculus to higher-dimensional spaces?" drives differential geometry, which underpins the theoretical framework for understanding gradient descent and loss landscapes.

---

## Cross-Thread Connections 🔗

- **[[../../Mathematics/02-Algebraic-Revolution/]]** — Galois (1832) showed group structure of polynomial roots; Hamilton (1843) showed algebra could be non-commutative. Both expand what "algebra" means.
- **[[../../Mathematics/05-Newton-Leibniz/]]** — the question of holomorphic functions over ℍ is: "can we do calculus in quaternion space?" Cauchy-Riemann is the complex-calculus side; Fueter equations are the quaternionic attempt.
- **[[../../Mathematics/07-Limits-Analysis/]]** — Cauchy's epsilon-delta definition (1821) and the Cauchy-Fueter equations (1935) share more than a name: both are about what "limit" means in an algebraic structure.

---

## Interesting Thread: Algebra → Geometry → Physics

```
Hamilton quaternions 1843 → 3D rotation representation
  └── Gibbs/Heaviside 1880s → stripped to 3D vectors (dot/cross product)
       └── modern vector calculus
            └── meanwhile: rotation quaternions survive in computer graphics, aerospace, robotics
                 └── Geometric Algebra (Clifford 1878) generalizes both
```

Note: Gibbs and Heaviside simplified Hamilton's quaternions into the "vectors" physicists use. They thought they were making things cleaner. In some sense they were (3D vector calculus is simpler to learn). But they lost the multiplicative structure. Then in the 20th century, rotations came back as quaternions in applied fields.

---

## References Worth Chasing

- **Hamilton (1843)** — original letter to son (reproduced in Buchmann's appendix) — *primary source*
- **Frobenius (1877)** — theorem on real division algebras — *the deeper result*
- **Clifford (1878)** — *"Applications of Grassmann's Extensive Algebra"* — generalizes quaternions to geometric algebra
- **Hurwitz (1898)** — normed division algebras theorem (ℝ, ℂ, ℍ, octonions) — *completes the classification*
- **Gentili & Struppa (2007)** — slice-regularity — *the modern resolution of quaternionic holomorphicity*
- **Dirac (1928)** — gamma matrices in the Dirac equation use a 4×4 algebra related to quaternions — *physics connection*

---

## Note on the Symplectic Paper (math_0606400)

The papers directory also contains `math_0606400.pdf` — *"On Fundamental Groups of Symplectically Aspherical Manifolds II"* by Kędra, Rudyak & Tralle (2006). This is differential topology, quite far from the current library lineage threads. It may have been queued by accident or as a distant "maybe someday." No note created — keeping the queue clean. If the math thread ever extends to topology and symplectic geometry (post-Cantor: algebraic topology, differential geometry, physics applications), this would be a natural connection point.

---

## Status

- [x] Survey paper read, note created (March 10, 2026)
- [ ] Find Hamilton's original 1843 paper via Philosophical Magazine or Cambridge archives
- [ ] Explore Clifford algebras — bridge between quaternions and modern geometric algebra
- [ ] Check: does the AI/ML thread ever need quaternion neural networks? (e.g., rotation-equivariant architectures)
