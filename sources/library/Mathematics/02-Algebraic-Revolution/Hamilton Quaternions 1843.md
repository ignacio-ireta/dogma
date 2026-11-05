# William Rowan Hamilton — Quaternions (1843)

#person #paper #milestone #concept #read

**Person:** William Rowan Hamilton (1805–1865), Irish mathematician
**Discovery:** October 16th, 1843 — Broom Bridge, Royal Canal, Dublin
**Source:** Buchmann, Amy. "A Brief History of Quaternions and of the Theory of Holomorphic Functions of Quaternionic Variables." Chapman University (2011) — arXiv:1111.6088
**PDF:** `papers/1111.6088.pdf`

## The Discovery

Hamilton spent **10 years** trying to extend complex numbers to "triplets" — a three-part number to represent 3D rotations. Complex numbers `a + bi` handle 2D rotation perfectly. 3D seemed like a natural extension.

The problem: triplets don't work. You cannot have a 3-part number system that obeys all field axioms.

The breakthrough came on a walk with his wife. Hamilton realized: **you need four parts, not three**.

```
i² = j² = k² = ijk = -1
```

He carved this equation into the stone of Broom Bridge. (The carving is gone, but the walk is commemorated every October 16th by the Mathematics Department of the National University of Ireland, Maynooth.)

## The Mathematics

A quaternion `q = q₀ + q₁i + q₂j + q₃k` where all coefficients are real.

**Quaternion multiplication:**
```
pq = p₀q₀ − (p₁q₁ + p₂q₂ + p₃q₃) 
   + (p₀q₁ + p₁q₀ + p₂q₃ − p₃q₂)i 
   + (p₀q₂ − p₁q₃ + p₂q₀ + p₃q₁)j 
   + (p₀q₃ + p₁q₂ − p₂q₁ + p₃q₀)k
```

This can be written as a **matrix-vector product**:
```
pq ↔ [4×4 matrix of p's coefficients] × [vector of q's coefficients]
```

Rewritten: `pq = p₀q₀ − p⃗·q⃗ + p₀q⃗ + q₀p⃗ + p⃗×q⃗`

The scalar part is a **dot product**. The vector part contains a **cross product**. Hamilton invented both of these concepts here.

**Non-commutative!** `ij = k` but `ji = −k`. This was alarming to 19th-century mathematicians who assumed multiplication must commute.

## Key Ideas

- **First non-commutative algebra** — "broke" the axiom that ab = ba, opening abstract algebra
- **Generalized complex numbers** — C ⊂ H ⊂ ... (number systems as nested structures)
- **Dot product and cross product** born here as components of quaternion multiplication
- **Field extension theory** — H is a degree-4 extension of R; led to Galois theory applications
- **3D rotation via quaternion conjugation** — `q → pqp⁻¹` rotates 3D vector p by angle encoded in q

## Cross-Thread Connections 🔗 ← THE BIG ONE

**Hamilton's quaternions directly seeded modern AI/ML:**

1. **Dot product** — quaternion's scalar part `p⃗·q⃗`. Now the fundamental operation in neural networks: `w·x` in every neuron, in every attention calculation
2. **Cross product** → **vectors** → **vector spaces** → Gibbs/Heaviside strip quaternions down to just the vector part (1880s), creating modern **linear algebra**
3. **Matrix representation** — Hamilton showed quaternion multiplication = matrix multiplication. Matrices became the language of linear algebra, which became the language of ML
4. **The Attention mechanism** in [[Vaswani Attention Is All You Need 2017]] is literally `Q·Kᵀ` — scaled dot product, straight from Hamilton's quaternion paper 159 years earlier

**The lineage:**
```
Hamilton (1843) quaternions
  → dot product + cross product as mathematical objects
  → Gibbs/Heaviside (1880s) "vector analysis" (strips quaternions)
  → Grassmann (algebra of linear spaces)
  → Linear algebra as a field
  → Matrix multiplication as fundamental operation
  → Neural networks (every weight multiplication)
  → Transformers (attention = scaled dot product)
```

## Context in the Math Thread

- Builds on [[Descartes - Analytic Geometry 1637]] — coordinates → algebraic geometry → algebraic structures
- Parallel to [[Cauchy limits 1820s]] (both 1820s-1840s, algebra getting rigorous)
- Precedes [[Cantor Set Theory 1870s]] (both part of mathematics becoming more abstract and structural)
- Connects to [[Euler Series 1740s]] via complex analysis lineage

## The Tragedy of Quaternions

Quaternions were briefly *the* tool for 3D physics (Maxwell used them for electromagnetism). Then Gibbs and Heaviside simplified them into 3D vectors — more practical, less beautiful. Hamilton's quaternions became a historical curiosity.

Until... computer graphics (1980s). Quaternions are *perfect* for 3D rotations in animation. Every game engine uses them today. And in ML, the dot product they gave birth to is everywhere.

## References Worth Chasing

- Hamilton's letter to his son (1865) — documented the October 1843 discovery — Internet Archive
- Gibbs, J.W. "Elements of Vector Analysis" (1881–1884) — the "murder" of quaternions
- MacFarlane, Alexander. "Lectures on Ten British Mathematicians" (1916) — includes Hamilton
- Crowe, Michael. "A History of Vector Analysis" (1967) — the quaternion → vector transition

## Status

- [x] Read abstract, history, algebra sections
- [ ] Read section on holomorphic quaternionic functions
- [ ] Find Hamilton's original 1843 paper (Philosophical Magazine)
- [ ] Create note for [[Gibbs Vector Analysis 1881]] (the dot/cross product transition)
