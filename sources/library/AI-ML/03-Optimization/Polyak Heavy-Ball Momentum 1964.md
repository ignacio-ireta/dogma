# Polyak — Heavy-Ball Momentum (1964)

*Gradient descent learned to carry velocity instead of forgetting every previous step.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"Some Methods of Speeding Up the Convergence of Iteration Methods"** — Boris T. Polyak, *USSR Computational Mathematics and Mathematical Physics*, 1964.

- Paper: `corpus/raw/library-papers/polyak-heavy-ball-1964.pdf`
- DOI: `10.1016/0041-5553(64)90137-5`
- Source: English translation of the original journal article
- Local PDF verified as a 17-page PDF document

## Key Ideas

1. **One-step methods throw useful motion away.**  
   Ordinary gradient descent computes a direction, takes a step, and begins the next iteration as if the previous movement never happened. Polyak starts from a broader question: if an iterative method is slow, why restrict the next point to information from only the current point?

2. **A two-step method adds inertia.**  
   For a differentiable objective with gradient \(P(x)\), Polyak studies:

   ```text
   x_(n+1) = x_n - α P(x_n) + β(x_n - x_(n-1))
   ```

   The first term is the usual negative-gradient step. The second carries part of the previous displacement forward. Modern notation calls this **momentum**; the paper describes its continuous analogue as a small heavy sphere moving through a potential field.

3. **Momentum attacks ill-conditioning.**  
   Gradient descent is painfully slow in a long, narrow valley. It oscillates across the steep direction while making small progress along the shallow one. The inertia term damps that repeated side-to-side correction and preserves motion along the valley floor.

4. **The speedup is a change of scale, not a cosmetic constant.**  
   Near a strongly convex minimum, let \(M\) and \(m\) bound the largest and smallest curvature, with condition number \(\kappa=M/m\). Optimally tuned gradient descent needs on the order of \(\kappa\) iterations for a fixed reduction in error. Polyak's two-step method needs on the order of \(\sqrt{\kappa}\). The worse the conditioning, the more consequential the difference.

5. **Acceleration costs almost nothing per iteration.**  
   The method needs one extra stored displacement and a weighted addition. It avoids the expensive curvature calculations of Newton's method while recovering much of the speed that plain first-order descent loses.

6. **The objective need not decrease at every step.**  
   Inertia can carry the iterate uphill temporarily. Polyak explicitly warns that short stretches of increasing objective value are not necessarily failure. This is an important break from the comforting but restrictive idea that every optimization step must improve the loss immediately.

7. **The parameters create a stability bargain.**  
   Too little momentum gives little acceleration; too much produces oscillation or divergence. Polyak gives local convergence conditions and recommends empirical tuning when curvature bounds are unknown. Momentum makes first-order optimization faster, but also turns optimizer state into a dynamical system that must be controlled.

## What Came Before → What It Enabled

### What came before

- Cauchy's steepest descent supplied the one-step rule: move against the gradient.
- [[Robbins Monro Stochastic Approximation 1951|Robbins-Monro]] made sample-based, noisy updates mathematically legitimate.
- [[Widrow ADALINE 1960|Widrow-Hoff / LMS]] made an error-derived gradient-like update concrete for an adaptive neuron.

### What broke

The gradient was useful, but the optimizer was amnesiac.

On a well-conditioned bowl, forgetting the previous step is tolerable. On an ill-conditioned surface, the gradient points mostly across the narrow valley rather than toward the distant minimum. Plain descent repeatedly reverses direction, wastes movement on oscillation, and crawls along the dimension that matters.

They could not just keep taking independent steepest-descent steps because local steepness was not the same thing as useful long-run direction. The optimizer needed memory.

Polyak gave it velocity.

### What it enabled

- **Momentum SGD** carries Polyak's deterministic two-step idea into minibatch neural-network training, where averaging directions also suppresses some gradient noise.
- [[Kingma Ba Adam 2015|Adam]] combines an exponential moving average of gradients with adaptive second-moment scaling. Its first-moment state is momentum's direct descendant.
- [[Loshchilov AdamW 2019|AdamW]] inherits the same stateful update while fixing weight decay for adaptive optimization.
- Nesterov acceleration, RMSProp, and later optimizer families all operate in the world Polyak helped establish: an optimizer is not merely the current gradient, but a system with memory.

## Why It Matters

Polyak's heavy-ball method is the missing bridge between gradient descent and the optimizer stack used to train deep networks.

Robbins-Monro explains why noisy updates can work. Widrow-Hoff shows how an error signal can update weights. But neither solves the geometric waste of treating every update as an isolated event. Deep-learning loss surfaces are high-dimensional and badly conditioned; without some memory of direction, first-order training spends enormous effort undoing its own oscillations.

Momentum changes the question from:

> Where is downhill right now?

to:

> Where has downhill been carrying us?

That small change is foundational. Adam's moving average, modern momentum SGD, and large-model optimization recipes all depend on the idea that past gradients contain information the current gradient alone does not.

Generative AI therefore owes part of its trainability to a 1964 observation about iterative methods: when the landscape forms a trough, do not restart your motion at every point. Carry speed along the floor.

## Lineage

- **Built on:** steepest descent, multistep numerical methods, strongly convex optimization, and the problem of ill-conditioned linear systems.
- **Leads to:** momentum SGD, Nesterov acceleration, RMSProp, [[Kingma Ba Adam 2015|Adam]], and [[Loshchilov AdamW 2019|AdamW]].
- **Modern echo:** the `momentum` buffer in SGD and the first-moment estimate \(m_t\) in Adam are both optimizer memory: present updates shaped by accumulated direction.

---

*Gradient descent knew the slope. Momentum remembered the journey.*
