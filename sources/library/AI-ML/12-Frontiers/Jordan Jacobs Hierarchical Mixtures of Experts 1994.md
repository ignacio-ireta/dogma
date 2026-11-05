# Jordan & Jacobs — Hierarchical Mixtures of Experts (1994)

*The MoE idea needed more than a flat set of specialists. It needed learned structure, probabilistic routing, and a training rule that could assign credit to both gates and experts.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"Hierarchical Mixtures of Experts and the EM Algorithm"** — Michael I. Jordan and Robert A. Jacobs, 1994.

- Venue: Proceedings of the 1993 International Joint Conference on Neural Networks
- Paper path: `corpus/raw/library-papers/jordan-jacobs-hierarchical-mixtures-experts-1994.pdf`
- Source PDF status: local file verified as `PDF document, version 1.3, 6 pages` on 2026-07-19
- Built from earlier work: [[Jacobs et al. — Adaptive Mixtures of Local Experts 1991|Adaptive Mixtures of Local Experts]]

## Key Ideas

1. **Flat mixtures were useful, but too shallow.**
   [[Jacobs et al. — Adaptive Mixtures of Local Experts 1991|Jacobs et al. 1991]] showed that a learned gate could make experts specialize on different regions of a task. But a single flat gate still treated specialization as one-level partitioning. Real problems often have nested structure: broad regimes first, finer subregions later.

2. **HME turns MoE into a tree.**
   Jordan and Jacobs propose a **tree-structured architecture**. Gating networks sit at internal nodes, experts sit at leaves, and predictions flow upward through gate-weighted sums. That makes routing hierarchical: a top gate can choose a coarse region, then lower gates route within that region.

3. **The boundaries are soft, not brittle.**
   The paper explicitly says the model divides the input space into nested regions with **soft boundaries**, so a data point can partly belong to multiple regions. This keeps the system differentiable and avoids the brittleness of hard decision-tree splits.

4. **Experts and gates are generalized linear models.**
   Both the mixture components and mixture coefficients are framed as GLIM-style models. For regression, experts can be linear; for binary classification, outputs can be interpreted through logistic probabilities. The contribution is not just "neural experts," but a statistical model for conditional specialization.

5. **Learning is maximum likelihood with EM.**
   The central training move is an **Expectation-Maximization algorithm**. The missing data are the unobserved choices of which expert generated each data point. The E-step estimates posterior responsibilities; the M-step updates expert and gating parameters.

6. **Online learning keeps the architecture operational.**
   The paper also develops an incremental online learning algorithm. That matters historically because modular specialization is only useful if it can be trained in realistic data streams, not only in small batch demonstrations.

## What Came Before -> What It Enabled

### What came before

- [[Jacobs et al. — Adaptive Mixtures of Local Experts 1991|Adaptive Mixtures of Local Experts]] introduced learned gates and competing local specialists.
- Decision-tree-like methods such as CART and MARS showed that divide-and-conquer learning could be interpretable and efficient, but their splits were usually more explicit and less neural/probabilistic.
- Classical mixture-model statistics supplied the latent-variable view: a data point can be explained by an unobserved component assignment.

### Why the previous approach broke

The flat 1991 MoE recipe proved the specialization instinct, but it did not fully solve structure. One gate over one set of experts can partition a task, yet it cannot naturally express nested decisions such as "first choose the broad physical regime, then choose the local linear approximation inside it." As tasks grow, a flat mixture risks becoming either too coarse to model local variation or too wide to train and interpret cleanly.

The next step was necessary because MoE needed a way to scale specialization without turning routing into an unstructured expert soup. Hierarchy gave the routing model shape; EM gave it a principled credit-assignment rule.

### What it enabled

- A formal bridge from early MoE intuition to **probabilistic conditional computation**.
- A clearer ancestor for later sparse MoE systems: experts, gates, routing probabilities, load of responsibility, and learned specialization all become explicit.
- The conceptual path toward [[Shazeer Sparsely-Gated Mixture-of-Experts 2017|Shazeer et al. 2017]], where the same "not every parameter should wake up" idea becomes sparse, distributed, and large-scale.
- A useful contrast with modern MoE: 1994 routes examples through a hierarchy; modern LLM MoE often routes tokens through flat top-k expert layers.

## Why It Matters

Jordan & Jacobs matters because it repairs the first MoE idea before the deep-learning scaling era arrives.

The 1991 paper made specialization learnable. The 1994 paper asks the next necessary question: if specialists are useful, how should they be organized and trained? The answer is a tree of gates and experts, interpreted as a hierarchical mixture model and trained by EM.

That is the missing middle between "multiple experts can compete" and "sparse expert layers can make trillion-parameter models economical." Without this step, the lineage jumps too quickly from a clever modular neural-network idea to modern GPU-scale sparse routing.

The later LLM story changes the hardware, the unit of routing, and the scale. But the deeper failure mode is the same: a single dense model doing the same computation for every input wastes capacity and blurs specialization. HME is one of the early places where the field says: structure the work, learn the routing, and let different parts of the model become good at different parts of the world.

## Lineage

- **Built on:** [[Jacobs et al. — Adaptive Mixtures of Local Experts 1991|Adaptive Mixtures of Local Experts]], mixture models, generalized linear models, EM
- **Leads to:** [[Shazeer Sparsely-Gated Mixture-of-Experts 2017|Sparsely-Gated MoE]], [[Fedus Switch Transformer 2021|Switch Transformer]], [[Du et al. — GLaM 2021|GLaM]], [[../13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]]
- **Connected to:** conditional computation, soft routing, modular networks, decision trees, local linear models

---

*1994 — MoE learns that specialization needs an organization chart, not just a room full of experts.*
