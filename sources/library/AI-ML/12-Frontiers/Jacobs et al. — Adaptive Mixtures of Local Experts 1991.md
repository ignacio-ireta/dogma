# Jacobs et al. — Adaptive Mixtures of Local Experts (1991)

*This is the missing first MoE note. Before sparse routing became the cost-saving trick of modern frontier models, this paper gave the field the core intuition: let several experts compete, and let the system learn who should handle which inputs.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Adaptive Mixtures of Local Experts"** — Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, Geoffrey E. Hinton, 1991.
- Neural Computation, Spring 1991
- DOI: 10.1162/neco.1991.3.1.79
- PubMed: <https://pubmed.ncbi.nlm.nih.gov/31141872/>
- PDF online: <https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf>

## Key Ideas

1. **One model does not need to learn the whole world uniformly.**
   The paper starts from a very practical observation: some problems are naturally made of subproblems. If that is true, forcing one monolithic network to learn everything can be awkward and inefficient.

2. **Use multiple experts, each specializing on part of the task.**
   Jacobs et al. propose a set of expert networks plus a gating network. The experts each produce an output, and the gate learns how much each expert should matter for a given input.

3. **The gate learns a soft partition of the input space.**
   This is the elegant move. Instead of hard-coding rules like “expert A handles vowels, expert B handles consonants,” the model learns those boundaries from data. Different regions of the problem space naturally pull different experts into prominence.

4. **Competition, not averaging, drives specialization.**
   The training objective is designed so experts are rewarded for being especially good on subsets of cases, not for all collapsing toward the same average behavior. That competitive pressure is what creates true division of labor.

5. **This is an early form of conditional computation.**
   Modern MoE language models activate only a few experts per token to save compute. This paper does not yet do top-k sparse routing the modern way, but the deep idea is already here: not every parameter needs to do work on every example.

## The Core Intuition

The paper is really about **learned specialization**.

If different examples call for different reasoning styles, then intelligence may scale better when a system can say:
- this example belongs more to *that* specialist,
- this other example belongs more to *this* one,
- and the routing itself should be learned, not manually designed.

That is the ancestor of modern MoE.

## What They Actually Demonstrated

The paper shows the method on a **vowel discrimination task**. The important part is not the benchmark itself; it is the behavior:
- the system decomposes the problem into subtasks,
- simple experts become competent in their own local regions,
- the gate learns which expert should dominate where.

So the contribution is not "state of the art on vowels" 😏 — it is the proof that modular specialization can emerge from supervised learning.

## Lineage

- **Built on:** supervised neural networks, competitive learning, modular connectionist ideas
- **Leads to:** [[Jordan Jacobs Hierarchical Mixtures of Experts 1994|Hierarchical Mixtures of Experts]], then much later [[Shazeer Sparsely-Gated Mixture-of-Experts 2017|sparse MoE]], [[Fedus Switch Transformer 2021|Switch Transformer]], [[Du et al. — GLaM 2021|GLaM]], and [[../13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]]
- **Connected to:** softmax gating, conditional computation, modularity, ensemble intuitions, expert specialization

## Why It Matters for the Lineage

This is the paper that turns “maybe specialists would help” into a real learning architecture.

The 1994 Jordan & Jacobs paper gives the more mature hierarchical and EM-based formulation. The 2017 Shazeer paper makes the idea scale in the deep-learning era. But **1991 is where the basic MoE instinct becomes explicit and operational**:
- multiple experts,
- a learned gate,
- specialization through competition,
- better fit by dividing the task.

Without this note, the MoE lineage jumps too abruptly from modern sparse systems back to the later hierarchical refinement.

## The Real Historical Role

If [[Vaswani Attention Is All You Need 2017|Transformers]] changed *how information mixes*, this paper is part of the older lineage about *who should think at all*.

That question keeps coming back because it is a real one:
**should every part of the model wake up for every problem, or should intelligence be conditional and specialized?**

Jacobs et al. gave one of the first durable answers.

## Open Questions

- When does soft routing outperform hard sparse routing?
- How much expert specialization is semantic versus just geometric partitioning?
- Could future agentic systems bring back richer modular decomposition beyond today's token-level expert routing?
- Are modern MoE models rediscovering older ideas under new hardware constraints?

---

*1991 — the moment neural networks first learned that division of labor can be a feature, not a bug.*
