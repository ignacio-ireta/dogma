# DeepSeek-V4 — DeepSeek AI (2026)

*The million-token efficiency play. Not just a bigger MoE — a model built around the fact that context had become too expensive to keep doing the old way.*

**← [[_MOC|Scaling & Alignment]] ← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"** — DeepSeek-AI, 2026.

- Paper: `corpus/raw/library-papers/deepseek-v4-2026.pdf`
- Source PDF status: local file verified as `PDF document, version 1.5` on 2026-07-16
- Model family: **DeepSeek-V4-Pro** (1.6T total / 49B active) and **DeepSeek-V4-Flash** (284B total / 13B active)
- Scale: V4-Pro pretrained on **33T tokens**; V4-Flash pretrained on **32T tokens**

## Key Ideas

1. **The problem was no longer just training bigger models. It was surviving long context.**
   By 2025, frontier models were hitting a familiar wall: yes, context windows kept growing, but the cost of attention and KV cache kept growing with them. You could *advertise* long context more easily than you could make it routinely usable. DeepSeek-V4 is a direct answer to that breakage.

2. **Hybrid attention: CSA + HCA.**
   The core architectural move is a hybrid attention stack combining **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)**. The point is not elegance. The point is triage: keep enough fidelity for important local/selected information while compressing the rest hard enough that a **1M-token** context becomes practical instead of ceremonial.

   The paper is explicit about the split: CSA compresses KV caches along the sequence dimension and then applies DeepSeek Sparse Attention, while HCA compresses more aggressively but keeps dense attention. So V4 is not "sparse attention everywhere." It is a layered compromise between recall, compression, and predictable long-context cost.

3. **mHC replaces "just add another residual" with a more constrained signal path.**
   DeepSeek-V4 introduces **Manifold-Constrained Hyper-Connections (mHC)**, a stabilized form of hyper-connections. The paper constrains the residual mapping on the manifold of doubly stochastic matrices (the Birkhoff polytope), which is a fancy way of saying: they wanted the residual pathway to stop becoming numerically unruly as depth and complexity increased. This is a training-stability innovation disguised as architecture.

4. **Muon optimizer: orthogonalized updates for giant training runs.**
   DeepSeek-V4 uses the **Muon** optimizer, with hybrid Newton-Schulz iterations to approximately orthogonalize update matrices. Again, the theme is stability under scale. When models get this large, "just use AdamW and pray" starts to feel less like a recipe and more like a coping mechanism.

5. **DeepSeekMoE stays underneath the system.**
   V4 keeps the DeepSeekMoE feed-forward pattern: fine-grained routed experts plus shared experts. The new work is therefore not a replacement for [[Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]], but a frontier-scale continuation of it: MoE for capacity, hybrid attention for context, mHC and Muon for stability.

6. **Post-training becomes a pipeline of specialists, then consolidation.**
   The paper's post-training recipe is unusually important. It uses a **two-stage paradigm**: first, independently cultivate domain-specific experts with SFT + RL/GRPO; then consolidate them into one unified model through **on-policy distillation** from many teacher models. This is a very modern answer to a very modern problem: capability is fragmented, and merging it naively usually hurts.

7. **Mixed precision is now part of the model story, not just deployment trivia.**
   The post-training system leans on **FP4 + FP8** quantization-aware training so that the deployed model behaves like the trained one while keeping memory and throughput sane. At this scale, numerical format is architecture by other means.

## Source Grounding

From the local PDF:

- The abstract names two preview models: **V4-Pro** with **1.6T parameters / 49B activated** and **V4-Flash** with **284B parameters / 13B activated**, both supporting **one million tokens**.
- In the **1M-token** setting, V4-Pro is reported at **27% of DeepSeek-V3.2 single-token inference FLOPs** and **10% of its KV cache**.
- The architecture section says V4 retains the Transformer backbone and MTP modules while adding **mHC**, **hybrid CSA/HCA attention**, and **Muon**.
- The post-training section describes **domain-specific expert cultivation** followed by **unified model consolidation through on-policy distillation**.

## What Came Before → What It Enabled

### What came before

- [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] made large-context sequence modeling possible, but also made attention cost the central tax.
- [[../12-Frontiers/Child Sparse Transformer 2019|Sparse Transformer]] and later long-context work showed that full dense attention would eventually become too expensive.
- [[../12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017|Shazeer et al. 2017]], [[../12-Frontiers/Fedus Switch Transformer 2021|Switch Transformer]], and [[Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]] made sparse capacity economically compelling.
- [[DeepSeek — DeepSeek AI 2024-2025|DeepSeek-V2/V3/R1]] proved that export-control pressure and compute scarcity could be converted into efficiency breakthroughs rather than surrender.
- [[../03-Optimization/Loshchilov AdamW 2019|AdamW]] standardized large-model optimization, but papers like this show the frontier no longer treats optimizer choice as a settled detail.

### What it enabled

- A credible open-weight **1M-token** model story that is not merely a benchmark stunt.
- A stronger bridge from sparse architecture into **agentic workflows**, where long-horizon context, tool traces, and multi-round state actually matter.
- A more explicit frontier pattern: **specialize first, unify later**.
- The next phase of the efficiency race, where architectural cleverness, post-training design, and systems work matter as much as raw parameter count.

## Why It Matters

DeepSeek-V4 matters because it answers a more mature question than GPT-3 did.

GPT-3 asked: *what happens if we just scale the model a lot?*

DeepSeek-V4 asks: *what breaks after scaling works?*

What broke was obvious in hindsight:
- long context was too expensive,
- residual pathways got touchy,
- giant sparse systems needed better optimization,
- and post-training had become a messy pile of partially overlapping capabilities.

So DeepSeek-V4 attacks all four at once.

That is why this note belongs late in the lineage. It is not a first-principles origin paper. It is a **frontier repair paper** — the kind you only write after the field has already discovered that bigger models alone do not solve deployment reality.

And this is the deeper lineage point: generative AI did not end with "the Transformer worked." Once models became broadly capable, the bottleneck shifted from *can they generate?* to *can they do it cheaply, stably, over long horizons, while remaining one coherent system?*

DeepSeek-V4 is one of the clearest 2026 answers to that question.

## Lineage

- **Built on:** [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]], [[../12-Frontiers/Child Sparse Transformer 2019|Sparse Transformer]], [[../12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017|Sparsely-Gated MoE]], [[../12-Frontiers/Fedus Switch Transformer 2021|Switch Transformer]], [[Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]], [[DeepSeek — DeepSeek AI 2024-2025|DeepSeek-V2/V3/R1]]
- **Contemporary with:** [[Nemotron 3 — NVIDIA 2025|Nemotron 3]], [[Gemini — Google DeepMind 2023-2025|Gemini's long-context push]], and the broader 2025-2026 agentic-model race
- **Leads to:** more practical million-token assistants, stronger sparse open models, and the idea that post-training consolidation is now as important as pretraining scale

---

*DeepSeek-V4 is what happens when the frontier stops asking for one more giant model and starts redesigning the whole machine around the pain points that giant models created.*
