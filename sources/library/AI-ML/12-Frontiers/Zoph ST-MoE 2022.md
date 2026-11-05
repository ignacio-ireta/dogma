# Zoph et al. — ST-MoE (2022)

*Sparse experts become stable enough to transfer.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Title

**ST-MoE: Designing Stable and Transferable Sparse Expert Models**

## Paper/Year

Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus — 2022.

- Google Brain / Google Research
- arXiv:2202.08906
- Version checked: arXiv v2, April 29, 2022

## Paper path

`corpus/raw/library-papers/2202.08906.pdf`

Verified locally as: `PDF document, version 1.5`

## Key Ideas

1. **MoE had a transfer problem, not just a scaling problem.**
   [[Fedus Switch Transformer 2021|Switch Transformer]] showed that sparse expert models could train faster and scale to trillion-parameter regimes. [[Du et al. — GLaM 2021|GLaM]] showed sparse language models could make a serious energy-efficiency argument. But earlier sparse models could still behave badly when fine-tuned: strong pretraining speedups did not reliably become downstream transfer gains.

2. **Training stability needed a router-specific fix.**
   The paper studies sparse-model instability directly and introduces the **router z-loss**, an auxiliary loss that penalizes large router logits. This stabilizes the gating softmax without the quality damage caused by blunt fixes such as overly tight update clipping or removing useful multiplicative components.

3. **Sparse and dense models need different fine-tuning protocols.**
   ST-MoE shows that reusing dense-model fine-tuning defaults can hide sparse-model gains. Sparse models are more prone to overfitting on small tasks, and their best learning-rate / batch-size choices can differ from dense baselines. The lesson is practical: MoE is not only a pretraining architecture; it changes the post-training workflow.

4. **Capacity factor, routing, precision, and communication costs are part of model quality.**
   The paper treats expert capacity and routing choices as engineering knobs with real speed-quality tradeoffs. Larger capacity can reduce token dropping but increases einsum, activation-memory, all-to-all, and model-parallel communication costs. Sparse models are only efficient if the routing system and hardware layout cooperate.

5. **ST-MoE-32B proves the design guide at scale.**
   The final model has **269B sparse parameters** with compute comparable to a **32B dense encoder-decoder Transformer**. The paper reports state-of-the-art transfer performance across reasoning, summarization, closed-book QA, and adversarial tasks, showing that sparse models can be both efficient and fine-tunable.

## What Came Before -> What It Enabled

- **Came before:** [[Shazeer Sparsely-Gated Mixture-of-Experts 2017|sparsely-gated MoE]], [[Lepikhin GShard 2020|GShard]], [[Fedus Switch Transformer 2021|Switch Transformer]], and [[Du et al. — GLaM 2021|GLaM]].
- **The gap:** sparse models had become large and efficient, but reliability lagged. Instability and poor fine-tuning transfer meant the architecture could look great during pretraining and disappointing on downstream tasks.
- **Enabled:** a more mature sparse-model recipe: router z-loss for stability, sparse-specific fine-tuning, capacity-factor tuning, precision awareness, and model design that optimizes upstream pretraining and downstream transfer together.
- **Leads to:** [[../13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]], DeepSeek-style expert specialization, and the broader frontier-model habit of treating architecture, optimizer stability, precision, routing, and post-training as one system.

## Why It Matters

ST-MoE is the bridge between "MoE can scale" and "MoE can be dependable."

That distinction matters. A sparse model that trains quickly but fails to transfer cleanly is not a frontier recipe; it is an expensive demonstration. Zoph et al. make the sparse-scaling story more operational by asking where sparse models break, how router dynamics destabilize training, and why downstream fine-tuning cannot be treated as an afterthought.

For the lineage, this note sits naturally between [[Du et al. — GLaM 2021|GLaM]] and [[../13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]]. GLaM makes the cost-quality case. ST-MoE makes the stability-transfer case. DeepSeekMoE then pushes expert specialization itself further.
