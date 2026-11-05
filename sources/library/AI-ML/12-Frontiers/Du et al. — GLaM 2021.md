# Du et al. — GLaM (2021)

*The paper where sparse MoE stops being merely trainable and starts looking economically superior to dense giants.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"** — Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P. Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen, Claire Cui, 2021.
- Google Research
- ICML 2022 (arXiv preprint December 2021)
- [arXiv:2112.06905](https://arxiv.org/abs/2112.06905)

## Key Ideas

1. **Dense scaling was getting financially rude.**
   [[Brown GPT-3 2020|GPT-3]] proved that scale could buy startling generality, but the bill was ugly. GLaM starts from that pain point directly: if dense models keep using every parameter for every token, capability and cost remain chained together.

2. **Use a trillion-parameter model, but only wake a fraction of it per token.**
   GLaM is a sparse Transformer with mixture-of-experts feed-forward layers interleaved through the stack. The headline model has **1.2 trillion total parameters**, but activates only about **97 billion parameters per token**. That is the MoE promise in its cleanest modern form: huge capacity, much smaller active compute.

3. **Routing is sparse, but not as aggressively simplified as Switch.**
   [[Fedus Switch Transformer 2021|Switch Transformer]] pushed MoE toward top-1 routing for stability and simplicity. GLaM chooses **top-2 routing across 64 experts** in each MoE layer. So historically, GLaM is interesting because it is not just “Switch, but bigger.” It reopens the question of whether a little more routing richness can pay for itself in quality.

4. **The paper turns MoE into an energy-efficiency argument.**
   GLaM reports better overall zero-shot and one-shot/few-shot performance than GPT-3 across 29 NLP tasks while using about **one-third of the training energy** and about **half the inference FLOPs**. That is the moment sparse scaling stops sounding like an architectural curiosity and starts sounding like a strategic cost advantage.

5. **This is a systems-and-economics paper disguised as model scaling.**
   Yes, it is about language modeling. But the deeper claim is that the frontier no longer belongs automatically to the densest brute-force model. Architecture can bend the cost-quality curve.

## What GLaM Changes in the MoE Story

Very roughly:
- **[[Shazeer Sparsely-Gated Mixture-of-Experts 2017|Shazeer 2017]]** — modern sparse MoE becomes real
- **[[Lepikhin GShard 2020|GShard]]** — distributed sharding makes giant sparse training operationally possible
- **[[Fedus Switch Transformer 2021|Switch]]** — top-1 routing makes sparse trillion-scale training simpler
- **GLaM 2021** — sparse LMs make an explicit efficiency case against dense frontier models
- **[[../13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]]** — specialization gets refined further

That middle step matters. GLaM is one of the clearest demonstrations that MoE was not merely a scaling trick for researchers. It could be a better economic regime.

## Lineage

- **Built on:** [[Shazeer Sparsely-Gated Mixture-of-Experts 2017|Sparsely-Gated MoE]], [[Lepikhin GShard 2020|GShard]], and [[Vaswani Attention Is All You Need 2017|Transformers]]
- **In conversation with:** [[Fedus Switch Transformer 2021|Switch Transformer]], [[Brown GPT-3 2020|GPT-3]], and [[Kaplan Scaling Laws 2020|scaling laws]]
- **Leads to:** Mixtral-style sparse open models, [[../13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]], [[../13-Scaling-and-Alignment/DeepSeek — DeepSeek AI 2024-2025|DeepSeek]], and the broader quality-per-dollar race in frontier AI
- **Connected to:** enterprise inference economics, energy-aware model design, and the recurring industry question: *when should architecture innovation beat raw dense scale?*

## Why It Matters

GLaM is one of the papers that makes the 2024-2025 efficiency wars easier to understand.

Without it, the story can look too clean:
- dense models get big,
- MoE exists,
- then DeepSeek arrives and everyone suddenly cares about sparse efficiency.

But GLaM is an important bridge. It showed, very explicitly, that sparse large language models could beat a dense prestige model on quality while spending much less training energy. That is not just engineering trivia. It changes what counts as progress.

It says the frontier is not only about *more compute*. It is also about *better allocation of compute*.

## The Real Historical Role

If [[Fedus Switch Transformer 2021|Switch]] is the simplification step, GLaM is the proof-of-consequences step.

It is the paper that says: *fine, sparse MoE is not only trainable. It is strategically better than dense scaling in at least some important regimes.*

That is why it belongs in the lineage. It is one of the first papers where MoE starts looking like the adult answer to frontier compute budgets.

## Open Questions

- When does top-2 routing justify its extra complexity over Switch-style top-1 routing?
- How much of GLaM's advantage comes from MoE itself versus broader training recipe and data choices?
- Will the long-run winners be pure sparse Transformers, hybrid MoE systems, or newer mixtures that combine MoE with architectures like [[Gu Dao Mamba 2023|Mamba]]?

---

*GLaM is the moment sparse language modeling stops saying “look, we can scale too” and starts saying “we may be scaling more intelligently.”*