# Lepikhin et al. — GShard (2020)

*The missing bridge between early MoE and Switch. Sparse experts meet large-scale systems engineering.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"** — Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen, 2020.
- Google Research
- [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
- Submitted June 30, 2020

## Key Ideas

1. **MoE needed a systems story, not just an architecture story.**
   [[Shazeer Sparsely-Gated Mixture-of-Experts 2017|Shazeer et al. 2017]] proved that sparse experts could unlock enormous model capacity without dense compute on every token. But training those models at truly giant scale was still painful. Routing is messy. Device placement is messy. Distributed programming is messy.

2. **GShard's real contribution is automatic sharding.**
   The paper introduces **GShard**, a set of lightweight annotation APIs plus an XLA compiler extension that automatically partitions large models across many accelerators. In plain language: instead of hand-writing heroic distributed-training code, you annotate a few tensors and let the compiler do much of the ugly work.

3. **Conditional computation finally scales into the hundreds of billions.**
   Using a Transformer with Sparsely-Gated MoE, the paper scales multilingual neural machine translation beyond **600 billion parameters**. That number matters historically. It showed sparse models were not just elegant toy ideas; they could survive at industrial scale.

4. **The paper is about the stack, not only the model.**
   GShard sits at the intersection of:
   - model architecture (MoE Transformers),
   - compiler infrastructure (XLA),
   - distributed systems (sharding across devices),
   - and training economics (making giant models practical enough to run).

5. **This is the hidden prehistory of trillion-parameter LLMs.**
   The paper reports training on **2048 TPU v3 accelerators in 4 days** for multilingual translation from 100 languages to English, with better quality than prior work. That is the real signal: sparse conditional computation plus tooling could produce models that were both massive and usable.

## What GShard Changed

The conceptual move is subtle but important.

- **Shazeer 2017** said: sparse experts can decouple capacity from per-token compute.
- **GShard 2020** said: now let's make that idea trainable across a ridiculous number of devices.
- **Switch 2021** said: and now let's simplify the routing further so the whole thing becomes easier to operationalize.

Without GShard, the MoE lineage has a hole in the middle. You can see the idea in 2017 and the simplification in 2021, but you miss the moment where sparse scaling became a serious systems program.

## Lineage

- **Built on:** [[Shazeer Sparsely-Gated Mixture-of-Experts 2017|Sparsely-Gated MoE]] and [[Vaswani Attention Is All You Need 2017|Transformers]]
- **Responds to:** the difficulty of scaling conditional computation across large accelerator fleets
- **Leads to:** [[Fedus Switch Transformer 2021|Switch Transformer]], GLaM-style sparse scaling, and later MoE production lines like [[../13-Scaling-and-Alignment/Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]]
- **Connected to:** [[Kaplan Scaling Laws 2020|Scaling Laws]] (the pressure to scale), multilingual foundation-model prehistory, and the compiler/systems side of frontier AI that people love to forget

## Why It Matters for the Lineage

GShard is the paper where MoE stops being mostly an architectural cleverness and becomes an infrastructure bet.

That matters because frontier AI is never just about the math. It is also about whether the training stack can carry the architecture. GShard showed that sparse models needed compiler and systems innovation to become real. The model alone was not enough.

This is exactly the kind of paper that disappears in casual retellings because it is less glamorous than "Transformers" or "GPT-3." But historically, it is one of the reasons trillion-parameter sparse models became believable.

## The Real Historical Role

If [[Shazeer Sparsely-Gated Mixture-of-Experts 2017|Shazeer]] invented modern sparse experts, GShard built them a machine room.

And if [[Fedus Switch Transformer 2021|Switch]] later made MoE simpler, GShard is the missing middle chapter that proved the hard scaling path was worth pursuing at all.

## Open Questions

- How much of later sparse-model success comes from better routing ideas versus better distributed systems tooling?
- Do future compiler stacks make sparse architectures easier enough that their operational complexity stops mattering?
- Will the next big efficiency leap come more from architecture, more from systems, or from the two being co-designed from the start?

---

*GShard is what happens when an architecture paper meets the very adult question: yes, but can 2048 machines survive it?*