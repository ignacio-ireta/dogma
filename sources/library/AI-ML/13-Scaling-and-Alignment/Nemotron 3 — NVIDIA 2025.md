# Nemotron 3 — NVIDIA (2025)

*Hybrid Mamba-Transformer MoE. The GPU maker's own models.*

**← [[../_MOC|Scaling & Alignment]] ← [[../../_MOC|AI/ML Lineage]]**

---

## The Family (Dec 2025 — ongoing)

Three models: **Nano**, **Super**, **Ultra** — all using a novel hybrid architecture.

### Nemotron 3 Nano
- 3.2B active params (31.6B total) — MoE with <half activation
- Outperforms GPT-OSS-20B and Qwen3-30B-A3B on benchmarks
- 3.3× higher throughput than Qwen3-30B-A3B on single H200
- 1M token context, open weights + training data

### Nemotron 3 Super (Apr 2026 release)
- ~120B active parameters, competitive with GPT-OSS-120B and Qwen3.5-122B
- **2.2× and 7.5× higher inference throughput** respectively (8K input / 64K output)
- First model to use **LatentMoE** (Elango et al., 2026) — novel hardware-aware expert design
- NVFP4 training for efficiency
- Optimized for collaborative agents and high-volume workloads (IT ticket automation, etc.)

### Nemotron 3 Ultra
- Largest model, SOTA accuracy and reasoning
- Full LatentMoE + MTP + NVFP4 stack

## Key Architectural Innovations

1. **Hybrid Mamba-Transformer MoE:** Not pure Transformer. interleaves Mamba (state-space model) layers with attention layers inside an MoE framework. Best-in-class throughput without sacrificing accuracy.
2. **LatentMoE (Super/Ultra):** A novel approach that routes experts in a learned latent space rather than token space — gains accuracy without throughput penalty.
3. **Multi-Token Prediction (MTP):** Predicts multiple tokens per forward pass. Improves long-form generation efficiency and model quality. (Inspired by Gloeckle et al., 2024; DeepSeek-AI, 2025)
4. **NVFP4:** NVIDIA's custom 4-bit floating point format. Trained natively in FP4 rather than quantizing post-training.
5. **Multi-environment RL post-training:** Multiple diverse RL environments (not just single reward model) for broader capability.
6. **Granular reasoning budget control:** Inference-time compute scaling — control how much "thinking" the model does per query.


## Apr 2026 Update — Nemotron 3 as an Agentic Stack

NVIDIA's April 2026 Nemotron 3 push reframes the family less as one chatbot and more as a **coordinated model stack for production agents**:

- **Nemotron 3 Super:** long-context reasoning and agentic tasks; hybrid Mamba-Transformer MoE; 120B total / 12B active in the public Super checkpoint; 1M-token context; configurable thinking budget.
- **Nemotron 3 Content Safety:** compact multimodal/multilingual safety classifier for inline guardrails.
- **Nemotron 3 VoiceChat:** low-latency full-duplex voice interface.
- **Nemotron RAG:** image/text embeddings and reranking for multimodal retrieval.
- **Nano Omni / Ultra:** positioned as coming models for multimodal understanding and higher reasoning accuracy.

The interesting development is architectural and product-level at once: NVIDIA is treating agents as an ecosystem problem — reasoning, retrieval, safety, voice, and deployment efficiency — not just as a bigger base model.

**Local PDF:** `../../papers/nvidia-nemotron-3-super-technical-report-2026.pdf`

## The Paper

**"NVIDIA Nemotron 3: Efficient and Open Intelligence"** — Dec 2025
- [arXiv:2512.20856](https://arxiv.org/abs/2512.20856)
- [White Paper (PDF)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf)
- [Super Tech Report (PDF)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

## Lineage

- **Architecture:** Hybrid of [[../../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] + Mamba (state-space models) + [[DeepSeek — DeepSeek AI 2024-2025|MoE efficiency]]
- **Alignment:** Multi-environment RL (extends [[../../12-Frontiers/Ouyang InstructGPT RLHF 2022|RLHF]] paradigm)
- **Scale thinking:** Follows [[../../12-Frontiers/Hoffmann Chinchilla 2022|Chinchilla]] data efficiency + [[../../12-Frontiers/Kaplan Scaling Laws 2020|scaling laws]]

## Why It Matters

NVIDIA isn't just building GPUs anymore — they're building the models that run on them. This is vertical integration: chips → architecture → training → models → deployment. The hybrid Mamba-Transformer approach is the most serious challenge to pure Transformer dominance since the architecture was introduced. If it works at scale, it could reshape how everyone builds LLMs.

Also: they're open-sourcing the weights, training recipes, AND data (Nemotron-CC-v2.1 = 2.5T tokens). That's unusually transparent for a company this size.

---

*The GPU company that decided to build the software too.*
