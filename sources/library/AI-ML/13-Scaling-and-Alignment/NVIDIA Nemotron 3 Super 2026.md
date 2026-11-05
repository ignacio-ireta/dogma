# NVIDIA Nemotron 3 Super (2026)

*Agentic models hit a throughput wall: long context, tool use, and reasoning traces were useful, but too expensive to serve if every layer kept behaving like a dense Transformer.*

**← [[_MOC|Scaling & Alignment]] ← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"** — NVIDIA, 2026.

- Paper: `corpus/raw/library-papers/2604.12374.pdf`
- arXiv: `2604.12374`
- Also mirrored locally as: `corpus/raw/library-papers/nvidia-nemotron-3-super-technical-report-2026.pdf`
- Model: **Nemotron 3 Super 120B-A12B** — 120B total parameters, about 12B active parameters

## Key Ideas

1. **Hybrid Mamba-Attention MoE.**
   Nemotron 3 Super is not a pure Transformer. It combines Mamba-2 blocks, a small number of global attention anchors, and sparse expert layers. The bet is simple: attention is still useful for global dependency modeling, but it is too expensive to spend everywhere during long-context generation.

2. **LatentMoE makes sparse routing hardware-aware.**
   Standard MoE saves compute by activating only a few experts, but it still pays painful memory-bandwidth and all-to-all communication costs. LatentMoE projects tokens into a smaller latent space before expert routing, reducing communication payloads and using the savings to activate more experts at roughly similar serving cost.

3. **Multi-token prediction is folded into the model, not bolted on afterward.**
   Nemotron 3 Super includes MTP layers so inference can use native speculative decoding. That matters for agents because long tool traces and coding sessions are dominated not only by answer quality, but by how many useful output tokens per second the system can produce.

4. **NVFP4 pretraining treats low precision as a training recipe.**
   The model is pretrained in NVIDIA's four-bit floating point format, not merely quantized after the fact. The paper reports stable 25T-token training and then produces FP8 and NVFP4 deployment checkpoints. In this era, numerical format is no longer a boring deployment detail; it is part of the architecture-economics story.

5. **Post-training targets agentic behavior directly.**
   The post-training stack uses SFT plus multi-environment RL across instruction following, reasoning, long context, tool use, terminal tasks, and software engineering. The paper's practical lesson is sharp: single-environment RL can improve one benchmark while damaging others, so the training environment has to look more like the deployment surface.

## What Came Before → What It Enabled

### What came before

- [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] made large-scale sequence modeling work, but full attention became the long-context tax.
- [[../12-Frontiers/Gu Dao Mamba 2023|Mamba]] and [[../12-Frontiers/Dao Gu Mamba-2 2024|Mamba-2]] reopened the architecture question by showing that state-space models could offer linear-time sequence processing.
- [[../12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017|Sparsely-Gated MoE]], [[../12-Frontiers/Fedus Switch Transformer 2021|Switch Transformer]], and [[Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]] made conditional computation a serious route to more capacity without dense compute.
- [[DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]] attacked the million-token problem with hybrid sparse attention, residual stabilization, and specialist-then-consolidate post-training.

### What it enabled

- A concrete open model recipe for **agentic throughput**: long context, tool use, coding, and reasoning without paying pure-Transformer costs at every layer.
- A stronger bridge between state-space models and frontier LLM deployment. Mamba is no longer just an alternative architecture paper; it becomes one component inside production-scale hybrid systems.
- A clearer hardware-software co-design line: model architecture, sparse routing, precision format, inference kernels, and post-training environments all have to be designed together.
- A practical open-weight competitor shaped around serving economics rather than only peak benchmark accuracy.

## Why It Matters

Nemotron 3 Super matters because it answers a different failure than GPT-3.

GPT-3 showed that scale could turn next-token prediction into broad capability. But once people started using models as agents, the bottleneck changed. The model had to keep long task state, call tools, write code, inspect outputs, revise plans, and generate many tokens before the work was done. Dense Transformer serving began to creak under that workload.

So NVIDIA's answer is not "just make the model larger." It is:

- use Mamba where recurrent state is cheaper than attention,
- keep attention only where global anchors are worth the price,
- use MoE for capacity without dense activation,
- make the MoE routing friendlier to hardware,
- train and deploy in low precision,
- and post-train against the actual agentic behaviors people now expect.

That is the lineage break: once models became collaborators instead of autocomplete boxes, throughput became capability. A slow agent is not merely inconvenient. It cannot afford enough search, revision, and tool interaction to behave intelligently in the first place.

Nemotron 3 Super is therefore a useful companion note to [[DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]]. DeepSeek-V4 says long context needs new attention and post-training consolidation. Nemotron 3 Super says agentic serving needs hybrid sequence models, hardware-aware MoE, and low-precision training as one system.

## Lineage

- **Built on:** [[../12-Frontiers/Dao Gu Mamba-2 2024|Mamba-2]], [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]], [[../12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017|Sparsely-Gated MoE]], [[Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]], [[DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]]
- **Contemporary with:** [[DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]], [[Gemini — Google DeepMind 2023-2025|Gemini long context]], [[Qwen — Alibaba 2023-2025|Qwen]], and production coding-agent stacks
- **Leads to:** open agentic model stacks where context length, reasoning budget, tool use, and inference throughput are first-class design constraints

---

*The frontier did not stop at "can the model think?" It moved to "can the model afford to keep thinking while doing real work?"*
