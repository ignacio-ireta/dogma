# 13 — Scaling & Alignment Era (2022–2026)

*From raw capability to useful, safe, multimodal intelligence.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Shift

After [[../../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] and [[../../12-Frontiers/Brown GPT-3 2020|GPT-3]], the field bifurcated:
- **Scaling:** Bigger models, more data, emergent capabilities
- **Alignment:** Making models useful, safe, and instruction-following

[[../../12-Frontiers/Christiano et al. — Deep RL from Human Preferences 2017|Christiano et al. (2017)]] supplied the core preference-learning template. [[../../12-Frontiers/Stiennon et al. — Learning to Summarize from Human Feedback 2020|Stiennon et al. (2020)]] proved that the recipe works for language generation. [[Schulman et al. — Proximal Policy Optimization 2017|PPO]] provided the practical optimizer that made reward-guided policy updates stable enough to scale. [[../../12-Frontiers/Ouyang InstructGPT RLHF 2022|RLHF]] (2022) fused those pieces for general instruction-following language models. [[Bai et al. — Constitutional AI 2022]] (Dec 2022) showed that AI could critique itself against principles. After that, the pace became explosive.

## The Modern Landscape

### Open-Weight Foundation Models
- [[Touvron et al. — LLaMA 2023]] — The source-paper hinge for open-weight foundation models: smaller GPT-style models trained longer on public data, making strong LLMs portable enough for research and local adaptation.
- [[Llama — Meta 2023–2025]] — Open weights that democratized LLMs
- [[Qwen — Alibaba 2023–2025]] — Multilingual excellence, Chinese-first but globally competitive
- [[DeepSeek — DeepSeek AI 2024–2025]] — Efficiency breakthroughs, MoE architecture
- [[DeepSeek-V4 — DeepSeek AI 2026]] — Source-grounded million-token efficiency note: V4-Pro (1.6T total / 49B active), V4-Flash (284B / 13B active), hybrid CSA/HCA attention, mHC, Muon, DeepSeekMoE, FP4/FP8-aware training, and specialist-then-consolidate post-training
- [[Gemma — Google DeepMind 2024-2026]] — Google open models for local, edge, multimodal, and agentic workflows
- [[Manik Wang Gemma Phi Qwen Tradeoffs 2026]] — Deployment-aware comparison of Gemma 4, Phi-4, and Qwen3 dense/MoE reasoning models across accuracy, latency, VRAM, tasks, and prompting regimes
- [[Nemotron 3 — NVIDIA 2025]] — Hybrid Mamba-Transformer MoE stack for agentic systems
- [[NVIDIA Nemotron 3 Super 2026]] — Source-paper note for the Super model: 120B total / 12B active, hybrid Mamba-Attention MoE, LatentMoE, MTP, NVFP4, 1M context, and multi-environment RL for agentic workloads
- [[GLM — Zhipu AI 2023–2025]] — Bilingual (EN/ZH) architecture

### Proprietary Frontier Models
- [[ChatGPT — OpenAI 2022–2025]] — The product that changed everything
- [[Claude — Anthropic 2023–2025]] — Constitutional AI, safety-first, long context
- [[Gemini — Google DeepMind 2023–2025]] — Multimodal from birth, natively
- [[Kimi — Moonshot AI 2024–2025]] — Long-context Chinese model

## Key Architectural Innovations
- [[../12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017|MoE (Mixture of Experts)]] — sparse activation, more capacity without proportional compute
- [[Dai et al. — DeepSeekMoE 2024|DeepSeekMoE]] — fine-grained experts + shared experts. A more deliberate path to expert specialization, and the architectural bridge into DeepSeek's efficiency story.
- [[DeepSeek-V4 — DeepSeek AI 2026]] — hybrid CSA/HCA attention, mHC residual stabilization, Muon optimization, DeepSeekMoE continuation, and the clearest open-weight 1M-context efficiency push so far.
- [[NVIDIA Nemotron 3 Super 2026]] — hybrid Mamba-Attention MoE, LatentMoE, MTP, NVFP4 pretraining, and multi-environment RL. The agentic-throughput answer to long-context serving costs.
- [[Manik Wang Gemma Phi Qwen Tradeoffs 2026]] — prompt-conditioned Pareto evaluation. The model-selection problem becomes workload + hardware + prompting, not just benchmark rank.
- **Human preferences → PPO-based RLHF → CAI (Constitutional AI) → DPO** — alignment techniques evolving rapidly
- [[Schulman et al. — Proximal Policy Optimization 2017|PPO]] — the practical RL workhorse inside early RLHF pipelines
- [[DeepSeek-AI — DeepSeek-R1 Reasoning via RL 2025|DeepSeek-R1 (2025)]] — reasoning incentivized by **pure RL against verifiable rewards** (GRPO), no SFT warm-up and no human reasoning traces. The moment test-time compute becomes a scaling axis and imitation stops being the only route to reasoning.
- **Multimodal fusion** — vision + language + audio in single models
- **Long context** — from 2K to 1M+ tokens
- **Tool use / function calling** — models that act, not just speak

---

#moc #ai-ml #library #scaling #alignment
