# Gemma — Google DeepMind (2024–2026)

*Open models from the Gemini lineage, sized for local and edge deployment.*

**← [[../_MOC|Scaling & Alignment]] ← [[../../_MOC|AI/ML Lineage]]**

---

## Timeline

- **Feb 2024:** Gemma 1 launches as Google DeepMind's lightweight open model family.
- **2025:** Gemma 3 expands multimodal and multilingual capability.
- **Apr 2026:** **Gemma 4** arrives as Google's strongest open model family so far, explicitly framed around reasoning, multimodality, and agentic workflows.

## Gemma 4 Family

Google released four main Gemma 4 sizes:

- **E2B:** effective 2B model for edge/mobile and low-latency local use.
- **E4B:** effective 4B model; notable in early independent benchmarking for strong accuracy-efficiency tradeoffs.
- **26B A4B MoE:** 26B total / ~3.8B active per token, optimized for latency and throughput.
- **31B Dense:** higher-quality dense model aimed at local workstation / single high-memory GPU use.

Capabilities highlighted by Google:

- advanced reasoning and multi-step planning
- function calling, structured JSON, and system instructions for agentic workflows
- code generation for local/offline developer use
- multimodal input: images/video across the family; audio emphasized for E2B/E4B
- long context: 128K for edge models, up to 256K for larger models
- 140+ languages
- Apache 2.0 license

## Paper / Evaluation Note

**Manik & Wang (2026), "Gemma 4, Phi-4, and Qwen3: Accuracy-Efficiency Tradeoffs in Dense and MoE Reasoning Language Models"**

This independent benchmark compares Gemma-4-E2B, Gemma-4-E4B, Gemma-4-26B-A4B, Phi-4 variants, and Qwen3 variants across ARC-Challenge, GSM8K, Math Level 1–3, and TruthfulQA MC1 under zero-shot, chain-of-thought, and few-shot CoT prompting.

Key result worth remembering: **Gemma-4-E4B with few-shot CoT had the best weighted overall score** in their setup (0.675 weighted accuracy, ~14.9 GB VRAM), while Gemma-4-26B-A4B was close in accuracy but much heavier (~48.1 GB VRAM). Their practical conclusion matters: sparse activation alone does not guarantee the best deployment point; the real tradeoff depends on architecture, prompting, task mix, latency, and memory.

**Local PDF:** `../../papers/manik-wang-gemma4-phi4-qwen3-accuracy-efficiency-2026.pdf`

## Lineage

- **Parent family:** [[Gemini — Google DeepMind 2023-2025|Gemini]] research stack, distilled/opened into smaller deployable forms.
- **Architecture:** Dense + MoE variants; part of the post-Transformer era where [[../12-Frontiers/Shazeer Sparsely-Gated Mixture-of-Experts 2017|MoE]] becomes a deployment strategy, not just a scaling trick.
- **Deployment philosophy:** Local-first / edge-first models, closer in spirit to practical systems engineering than pure leaderboard chasing.

## Why It Matters

Gemma is Google's open pressure valve: the proprietary frontier model remains Gemini, but Gemma lets developers run useful Google-derived intelligence locally, fine-tune it, and build offline or edge workflows. Gemma 4 is especially interesting because it suggests that "frontier" capability is fragmenting into **right-sized model families**: tiny multimodal edge models, fast MoEs, and dense quality anchors.

For our research library, Gemma 4 belongs next to Nemotron 3 because both are about the same 2026 shift: not just bigger models, but **deployable model ecosystems**.

---

*Gemini's open, practical younger sibling.*
