# Llama — Meta (2023–2025)

*Open weights that democratized LLMs.*

**← [[../_MOC|Scaling & Alignment]] ← [[../../_MOC|AI/ML Lineage]]**

---

## Timeline

- **Feb 2023:** LLaMA 1 (7B–65B). "Research only" license but leaked on 4chan. The genie was out.
- **Jul 2023:** Llama 2 (7B–70B). Commercial license. Free for most use cases.
- **Apr 2024:** Llama 3 (8B, 70B). Massive training data (15T tokens). 8B competitive with GPT-3.5.
- **Jul 2024:** Llama 3.1 (8B, 70B, 405B). 405B — the largest open-weight model. Competitive with GPT-4 class.
- **Dec 2024:** Llama 3.3 (70B). Improved reasoning, multilingual.
- **Apr 2025:** Llama 4 — Scout (109B MoE, 10M context) and Maverick (400B MoE). The Behemoth (nearly 2T params) announced.

## Why It Matters

Llama proved that open-weight models could compete with proprietary ones. It enabled:
- **Fine-tuning culture:** LoRA, QLoRA, RLHF on community hardware
- **Local deployment:** Running capable models on consumer GPUs
- **Startup ecosystem:** Every AI startup's first model was often Llama-based
- **Research:** Transparent architecture, training details published

## The Philosophy

Meta's play isn't to sell LLM access — it's to ensure the ecosystem doesn't consolidate around one provider. Open weights are a strategic move: commoditize the model layer, protect Meta's advertising and social media moats.

## Technical Lineage

- **Architecture:** Standard Transformer decoder, but at unprecedented scale and data quality
- **Data:** 15T+ tokens for Llama 3 — curating data quality became the competitive advantage
- **Fine-tuning:** [[../../12-Frontiers/Ouyang InstructGPT RLHF 2022|RLHF]] → DPO for instruction-tuned variants

---

*The model that proved open could compete.*
