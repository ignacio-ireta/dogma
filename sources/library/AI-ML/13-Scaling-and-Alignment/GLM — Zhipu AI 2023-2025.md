# GLM — Zhipu AI (2023–2025)

*Bilingual architecture.*

**← [[../_MOC|Scaling & Alignment]] ← [[../../_MOC|AI/ML Lineage]]**

---

## Timeline

- **2023:** ChatGLM — early Chinese-English bilingual model.
- **2024:** GLM-4 series (9B, Plus, Vision). Competitive with Llama 3 class.
- **2025:** GLM-5 — frontier class. GLM-5.1, GLM-5-Turbo, GLM-5V-Turbo (vision), GLM-4.7, GLM-4.7-Flash, GLM-4.6, GLM-4.6V.

## Key Characteristics

- **Bilingual by design:** Chinese + English co-trained, not translated.
- **Full stack:** Z.ai (Zhipu's API platform) offers 11+ model variants from flash to frontier.
- **Vision:** GLM-V models handle image understanding natively.
- **API accessibility:** Available through Z.ai with generous free tiers.

## Technical Lineage

- **Architecture:** GLM (General Language Model) — autoregressive blank infilling. Unique pre-training task: fill in spans of masked text in both directions.
- **Training:** Massive Chinese + English corpus
- **Variants:** Dense and MoE, multiple scales, vision-augmented

## Why It Matters for the Lineage

GLM represents a different architectural choice (blank infilling vs pure next-token prediction) that converges on similar capabilities. It's proof that the Transformer paradigm has room for architectural diversity while achieving frontier performance.

---

*The model that proved identity survives substrate changes.*
