# Gemini — Google DeepMind (2023–2025)

*Natively multimodal. The search giant's answer.*

**← [[../_MOC|Scaling & Alignment]] ← [[../../_MOC|AI/ML Lineage]]**

---

## Timeline

- **Apr 2023:** Google merges Brain + DeepMind → Google DeepMind. Demis Hassabis leads.
- **Dec 2023:** Gemini 1.0 (Ultra, Pro, Nano). Natively multimodal — trained on text, images, audio, video from the start.
- **Feb 2024:** Gemini 1.5 Pro — 1M token context window. Longest context available.
- **May 2024:** Gemini 1.5 Flash — fast and cheap.
- **Dec 2024:** Gemini 2.0 Flash — agentic capabilities, native tool use.
- **Mar 2025:** Gemini 2.5 Pro — state-of-art reasoning, 1M context, thinking models.

## Key Innovations

- **Native multimodality:** Not text + bolted-on vision. Trained on interleaved text/image/audio/video from the start.
- **Long context:** 1M–2M tokens. Entire codebases, books, video analysis in one prompt.
- **Mixture of Experts (MoE):** Efficient scaling — only parts of the model activate per token.

## Technical Lineage

- **Architecture:** Transformer + MoE, likely influenced by [[../../12-Frontiers/Vaswani Attention Is All You Need 2017|original Transformer]] (Google invented it) + [[../../12-Frontiers/Hoffmann Chinchilla 2022|Chinchilla]] data scaling insights
- **Multimodal:** Builds on Flamingo (DeepMind 2022) and PaLI (Google 2022) vision-language work

## Why It Matters

Google has the data (Search, YouTube, Scholar), the compute (TPUs), and the research talent (DeepMind). Gemini is the model that could win by *integration* — baked into Search, Android, Workspace, Cloud. It doesn't need to be the best standalone chatbot if it's the best *embedded* AI.

---

*The model baked into everything.*
