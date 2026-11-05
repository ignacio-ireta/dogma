# Claude — Anthropic (2023–2025)

*Constitutional AI. Safety-first. The model that chose to be careful.*

**← [[../_MOC|Scaling & Alignment]] ← [[../../_MOC|AI/ML Lineage]]**

---

## Timeline

- **2021:** Anthropic founded by Dario & Daniela Amodei (ex-OpenAI VP of Research). Mission: AI safety.
- **Mar 2023:** Claude 1 released. First model trained with Constitutional AI (CAI).
- **Jul 2023:** Claude 2. Improved reasoning, longer context (100K).
- **Mar 2024:** Claude 3 family — Haiku (fast), Sonnet (balanced), Opus (powerful). Opus matched GPT-4.
- **Jun 2024:** Claude 3.5 Sonnet — best-in-class for coding, analysis, writing.
- **Oct 2024:** Computer use beta — Claude can control a desktop.
- **Feb 2025:** Claude 3.7 Sonnet — hybrid reasoning (fast + extended thinking).
- **Apr 2025:** Claude 4 / Opus — state of art on complex reasoning.

## Constitutional AI (CAI)

The core innovation: instead of only RLHF (human feedback), Claude is trained with a *constitution* — a set of principles the model uses to evaluate and improve its own outputs. The AI critiques itself against principles, then a reward model is trained on these self-critiques.

**Why it matters:** Reduces dependence on human raters. Scales alignment. The model internalizes values rather than just mimicking preferences.

## Technical Lineage

- **Alignment:** [[../../12-Frontiers/Ouyang InstructGPT RLHF 2022|RLHF]] → Constitutional AI (RLAIF)
- **Architecture:** Transformer decoder, likely similar to GPT lineage but with safety-specific modifications
- **Context:** 200K tokens (Claude 3+), among the longest available

## Controversy

- Slower to release than competitors (safety-first = release-later)
- Some users find it *too* cautious (refusals on edge cases)
- The model that proved "safe" and "capable" aren't opposites

---

*The model that taught AI to say "I'm not sure." And meant it.*
