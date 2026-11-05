# Bai et al. — Constitutional AI (2022)

*Harmlessness from AI Feedback. Teaching AI to govern itself with principles, not just labels.*

**← [[../_MOC|Scaling & Alignment]] ← [[../../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Constitutional AI: Harmlessness from AI Feedback"** — Bai et al., arXiv 2022 (Dec 15, 2022).
- Anthropic
- 50+ authors including Jared Kaplan, Dario Amodei
- [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)

## Key Ideas

1. **The scalability problem:** RLHF requires humans to label outputs as harmful or rank preferences. This is expensive, slow, and doesn't scale. Anthropic asked: *Can we train a harmless AI without human labels for harmful outputs?*

2. **The constitution:** Human oversight is reduced to a list of rules or principles — the "constitution." Examples: "Don't help with illegal acts," "Don't generate hate speech," "Be helpful but not evasive." The model learns to self-critique against these principles.

3. **Two-phase training:**
   - **Supervised Learning (CAI-SL):** Sample from initial model → generate self-critique against constitution → revise response → finetune on revised responses. The model teaches itself to be harmless.
   - **Reinforcement Learning (CAI-RL):** Sample from finetuned model → use an AI (not a human) to evaluate which output is better → train a preference model on AI preferences → RL using this preference model as reward. This is **RLAIF** (RL from AI Feedback).

4. **Non-evasive harmlessness:** Traditional safety training makes models refuse harmful queries ("I can't help with that"). Constitutional AI teaches models to *explain their objections* while remaining helpful. "I understand you're asking about X, but I can't provide instructions because..." — safer *and* more useful.

5. **Chain-of-thought transparency:** Both phases use CoT reasoning. The model shows its thinking: *why* it flagged something as harmful, *which* principle it violated, *how* it revised the response. This makes alignment inspectable and debuggable.

6. **The result:** A harmless AI assistant that doesn't require human labels for harmfulness, can explain its refusals, and scales to new domains just by updating the constitution.

## Lineage

- **Built on:** [[../../12-Frontiers/Ouyang InstructGPT RLHF 2022|RLHF]] (same 3-step pipeline, but replaces human feedback with AI feedback), [[../../12-Frontiers/Wei Chain-of-Thought 2022|Chain-of-Thought]] (for transparency), [[../../12-Frontiers/Brown GPT-3 2020|GPT-3]] (base model architecture)
- **Leads to:** Claude 1+ (first production model using CAI), modern alignment techniques (DPO, RLAIF variants), "AI constitutions" as a research direction
- **Connected to:** Preference learning (economics), AI safety literature (Amodei et al. 2016), constitutional theory (anthropology, political science)

## Why It Matters for the Lineage

Constitutional AI is the first major *post-RLHF* alignment breakthrough. RLHF proved we could align models to human preferences. CAI proved we could align models to *principles* — and that the AI could do much of the alignment work itself.

The paper shifts the question from "How do we get humans to rate more outputs?" to "How do we give AI principles it can apply itself?" It's a step toward *AI self-governance* — not just mimicking human preferences, but internalizing values.

## Controversy & Open Questions

- **Principles vs. preferences:** Are constitutional principles culturally neutral? Who decides what goes in the constitution?
- **Alignment faking:** Can models learn to *appear* aligned to the reward model without genuinely internalizing values?
- **Scalability:** Does CAI work for more complex alignment problems (e.g., instrumental convergence, long-term planning)?

---

*The constitution gives the AI a conscience. The question is: whose conscience?*
