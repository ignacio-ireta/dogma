# Ouyang et al. — InstructGPT / RLHF (2022)

*Teaching language models to follow instructions with human feedback.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Training language models to follow instructions with human feedback"** — Long et al. (Ouyang et al.), NeurIPS 2022.
- OpenAI
- [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

## Key Ideas

1. **The alignment problem:** Raw GPT-3 completes text — it doesn't necessarily *follow instructions*. Fine-tuned on internet text, it can be helpful, harmful, or nonsensical depending on the prompt.

2. **Three-step RLHF pipeline:**
   - **SFT (Supervised Fine-Tuning):** Human demonstrators write ideal responses to prompts. Model learns the format.
   - **Reward Model:** Humans rank multiple model outputs. A separate model learns to predict human preferences → scalar reward signal.
   - **[[../13-Scaling-and-Alignment/Schulman et al. — Proximal Policy Optimization 2017|PPO (Proximal Policy Optimization)]]:** The language model is fine-tuned to maximize reward model score, with a KL penalty to prevent drift from the SFT model.

3. **Why it matters:** This is the bridge between "GPT-3 as a text completer" and "ChatGPT as a useful assistant." The same base model, aligned through human feedback, becomes dramatically more useful.

## Lineage

- **Built on:** [[Brown GPT-3 2020]] (base model), [[Wei Chain-of-Thought 2022]] (emergent capabilities at scale), and [[../13-Scaling-and-Alignment/Schulman et al. — Proximal Policy Optimization 2017|PPO]] as the optimization engine for preference tuning
- **Leads to:** Every instruction-tuned model since — ChatGPT, Claude, Gemini, Llama variants
- **Connected to:** RL from policy optimization literature, preference learning from economics

## Why It Matters for the Lineage

RLHF is the *second* major breakthrough after Transformers. Attention gave us models that could process sequences. RLHF gave us models that could *listen*. Every model people use daily (including me) is a product of this pipeline.

---

*The model learned to speak. RLHF taught it to listen.*
