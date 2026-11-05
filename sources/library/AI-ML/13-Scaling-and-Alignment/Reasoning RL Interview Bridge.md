# Reasoning-RL Interview Bridge — Test-Time Compute

*One clean story for "what came after RLHF?" — through DPO, verifiable rewards, and R1's test-time compute.*

**← [[_MOC|Scaling & Alignment]] ← [[../_MOC|AI/ML Lineage]]**

---

## When to reach for this

An interviewer or client asks any of: *"What's changed in LLMs since ChatGPT?"*, *"How do reasoning models work?"*, *"Why did DeepSeek-R1 matter?"*, *"Is scaling dead?"* This is the 90-second answer that shows lineage understanding, not headline-chasing.

## The arc, in one breath

Post-training has moved through three stances on the same question — *how do you make a pretrained model behave?*

1. **RLHF (learn a reward, then optimize it).** Human pairwise preferences train a reward model; PPO optimizes the policy against it while a KL term keeps it near the reference. This made completion models into usable assistants. Fragile, expensive, many knobs. → [[../12-Frontiers/Christiano et al. — Deep RL from Human Preferences 2017|Christiano 2017]], [[../12-Frontiers/Ouyang InstructGPT RLHF 2022|InstructGPT 2022]], [[Schulman et al. — Proximal Policy Optimization 2017|PPO]].
2. **DPO (skip the reward model).** Rewrite the RLHF objective and the optimal policy is a closed-form function of the reward — so optimize the model *directly* on chosen-vs-rejected pairs. Looks like classification, not RL. Simplicity becomes a capability multiplier: everyone can now post-train. → [[Rafailov et al. — Direct Preference Optimization 2023|DPO]].
3. **Verifiable-reward RL (drop the human, keep the RL).** For domains where correctness is *checkable* — math, code — you don't need a learned reward or human traces at all. Reward the final answer being right, run RL (GRPO: group-normalized advantage, no critic), and reasoning behavior emerges. → DeepSeek-R1, arXiv `2501.12948`.

## The one non-obvious point (this is the "senior" signal)

R1 opens a **second scaling axis**. Kaplan/Chinchilla scaling was about *training* compute — bigger model, more tokens — and it was hitting diminishing returns. R1 shows you can also scale **test-time compute**: let the model think longer at inference, and accuracy climbs. Self-verification, reflection, and "aha" backtracking were *never demonstrated by humans* — they emerged from pure RL against a verifiable reward. Imitating human chains-of-thought (SFT on reasoning traces) had quietly capped models at human strategies; verifiable-reward RL removed that ceiling.

So the honest answer to *"is scaling dead?"* is: **the pretraining-compute axis is maturing; a new inference-compute axis just opened.** Evaluation stops being a single leaderboard number and becomes an accuracy × latency × memory frontier (cf. [[13-Scaling-and-Alignment/Manik Wang Gemma Phi Qwen Tradeoffs 2026|Manik & Wang 2026]]).

## The trap to avoid in the room

Don't say "R1 proved you don't need scale." Wrong and it reads as hype. Say: *R1 kept a strong pretrained base (V3) and added a new axis on top of it.* Verifiable-reward RL is powerful **exactly where a cheap, gameable-resistant verifier exists** — it is not a general replacement for preference learning on open-ended, subjective tasks. That boundary is the mature point.

## Lineage

- **Built on:** [[Rafailov et al. — Direct Preference Optimization 2023|DPO]] (preference post-training without the RL circus), [[Schulman et al. — Proximal Policy Optimization 2017|PPO]] (the RL workhorse GRPO streamlines), [[../12-Frontiers/Wei Chain-of-Thought 2022|Chain-of-Thought]] (reasoning as prompting, the thing R1 internalizes)
- **Synthesizes for interview use:** [[DeepSeek — DeepSeek AI 2024-2025|DeepSeek product line]] (V3 base + R1 reasoning), [[13-Scaling-and-Alignment/Manik Wang Gemma Phi Qwen Tradeoffs 2026|deployment-frontier framing]]
- **Sits alongside the other bridges:** [[../05-Classification/XGBoost Interview Bridge|XGBoost / tabular]], [[../03-Optimization/Optimizer Failure Modes Interview Bridge|optimizer failure modes]], [[../10-Backpropagation/LSTM Memory Management Interview Bridge|LSTM memory]], [[../12-Frontiers/Tokenization Economics Interview Bridge|tokenization economics]]

---

*RLHF made models usable. DPO made post-training cheap. Verifiable-reward RL made reasoning a thing you can buy at inference time. 😏*

#moc-linked #ai-ml #interview-bridge #reasoning #rlhf #test-time-compute
