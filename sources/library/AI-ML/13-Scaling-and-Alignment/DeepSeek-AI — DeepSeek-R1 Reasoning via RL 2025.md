# DeepSeek-AI — DeepSeek-R1: Incentivizing Reasoning via Reinforcement Learning (2025)

*Stop scaling pretraining. Let the model think longer — and let RL, not humans, teach it how.*

**← [[_MOC|Scaling & Alignment]] ← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"** — DeepSeek-AI, January 2025.
- [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (v1 Jan 2025; peer-reviewed version in *Nature*, 2025)
- Paper: `corpus/raw/library-papers/2501.12948.pdf` (verified PDF, 17 pages)

## The Break — Why They Couldn't Keep Doing What They Were Doing

By 2024 the reasoning recipe had a ceiling built into it. Chain-of-Thought (Wei et al., 2022) showed that models could reason *if you prompted them to*, and post-training on curated multi-step traces (InstructGPT-style SFT, then o1's private recipe) pushed that further. But every gain still flowed through **human-annotated reasoning demonstrations**. That dependency has two hard costs the paper names directly:

1. It doesn't scale — expert reasoning traces for competition math and code are scarce and expensive.
2. It **caps the model at human performance**. Training a model to imitate human thought processes forbids it from discovering *non-human* reasoning pathways that might be better.

Meanwhile the other lever — scale pretraining harder (Kaplan, Chinchilla) — was hitting diminishing returns against a finite supply of tokens and rising compute bills. Both roads were narrowing at once. The question became: is there a source of reasoning improvement that is *neither* more pretraining data *nor* more human demonstrations?

## Key Ideas

1. **Pure RL, no SFT warm-up (R1-Zero).**
   Start from DeepSeek-V3-Base and apply reinforcement learning directly — deliberately skipping the conventional supervised fine-tuning stage. The hypothesis: human-defined reasoning patterns constrain exploration, so remove them and let the model self-evolve.

2. **The reward is only correctness.**
   No learned reward model, no human preference labels, no process supervision. For math and code the answer is checkable against ground truth, so the reward is *rule-based*: did the final answer verify? Nothing constrains *how* the model reaches it. This is the whole trick — a **verifiable reward** replaces the human annotator.

3. **GRPO instead of PPO.**
   Group Relative Policy Optimization (Shao et al., 2024) drops PPO's separate value/critic network. For each question it samples a *group* of G outputs from the old policy and computes each output's advantage as its reward normalized within the group (subtract the group mean, divide by the group std). Baseline-by-group means far less memory and compute than PPO's critic — which is what makes reasoning-scale RL affordable.

4. **Reasoning emerges; nobody programmed it.**
   Under RL, R1-Zero spontaneously starts generating *longer* responses containing self-verification, reflection, and the exploration of alternative approaches — the famous "aha moment" where it re-examines its own work. These behaviors were never demonstrated to it. They are what maximizing a correctness reward *produces* when the model is free to use as many tokens as it wants.

5. **Test-time compute becomes the scaling axis.**
   The model learns, on its own, that thinking longer pays off. Response length grows over training. This is the paradigm shift in one sentence: **you buy more capability at inference time by letting the model reason, not only at training time by growing parameters.**

6. **R1 = R1-Zero made usable.**
   R1-Zero reasons brilliantly but has poor readability and mixes English and Chinese mid-thought, and its rule-based RL only covers verifiable domains (weak at open writing/QA). So R1 wraps the idea in a **multi-stage pipeline**: a small cold-start SFT for readable format → reasoning RL → rejection sampling to harvest good traces → a final SFT+RL pass that folds in general (non-reasoning) data and human preferences.

7. **Distillation transfers reasoning downward.**
   The long-CoT behavior of the big model can be distilled into small dense models (Qwen/Llama backbones), and the distilled students **beat their original instruction-tuned versions**. Reasoning is a transferable skill, not just an artifact of scale.

## What Came Before → What It Enabled

- **Built on:** [[../12-Frontiers/Wei Chain-of-Thought 2022|Chain-of-Thought]] (reasoning as an autoregressive scratchpad), [[Schulman et al. — Proximal Policy Optimization 2017|PPO]] and [[../12-Frontiers/Ouyang InstructGPT RLHF 2022|InstructGPT RLHF]] (the RL post-training machinery R1 both inherits and simplifies), [[../12-Frontiers/Kaplan Scaling Laws 2020|Scaling Laws]] / [[../12-Frontiers/Hoffmann Chinchilla 2022|Chinchilla]] (the pretraining road whose diminishing returns motivated a new axis), and DeepSeek-V3-Base as the substrate.
- **Responds to:** the ceiling of human-demonstration imitation and the diminishing returns of pure pretraining scale.
- **Leads to:** the inference-time-scaling era — o1/o3-style reasoning models, verifiable-reward RL as a standard post-training stage, GRPO's spread across open models, and the reasoning capability that later frontier systems (including [[DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]]) assume as table stakes. It also reframes evaluation: once reasoning is cheap to add, model choice becomes the accuracy-latency-memory frontier question of [[Manik Wang Gemma Phi Qwen Tradeoffs 2026|Manik & Wang 2026]].

## Why It Matters for the Lineage

This is where the field's dominant lever *moves*. For a decade "make it better" meant "make it bigger" — more parameters, more tokens, Kaplan's straight lines on a log plot. R1 demonstrates a second, orthogonal source of capability: **spend compute at inference, and train that spending with RL against rewards you can check.** No human in the loop for the reasoning itself.

It also lands a sharper philosophical point than DPO did. DPO simplified *alignment*. R1 challenges *imitation itself*: by refusing to show the model human reasoning traces, it lets the model find strategies humans wouldn't — and they work better on verifiable tasks. That is the first large, public demonstration that an LLM can improve at reasoning by **self-play-like exploration** rather than by copying us.

## The Real Shift

- **Old axis:** scale pretraining → lower cross-entropy → hope reasoning emerges, then teach it with human traces.
- **New axis:** freeze a capable base → RL against verifiable rewards → the model *learns to think longer* and discovers its own reasoning, no human traces required.

Same goal — better reasoning. Completely different fuel. 😏

## Open Questions

- How far does verifiable-reward RL generalize *beyond* domains with checkable answers?
- Does emergent reasoning transfer to long-horizon agent tasks, where credit assignment spans many tool calls?
- Is the "aha moment" genuine strategy discovery, or sophisticated pattern completion the reward happens to favor? The paper shows the behavior; the mechanism is still open.

---

*Pretraining taught models what to say. RLHF taught them what we prefer. R1 asked a colder question: if we only score the answer and let go of the reasoning, what does the model teach itself?*
