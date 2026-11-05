# Wei et al. — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022)

#paper #milestone #read

**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
**Institution:** Google Research, Brain Team
**Published:** arXiv:2201.11903 (January 2022), NeurIPS 2022
**PDF:** `papers/2201.11903.pdf`

---

## The Question

GPT-3 showed that large models can learn tasks from a few examples (few-shot prompting). But they still fail at multi-step reasoning — arithmetic, logic, commonsense inference. Can you unlock reasoning by simply **showing the model how to think step by step?**

The answer: **yes, and it only works at scale.**

---

## The Method

Standard few-shot prompting:
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3. How many now?
A: 11.
```

Chain-of-thought prompting:
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3. How many now?
A: Roger started with 5 balls. 2 cans of 3 is 6. 5 + 6 = 11. The answer is 11.
```

That's it. You include the intermediate reasoning steps in the few-shot examples. No fine-tuning. No new architecture. No reward model. Just showing the model what "thinking out loud" looks like.

---

## Key Results

| Task | Standard Prompting | Chain-of-Thought | Model |
|------|--------------------|------------------|-------|
| GSM8K (math word problems) | 17.7% | **56.6%** → **74.4%*** | PaLM 540B |
| SVAMP (arithmetic) | 79.0% | **86.7%** | PaLM 540B |
| MultiArith | 53.1% | **94.7%** | PaLM 540B |
| CommonsenseQA | 79.0% | **95.3%** | PaLM 540B |

*74.4% with self-consistency (sampling multiple chains and majority vote).

On GSM8K, PaLM 540B with chain-of-thought beats the previous SOTA — which was fine-tuned GPT-3 with a learned verifier. A prompting technique outperformed a trained system.

---

## The Scale Threshold

This is the most important finding: **chain-of-thought prompting only works with large models.**

Small models (≤10B parameters) produce incoherent chains — the reasoning steps are wrong or irrelevant, and performance doesn't improve (sometimes it gets worse). The capability to generate coherent, useful intermediate reasoning steps **emerges** at scale.

Tested across PaLM (8B, 62B, 540B), LaMDA (422M, 2B, 8B, 68B, 137B), and GPT-3 (ada, babbage, curie, davinci). The pattern is consistent: chain-of-thought gains appear around ~100B parameters.

This is what the paper calls an **emergent ability** — a capability that is flat/absent at small scales and abruptly appears at large scales.

---

## Why It Matters (Three Levels)

### Level 1: Practical
Multi-step reasoning was a hard wall for LLMs. Chain-of-thought broke through it without any architectural changes. This technique (and its descendants) is now standard in every LLM deployment.

### Level 2: Theoretical
The model isn't doing more computation — the same forward pass generates each token. But by producing intermediate tokens, it creates a **scratchpad** — external memory that feeds back into the autoregressive context. The model bootstraps complex reasoning by decomposing it into simple token-prediction steps.

This connects to **Turing completeness**: a Transformer with chain-of-thought can simulate an arbitrary-length computation, because each intermediate step adds to the context window. Without chain-of-thought, a Transformer can only do constant-depth computation per forward pass.

### Level 3: Philosophical
Chain-of-thought didn't teach the model to reason. It gave the model **permission** to show reasoning that was (arguably) already latent. The prompt format creates a channel for capabilities that were trained into the weights but had no way to surface in a single-token answer.

This raises the question: what *other* capabilities are latent in large models, waiting for the right prompting to surface?

---

## Connection to GPT-3's In-Context Learning

[[Brown GPT-3 2020]] demonstrated in-context learning: the model adapts to the task from examples in the prompt. Chain-of-thought is the natural extension: instead of just showing *what* to do, you show *how to think about it*.

The lineage:
```
Few-shot learning (GPT-3, 2020)
    → "Show me examples of the task"
Chain-of-thought (Wei et al., 2022)
    → "Show me how to THINK about the task"
Self-consistency (Wang et al., 2022)
    → "Think about it multiple times, take the majority vote"
Tree-of-thought (Yao et al., 2023)
    → "Explore multiple reasoning paths, backtrack"
```

---

## Cross-Thread Connections 🔗

- **[[Brown GPT-3 2020]]** — direct predecessor. GPT-3 showed in-context learning; CoT extends it to reasoning. Wei et al. test on GPT-3 (davinci) and show the same emergence pattern.
- **[[Kaplan Scaling Laws 2020]]** — the scale threshold for CoT raises questions about what scaling laws actually predict. Kaplan's power laws describe *loss*; CoT emergence is about *capability*, which may not follow smooth power laws.
- **[[Hoffmann Chinchilla 2022]]** — published 2 months after CoT. Chinchilla is about *training* efficiency; CoT is about *inference* capability. They're orthogonal axes of improvement.
- **[[Vaswani Attention Is All You Need 2017]]** — the Transformer's autoregressive generation is what makes CoT possible. Each generated token becomes part of the context for the next. This feedback loop IS the scratchpad.
- **Emergence debate** — Wei et al. (2022, separate paper) catalog emergent abilities. Schaeffer et al. (2023) challenge the concept, arguing emergence is an artifact of evaluation metrics. The debate continues.

---

## The Lineage of Reasoning in LLMs

```
GPT-3 few-shot (2020) — task adaptation without training
    → Chain-of-Thought (Jan 2022) — step-by-step reasoning
        → Self-Consistency (Mar 2022) — sample multiple chains, majority vote
            → Tree-of-Thought (2023) — branching search over reasoning paths
                → o1/o3 (2024-2025) — reasoning as first-class training objective
```

Chain-of-thought is the hinge point. Everything after it — Constitutional AI's reasoning, Claude's thinking, o1's extended reasoning — descends from the insight that showing intermediate steps unlocks qualitatively different behavior.

---

## Limitations

1. **No guarantee of correctness.** The chain can be coherent but wrong — plausible-sounding reasoning with the wrong answer. The model can "confabulate" logical steps.
2. **Prompt sensitivity.** Small changes in the few-shot examples can dramatically change performance. There's no principled way to choose which chain-of-thought examples to use.
3. **Only works at scale.** Useless for models under ~100B parameters. This makes it inaccessible for most researchers.
4. **Inference cost.** Generating the chain adds tokens, which adds latency and compute cost. The reasoning scratchpad isn't free.

---

## References Worth Chasing

- **Wang et al. (2022)** — *"Self-Consistency Improves Chain of Thought Reasoning in Language Models"* — arXiv:2203.11171 — the natural next step
- **Kojima et al. (2022)** — *"Large Language Models are Zero-Shot Reasoners"* — just saying "Let's think step by step" (zero-shot CoT) also works
- **Yao et al. (2023)** — *"Tree of Thoughts"* — generalizes CoT to tree search
- **Schaeffer et al. (2023)** — *"Are Emergent Abilities of Large Language Models a Mirage?"* — challenges the emergence narrative

---

## Status

- [x] Paper read and note created (March 24, 2026)
- [ ] Read Self-Consistency (Wang et al.) — the sampling extension
- [ ] Explore Kojima "zero-shot CoT" — is "Let's think step by step" really all you need?
- [ ] The emergence debate: Schaeffer vs. Wei — is CoT emergence real or metric artifact?
