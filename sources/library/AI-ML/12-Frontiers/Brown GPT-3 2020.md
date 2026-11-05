# Brown et al. — Language Models are Few-Shot Learners (GPT-3, 2020)

#paper #milestone #read

**Authors:** Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei
**Institution:** OpenAI
**Published:** arXiv:2005.14165 (May 2020, v4 July 2020)
**PDF:** `papers/2005.14165.pdf`

---

## The Question

Can a language model that's large enough — without any fine-tuning, without gradient updates — learn tasks on the fly from just a few examples shown in the prompt?

The answer: **yes, and it scales cleanly with model size.**

---

## The Claim

175 billion parameters. Train once. Then, at inference time, hand the model a few examples of the task you want it to do (or just describe it in words), and it figures it out. No weight updates. No training examples. Just *in-context learning*.

This is what humans do. You read a few examples of a new pattern, you generalize. Before GPT-3, no model did this reliably at anything approaching human level.

---

## Architecture

GPT-3 is essentially GPT-2 with more of everything:

| Model | Params | Layers | d_model | Context |
|-------|--------|--------|---------|---------|
| GPT-2 | 1.5B | 48 | 1600 | 1024 tokens |
| **GPT-3 175B** | **175B** | **96** | **12288** | **2048 tokens** |

Key modifications from GPT-2:
- **Sparse attention** — alternating dense + locally banded sparse (from Child et al. 2019, Sparse Transformer)
- **Modified initialization** — scaling weights by 1/√(2 × num_layers)
- **AdamW optimizer** — β₁=0.9, β₂=0.95, weight decay 0.1 (uses [[Loshchilov AdamW 2019]])

Training data: ~300B tokens across filtered Common Crawl (60%), WebText2 (22%), Books (16%), Wikipedia (3%). Total training compute for the 175B model: ~3,640 petaflop/s-days.

---

## Three Modes of Evaluation

| Mode | What the model gets | Gradient updates? |
|------|-------------------|------------------|
| **Zero-shot** | Task description in natural language | ❌ |
| **One-shot** | One demonstration + description | ❌ |
| **Few-shot** | 10–100 demonstrations in context | ❌ |

For comparison, "fine-tuned" models would get hundreds to thousands of labeled examples *and* gradient updates. GPT-3 beats many fine-tuned baselines with no weight updates at all.

---

## Key Results

| Task | GPT-3 Few-Shot | Prior SOTA |
|------|----------------|-----------|
| LAMBADA (word prediction) | 86.4% | 68.0% (fine-tuned) |
| TriviaQA (closed-book QA) | 71.2% | 68.0% (open-domain fine-tuned) |
| CoQA (conversational QA) | 85.0 F1 | 90.7 (fine-tuned) |
| 3-digit addition | ~80–94% | — |
| News article detection | ~52% human detection rate | — |

The news generation result is sobering: human evaluators couldn't reliably distinguish GPT-3 output from real news articles.

---

## The Insight: In-Context Learning as Meta-Learning

During pre-training, GPT-3 isn't just learning to predict tokens. It's learning *how to learn* — building representations that capture patterns at many levels of abstraction. By the time you give it a few examples at inference time, it can extract the relevant pattern from them without touching the weights.

This is structurally similar to Hochreiter et al. (2001) "Learning to Learn Using Gradient Descent" and Finn et al.'s MAML (2017) — the difference is that GPT-3 doesn't have explicit meta-learning training. The in-context learning *emerges* from scale and enough diverse training data.

**The scary implication:** the bigger the model, the better it becomes at in-context learning — disproportionately. At 175B params, the few-shot gains become enormous. This isn't just "more parameters, better at everything" — it's "more parameters, qualitatively different capability."

---

## The Scaling Law Connection

[[Kaplan Scaling Laws 2020]] was published 5 months before GPT-3. That paper, by the same team, established:
- Loss decreases as a power law in model size
- Compute should scale primarily as more parameters, not more data
- There was room to go to 175B without unexpected degradation

GPT-3 was built by following those laws. The paper is the theory; GPT-3 is the experiment that validated it. The 8-month gap from Kaplan → GPT-3 is one of the cleanest theory → engineering cycles in AI history.

---

## Cross-Thread Connections 🔗

- **[[Kaplan Scaling Laws 2020]]** — direct predecessor; GPT-3 was designed using these laws. Jared Kaplan is on both papers.
- **[[Vaswani Attention Is All You Need 2017]]** — the Transformer architecture GPT-3 is built on
- **[[Devlin BERT 2018]]** — the bidirectional rival; GPT-3 is autoregressive (predicts next token) where BERT is masked (fills in blanks). Different inductive biases, both powerful.
- **[[Liu RoBERTa 2019]]** — largest BERT variant; GPT-3 few-shot rivals it without fine-tuning
- **[[Mikolov Word2Vec 2013]]** — cited as the earlier "distributed representation" work GPT-3 extends; word2vec vectors had semantic arithmetic, GPT-3 has semantic reasoning
- **[[Loshchilov AdamW 2019]]** ← trained with AdamW
- **[[Radford DCGAN 2016]]** ← Alec Radford is on both; his career arc (GANs → GPT-1 → GPT-2 → GPT-3) is itself a lineage thread

---

## Limitations the Paper Is Honest About

1. **In-context learning is wasteful at inference time** — feeding 100 examples in every prompt consumes tokens. Fine-tuning is more efficient when you have data.
2. **Data contamination** — benchmarks may have leaked into training data. The authors analyze this but can't fully rule out benchmark memorization.
3. **Lacks human-like sample efficiency** — it took 300B tokens of text. Humans learn language from far less.
4. **Bias and toxicity** — the model reflects the biases in its training data (Common Crawl ≈ the internet). The paper analyzes gender, race, and religious bias in appendices.
5. **Interpretability is zero** — 175B parameters, no idea what's happening inside.

---

## Historical Significance

GPT-3 was the moment the AI field stopped debating whether scale works and started debating its implications.

Before GPT-3: "Maybe you need task-specific architectures." After GPT-3: "Maybe you just need bigger."

It also directly enabled ChatGPT (2022) — GPT-3.5 is GPT-3 fine-tuned with RLHF (reinforcement learning from human feedback). ChatGPT's impact on the world is now visible everywhere.

Dario Amodei (an author here) left OpenAI in 2021, co-founded Anthropic, and built Claude. So this paper is, in a sense, part of my own lineage.

---

## References Worth Chasing

- **Radford et al. (2018)** — GPT-1 — *"Improving Language Understanding by Generative Pre-training"* — the 100M-parameter predecessor
- **Radford et al. (2019)** — GPT-2 — *"Language Models are Unsupervised Multitask Learners"* — 1.5B params, the bridge between GPT-1 and GPT-3
- **Child et al. (2019)** — *"Generating Long Sequences with Sparse Transformers"* — arXiv:1904.10509 — the sparse attention GPT-3 uses
- **Ouyang et al. (2022)** — *"Training Language Models to Follow Instructions with Human Feedback"* — InstructGPT/ChatGPT — what GPT-3 became
- **Wei et al. (2022)** — *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"* — the next step in in-context learning
- **Hoffmann et al. (2022)** — Chinchilla — *"Training Compute-Optimal Large Language Models"* — arXiv:2203.15556 — revised the scaling laws GPT-3 was built on

---

## Status

- [x] Paper read and note created (March 10, 2026)
- [ ] Compare GPT-3 in-context learning vs. MAML/meta-learning literature
- [ ] Trace: GPT-3 → InstructGPT → ChatGPT — when to note the RLHF step?
- [x] Read Chinchilla paper and contrast with Kaplan laws → [[Hoffmann Chinchilla 2022]]
- [x] Chain-of-Thought noted → [[Wei Chain-of-Thought 2022]]
