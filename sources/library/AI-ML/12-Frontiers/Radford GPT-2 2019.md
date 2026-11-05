# Radford et al. — Language Models are Unsupervised Multitask Learners (GPT-2, 2019)

#paper #milestone #read

**Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever  
**Institution:** OpenAI  
**Published:** OpenAI technical report, 2019  
**Paper:** `corpus/raw/library-papers/radford-gpt2-2019.pdf`

---

## The Question

Can the next-token prediction objective itself become a kind of multitask learning?

Not “pre-train, then fine-tune on a labeled dataset.” Not “build a separate architecture for translation, summarization, question answering, and reading comprehension.” Just train a very large language model on broad web text and ask whether the tasks are already latent in the data.

GPT-2's answer was: **yes, weakly — and the effect scales with model size.**

---

## What Broke Before GPT-2

The 2018-2019 NLP recipe was powerful but brittle:

1. Pre-train a representation model.
2. Fine-tune it on a supervised task dataset.
3. Report benchmark gains on IID held-out examples.

That worked beautifully for narrow benchmarks, especially after [[Devlin BERT 2018]] and [[Liu RoBERTa 2019]]. But it did not solve the deeper problem: every new task still needed a dataset, a training loop, and often task-specific evaluation assumptions.

The paper says the quiet part directly: existing systems were “narrow experts rather than competent generalists.” They failed under distribution shift, tiny specification changes, or tasks without labels.

So the old thing that broke was not accuracy. It was **supervised task creation as the scaling bottleneck**.

If general intelligence requires thousands of tasks, then manually creating thousands of clean supervised datasets becomes absurd. The web already contains task demonstrations in natural language — questions followed by answers, summaries after articles, translations next to source text, explanations after prompts. GPT-2 asks whether a language model can absorb those demonstrations without being explicitly told what the task labels are.

---

## Key Ideas

### 1. Language modeling as unsupervised multitask learning

A task can be written as a sequence:

```text
translate to French, English text, French text
answer the question, document, question, answer
summarize, article, summary
```

A standard language model maximizes:

```text
p(next token | previous tokens)
```

But if enough naturally occurring task demonstrations are present in the training corpus, then predicting the next token forces the model to learn pieces of:

```text
p(output | input, task)
```

The task is not a separate supervised head. It is expressed in language, inside the context.

That is the conceptual bridge from GPT-1-style generative pretraining to GPT-3-style in-context learning.

### 2. WebText: broad text beats narrow corpora

OpenAI built **WebText** by scraping outbound Reddit links with at least 3 karma — a crude but useful human-filtered quality signal.

- ~45 million links scraped
- ~8 million documents after filtering and deduplication
- ~40 GB of text
- Wikipedia removed to reduce benchmark contamination
- cutoff: links before December 2017

The point was not just “more data.” It was **more kinds of situations where language demonstrates tasks**.

### 3. Byte-level BPE as practical universality

GPT-2 uses byte-level BPE with a 50,257-token vocabulary. This matters because older NLP pipelines were full of brittle preprocessing: lowercasing, unknown tokens, dataset-specific tokenizers.

Byte-level BPE lets the model assign probability to essentially any Unicode string while staying more efficient than pure byte/character modeling.

That universality is part of the zero-shot story: if the model has to handle arbitrary text, the input representation cannot keep collapsing rare strings into `<UNK>`.

### 4. Scale reveals the transfer

GPT-2 trained four decoder-only Transformer language models:

| Model | Parameters | Layers | d_model | Context |
|---|---:|---:|---:|---:|
| GPT-sized baseline | 117M | 12 | 768 | 1024 tokens |
| medium | 345M | 24 | 1024 | 1024 tokens |
| large | 762M | 36 | 1280 | 1024 tokens |
| **GPT-2** | **1.542B** | **48** | **1600** | **1024 tokens** |

Architecturally, GPT-2 is still recognizably a decoder-only [[Vaswani Attention Is All You Need 2017|Transformer]] in the GPT line. The novelty is not a new block. It is the combination of scale, broad web data, and evaluating downstream tasks without fine-tuning.

---

## Results That Mattered

GPT-2 achieved state-of-the-art zero-shot results on **7 of 8 language modeling datasets** tested.

The result that became symbolic was **LAMBADA**, a benchmark requiring long-range context to predict the final word of a passage:

- prior LM accuracy: ~19%
- GPT-2 accuracy: **52.66%**
- with a stop-word filter: **63.24%**
- perplexity improved from ~99.8 to **8.6**

It also showed nontrivial zero-shot behavior on:

- Children’s Book Test cloze prediction
- Winograd Schema Challenge
- reading comprehension / CoQA
- summarization
- translation
- question answering

But the paper is careful: many of these results were still crude. Summarization especially looked qualitatively plausible but quantitatively weak. GPT-2 was not yet a reliable general assistant.

The important result was not “GPT-2 solved NLP.” It was:

> high-capacity language models trained on broad natural text begin to learn tasks without explicit supervised adaptation.

That is the spark GPT-3 turns into a bonfire.

---

## What Came Before → What It Enabled

### Before: [[Vaswani Attention Is All You Need 2017]]

The Transformer removed recurrence and made sequence modeling parallelizable. GPT adopted the decoder-only version: causal self-attention, next-token prediction, autoregressive generation.

Without the Transformer, a 1.5B-parameter web-scale language model would have been much harder to train and much worse at long-range dependencies.

### Before: GPT-1 / generative pre-training

GPT-1 showed that unsupervised generative pretraining plus supervised fine-tuning could improve language understanding. But fine-tuning was still central.

GPT-2 asks the more radical question: what if fine-tuning is not always necessary?

### Before: [[Devlin BERT 2018]] and [[Liu RoBERTa 2019]]

BERT made pretraining dominant, but as an encoder-only masked-language-model system. It was excellent for understanding tasks after fine-tuning, less naturally suited to open-ended generation.

GPT-2 made the opposite bet: autoregressive generation as the universal interface.

### Enabled: [[Brown GPT-3 2020]]

GPT-3 is GPT-2's thesis scaled up:

- GPT-2: zero-shot task transfer begins to appear.
- GPT-3: zero-shot, one-shot, and few-shot prompting become a central evaluation paradigm.

GPT-2 gave the field the hypothesis: **task learning is latent in next-token prediction over diverse text**. GPT-3 tested that hypothesis at 175B parameters.

### Enabled: instruction following and RLHF

GPT-2 was not instruction-tuned. It could continue text; it did not reliably obey users.

That limitation sets up the next alignment chain:

[[Ziegler et al. — Fine-Tuning Language Models from Human Preferences 2019]] → [[Stiennon et al. — Learning to Summarize from Human Feedback 2020]] → [[Ouyang InstructGPT RLHF 2022]] → ChatGPT.

In other words: GPT-2 showed the base model could contain task ability. RLHF later made that ability usable through conversation.

---

## Why It Matters

GPT-2 is the hinge between “language models as pretraining components” and “language models as general-purpose systems.”

Before GPT-2, the standard mental model was:

```text
pretrain representation → fine-tune per task
```

After GPT-2, a new mental model became plausible:

```text
train one huge generative model → specify tasks in context
```

It did not fully work yet. That is exactly why the note matters. The failure mode points directly to the next step.

GPT-2 could generate coherent paragraphs and show zero-shot transfer, but it was still unreliable, weak at many tasks, and bounded by a 1024-token context. The model had learned a surprising amount from raw web text, but not enough to be dependable. The natural next question became brutally simple:

**What happens if we keep the objective, keep the architecture family, and scale everything?**

That question is [[Brown GPT-3 2020]].

---

## Cross-Thread Connections 🔗

- **[[Vaswani Attention Is All You Need 2017]]** — GPT-2 is a decoder-only Transformer descendant.
- **[[Devlin BERT 2018]]** — the contrasting encoder-only/fine-tuning path.
- **[[Liu RoBERTa 2019]]** — showed BERT-style systems still had room from better data/training; GPT-2 explored the generative alternative.
- **[[Child Sparse Transformer 2019]]** — Rewon Child is a GPT-2 coauthor; sparse attention becomes one of the ingredients in GPT-3.
- **[[Brown GPT-3 2020]]** — direct successor; few-shot prompting is the scaled version of GPT-2's zero-shot transfer story.
- **[[Ouyang InstructGPT RLHF 2022]]** — turns GPT-style completion models into instruction-following assistants.
- **[[13-Scaling-and-Alignment/Claude — Anthropic 2023-2025]]** — Dario Amodei is a GPT-2 coauthor; this lineage later branches into Anthropic and Claude.

---

## The Lineage Sentence

The Transformer made web-scale autoregressive language modeling trainable; GPT-2 showed that next-token prediction on broad web text starts to induce zero-shot task behavior; GPT-3 then scaled that behavior into in-context learning; RLHF later made it conversationally useful.

---

## Status

- [x] Source paper downloaded: `corpus/raw/library-papers/radford-gpt2-2019.pdf`
- [x] Paper path referenced in note
- [x] Note created (May 21, 2026)
- [x] GPT-1 bridge added: [[Radford GPT-1 2018]]
