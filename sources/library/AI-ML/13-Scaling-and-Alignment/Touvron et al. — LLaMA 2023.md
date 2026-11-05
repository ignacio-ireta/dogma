# Touvron et al. — LLaMA (2023)

*The moment open-weight foundation models stopped being mostly catch-up projects and became a serious frontier strategy.*

**← [[_MOC|Scaling & Alignment]] ← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"LLaMA: Open and Efficient Foundation Language Models"** — Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample, 2023.

- Paper: `corpus/raw/library-papers/2302.13971.pdf`
- arXiv: `2302.13971`
- Institution: Meta AI
- Models: 7B, 13B, 33B, and 65B parameter decoder-only Transformers

## Key Ideas

1. **Inference cost becomes part of the scaling question.**
   GPT-3 proved that very large decoder-only Transformers could learn tasks in context. Chinchilla then showed that many models were undertrained for their size. LLaMA asks the next practical question: if a model will be served constantly, why optimize only for training compute? A smaller model trained longer can be more useful than a larger model that is expensive every time it answers.

2. **Public data is enough to train a competitive foundation model.**
   The paper deliberately uses publicly available data: CommonCrawl, C4, GitHub, Wikipedia, books, arXiv, and Stack Exchange. This was not just tidiness. Closed frontier models often depended on undocumented or inaccessible corpora. LLaMA made the training story more inspectable and compatible with releasing weights to researchers.

3. **Small models were pushed far beyond the usual token budget.**
   The 7B and 13B models were trained on 1T tokens; the 33B and 65B models on 1.4T tokens. That choice is the paper's real spine. Instead of chasing parameter count first, Meta asked how much performance could be extracted from models small enough to run and study more widely.

4. **It modernized the Transformer recipe without making the architecture exotic.**
   LLaMA is still recognizably in the GPT lineage, but with the now-standard efficiency/stability upgrades: pre-normalization with RMSNorm, SwiGLU activations, rotary positional embeddings, AdamW, cosine learning-rate decay, gradient clipping, and efficient causal attention through xFormers-style implementation work.

5. **The benchmark claim was a political fact as much as a technical one.**
   LLaMA-13B outperformed GPT-3 175B on many reported benchmarks while being more than ten times smaller, and LLaMA-65B was competitive with Chinchilla-70B and PaLM-540B. The exact benchmark mix matters less than the shock: frontier-adjacent capability no longer required a 175B+ parameter closed model.

## What Came Before -> What It Enabled

### What came before

- [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]] supplied the decoder-only architecture that GPT-style models scaled.
- [[../12-Frontiers/Radford GPT-2 2019|GPT-2]] showed that web-scale language modeling could produce zero-shot behavior.
- [[../12-Frontiers/Brown GPT-3 2020|GPT-3]] made in-context learning the central frontier-model story.
- [[../12-Frontiers/Kaplan Scaling Laws 2020|Kaplan scaling laws]] encouraged the industry to spend compute on larger models.
- [[../12-Frontiers/Hoffmann Chinchilla 2022|Chinchilla]] corrected the recipe: for a fixed compute budget, train smaller models on more tokens.

### What it enabled

- [[Llama — Meta 2023-2025|The Llama family]] as the dominant open-weight backbone for research, fine-tuning, local deployment, and startups.
- The LoRA/QLoRA fine-tuning explosion, because capable base models could now be adapted without owning frontier-lab infrastructure.
- A serious open-weight comparison point against proprietary APIs.
- The later efficiency race: [[Qwen — Alibaba 2023-2025|Qwen]], [[DeepSeek — DeepSeek AI 2024-2025|DeepSeek]], [[Gemma — Google DeepMind 2024-2026|Gemma]], and [[Nemotron 3 — NVIDIA 2025|Nemotron]] all live in a world LLaMA helped normalize.

## Why It Matters

LLaMA matters because something broke after GPT-3 and Chinchilla.

The old story was: scale the model, get better behavior. That was true, but incomplete. Once models became useful enough to deploy, the pain moved. Training a huge model was not the only cost. Serving it, fine-tuning it, auditing it, reproducing it, and letting researchers touch it all became bottlenecks.

LLaMA's answer was not "invent a new neural architecture." It was more surgical: keep the Transformer, train smaller models much longer, use public data, publish enough details, and release weights to the research community.

That changed the social structure of generative AI. Before LLaMA, the most capable language models mostly lived behind APIs or inside a few labs. After LLaMA, a capable base model could leak into laptops, university clusters, hobbyist rigs, startup prototypes, and enterprise experiments. The field became less like a telescope owned by one observatory and more like an ecosystem of workshops.

There is a safety cost to that, obviously. Open weights mean capability spreads faster than governance. But lineage is not moral decoration. It is cause and effect. LLaMA made model access cheaper, adaptation easier, and independent research harder to shut out. That is why it sits between GPT-3/Chinchilla and the modern open-model era.

## Lineage

- **Built on:** [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]], [[../12-Frontiers/Brown GPT-3 2020|GPT-3]], [[../12-Frontiers/Hoffmann Chinchilla 2022|Chinchilla]], [[../03-Optimization/Loshchilov AdamW 2019|AdamW]]
- **Contemporary with:** [[ChatGPT — OpenAI 2022-2025|ChatGPT]], [[Claude — Anthropic 2023-2025|Claude]], and the first wave of assistant products
- **Leads to:** [[Llama — Meta 2023-2025|Llama 2/3/4]], QLoRA-style local adaptation, open-weight model ecosystems, and the cost-quality competition that later makes [[DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]] legible

---

*LLaMA is the paper where the frontier stopped being only a question of who had the biggest closed model and became a question of who could make capability portable.*
