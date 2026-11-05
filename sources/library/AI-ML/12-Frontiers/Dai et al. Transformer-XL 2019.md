# Dai et al. — Transformer-XL (2019)

*The Transformer could attend across a segment, but it forgot everything at the segment boundary. Transformer-XL made attention carry memory forward.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**“Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context”** — Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov, 2019.

- Paper: `corpus/raw/library-papers/1901.02860.pdf`
- arXiv: [1901.02860](https://arxiv.org/abs/1901.02860)
- Source PDF status: local file verified as `PDF document, version 1.5` on 2026-08-27

## Key Ideas

1. **Self-attention had escaped recurrence, but not the context window.**
   [[Vaswani Attention Is All You Need 2017|The original Transformer]] connected every token directly to every other token *inside the input it was given*. Language-model training still had to split long text into manageable fixed-length segments. Those segments were treated independently, so no information crossed their boundaries.

2. **Independent segments created two different failures.**
   - **Hard context ceiling:** the model could not learn a dependency longer than one training segment.
   - **Context fragmentation:** the first tokens in every segment were predicted without the preceding text, even when the split cut through a sentence or topic.

   Simply making segments longer was not a real repair: dense attention cost and memory grow quadratically with segment length.

3. **Segment-level recurrence reuses hidden states as memory.**
   At each layer, Transformer-XL caches hidden states from the previous segment, applies stop-gradient to them, and concatenates them with the current segment’s states as keys and values. Current tokens can therefore attend into the cached past without recomputing it. Repeating this across layers and segments creates an effective receptive field far longer than any single segment while preserving parallel computation *within* each new segment.

4. **Absolute positions break when old states are reused.**
   If each segment starts again at positions `1…L`, a cached token and a current token can carry indistinguishable absolute position labels even though one came earlier. Transformer-XL therefore reparameterizes attention around **relative distance**: content-to-content, content-to-position, a global content bias, and a global positional bias. The model asks how far apart two tokens are rather than pretending that every segment occupies the same absolute slots.

5. **The recurrence fixes evaluation waste too.**
   A vanilla Transformer evaluated with a sliding context repeatedly recomputes almost the same hidden states. Transformer-XL caches them once and reuses them. The paper reports evaluation up to **1,800+ times faster** than that vanilla sliding-window procedure.

6. **The repair is powerful, but deliberately incomplete.**
   Transformer-XL does not make attention cheap at arbitrary length. Cached memory is finite; old states are frozen rather than revised; and each current segment still attends over the retained memory. It repaired discontinuity and redundant computation, not the full scaling cost of long context. That remaining break motivates sparse attention, compressed KV memory, retrieval, and state-space models.

## Source Grounding

From the paper:

- Transformer-XL combines a **segment-level recurrence mechanism** with a new **relative positional encoding** scheme; the authors state that either one alone is insufficient to solve fixed-length context cleanly.
- The recurrence caches a sequence of prior hidden states rather than one recurrent vector, preserving token-level information while applying `SG(·)`—stop-gradient—to the cached segment.
- The reported relative effective context length is **80% longer than recurrent networks** and **450% longer than vanilla Transformers**.
- Reported test results include **0.99 bits per character on enwik8**, **1.08 on text8**, and **18.3 perplexity on WikiText-103**.
- A model trained only on WikiText-103 generated coherent novel text extending for thousands of tokens, demonstrating that the added memory changed generation behavior rather than merely one benchmark score.

## What Came Before → What It Enabled

### What came before

- [[../10-Backpropagation/Hochreiter Schmidhuber LSTM 1997|LSTM]] carried state through time, but compressed the past into a recurrent state and trained sequentially.
- [[Bahdanau Attention 2015|Bahdanau attention]] let a decoder retrieve encoder states instead of trusting one fixed vector.
- [[Vaswani Attention Is All You Need 2017|The Transformer]] replaced recurrent processing with direct self-attention and parallel training, but practical language models still chopped continuous text into disconnected windows.

### What broke

The Transformer made relationships inside a window easy to learn, then treated the edge of that window as if history had ended. Long documents did not agree. Increasing the window attacked the symptom while paying the quadratic attention bill.

### What it enabled

- Long-range neural language modeling without returning to token-by-token recurrent training.
- Reusable hidden-state caching as an architectural memory mechanism, not merely a serving optimization.
- Relative positional representations that remain meaningful when attention reaches beyond the segment seen during training.
- A second long-context branch beside [[Child Sparse Transformer 2019|Sparse Transformer]]: **reuse the past** as well as **look at less of it**.
- The conceptual path toward modern systems that combine recurrent or compressed memory, sparse attention, and million-token context management, including [[../13-Scaling-and-Alignment/DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]].

## Why It Matters

Transformer-XL closes an easy-to-miss gap in the lineage.

The 2017 Transformer paper explains why recurrence was not necessary for sequence computation. It does **not** explain how a practical language model remembers beyond the chunk currently in memory. Without Transformer-XL, the history can sound magical: attention appears, then models somehow begin writing coherent long passages.

The actual hinge was less glamorous and more important: **do not throw away a useful computation just because a batching boundary was reached.** Cache it, let the next segment attend to it, and represent time in a way that survives reuse.

That repair also clarifies why later long-context work was necessary. Transformer-XL proved that context boundaries were an implementation choice, not a law of the architecture. But once the boundary moved, cost became the next wall. [[Child Sparse Transformer 2019|Sparse Transformer]] reduced how much history each token had to inspect; later retrieval, state-space, and compressed-attention systems changed how history was stored and selected.

Generative AI needed both lessons: **memory must persist, and persistence must eventually become affordable.**

## Lineage

- **Built on:** [[Vaswani Attention Is All You Need 2017|Transformer self-attention]], [[../10-Backpropagation/Hochreiter Schmidhuber LSTM 1997|recurrent memory]], and relative-position work in attention
- **Repairs:** fixed-length context, segment-boundary amnesia, temporal confusion under state reuse, and sliding-window recomputation
- **Runs alongside:** [[Child Sparse Transformer 2019|structured sparse attention]]
- **Leads toward:** long-context language models, cached/compressed attention state, and frontier context systems such as [[../13-Scaling-and-Alignment/DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]]

---

*The Transformer had learned where to look. Transformer-XL taught it not to burn yesterday’s map at every segment boundary.*
