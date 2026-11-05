# Dao et al. — FlashAttention (2022)

*The quadratic attention matrix did not only cost arithmetic. Moving it through GPU memory was often the slower part.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**“FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”** — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré, 2022.

- Paper: `corpus/raw/library-papers/2205.14135.pdf`
- arXiv: [2205.14135](https://arxiv.org/abs/2205.14135)
- Source PDF status: local file verified as `PDF document, version 1.5` on 2026-09-01

## Key Ideas

1. **The field was optimizing the wrong cost model.**
   Standard self-attention forms an `N × N` score matrix, applies softmax, and multiplies by the values. Earlier efficient-attention work therefore focused on reducing FLOPs through sparse, low-rank, or kernel approximations. Yet lower asymptotic compute did not reliably produce lower wall-clock time. On a GPU, repeatedly reading and writing large intermediates between high-bandwidth memory (HBM) and much smaller, faster on-chip SRAM can dominate the arithmetic.

2. **FlashAttention keeps exact attention and changes its execution order.**
   The algorithm tiles `Q`, `K`, and `V` into blocks that fit in SRAM. It computes each block of scores on chip and accumulates the output without ever materializing the full attention or probability matrix in HBM. This is not a new approximation to attention: aside from ordinary numerical roundoff, it computes the same result.

3. **Online softmax makes blockwise exactness possible.**
   A softmax denominator normally appears to require an entire row at once. FlashAttention maintains a running row maximum and normalization sum as successive key blocks arrive, rescaling earlier partial results when the maximum changes. The final normalization is therefore identical to a full-row softmax without storing the full row.

4. **Recomputation is cheaper than storage when memory traffic is the bottleneck.**
   The backward pass saves compact normalization statistics, then recomputes score/probability blocks in SRAM instead of loading an `N × N` intermediate from HBM. That performs more FLOPs, but less expensive IO. The counterintuitive result is central: **doing more arithmetic can finish sooner when it avoids slow data movement.**

5. **It changes memory complexity, not the mathematical compute complexity of dense attention.**
   Exact dense FlashAttention still performs quadratic attention work. Its major repair is reducing HBM traffic and avoiding quadratic-size stored intermediates, making additional context fit and run faster. The paper also implements block-sparse FlashAttention for cases willing to trade exact dense connectivity for still lower cost.

6. **The paper grounds the systems claim empirically and theoretically.**
   The authors derive lower HBM-access complexity than standard attention and prove IO optimality for a range of SRAM sizes. Their implementation reports a `7.6×` speedup for the attention kernel on GPT-2, `3×` end-to-end GPT-2 training speedup at sequence length 1K, `15%` faster BERT-large training than the cited MLPerf 1.1 record, and `2.4×` speedup on Long Range Arena tasks. Longer feasible sequences also improved model quality rather than merely benchmark throughput.

## What Came Before → What It Enabled

### What came before

- [[Vaswani Attention Is All You Need 2017|The Transformer]] made dense all-to-all attention the core sequence operation, but its `N × N` intermediates grew rapidly with context length.
- [[Child Sparse Transformer 2019|Sparse Transformer]] reduced the number of token pairs evaluated through structured patterns, accepting restricted connectivity to reduce asymptotic cost.
- [[Dai et al. Transformer-XL 2019|Transformer-XL]] reused prior hidden states across segment boundaries, but retained memory still had to be attended over efficiently.

### What broke

The field could not just keep inventing lower-FLOP attention formulas. Many approximations changed model behavior, added irregular kernels, or saved theoretical operations without becoming faster on actual accelerators. Even ordinary exact attention spent too much time materializing and transporting intermediates that the next operation immediately consumed.

The missing question was not only *“How many multiplications?”* It was *“Where does every byte travel?”*

### What it enabled

- Longer exact-attention training within the same accelerator-memory budget.
- Fused attention kernels becoming standard infrastructure rather than an optional model-specific trick.
- A hardware-aware design vocabulary—tiling, kernel fusion, online reductions, recomputation, and memory hierarchy—for frontier-model architecture.
- Later FlashAttention revisions and framework-native memory-efficient scaled dot-product attention.
- Long-context systems that combine an efficient exact local/dense primitive with sparse, sliding-window, compressed, or retrieved context rather than paying naive attention overhead everywhere.

## Why It Matters

FlashAttention closes the gap between an algorithm that is elegant on paper and one that survives the memory hierarchy of a real GPU.

That distinction matters to the generative-AI lineage because scaling did not happen through model ideas alone. GPT-style models needed longer contexts and larger training runs, but the standard implementation wrote an enormous attention matrix to HBM, read it back for softmax, wrote the probabilities, and read them again. The mathematics was correct; the schedule was wasteful.

FlashAttention showed that exactness and efficiency were not necessarily opposites. By reorganizing the same computation around the hardware, it recovered speed and memory without asking the model to forget token relationships. Sparse and approximate methods remained necessary when quadratic arithmetic itself became the wall, but exact dense attention gained much more runway first.

This also sharpens the chain into frontier long-context models such as [[../13-Scaling-and-Alignment/DeepSeek-V4 — DeepSeek AI 2026|DeepSeek-V4]]. Sparse or compressed attention decides **which relationships to compute**; FlashAttention-style IO awareness decides **how to compute the chosen relationships without drowning in memory traffic**. Million-token context ultimately needs both architectural selection and systems execution.

## Lineage

- **Built on:** [[Vaswani Attention Is All You Need 2017|dense Transformer attention]], GPU tiling and kernel fusion, and the long-context pressure exposed by [[Dai et al. Transformer-XL 2019|Transformer-XL]] and [[Child Sparse Transformer 2019|Sparse Transformer]]
- **Repairs:** quadratic stored intermediates, excessive HBM traffic, and the mismatch between FLOP counts and wall-clock speed
- **Preserves:** exact dense attention semantics while reducing memory traffic and activation storage
- **Leads toward:** production fused-attention kernels, longer contexts, hardware-aware model design, and hybrid long-context systems

---

*Attention had learned to look everywhere. FlashAttention taught it to stop carrying the whole view back and forth through the slowest door.*
