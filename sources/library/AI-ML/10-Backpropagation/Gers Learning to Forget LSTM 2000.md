# Gers, Schmidhuber & Cummins - Learning to Forget with LSTM (2000)

*The forget gate: the small repair that turned LSTM from "can remember" into "can decide when memory is stale."*

**<- [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Learning to Forget: Continual Prediction with LSTM"** - Felix A. Gers, Jurgen Schmidhuber, Fred Cummins, 2000.
- IDSIA technical report IDSIA-01-99, later published in *Neural Computation* 12(10), 2451-2471
- DOI: 10.1162/089976600300015015
- Paper: `corpus/raw/library-papers/gers-learning-to-forget-lstm-2000.pdf`

## Key Ideas

1. **Original LSTM could remember, but not reliably clear itself.**
   [[Hochreiter Schmidhuber LSTM 1997]] fixed vanishing gradients with the Constant Error Carousel: a linear self-connection with weight 1.0, protected by input and output gates. That lets information persist across long time lags.

   But persistence has a shadow. If the input stream is continual, with no external reset between subsequences, the cell state can keep accumulating. The model has memory, but not hygiene.

2. **The failure mode is continual input, not short toy sequences.**
   Earlier LSTM experiments often reset internal states at known sequence boundaries. Real streams are rarely so polite. Speech, sensor readings, text, and behavioral logs arrive as continuous flow. If the network depends on a human-provided "sequence ends here" signal, the memory system is not fully learned.

3. **The forget gate learns a multiplicative reset.**
   Gers et al. replace the fixed self-recurrent CEC weight of 1.0 with a learned gate value between 0 and 1:

   - near 1: preserve the cell state,
   - near 0: reset it,
   - in between: let it decay gradually.

   The cell state is no longer just "old state plus new write." It becomes:

   `c_t = f_t * c_{t-1} + i_t * candidate_t`

   That one term, `f_t`, is the difference between memory as accumulation and memory as controlled state.

4. **Forgetting also clears credit assignment history.**
   The paper's nice line, technically, is that when forget-gate activation goes to zero, it resets not only the cell state but also the relevant partial derivatives. Forgetting includes forgiving previous mistakes. The network can reuse internal resources instead of dragging old context through every future update.

5. **The result is self-segmentation.**
   On continual variants of benchmark sequence tasks, standard LSTM fails even though it succeeds with external resets. LSTM with forget gates learns when to reset itself. That matters because the model is no longer assuming the world arrives pre-chopped into training examples.

## What Came Before -> What It Enabled

**Before:** Backpropagation through time could not preserve useful gradient signal over long delays. Standard RNNs either forgot too quickly or blew up. Hochreiter and Schmidhuber's 1997 LSTM repaired that with a memory cell whose error signal could circulate without decay.

**What broke:** A perfect memory path is not automatically a good memory policy. If an LSTM sees a continual stream with no explicit boundary markers, the cell state can grow without bound. Resetting externally works in laboratory tasks, but it smuggles in segmentation knowledge that the network should learn for itself.

**After:** The forget gate made LSTM a genuinely adaptive memory machine. It could preserve information across long gaps, erase stale information when a subtask ended, and reuse cells for the next segment of a continuous stream.

That repair became the LSTM people now recognize. Modern diagrams usually show input, output, and forget gates as if all three were born together. They were not. The forget gate is the 2000 correction that made the architecture much closer to a practical sequence model.

## Why It Matters for the Lineage

The lineage from backprop to Transformers is not just "RNNs were too slow, attention won." There is a deeper story about **information flow**.

- [[Rumelhart Hinton Williams Backpropagation 1986]] gave neural nets credit assignment.
- [[Hochreiter Schmidhuber LSTM 1997]] protected credit assignment through time.
- **Gers et al. 2000** added learned forgetting, so protected memory would not become uncontrolled accumulation.
- [[Cho Encoder-Decoder GRU 2014]] simplified the gating idea.
- [[Bahdanau Attention 2015]] showed that even gated recurrence still bottlenecked translation through compressed state.
- [[Vaswani Attention Is All You Need 2017]] removed recurrence and made direct token-to-token access the main computation.

So the forget gate is not a side branch. It is the moment where long-term memory becomes **managed state**. That idea survives after LSTMs: residual streams, attention masks, normalization, gating, router decisions, memory tokens, and long-context architectures all ask the same question in different forms:

*what should persist, what should update, and what should be allowed to disappear?*

## The Real Shift

Original LSTM answered: *how do we keep a signal alive across time?*

Gers et al. answered the next, less glamorous, more grown-up question:

*how do we stop keeping it once it is no longer useful?*

That is the whole lesson. Memory without forgetting is not intelligence. It is clutter with gradients.

---

## Status

- [x] Source PDF saved: `corpus/raw/library-papers/gers-learning-to-forget-lstm-2000.pdf`
- [x] Note created: July 2, 2026
- [ ] Later: add Sutskever, Vinyals & Le 2014 as the LSTM sequence-to-sequence branch before Bahdanau attention
