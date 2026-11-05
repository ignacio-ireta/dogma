# Sutskever, Vinyals & Le — Sequence to Sequence Learning (2014)

*Deep networks could classify fixed-size inputs, and LSTMs could remember through time. But translation was neither: one variable-length sequence had to become another, with different length and no known alignment.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**“Sequence to Sequence Learning with Neural Networks”** — Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, 2014.

- Paper: `corpus/raw/library-papers/1409.3215.pdf`
- arXiv: `1409.3215`
- Venue: NeurIPS 2014
- Source PDF is present locally and verified as a 9-page PDF document.

## Key Ideas

### 1. Fixed-dimensional deep networks broke on variable-length problems

A conventional feed-forward network expects an input vector and an output vector whose sizes are known in advance. That works for image classification. It does not naturally express translation, question answering, or dialogue, where both sides are sequences and their lengths can differ.

A recurrent network can process a sequence when input and output positions are already aligned. Translation is harder: the third English word need not correspond to the third French word, and one language may need more tokens than the other. The old recipe could not simply be stretched.

### 2. The encoder–decoder makes sequence transduction end-to-end

The paper uses two deep LSTMs:

1. an **encoder** reads the source sequence and compresses it into a fixed-dimensional vector;
2. a **decoder** is initialized from that vector and generates the target sequence token by token;
3. an end-of-sequence token tells the decoder when to stop.

This factorizes translation as

\[
p(y_1,\ldots,y_{T'}\mid x_1,\ldots,x_T)
= \prod_{t=1}^{T'} p(y_t\mid v,y_1,\ldots,y_{t-1}),
\]

where \(v\) is the encoder’s representation of the whole source sentence. One differentiable model can therefore learn both representation and generation directly from sentence pairs.

### 3. LSTM repairs the long-delay training problem — enough to make the idea work

A plain RNN would have to carry information across the entire source sentence and into the target sentence. [[Bengio Simard Frasconi Long-Term Dependencies 1994|Bengio, Simard & Frasconi]] had already shown why gradients across such delays become unstable or vanish. [[Hochreiter Schmidhuber LSTM 1997|LSTM]] supplied a gated memory path designed for that failure.

Sutskever and colleagues turned that memory mechanism into a general sequence interface. Their four-layer LSTMs had 384 million parameters, were trained with SGD and gradient clipping, and directly outperformed the phrase-based statistical machine-translation baseline on WMT’14 English→French: **34.81 BLEU versus 33.30**.

That result mattered more than the exact score. A neural system that made very few assumptions about linguistic structure could learn translation end to end at production-relevant scale.

### 4. Reversing the source sentence exposed optimization as the hidden constraint

The paper’s strangest and most revealing trick was to reverse the source sentence while leaving the target sentence in normal order. This moved the beginning of the source close to the beginning of the target, creating short dependency paths that SGD could establish first.

The change reduced test perplexity from **5.8 to 4.7** and raised single-model BLEU from **25.9 to 30.6**. The architecture had not gained information. The problem had been re-encoded so gradients had shorter routes.

This is a durable lesson: representational capacity is not the same as trainability. Sometimes the model can express the solution but cannot reliably reach it through the available optimization path.

### 5. The model learned sentence representations, not just phrase-table substitutions

The encoder’s vectors were sensitive to word order while placing active and passive paraphrases near each other. Translation supervision forced the network to construct a distributed representation of sentence meaning useful enough for the decoder to regenerate that meaning in another language.

This is one of the cleanest bridges from learned representations to conditional generation: the hidden state is no longer merely a classifier feature. It is the condition from which a new sequence is generated.

## What Came Before → What It Enabled

### What came before

- [[Rumelhart Hinton Williams Backpropagation 1986|Backpropagation]] made multilayer representations trainable, but ordinary DNNs still expected fixed-size inputs and targets.
- Recurrent neural networks introduced state across time, but long credit-assignment paths were hard to optimize.
- [[Bengio Simard Frasconi Long-Term Dependencies 1994|Bengio et al. 1994]] diagnosed the vanishing-gradient conflict between robust memory and trainability.
- [[Hochreiter Schmidhuber LSTM 1997|LSTM]] created a gated path for preserving information over long delays.
- [[Cho Encoder-Decoder GRU 2014|Cho et al. 2014]] independently developed an RNN encoder–decoder with the GRU and showed that sentence vectors could improve statistical machine translation.

### What it enabled

- A general neural recipe for mapping **variable-length sequences to variable-length sequences**.
- Direct neural machine translation without a hand-built phrase table or a fixed monotonic alignment.
- The modern encoder–decoder abstraction used in translation, summarization, dialogue, speech, and conditional generation.
- A clear failure for the next paper to repair: every source token still had to pass through **one fixed-dimensional vector**.

That last point is the hinge. Reversing the source reduced the optimization distance, and LSTM made long memory possible, but neither removed the compression bottleneck. As sentences and tasks grew, one vector had to preserve every detail the decoder might later need.

[[../12-Frontiers/Bahdanau Attention 2015|Bahdanau, Cho & Bengio]] answered by letting the decoder look back at all encoder states and learn a soft alignment at each output step. Attention did not appear from nowhere. It was the repair demanded by successful sequence-to-sequence learning.

## Why It Matters

This paper is where neural networks stop being merely systems that label an input and become systems that **transform one structured sequence into another**.

That sounds obvious now because generative AI made sequence generation ordinary. It was not obvious in 2014. Statistical machine translation was a mature pipeline of alignments, phrase tables, language models, and search. Feed-forward neural networks could improve pieces of that pipeline, but they did not provide a clean replacement for the whole mapping.

The encoder–decoder changed the unit of learning. Instead of engineering how phrases correspond, the system learned a conditional distribution over complete output sequences. Instead of treating generation as a collection of local classification decisions with externally supplied structure, it learned the representation and the generator together.

But the breakthrough carried its own breakage inside it. The encoder compressed a whole sentence into one vector. The more successful the model became, the more obvious that bottleneck became. Long or information-dense inputs should not have to squeeze through a single state and hope the decoder remembers everything.

That is why this note belongs directly before attention:

- backpropagation made hidden representations learnable;
- LSTM made temporal credit assignment survivable;
- sequence-to-sequence learning made end-to-end conditional generation work;
- the fixed-vector bottleneck made attention necessary;
- attention eventually made the Transformer possible.

Generative AI did not arrive because someone suddenly decided models should write. It arrived through a sequence of repairs. This paper supplied the first broadly convincing end-to-end generator — and, by succeeding, exposed exactly what had to break next.

## Lineage

- **Built on:** [[Rumelhart Hinton Williams Backpropagation 1986|backpropagation]], [[Bengio Simard Frasconi Long-Term Dependencies 1994|the long-term dependency diagnosis]], [[Hochreiter Schmidhuber LSTM 1997|LSTM]], and [[Cho Encoder-Decoder GRU 2014|the parallel GRU encoder–decoder line]]
- **Leads to:** [[../12-Frontiers/Bahdanau Attention 2015|neural attention]], [[../12-Frontiers/Vaswani Attention Is All You Need 2017|Transformers]], neural translation, summarization, dialogue, and modern conditional generation

---

*Seq2seq proved that one neural system could read a sentence and write another. Attention arrived because one vector was too small a doorway for everything the reader had seen.*
