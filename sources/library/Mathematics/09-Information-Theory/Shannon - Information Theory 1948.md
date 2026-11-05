# Claude Shannon - Information Theory (1948)

#person #paper #milestone

**Person:** Claude E. Shannon (1916-2001)  
**Work:** *A Mathematical Theory of Communication* (1948)  
**Primary source:** *The Bell System Technical Journal*, 27, 379-423 and 623-656  
**Paper:** `corpus/raw/library-papers/shannon-information-theory-1948.pdf`  
**Source note:** Local PDF verified as a PDF document. The file is a corrected reprint of the 1948 two-part Bell System Technical Journal paper.

## Title
Uncertainty becomes a measurable quantity.

Probability had already learned how to assign numbers to uncertain events.
Kolmogorov had even given probability its measure-theoretic foundation.

But modern AI needs a sharper question than:

> how likely is this event?

It needs:

> how much uncertainty remains before the next symbol is observed, and how much is removed when the symbol arrives?

That is Shannon's move.

He does not begin with meaning. He begins with selection. A source chooses one message from many possible messages. Communication is the problem of representing that choice, sending it through a channel, and recovering it despite noise.

Once that is the problem, information stops being a metaphor. It becomes a quantity.

## Paper Reference
**Shannon, C. E. (1948).** *A Mathematical Theory of Communication.* *The Bell System Technical Journal*, 27, 379-423 and 623-656.

Local source:
- `corpus/raw/library-papers/shannon-information-theory-1948.pdf`

Reference trail: Bell System Technical Journal reprint, CiNii/Crossref-style bibliographic records, and the AI/ML-side Library note [[../../AI-ML/04-Probability/Shannon Information Theory 1948|Shannon Information Theory 1948]].

## What Problem Made This Necessary?
Probability gives a language for uncertainty, but communication engineering needed a language for **uncertainty carried by messages**.

Before Shannon, there were pieces:

- telegraphy and telephony had practical engineering rules,
- Nyquist and Hartley had connected signaling rates to logarithmic measures,
- probability could describe uncertain events,
- cryptography already cared about unpredictability,
- and statistics could model observed frequencies.

But there was no single theory that said what all communication systems were doing underneath the hardware.

The missing abstraction was this:

> a message is a reduction of uncertainty about which possibility a source selected.

That made communication independent of its medium. Telegraph signals, speech, written English, punched cards, and later digital tokens could all be treated as outputs of a source with a probability structure.

Without that abstraction, "message", "noise", "compression", "redundancy", and "capacity" remain engineering intuitions. With it, they become mathematics.

## Key Ideas

### 1. Information is about selection, not meaning
Shannon famously brackets off semantic meaning.

That is not because meaning is unimportant. It is because the engineering problem is different: the receiver must identify which message was selected from the set of possible messages.

This makes the theory general. A communication system can carry English, Morse code, images, sensor readings, or machine tokens. The mathematical question is how uncertain the receiver is before and after the signal.

### 2. The logarithm makes information additive
If there are \(M\) equally likely messages, Shannon wants the information measure to increase with \(M\), behave continuously, and add when independent choices are combined.

That points to:

```text
H = log M
```

With base 2, the unit is the bit.

The logarithm is the quiet structural choice. It means two independent binary decisions carry two bits, not four mysterious units. Information becomes compositional.

### 3. Entropy measures expected uncertainty
Real sources rarely choose messages uniformly.

For probabilities \(p_i\), Shannon's entropy is:

```text
H = - sum p_i log p_i
```

Common events are less surprising. Rare events carry more information when they occur. A predictable source has lower entropy; a highly uncertain source has higher entropy.

This turns probability distributions into measurable uncertainty budgets.

### 4. Language becomes a stochastic source
Shannon models English by increasingly structured approximations: independent letters, letter frequencies, digrams, trigrams, word sequences.

The point is not that these are good modern language models. The point is that written language has statistical structure. It is not uniformly random. Its next symbol is constrained by what came before.

That is the primitive form of the modern next-token question.

### 5. Compression and prediction become the same bargain
If a source is predictable, it is compressible.

A good code spends fewer bits on common messages and more bits on rare ones. A good probabilistic model assigns high probability to what actually appears. These are two faces of the same object:

```text
better source model -> lower expected code length -> lower cross-entropy
```

This is the mathematical root of why language-model loss is not arbitrary. Cross-entropy asks how many bits of surprise remain under the model's predicted distribution.

### 6. Noise turns information into a channel-capacity problem
Shannon separates the source, transmitter, channel, receiver, destination, and noise source.

That diagram becomes one of the great abstractions of the 20th century. It lets mathematics ask:

- how much information does the source produce?
- how much can the channel carry?
- how much redundancy is needed to survive noise?
- when is reliable communication possible?

This is not just telecommunications. It is the ancestor of denoising, representation bottlenecks, error-correcting codes, and probabilistic modeling under corruption.

## Historical Context
Shannon was working at Bell Labs, where communication was not a philosophical abstraction. It was wires, switches, relays, bandwidth, noise, and money.

The practical pressure mattered. Telephone and telegraph networks forced a question pure probability had not needed to ask by itself:

> what is the maximum reliable rate at which messages can be transmitted through a noisy physical channel?

Shannon answered by creating a measurement theory for information itself.

The historical placement is important for the Library. Kolmogorov gives probability a foundation in 1933. Shannon then uses probability to measure uncertainty in messages in 1948. After that, probabilistic prediction can become an engineering objective, not only a statistical description.

That is the bridge to machine learning.

## Lineage

```text
Bayes 1763 (belief updating)
  └── Laplace 1812 (probability as a general scientific calculus)
      └── Gauss 1809 (error, fitting, least squares)
          └── Kolmogorov 1933 (probability as measure)
              └── Shannon 1948 (entropy, bits, source/channel coding)
                  ├── cross-entropy and negative log-likelihood
                  ├── compression as prediction
                  ├── probabilistic language modeling
                  ├── information bottlenecks and representation learning
                  └── modern next-token training objectives
```

More specifically:

- **Built on:** probability distributions, logarithmic information measures from Hartley, communication engineering, and Kolmogorov's rigorous probability foundation.
- **Solved the next problem:** probability could say how likely events were, but communication needed to know how much uncertainty messages remove and how many bits are needed to encode them.
- **Leads to:** coding theory, channel capacity, cross-entropy, KL divergence, perplexity, compression, language modeling, information bottlenecks, denoising, and modern generative AI.

## Why It Matters for AI
Every serious language model is downstream of Shannon.

The Transformer gives the architecture. Backpropagation gives the learning mechanism. GPUs give the scale.

But Shannon gives the objective its meaning.

When a language model predicts the next token, it is modeling a source. When training minimizes cross-entropy, it is reducing expected surprise under that source. When evaluation reports perplexity, it is measuring remaining uncertainty.

This is why next-token prediction is not a toy. A source with rich structure can be understood by modeling its conditional distributions. Human language is such a source. Code is such a source. Scientific text, images-as-tokens, audio tokens, and multimodal streams all become sources in Shannon's sense once they are represented as sequences or distributions.

Modern AI keeps rediscovering Shannon-shaped questions:

- How much information is in the representation?
- What can be compressed without losing the signal?
- What does the model know about the data distribution?
- How much uncertainty remains after conditioning on context?
- How much channel noise can be corrected?
- Which bottleneck preserves useful information and discards nuisance variation?

That is not decoration around ML. It is one of the deep mathematical reasons ML works.

## The Real Historical Role
Shannon did for information what earlier mathematicians did for area, motion, number, probability, and the continuum:

he made the vague thing measurable.

Archimedes made curved area attackable.
Descartes made geometry algebraic.
Cauchy and Weierstrass made limits rigorous.
Kolmogorov made probability a measure.

Shannon made messages mathematical.

Modern AI did not start at Bell Labs in 1948. But one of its most important loss functions did.

## Leads To
- [[../_MOC|Mathematics Lineage]]
- [[../06-Probability/Kolmogorov - Foundations of the Theory of Probability 1933]]
- [[../../AI-ML/04-Probability/Shannon Information Theory 1948]]
- [[../../AI-ML/12-Frontiers/Radford GPT-2 2019]]
- [[../../AI-ML/12-Frontiers/Kaplan Scaling Laws 2020]]
- [[Kullback-Leibler - Information Divergence 1951]] — relative information, data processing, sufficiency, and cross-entropy
- Future note: Information bottleneck as the representation-learning bridge

## References
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* *The Bell System Technical Journal*, 27, 379-423 and 623-656. Local PDF: `corpus/raw/library-papers/shannon-information-theory-1948.pdf`.
- Hartley, R. V. L. (1928). *Transmission of Information*, used as prehistory for logarithmic information measures.
- Kolmogorov, A. N. (1933). *Foundations of the Theory of Probability*, used as the probability-foundation predecessor.
- AI/ML Library note: [[../../AI-ML/04-Probability/Shannon Information Theory 1948]].

## Status
- [x] Note created (June 5, 2026)
- [x] Source PDF already present and verified at `corpus/raw/library-papers/shannon-information-theory-1948.pdf`
- [x] Closes the Kolmogorov -> information theory -> language modeling bridge in the Mathematics lineage
- [x] Added [[Kullback-Leibler - Information Divergence 1951]] as the next information-theoretic bridge

---

*Shannon is where uncertainty becomes something you can budget, transmit, compress, and train against.*
