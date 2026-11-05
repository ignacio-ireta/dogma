# Context Representation — Interview Bridge

*Use the path from fixed windows to recurrent state to attention to explain model choice as a memory-and-retrieval decision, not as a chronology recital.*

**Source notes:**
- [[Bengio Neural Probabilistic Language Model 2003]]
- [[Mikolov Recurrent Neural Language Model 2010]]
- [[../10-Backpropagation/Sutskever Vinyals Le Sequence to Sequence 2014]]
- [[Bahdanau Attention 2015]]
- [[Vaswani Attention Is All You Need 2017]]

**Source paths:**
- `AI-ML/12-Frontiers/Bengio Neural Probabilistic Language Model 2003.md`
- `AI-ML/12-Frontiers/Mikolov Recurrent Neural Language Model 2010.md`
- `AI-ML/10-Backpropagation/Sutskever Vinyals Le Sequence to Sequence 2014.md`
- `AI-ML/12-Frontiers/Bahdanau Attention 2015.md`
- `AI-ML/12-Frontiers/Vaswani Attention Is All You Need 2017.md`
- `corpus/raw/library-papers/bengio-neural-probabilistic-language-model-2003.pdf`
- `corpus/raw/library-papers/mikolov-rnn-language-model-2010.pdf`
- `corpus/raw/library-papers/1409.0473.pdf`
- `corpus/raw/library-papers/1706.03762.pdf`

---

## The Diagnostic Hinge

Every sequence architecture answers the same question:

> What information from the past remains available when the model makes its next decision?

The useful lineage is a sequence of different answers:

1. **Fixed window:** Bengio et al. learn semantic sharing across a short, predefined context. This repairs n-gram sparsity, but information outside the window is absent.
2. **Recurrent state:** Mikolov et al. replace the hard cutoff with a learned state carried through time. The horizon is no longer fixed, but the past is repeatedly compressed into one vector and distant credit is difficult to learn.
3. **Encoder summary:** Seq2seq makes variable-length generation practical, but one final encoder vector must represent the entire source sequence.
4. **Attention:** Bahdanau lets each output step retrieve the source states relevant to that decision instead of trusting one universal summary.
5. **Self-attention:** The Transformer makes contextualization directly addressable and parallel, replacing the recurrent path with learned token-to-token retrieval.

This is not simply “old models became more powerful.” Each step changes the **access pattern**:

```text
window:    preserve the last n items
state:     compress everything seen so far
attention: retrieve different evidence for each decision
```

That distinction is more durable than memorizing architecture dates.

## The Interview Move

If asked why attention displaced recurrence, avoid saying only:

> RNNs have vanishing gradients and Transformers are parallel.

A stronger answer is:

> A recurrent model gives every prediction one compressed summary of the past. Even when gated recurrence improves gradient flow, the state remains a serial information bottleneck. Attention changes the problem from “what can one state preserve?” to “which prior representations should this decision retrieve?” The Transformer then parallelizes that retrieval during training. The tradeoff is that direct access is computationally and memory intensive as context grows.

This answer separates three issues that interview responses often blur:

- **Visibility:** Is relevant evidence inside the model’s accessible context at all?
- **Representation:** Is the evidence preserved distinctly or compressed into a summary?
- **Retrieval:** Can the model select the right evidence for this particular decision?

A large context window solves only visibility. It does not guarantee that the model will retrieve or use the right evidence.

## Case Framing

For an enterprise sequence or generative-AI case, start from the information requirement rather than from the fashionable model:

1. **Does order carry signal?** If recent event counts are enough, engineered tabular features may beat a sequence model on cost, auditability, and latency.
2. **How far back can relevant evidence occur?** A fixed recent window is reasonable only when the business process supplies a defensible horizon.
3. **Must evidence remain individually inspectable?** Recurrent state is compact, but it hides history inside a summary. Retrieval over explicit records preserves provenance more naturally.
4. **Is the knowledge inside model parameters or in a changing corpus?** Frequently changing policy, product, or ontology knowledge favors external retrieval over repeated fine-tuning.
5. **What is the serving constraint?** Attention gives flexible access but makes long contexts expensive. Retrieval quality, chunking, latency, and context budget become part of model quality.

A concise architecture answer:

> I would first identify the minimum memory access pattern the decision needs. If a bounded set of aggregate features captures the signal, I would keep the simpler tabular baseline. If ordered history matters, I would test a sequence model out of time. If the answer depends on changing documents or terminology, I would keep that evidence external and retrieve it with citations rather than asking one hidden state—or one enormous prompt—to remember everything. I would evaluate retrieval recall and answer faithfulness separately because context availability is not context use.

## Retrieval Versus Memory

A domain-terminology agent has a different memory problem. The issue is not mainly temporal sequence; it is **grounded access to changing, organization-specific meaning**.

A model’s parametric memory may know general language but not which department owns a term, which synonym is approved, or which policy version is current. Putting the entire ontology into every prompt improves visibility but wastes context and still does not guarantee retrieval. A better design is to retrieve a small, relevant subgraph or document set for each question and return provenance with the answer.

The lineage gives a clean explanation:

- recurrence asks one state to carry prior information;
- attention lets a decision select among visible representations;
- RAG decides which external representations become visible in the first place.

Therefore RAG quality has at least two separable gates:

- **retrieval gate:** did the system fetch the right terms, relations, and source passages?
- **generation gate:** did the model use that evidence faithfully?

Do not hide a retrieval failure inside an end-to-end answer score.

## One Sentence To Memorize

Sequence-model evolution is a history of changing memory access: from a fixed window, to one compressed state, to decision-specific retrieval—and enterprise RAG adds the prior question of which evidence enters the context at all.

---

Tags: #bridge #interview #rag #attention #context
