# Lewis et al. — Retrieval-Augmented Generation (2020)

*RAG is what happens when a language model stops pretending its parameters should contain the whole library.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela, NeurIPS 2020.
- Facebook AI Research
- [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

## Key Ideas

1. **Parametric memory is powerful, but annoyingly sealed shut.**
   Large language models store a lot of factual knowledge in their weights, but that knowledge is hard to inspect, hard to update, and easy to hallucinate around. If the world changes, you usually have to retrain or fine-tune the model.

2. **RAG splits knowledge into two kinds of memory.**
   - **Parametric memory:** what the seq2seq model already knows in its parameters
   - **Non-parametric memory:** an external document store, here a dense Wikipedia index

   The move is conceptually beautiful: do not force the model to memorize everything. Let it *look things up*.

3. **Retrieval becomes part of generation, not a separate pre-processing step.**
   The model uses a neural retriever to fetch relevant passages, then conditions generation on those passages. This means factual support can flow directly into the answer instead of hoping the model remembers correctly from pretraining alone.

4. **The paper proposes two RAG variants.**
   - **RAG-Sequence:** retrieve a set of passages once, use them for the whole generated answer
   - **RAG-Token:** allow different passages to matter at different decoding steps

   RAG-Token is more flexible. RAG-Sequence is simpler and cheaper. That tradeoff is still alive in production systems now.

5. **The real win is not just accuracy. It is provenance and updatability.**
   Because the answer is grounded in retrieved documents, you can inspect what the model used, swap out the knowledge base, and refresh the system's world knowledge without redoing pretraining.

## Lineage

- **Built on:** [[Devlin BERT 2018|BERT]] and the dense-retrieval turn in representation learning, plus the broader seq2seq/pretraining wave that culminates in [[Brown GPT-3 2020|GPT-3]]
- **Responds to:** the factual brittleness of parametric-only language models and the awkward gap between retrieve-then-read QA systems and fluent text generation
- **Leads to:** modern enterprise RAG stacks, grounded copilots, retrieval-backed agents, and most practical "chat with your documents" systems
- **Connected to:** [[Ouyang InstructGPT RLHF 2022|RLHF]] and later alignment work, because once models become more helpful conversationally, grounding their answers becomes even more important

## Why It Matters

RAG is one of those papers whose historical role is bigger than its benchmark table.

This is the paper that made a crucial institutional compromise intellectually respectable:

**maybe the model should not be the database.**

That sounds obvious now, but it was not obvious in 2020. The frontier mood then was still very "just scale the parameters and the knowledge will be inside." Lewis et al. say: yes, but external memory is not a failure of the model. It is a design choice.

That choice turned out to be incredibly important for real systems:
- lower hallucination risk,
- fresher knowledge,
- inspectable sources,
- domain adaptation without retraining a giant base model.

In other words: this is the bridge from glamorous foundation models to useful enterprise AI.

## The Real Shift

Before RAG, the question was mostly:
- *How much can a model remember?*

After RAG, a better question became:
- *What should the model remember, and what should it retrieve?*

That is a much more grown-up systems question. And the whole industry is still answering it.

---

*RAG did not make language models smarter in the mystical sense. It made them more willing to check the shelf before speaking.*
