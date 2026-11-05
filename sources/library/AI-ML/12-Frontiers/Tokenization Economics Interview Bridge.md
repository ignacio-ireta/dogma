# Tokenization Economics — Interview Bridge

*Use subword tokenization to turn “context window” from a model-spec slogan into a measurable coverage, cost, and latency decision.*

**Source notes:**
- [[Sennrich Haddow Birch Subword BPE 2016]]
- [[Radford GPT-2 2019]]

**Source paths:**
- `AI-ML/12-Frontiers/Sennrich Haddow Birch Subword BPE 2016.md`
- `AI-ML/12-Frontiers/Radford GPT-2 2019.md`
- `corpus/raw/library-papers/1508.07909.pdf`

---

## The Diagnostic Hinge

A model does not receive words, pages, or database rows. It receives tokens chosen by its tokenizer.

Sennrich et al. solved a coverage problem: whole-word vocabularies lose unseen forms, while character sequences cover everything at excessive length. Learned subwords keep frequent forms compact and compose rare forms from reusable pieces. The same compromise now affects production economics:

```text
text → tokenizer → token count → usable context, latency, and cost
```

The durable interview distinction is:

- **Coverage:** Can every name, acronym, language, and domain term be represented without an unknown-token escape hatch?
- **Efficiency:** How many tokens does representative text require?
- **Capacity:** After instructions, history, and output reserve, how much evidence actually fits?
- **Quality:** Does chunking preserve the semantic units needed to answer the question?

A published “128k context window” answers none of these for a particular enterprise corpus until the corpus is tokenized with the candidate model’s actual tokenizer.

## The Interview Move

If asked how to choose a model or size a RAG prompt, avoid assuming a fixed words-per-token conversion.

A stronger answer is:

> I would measure token counts on representative production documents, segmented by language and document type. Then I would budget the window explicitly across system instructions, user history, retrieved evidence, and output reserve. Tokenization changes both cost and effective capacity, especially for multilingual text, acronyms, identifiers, and specialized terminology, so I would compare coverage and answer quality on the real corpus rather than infer them from the model’s headline context size.

This makes tokenization a systems variable rather than preprocessing trivia.

## Case Framing

For a multilingual enterprise assistant, evaluate the text interface before debating fine-tuning or a larger context window:

1. **Profile the corpus.** Measure token-per-character or token-per-word distributions by language, source system, and document class; report tails, not just an average.
2. **Inspect costly fragments.** Product codes, department acronyms, names, tables, and mixed Spanish–English text may split very differently from ordinary English prose.
3. **Set a real prompt budget.** Reserve capacity for instructions, conversation state, citations, and the answer before deciding retrieval `top_k` or chunk size.
4. **Chunk semantically, count mechanically.** Preserve headings, definitions, and table rows where possible, then enforce the model’s token limit. Fixed character chunks can sever meaning and still vary wildly in token count.
5. **Test end to end.** Compare retrieval recall, grounded answer quality, p95 latency, and cost per successful answer. Fewer tokens are useful only if the evidence still survives.
6. **Re-measure on model changes.** A new model or tokenizer can change prompt occupancy even when the documents are identical.

A concise architecture answer:

> Context size is denominated in tokenizer-specific units, so I would not translate a model card into document capacity using a generic words-per-token rule. I would tokenize the actual multilingual corpus, reserve the non-evidence parts of the prompt, and choose chunking and retrieval depth against grounded-task quality, latency, and cost.

## One Sentence To Memorize

A context window is not a document limit—it is a tokenizer-specific budget, so measure the real corpus before promising coverage, cost, or latency.

---

Tags: #bridge #interview #rag #tokenization #multilingual #cost
