# Day 6 Answers — Fine-Tuning + Integration
**Date:** March 5, 2026

---

## Spaced Retrieval Answers

**Q1 (Chunking strategies):**
- **Fixed-size chunking:** Split every N tokens/characters. Simple, predictable. Works for uniform docs. Tradeoff: may split mid-sentence (context loss) or include irrelevant material.
- **Sentence/paragraph chunking:** Split at natural boundaries. Better context preservation. Tradeoff: variable chunk sizes, harder to manage embeddings.
- **Semantic chunking:** Split where topic/meaning changes (embedding-based). Best context quality. Tradeoff: computationally expensive, requires embedding each potential split.
- Key tradeoff: Smaller chunks → higher retrieval precision (more specific) but lose context. Larger chunks → more context but lower precision and higher token cost.

**Q2 (RAG evaluation metrics):**
- **Groundedness:** Does the answer come from the retrieved context (not hallucinated)?
- **Relevance:** Is the retrieved content relevant to the user's question?
- **Coherence:** Is the generated response logically consistent and well-structured?
- **Fluency:** Is the language natural, grammatically correct, and readable?

**Q3 (Prompt variant vs template):**
- **Prompt variant:** A specific version of a prompt with different wording/structure to A/B test. Multiple variants of the same node. SDK: `PromptFlowClient.run()` with a dataset targeting the flow.
- **Prompt template:** A reusable Jinja2 template with variables (e.g., `{{context}}`, `{{question}}`). Defines the prompt structure — variants are different parameterizations of a template.
- Comparing variants: `client.runs.stream(flow=..., data=..., column_mapping=...)` run multiple times with different prompt node versions, then compare metrics.

**Q4 (Prompt Flow tracing):**
- Use tracing to debug complex multi-step flows where intermediate node outputs matter.
- Tracing captures: (1) exact input/output for each node in the execution graph, (2) latency per node (identify bottlenecks). You can't see these from aggregate eval scores alone.

**Q5 (Managed Online Endpoint vs Serverless API):**
- **Managed Online Endpoint:** You provision dedicated compute (VM size, instance count, auto-scale rules). You pay for compute whether it's used or not. Full control over hardware, scaling, deployment ring.
- **Serverless API (Models as a Service / MaaS):** Pay-per-token consumption. No provisioned capacity. No control over underlying infrastructure. Best for variable workloads or evaluation/testing.

---

## Scenario Question Answers

**Q1: C) Fine-tuning**
- Why: SOAP format is a stable behavioral pattern (never changes) + they have 300 examples (good dataset). This is exactly what fine-tuning is for.
- A is wrong because: prompt engineering adds SOAP to every call (token waste) and is unreliable — the model may still deviate from format, especially for complex notes.
- B is wrong because: RAG retrieves facts/documents, not format patterns. Clinical guidelines are knowledge, not format behavior.
- D is wrong because: the model catalog helps you select a base model, not adapt behavior. There's no "SOAP-format" model to download.

**Q2: B) Overfitting — reduce epochs or add more diverse training data**
- Training loss ↓↓ while validation loss ↑↑ is the textbook overfitting signature.
- A is wrong because: underfitting = both losses remain high. Here training loss is very low.
- C is wrong because: a validation loss of 0.45 vs training loss of 0.02 is a massive gap — the model has memorized training data.
- D is wrong because: a wrongly formatted validation file would cause an upload error, not produce loss metrics.

**Q3: B) The file is formatted as a JSON array instead of JSONL**
- Fine-tuning requires one JSON object per line (JSONL). Wrapping in `[...]` makes it a JSON array — invalid.
- A is wrong because: "messages" is the correct key for chat-completions fine-tuning format.
- C is wrong because: a system message is recommended but not required.
- D is wrong because: encoding would cause a different error (garbled characters), not a structure validation failure.

**Q4: B) GPT-4.1 (for complex language tasks)**
- Microsoft explicitly recommends GPT-4.1 for "complex skills like language translation, domain adaptation, or advanced code generation."
- A is wrong because: GPT-4.1-mini is for "focused tasks (classification, sentiment analysis)" — translation is not focused, it's complex.
- C is wrong because: Phi-4 is efficient but not recommended for complex translation.
- D is wrong because: GPT-4o is multimodal (text+images) — no image data here, no need for multimodal.

**Q5: Correct order: D → A → C → B → F → E**
1. D) Upload training JSONL file
2. A) Upload validation JSONL file
3. C) Create fine-tuning job with hyperparameters
4. B) Monitor training loss and validation loss curves
5. F) Evaluate on held-out test set using Azure AI Foundry Evaluations
6. E) Deploy and test the fine-tuned model

**Q6: B) The content filter blocks the input and returns an error before the model processes it**
- Content filtering runs on INPUT before it reaches the model AND on OUTPUT before it reaches the user.
- A is wrong because: content filtering absolutely applies to fine-tuned models — there are no exceptions.
- C is wrong because: filtering happens at input stage, before the model processes anything.
- D is wrong because: the request is blocked, not just logged.

**Q7: C) Groundedness**
- Groundedness measures whether the model's response is supported by retrieved context.
- A is wrong because: Relevance measures whether the *retrieved documents* are relevant to the query — not whether the answer uses them.
- B is wrong because: Coherence measures logical consistency of the response text, not its connection to source docs.
- D is wrong because: Fluency measures grammatical quality of the response text.

**Q8: B) Use `PromptFlowClient.run()` with a dataset JSONL file**
- The correct pattern for batch evaluation of a Prompt Flow is to run it with a dataset referencing column mappings. The `PromptFlowClient.run()` method handles this.
- A is wrong because: manual testing in the portal doesn't scale to 500 cases.
- C is wrong because: you don't need to re-deploy as a batch endpoint; a run can execute the flow on a dataset directly. BatchDeployment is for AML ML pipelines, not Prompt Flow evaluation.
- D is wrong because: no need to export to AML; Foundry handles this natively.

**Q9: B) Safety evaluation during fine-tuning (adversarial attack testing)**
- Azure AI Foundry automatically runs safety evaluation during fine-tuning by provisioning a separate fine-tuned GPT-4o model to orchestrate adversarial attacks against your model.
- A is wrong because: Azure Content Safety is a separate resource for general content moderation, not fine-tuning safety eval.
- C is wrong because: Responsible AI dashboard is for traditional ML models (fairness, explainability) — not LLM safety post-fine-tuning.
- D is wrong because: benchmark comparison is for choosing between models in the catalog, not evaluating safety after fine-tuning.

**Q10: C) Fine-tuning + RAG**
- Fine-tune on the 1,000 Q&A examples → permanently encodes the formal tone + regulatory disclaimer pattern
- RAG for monthly-changing regulations → keeps knowledge current + enables citation of specific documents
- A is wrong because: fine-tuning on regulation documents makes the model's knowledge stale next month. Fine-tuning cannot keep up with monthly updates.
- B is wrong because: RAG alone doesn't teach the consistent formal tone and automatic disclaimer inclusion behavior — that requires fine-tuning.
- D is wrong because: adding all regulations to the system prompt hits context window limits, is expensive per query, and doesn't scale to a large regulation library. Also doesn't address tone/disclaimer.
