# Day 5 — Answer Key: RAG
**Date:** March 4, 2026
**Read ONLY after attempting all questions.**

---

## SPACED RETRIEVAL ANSWERS (from Days 1-4)

**Q1.** Prompt Node vs LLM Node:
- **Prompt Node** = renders a Jinja2 template string only (NO LLM call, no API request)
- **LLM Node** = sends the prompt to a deployed model and receives a response
- Exam trap: "Add a Prompt Node to make an LLM call" is WRONG — you need an LLM Node

**Q2.** Three Prompt Flow types:
- **Standard Flow** = general-purpose DAG — most flexible, for RAG too
- **Chat Flow** = adds conversation history management (multi-turn chat)
- **Evaluation Flow** = takes a flow's output + ground truth → computes metrics
- RAG is typically built with a **Standard Flow** (or Chat Flow for multi-turn)

**Q3.** Testing different system prompts = **variants** on an LLM node. In Python SDK: define multiple `PromptVariant` objects or use the `promptflow` library's variant syntax in YAML. In the portal: "Add variant" button on an LLM node.

**Q4.** Three Model Catalog deployment types:
- **Managed online endpoint** — you provision the compute, you control it
- **Serverless API** — pay-per-token, no capacity reservation, Microsoft hosts
- **Azure AI Services** — via Azure OpenAI or other AI Services
- Pay-as-you-go without pre-provisioning = **Serverless API**

**Q5.** Registering an MLflow model: Both of these work:
- `ml_client.models.create_or_update(Model(name="...", path="runs:/RUN_ID/model", type=AssetTypes.MLFLOW_MODEL))`
- `mlflow.register_model(f"runs:/{run_id}/model", "my-model")`
- `mlflow_model` source type = the model was logged via MLflow's model format (has MLmodel file, conda.yaml, etc.)

---

## SCENARIO QUESTION ANSWERS

**Q1 — Answer: B**
- ✅ B is correct: indexer with continuous schedule keeps index current; `text-embedding-3-small` is a solid embedding choice; HNSW is the correct default vector algorithm
- ❌ A is wrong: Monthly schedule means stale data between updates; `exhaustiveKNN` is too slow for production
- ❌ C is wrong: Cosmos DB is not Azure AI Search — you'd lose the vector search capability
- ❌ D is wrong: Azure ML Feature Store is for ML feature engineering, not document embeddings

---

**Q2 — Answer: B**
- ✅ B is correct: Groundedness measures whether the answer is supported by retrieved context. Low groundedness = LLM is making things up. Fix: stronger system prompt grounding instruction.
- ❌ A is wrong: Relevance measures the retrieval quality (is context relevant to query?), not hallucination
- ❌ C is wrong: Coherence is about answer structure, not factual accuracy
- ❌ D is wrong: Fluency is about grammar, not factual grounding; embedding dimensions don't affect post-generation quality

---

**Q3 — Drag-and-Drop Answer:**
1 → B (Fixed-size = splits every N tokens with overlap)
2 → C (Sentence/paragraph = natural language boundaries, variable sizes)
3 → A (Semantic = topic shifts via embedding model, highest quality, highest cost)

---

**Q4 — Answer: C**
- ✅ C is correct: Hybrid search combines keyword matching (catches exact abbreviations like "VPN") with semantic vector search (catches related concepts). Semantic ranker re-ranks for even better precision.
- ❌ A is wrong: Pure vector search is great for semantic similarity but may miss exact abbreviation matches if they're not close in embedding space to the full term
- ❌ B is wrong: Pure keyword search misses paraphrases and semantic meaning
- ❌ D is wrong: Chunk overlap addresses context loss at chunk boundaries, not abbreviation matching

---

**Q5 — Answer: B**
- ✅ B is correct: RAG with Azure AI Search is the right pattern for:
  - Frequently updating content (re-index vs. retrain)
  - Grounding requirement (strict system prompt)
  - Scale (Azure AI Search handles this)
- ❌ A is wrong: Fine-tuning bakes knowledge into weights — monthly retraining is expensive and the knowledge is always stale between retrains
- ❌ C is wrong: Azure OpenAI Assistants API with file uploads works for individual users but not 10,000 concurrent users with shared knowledge
- ❌ D is wrong: Building a custom embedding model is unnecessary complexity; pretrained models work fine

---

**Q6 — Answer: B**
- ✅ B is correct: Groundedness evaluator needs the **answer** AND the **context** that was retrieved. It checks: "Is this answer supported by this context?" No context → can't check groundedness.
- ❌ A is wrong: `ground_truth` is used for accuracy metrics (like F1 or similarity to expected answer), not groundedness
- ❌ C is wrong: Embedding vectors are used during retrieval, not evaluation
- ❌ D is wrong: Source URLs help with citation tracking but aren't required for groundedness scoring

---

**Q7 — Answer: B**
- ✅ B is correct: Variants in Prompt Flow can be defined on ANY node, including Vector DB Lookup nodes. You define variant 1 with top-k=3, variant 2 with top-k=5, variant 3 with top-k=10 — then run them in a comparison evaluation.
- ❌ A is wrong: Creating three separate flows is possible but wasteful — variants are the designed mechanism
- ❌ C is wrong: Random selection doesn't enable systematic comparison
- ❌ D is wrong: Evaluation flows evaluate existing flows — they don't configure retrieval variants

---

**Q8 — Answer: B**
- ✅ B is correct: HNSW (Hierarchical Navigable Small World) is the default ANN algorithm in Azure AI Search. It builds a graph of vector neighbors and navigates it efficiently at query time. Best trade-off of speed vs. accuracy for production use.
- ❌ A is wrong: ExhaustiveKNN checks ALL vectors (brute force) — 100% accurate but O(n) slow. Use only for small indexes or quality validation.
- ❌ C is wrong: BM25 is the keyword ranking algorithm, not a vector search algorithm
- ❌ D is wrong: RRF (Reciprocal Rank Fusion) is the fusion algorithm that combines keyword + vector scores in hybrid search — it's not a search algorithm itself

---

**Q9 — Answer: D (Both B and C)**
- ✅ D is correct: Both approaches work for registering an MLflow model:
  - B uses Azure ML SDK v2 (`ml_client.models.create_or_update`)
  - C uses the MLflow client directly (`mlflow.register_model`)
  - If connected to the same workspace, both register to the Azure ML Model Registry
- ❌ A is wrong: `az ml model register` syntax is wrong — it would be `az ml model create`, and the path format is different

---

**Q10 — Answer: C**
- ✅ C is correct: Groundedness is 4.8/5 (excellent) → the LLM is staying grounded in what it was given. But Relevance is 2.1/5 (poor) → the **retrieved chunks are not relevant** to the user's query. The LLM faithfully answers from irrelevant chunks = unhelpful but technically "grounded." Fix: improve retrieval (chunking strategy, embedding model, search configuration, or top-k setting).
- ❌ A is wrong: High groundedness means the system prompt is working well — loosening it would make things worse
- ❌ B is wrong: Embedding mismatch would cause catastrophically bad results across ALL metrics, not just relevance
- ❌ D is wrong: Chunk size affects whether context fits in the prompt, but the root cause here is retrieval returning the wrong chunks entirely

---

## HANDS-ON CHALLENGE DISCUSSION

**Why "free shipping for premium members" scores high for "How long does delivery take?"**

Both documents are about shipping. The embedding model encodes them in a similar region of vector space because they share vocabulary (shipping, delivery, orders). This is semantic overlap — the retrieval returns "related" documents, not necessarily "answer-containing" documents.

**Real-world fix for this:**
1. **Better chunking** — ensure each chunk has a clear, singular topic
2. **Semantic ranker** — adds a cross-encoder that re-ranks results for precise question-answer alignment
3. **Metadata filtering** — tag chunks by category, filter at query time
4. **Query rewriting** — before retrieval, have an LLM rewrite the user query to be more specific

---

*These are the RAG failure modes. Tomorrow: Fine-tuning. Day 5 complete. 🎯*
