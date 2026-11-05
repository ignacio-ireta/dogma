# Day 4 — Prompt Flow Answers
**Date:** March 3, 2026

---

## Scenario Question Answers

### Q1 — ANSWER: B) Prompt variants within a single flow

**Why B is correct:** Variants are specifically designed for this use case — comparing multiple prompt versions in a single batch run, with side-by-side metrics.

**Why A is wrong:** Running two separate flows loses the native comparison UI and variant tracking. You'd have to manually compare run outputs.

**Why C is wrong:** Evaluation flows GRADE outputs — they don't test different prompts. You'd use variants first, THEN optionally run an evaluation flow on the results.

**Why D is wrong:** MLflow tracks training experiments, not prompt variant comparisons. Wrong tool entirely.

---

### Q2 — ANSWER: B) The `PF_DISABLE_TRACING` environment variable is set to `"true"`

**Why B is correct:** `PF_DISABLE_TRACING` defaults to `"true"` (tracing disabled). You must explicitly set it to `"false"`. This is a classic gotcha — tracing "doesn't work" because it was never turned on.

**Why A is wrong:** Tracing works on all node types (LLM, Python, Tool, Prompt) — not just Python nodes.

**Why C is wrong:** Tracing works in both the portal AND the SDK. The SDK captures traces when the env var is set.

**Why D is wrong:** Evaluation flows do support tracing — you need it to debug eval node inputs/outputs.

---

### Q3 — ANSWER: C) Groundedness

**Why C is correct:** Groundedness specifically measures: "Is the answer factually supported by the retrieved context/documents?" — exactly what RAG faithfulness requires.

**Why A is wrong:** Fluency measures grammatical correctness — doesn't check factual grounding.

**Why B is wrong:** Similarity measures how close the answer is to a *ground truth answer* (token overlap) — requires you to already know the right answer. Doesn't measure if it's supported by retrieved docs.

**Why D is wrong:** Coherence measures logical structure and readability — not factual grounding.

---

### Q4 — ANSWER: B) `log_metric("accuracy", value)` from `promptflow`

**Why B is correct:** Evaluation flows use `log_metric` imported from the `promptflow` package. This sends the metric to the Foundry evaluation dashboard and associates it with the run.

**Why A is wrong:** `mlflow.log_metric()` is for Azure ML training jobs (registered via MLflow tracking). In an evaluation flow, this does NOT appear in the Foundry evaluation dashboard.

**Why C is wrong:** `pf.log_metric()` doesn't exist — `pf` is the PFClient for managing runs, not logging metrics from inside a node.

**Why D is wrong:** `azure.ai.evaluation.log()` is not a real API in this context.

---

### Q5 — ANSWER: B) `pf.run(flow="./my_flow", data="./test.jsonl")`

**Why B is correct:** `PFClient.run()` is the primary SDK method for executing a Prompt Flow batch run — works in CI/CD pipelines, notebooks, or scripts.

**Why A is wrong:** `ml_client.jobs.create_or_update()` is for Azure ML training/pipeline jobs — not prompt flow runs.

**Why C is wrong:** `pf.stream()` is for streaming the output of a run that's already been initiated — not for starting one.

**Why D is wrong:** `ml_client.flows.begin_run()` doesn't exist as an SDK method.

---

### Q6 — ANSWER: B) Tool Node → Prompt Node → LLM Node

**Why B is correct:** The logical flow is:
1. **Tool Node** (Azure AI Search) retrieves relevant documents
2. **Prompt Node** formats the documents + question into a Jinja2 prompt template
3. **LLM Node** sends the rendered prompt to GPT-4o and returns the answer

**Why A is wrong:** LLM Node first makes no sense — you don't have documents yet to include in the prompt.

**Why C is wrong:** Python Node first could work for pre-processing, but doesn't search. And putting LLM before retrieval is wrong.

**Why D is wrong:** Tool Node → LLM Node → Python Node skips the prompt formatting step — the LLM would receive unformatted search results.

---

### Q7 — ANSWER: B) Model Catalog Benchmarks leaderboard

**Why B is correct:** The Microsoft Foundry Model Catalog includes a **Benchmarks** tab where you can compare models across standard datasets (MMLU, HumanEval, HellaSwag, etc.) by quality, latency, and cost — all BEFORE deploying.

**Why A is wrong:** Azure Monitor shows operational metrics of deployed resources — nothing about model quality before deployment.

**Why C is wrong:** Prompt Flow evaluation runs require you to DEPLOY or test the models first — you can't compare models you haven't deployed yet.

**Why D is wrong:** MLflow experiment comparison tracks your own training runs — not catalog model benchmarks.

---

### Q8 — ANSWER: A) The template variable name must match the mapping key; fix by updating one side

**Why A is correct:** In Prompt Flow, Jinja2 template variables (`{{question}}`) must match what the column mapping provides. If data has `"user_query"` but template expects `"question"`, the variable renders as empty/null. Fix: either change `{{question}}` to `{{user_query}}` in the template, OR update the column mapping to `"question": "${data.user_query}"`.

**Why B is wrong:** Jinja2 absolutely supports variable interpolation — it's the core templating mechanism.

**Why C is wrong:** Empty outputs from a variable name mismatch aren't an LLM temperature issue. Temperature affects response creativity, not whether variables are populated.

**Why D is wrong:** Adding a Python node is unnecessarily complex when a simple name alignment fix solves it.

---

### Q9 — ANSWER: B) Connect the Foundry project to an Azure Storage Account and enable run history logging

**Why B is correct:** Foundry projects store run history (inputs, outputs, traces) in Azure Storage. Configuring an appropriate storage account with a 90-day retention policy ensures auditability and reproducibility.

**Why A is wrong:** Setting `PF_DISABLE_TRACING=false` captures node-level debug info for a session, but doesn't guarantee 90-day storage retention or organizational auditability.

**Why C is wrong:** MLflow tracking is for Azure ML training experiments — not Prompt Flow runs in Foundry.

**Why D is wrong:** Manual CSV exports are not auditable/automated, prone to human error, and don't scale.

---

### Q10 — ANSWER: C) One endpoint with two deployments + traffic allocation

**Why C is correct:** This is exactly the canary/A-B testing pattern for managed online endpoints. You create:
- One endpoint
- Two deployments (current_flow and new_flow)
- Traffic allocation: 50%/50%
Real production traffic is split automatically. This is the same pattern as model A/B testing.

**Why A is wrong:** Prompt variants + evaluation flow is for OFFLINE testing with your own test data — not splitting real user traffic.

**Why B is wrong:** Two separate endpoints + APIM works but is unnecessarily complex and expensive. The built-in traffic allocation feature of managed online endpoints handles this natively.

**Why D is wrong:** Batch endpoints process data asynchronously in bulk — they don't serve real-time user interactions.

---

## Spaced Retrieval Answers (from Days 1-3)

**Q1** (deployment type): **Standard deployment** (Foundry resource) — gives Microsoft SLA + support. Serverless = pay-as-you-go, no SLA. Managed compute = you manage the infra.

**Q2** (when to use what):
- Prompt engineering: existing model works, just needs better instructions (e.g., "format output as JSON")
- RAG: model needs YOUR data without retraining (e.g., internal knowledge base Q&A)
- Fine-tuning: model needs to learn a NEW style/task that prompts can't fix (e.g., domain-specific tone)

**Q3** (batch endpoint classes):
- `BatchEndpoint` — defines the endpoint
- `BatchDeployment` — defines the deployment under it
- Invoke: `ml_client.batch_endpoints.invoke(endpoint_name, input=...)` — returns a job

**Q4** (mlflow.autolog): Captures params, metrics, model artifacts automatically for supported frameworks (sklearn, XGBoost, PyTorch, etc.). Fails to log correct signature when the model has custom input formats or when called before the first `fit()`.

**Q5** (Serverless API vs Azure OpenAI): NOT the same.
- Azure OpenAI = direct Microsoft service, your own quotas, full enterprise features, private endpoint option
- Serverless API from Catalog = pay-as-you-go, billed per token to your subscription, but the model is hosted by Microsoft/provider with shared infrastructure. No private endpoint in most cases.
