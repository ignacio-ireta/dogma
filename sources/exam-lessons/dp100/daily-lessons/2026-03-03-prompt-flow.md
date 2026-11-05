# Day 4 — Domain 4, Part 2: Prompt Flow Deep Dive
**Date:** March 3, 2026 (Tuesday)
**Exam:** DP-100 — March 7, 2026 (4 days out)
**Today's Focus:** Prompt Flow — the topic that burned you last time. We're going deep, systematic, and SDK-level. This is the gap. Today we close it. 🎯

---

## 🧠 SPACED RETRIEVAL (5 min)
*From Days 1-3 — don't look. Write your answers first, then check.*

**Q1.** You need to deploy a model from the Azure/Microsoft Foundry Model Catalog to a production endpoint with a guaranteed SLA and Microsoft support. Which of the three deployment types do you choose?

**Q2.** What's the decision rule: when do you use **prompt engineering** vs **RAG** vs **fine-tuning**? Give one real scenario for each.

**Q3.** In Azure ML SDK v2, what's the Python class used to define a **batch endpoint**? And what's the class used to invoke it (trigger a batch job)?

**Q4.** What does `mlflow.autolog()` capture automatically, and what's one scenario where it won't log the model signature correctly?

**Q5.** A colleague says: "I deployed GPT-4o from the Model Catalog with a **Serverless API** endpoint — it's the same as using Azure OpenAI directly." Are they correct? What's the key difference?

*(Answers in yesterday's notes + Days 1-2. Struggle first — getting it wrong is productive.)*

---

## 📊 CONCEPT MAP: Prompt Flow as a SYSTEM (15 min)

### THE BIG PICTURE — What Is Prompt Flow?

Prompt Flow is Microsoft's **LLM application development framework**. It lets you:
1. **Build** chains of LLM calls + logic + tools as a directed graph (DAG)
2. **Evaluate** those chains systematically across many inputs
3. **Deploy** the tested flow as a managed API endpoint

> **TERMINOLOGY ALERT (CRITICAL for exam):**
> - **Microsoft Foundry portal** = the new portal (formerly "Azure AI Studio" → "Azure AI Foundry" → "Microsoft Foundry"). URL: `ai.azure.com` or `foundry.azure.com`
> - **Microsoft Foundry (classic)** = the hub-based experience still in use for exam questions
> - Prompt Flow is available as **open-source** (`promptflow` pip package) AND embedded in both AML Studio and Microsoft Foundry portal
> - The exam uses both — know how to do it in **the portal GUI** AND via the **SDK**

---

### PROMPT FLOW ANATOMY — The 4 Node Types

A flow is a **Directed Acyclic Graph (DAG)** of nodes. Every node is one of:

```
Flow Input
    │
    ▼
┌─────────────┐
│ LLM Node    │  ← calls an LLM (GPT-4o, Phi, etc.) with a prompt template
└─────────────┘
    │
    ▼
┌─────────────┐
│ Python Node │  ← runs arbitrary Python (pre/post-processing, parsing)
└─────────────┘
    │
    ▼
┌─────────────┐
│ Prompt Node │  ← renders a Jinja2 template (no LLM call — just template)
└─────────────┘
    │
    ▼
┌─────────────┐
│ Tool Node   │  ← built-in tools: Azure AI Search, Bing, Content Safety
└─────────────┘
    │
    ▼
Flow Output
```

**⚠️ EXAM TRAP:** A "Prompt Node" is NOT the same as an "LLM Node."
- **Prompt Node** = just renders a Jinja2 template string (no API call)
- **LLM Node** = sends that rendered prompt TO the model and gets a response

---

### FLOW TYPES — Know All Three

| Flow Type | Purpose | When to Use |
|-----------|---------|-------------|
| **Standard flow** | Build your actual LLM app logic | Default for chat, Q&A, analysis apps |
| **Chat flow** | Standard flow + built-in chat UI | When you need conversation history |
| **Evaluation flow** | Takes outputs of another flow + grades them | Quality assessment, batch evaluation |

**⚠️ EXAM TRAP:** Evaluation flows have a special `@aggregation_node` concept — they run once per item, then aggregate all results. Confusing this with a standard flow is a common mistake.

---

### PROMPT VARIANTS — The Exam's Favorite Prompt Flow Topic

**Variants** let you define multiple versions of a **node** (usually a prompt template or LLM settings) and compare them in a single run.

**How it works:**
1. You create a base node (e.g., an LLM node with a system prompt)
2. You add **variants** — alternative prompt texts or model params (temperature, etc.)
3. You run a **batch run** with all variants enabled
4. Foundry shows you metrics side by side

**In the portal:** Node → "+" → "Add Variant" → edit the template or params

**In the SDK:**
```python
from promptflow import PFClient

pf = PFClient()

# Run with all variants activated
run = pf.run(
    flow="./my_flow",
    data="./test_data.jsonl",  # your eval dataset
    column_mapping={
        "question": "${data.question}",
    },
    # variants auto-detected from flow.dag.yaml
)

# Check results per variant
pf.get_metrics(run.name)
```

In `flow.dag.yaml`, variants look like:
```yaml
nodes:
- name: chat_node
  type: llm
  source:
    type: code
    path: chat_node.jinja2
  inputs:
    temperature: 0.7
  activate:
    when: True
  use_variants: true  # ← enables variant mode
  variants:
    variant_0:
      inputs:
        temperature: 0.3
        prompt: "You are a helpful assistant. Answer concisely."
    variant_1:
      inputs:
        temperature: 0.7
        prompt: "You are a friendly assistant. Give detailed explanations."
```

---

### PROMPT TEMPLATES — Jinja2 Syntax

Prompt Flow templates use **Jinja2**:

```jinja
# chat_node.jinja2
system:
You are a helpful assistant. Today's date is {{current_date}}.

{% for item in chat_history %}
user:
{{item.inputs.question}}
assistant:
{{item.outputs.answer}}
{% endfor %}

user:
{{question}}
```

Key Jinja2 features you'll see on the exam:
- `{{variable}}` — interpolate a variable
- `{% for item in list %}` / `{% endfor %}` — loop
- `{% if condition %}` / `{% endif %}` — conditional

**⚠️ EXAM TRAP:** The system message, user turns, and assistant turns are declared with labels (`system:`, `user:`, `assistant:`). Without the correct labels, the LLM gets confused. The exam may show a broken template and ask what's wrong.

---

### TRACING — Debug Your Flow

Tracing captures **every input/output and duration** for each node in a flow run. It's how you debug what went wrong.

**Enabling tracing (code):**
```python
import os
os.environ["PF_DISABLE_TRACING"] = "false"  # tracing is OFF by default!

# Or when running:
from promptflow import PFClient
pf = PFClient()
run = pf.run(
    flow="./my_flow",
    data="./test_data.jsonl",
    # tracing auto-captured when env var set
)

# View trace
pf.stream(run)  # streams output as it runs
```

**In the portal:** Run a flow → "Trace" tab → click any node to see input/output/latency

**⚠️ EXAM TRAP:** `PF_DISABLE_TRACING` is `true` by default — you must explicitly set it to `"false"` to enable. The exam may show a debug scenario where tracing is mysteriously empty.

---

### EVALUATION FLOWS — Grading Your LLM App

This is where Prompt Flow gets powerful (and where the exam goes deep):

```
Your test data (question + ground_truth)
         │
         ▼
   [Your Flow] ← standard/chat flow
         │
         ▼
   (generated answers)
         │
         ▼
 [Evaluation Flow] ← takes (question, answer, ground_truth) as input
         │
         ▼
   Score per item + aggregate metrics
```

**Built-in evaluators in Microsoft Foundry:**
| Evaluator | What It Measures | Input Required |
|-----------|-----------------|----------------|
| **Groundedness** | Is the answer based on the context? | question, answer, context |
| **Relevance** | Is the answer relevant to the question? | question, answer, context |
| **Coherence** | Is the answer logically structured? | question, answer |
| **Fluency** | Is the answer grammatically correct? | question, answer |
| **Similarity** | How close to ground truth? | question, answer, ground_truth |
| **F1 Score** | Token-level overlap with ground truth | answer, ground_truth |

**SDK: Running an evaluation:**
```python
from promptflow import PFClient

pf = PFClient()

# First, run your actual flow to get outputs
base_run = pf.run(
    flow="./my_chat_flow",
    data="./eval_data.jsonl",
    column_mapping={
        "question": "${data.question}",
        "chat_history": "${data.chat_history}",
    }
)

# Then run an evaluation flow AGAINST those outputs
eval_run = pf.run(
    flow="./groundedness_eval_flow",   # evaluation flow
    run=base_run,                      # reference your base run!
    column_mapping={
        "question": "${run.inputs.question}",
        "answer": "${run.outputs.answer}",
        "context": "${run.outputs.context}",
    }
)

metrics = pf.get_metrics(eval_run.name)
print(metrics)
# e.g. {"groundedness": 4.2, "relevance": 3.8}
```

**In an evaluation flow node (Python):**
```python
from promptflow import tool, log_metric

@tool
def calculate_score(grades: list, variant_ids: list) -> dict:
    for i, grade in enumerate(grades):
        log_metric("accuracy", grade, variant_id=variant_ids[i])
    return {"mean_score": sum(grades) / len(grades)}
```

**⚠️ EXAM TRAP:** `log_metric` is for **evaluation flows only**. Don't confuse with `mlflow.log_metric()` (used in training jobs). They are DIFFERENT.

---

### DEPLOYING A PROMPT FLOW

Once tested, you can deploy a flow as a **managed online endpoint**:

**Portal path:** Your Flow → "Deploy" button → Choose endpoint name + compute

**SDK:**
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

ml_client = MLClient(...)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="my-flow-endpoint",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Deploy the flow
# (in Foundry portal: simpler UI-based deployment)
```

**⚠️ EXAM TRAP:** Prompt flow deployments use the SAME `ManagedOnlineEndpoint` infrastructure as regular model deployments. The exam may ask if you can deploy a prompt flow — YES, you can, via the Foundry portal or AML SDK.

---

### MANUAL EVALUATION vs AUTOMATED EVALUATION

| | Manual Evaluation | Automated Evaluation |
|---|---|---|
| **How** | Human raters score outputs in Foundry UI | Evaluation flow with LLM-as-judge or metrics |
| **When** | Small datasets, subjective quality, initial testing | Large datasets, CI/CD pipelines, regression testing |
| **Portal** | "Evaluation" → "Manual evaluation" tab | "Evaluation" → run an evaluation flow |
| **Cost** | Human time | LLM API calls (for LLM-as-judge) |
| **Scale** | Limited (100s of samples) | Unlimited (1000s of samples) |

---

### PROMPT FLOW vs DIRECT API — When to Use Which

**⚠️ HIGH-PROBABILITY EXAM SCENARIO:**

| Use Prompt Flow when... | Use Direct API when... |
|------------------------|----------------------|
| You need to chain multiple LLM calls + tools | Single, simple LLM call |
| You want built-in evaluation & variants | Latency-critical, no extra hops |
| You're building a structured, reproducible pipeline | Quick prototype / scripting |
| You need to compare prompt variants systematically | Full control over the HTTP request |
| Team collaboration on prompts | Solo developer, no governance needed |

---

## 📝 SCENARIO QUESTIONS (20 min)
*Questions only. Write your answers, then check `2026-03-03-prompt-flow-answers.md`*

**Q1.** You are building a customer support chatbot using Prompt Flow. You want to test two different system prompts — one formal, one friendly — across 500 sample questions to see which produces higher customer satisfaction scores. Which Prompt Flow feature do you use?

A) Batch runs with two separate flows  
B) Prompt variants within a single flow  
C) Two separate evaluation flows  
D) MLflow experiment tracking with two runs  

---

**Q2.** Your Prompt Flow tracing is enabled but the Trace tab shows empty results for all nodes. What is the MOST likely cause?

A) The flow has no Python nodes  
B) The `PF_DISABLE_TRACING` environment variable is set to `"true"`  
C) Tracing only works in the portal, not the SDK  
D) Evaluation flows don't support tracing  

---

**Q3.** You need to measure whether your RAG chatbot's answers are factually supported by the retrieved documents. Which built-in evaluator metric should you use?

A) Fluency  
B) Similarity  
C) Groundedness  
D) Coherence  

---

**Q4.** In an evaluation flow's Python tool node, you want to report an aggregate accuracy metric that will appear in the Foundry evaluation dashboard. Which function do you call?

A) `mlflow.log_metric("accuracy", value)`  
B) `log_metric("accuracy", value)` from `promptflow`  
C) `pf.log_metric("accuracy", value)`  
D) `azure.ai.evaluation.log("accuracy", value)`  

---

**Q5.** You have a Prompt Flow that works perfectly in the Foundry portal. You now want to run it as part of an automated CI/CD pipeline via Python. Which SDK call initiates a batch run of the flow against a test dataset?

A) `ml_client.jobs.create_or_update(flow_job)`  
B) `pf.run(flow="./my_flow", data="./test.jsonl")`  
C) `pf.stream(flow="./my_flow", inputs=data)`  
D) `ml_client.flows.begin_run(flow_name, data)`  

---

**Q6.** You are designing a Prompt Flow that must: (1) call Azure AI Search to retrieve relevant documents, (2) format the documents into a prompt, and (3) call GPT-4o with that prompt. What is the CORRECT sequence of node types?

A) LLM Node → Python Node → Tool Node  
B) Tool Node → Prompt Node → LLM Node  
C) Python Node → LLM Node → Tool Node  
D) Tool Node → LLM Node → Python Node  

---

**Q7.** *(From Day 3 — interleave)* You're comparing GPT-4o and Phi-4 for a customer summarization task. Both are available in Microsoft Foundry. What feature lets you compare their performance benchmarks BEFORE deploying either?

A) Azure Monitor metrics  
B) Model Catalog Benchmarks leaderboard  
C) Prompt Flow evaluation runs  
D) MLflow experiment comparison  

---

**Q8.** A Jinja2 prompt template contains `{{question}}` but the flow's column mapping provides `"user_query"` as the data column name. The flow runs but produces empty outputs. What's wrong and how do you fix it?

A) The template variable name must match the column mapping key; change template to `{{user_query}}` OR update the column mapping to map `"question"` to `"${data.user_query}"`  
B) Jinja2 doesn't support variable interpolation in LLM nodes  
C) Empty outputs mean the LLM returned a null — retry with a higher temperature  
D) Add a Python node before the LLM node to rename the variable  

---

**Q9.** Your organization requires that prompt flow evaluation runs be auditable and reproducible, with all inputs/outputs stored for 90 days. What should you configure?

A) Set `PF_DISABLE_TRACING=false` only  
B) Connect the Foundry project to an Azure Storage Account and enable run history logging  
C) Use MLflow tracking server with 90-day retention policy  
D) Export evaluation results manually to a CSV after each run  

---

**Q10.** *(Case study)* A data scientist says: "I built a chat flow in Prompt Flow. My manager wants to A/B test it in production — 50% of real users on the current version, 50% on the new version with a better system prompt. How do I do this?"

Which combination of features achieves this?

A) Prompt variants + evaluation flow  
B) Deploy two separate managed online endpoints + Azure API Management with traffic splitting  
C) One endpoint with two deployments + traffic allocation (like a canary deployment)  
D) Batch endpoints with two different flows  

---

## 🛠️ HANDS-ON CHALLENGE (10 min)

**Task:** Create and run a minimal Prompt Flow locally using the SDK.

If you have the `promptflow` package installed (`pip install promptflow`), do this in a Python script or notebook:

```python
# Install first: pip install promptflow promptflow-tools

from promptflow import PFClient
pf = PFClient()

# 1. List available flows in current directory
# (assumes you've cloned: https://github.com/microsoft/promptflow/tree/main/examples)

# 2. Run the built-in "basic" example flow
run = pf.run(
    flow="path/to/basic_flow",  # or create your own minimal flow
    data="path/to/data.jsonl",
)

# 3. View outputs
details = pf.get_details(run.name)
print(details)

# 4. Check metrics (if an eval flow was attached)
metrics = pf.get_metrics(run.name)
print(metrics)
```

**Alternative (portal):** If you have a Foundry project:
1. Go to `ai.azure.com` → your project → **Prompt flow** (left nav)
2. Click **"+ Create"** → choose **"Chat flow"** template
3. Explore the default nodes — identify the LLM node and its Jinja2 template
4. Click **"Add variant"** on the LLM node and change the temperature
5. Run the flow with the sample data

**What to observe:** How the flow DAG is structured, how variants appear in run results.

---

## ✅ KEY TAKEAWAYS (Memorize These)

1. **4 node types:** LLM Node (calls model), Prompt Node (renders Jinja2 only), Python Node (code), Tool Node (built-ins like AI Search)

2. **Variants = multiple prompt templates or LLM configs tested in a single batch run.** Use them to compare system prompts, temperatures, or models systematically.

3. **Tracing is OFF by default.** Must set `PF_DISABLE_TRACING="false"` to capture node-level debug info.

4. **Evaluation flows use `log_metric()` from `promptflow`, NOT `mlflow.log_metric()`.** This is a very common exam trap.

5. **Evaluation flow structure:** Your standard flow runs first → outputs feed into evaluation flow → evaluation flow grades each output → metrics aggregate. Two separate flows, linked by `run=base_run`.

6. **Manual evaluation** = humans grade in Foundry portal (small scale). **Automated evaluation** = evaluation flow with LLM-as-judge or token metrics (large scale, CI/CD).

7. **Prompt Flow can be deployed as a managed online endpoint** — same infrastructure as model deployments. Portal has a "Deploy" button; SDK uses `ManagedOnlineEndpoint`.

---

## 🃏 Anki Card Suggestions

**Card 1:**
Front: What Python call initiates a Prompt Flow batch run against a JSONL dataset?
Back: `pf.run(flow="./my_flow", data="./test_data.jsonl", column_mapping={...})`

**Card 2:**
Front: What environment variable must be set (and to what value) to enable Prompt Flow tracing?
Back: `PF_DISABLE_TRACING = "false"` (default is `"true"` = tracing off)

**Card 3:**
Front: In an evaluation flow Python node, what function logs a metric to the Foundry dashboard?
Back: `log_metric("metric_name", value)` imported from `promptflow` — NOT mlflow

**Card 4:**
Front: Prompt Node vs LLM Node — what's the difference?
Back: Prompt Node = renders Jinja2 template (no API call). LLM Node = sends rendered prompt to model and returns response.

**Card 5:**
Front: Name 3 built-in evaluation metrics for RAG pipelines and what each measures.
Back: Groundedness (answer supported by context), Relevance (answer relevant to question), Coherence (answer logically structured)

**Card 6:**
Front: How do you link an evaluation flow run to the standard flow run it evaluates (SDK)?
Back: Pass `run=base_run` parameter when calling `pf.run()` for the evaluation flow

---

## 🎯 Anxiety Calibration

You failed this exam once. That hurt. But here's what's true:

- **You already own Domains 1-3.** That's 70-75% of the exam. You walked in prepared there.
- **Domain 4 was restructured to 25-30% AFTER you studied.** You didn't fail — you got blindsided by a moving target.
- **You've been closing this gap systematically** for 4 days now. Model Catalog yesterday, Prompt Flow today, RAG tomorrow, Fine-Tuning Thursday.
- **By Friday you'll have covered all of Domain 4.** No blind spots.

You're not behind. You're exactly where you need to be. 🏔️

---

*Tomorrow: Day 5 — RAG (Retrieval Augmented Generation). The full pipeline from chunking to evaluation.*
*Answers to today's questions: `2026-03-03-prompt-flow-answers.md`*
