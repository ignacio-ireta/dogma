# Day 6 — Domain 4, Part 4: Fine-Tuning + Full Integration
**Date:** March 5, 2026 (Thursday)
**Exam:** DP-100 — March 7, 2026 (2 days out)
**Today's Focus:** Fine-tuning (data prep, model selection, config, evaluation) + Responsible AI for LLMs + **The Big Picture: All of Domain 4 connected**

> 💙 *Two days out. You know Domains 1-3. You've systematically rebuilt Domain 4 from scratch over the last 4 days: Model Catalog → Prompt Flow → RAG → Fine-Tuning. Today you connect the whole picture and fill the last gaps. This time is different.*

---

## 🧠 SPACED RETRIEVAL (5 min)
*From Days 3–5 — write answers before checking. Wrong answers are signal, not failure.*

**Q1.** From Day 5 (RAG): Name the **three chunking strategies** and when you'd choose each. What's the key tradeoff between chunk size and retrieval quality?

**Q2.** From Day 5 (RAG): What are the **four evaluation metrics** for a RAG solution in Azure AI Foundry? Give a one-sentence definition of each.

**Q3.** From Day 4 (Prompt Flow): What's the difference between a **Prompt variant** and a **Prompt template**? How do you compare variants in the SDK?

**Q4.** From Day 4 (Prompt Flow): When would you use `PromptFlow` tracing? Name **two things** tracing captures that you can't see from just reading evaluation scores.

**Q5.** From Day 3 (Model Catalog): What is a **Managed Online Endpoint** in the context of the Model Catalog? How is it different from a **Serverless API** endpoint in terms of cost model and control?

*(Struggle first. 5 minutes max. Then move on.)*

---

## 📊 CONCEPT MAP: Fine-Tuning as a SYSTEM (15 min)

### THE ONE-SENTENCE DEFINITION

> **Fine-tuning = updating model weights** on your labeled data so the model *permanently* changes its behavior — vs. RAG (retrieving facts at runtime) and prompt engineering (shaping behavior at inference time without changing weights).

This is the core exam distinction. Burn it in.

---

### THE DECISION TREE: Prompt Engineering vs RAG vs Fine-Tuning

```
                    ┌─────────────────────────────────────────────────────┐
                    │ WHAT'S THE PROBLEM?                                 │
                    └─────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼──────────────────────────────┐
              │                           │                              │
     "Needs real-world facts"    "Needs specific style,       "Base model works fine,
     that change over time        format, or behavior          just needs better context
     (docs, policies, FAQs)"      baked in permanently"        for each query"
              │                           │                              │
              ▼                           ▼                              ▼
            RAG                    FINE-TUNING                PROMPT ENGINEERING
  (retrieval at query time)    (weight update, one-time)   (better system prompt/few-shot)
```

**Rule of thumb for exam scenarios:**
| Signal in scenario | → Choose |
|---|---|
| "Needs updated info from our documents" | RAG |
| "Always responds in our brand voice" | Fine-tuning |
| "Model doesn't know our terminology / jargon" | Fine-tuning |
| "Need consistent structured output format" | Fine-tuning |
| "Need to reference current inventory prices" | RAG |
| "Cost is a concern, want smaller model to perform like GPT-4" | Fine-tuning (knowledge distillation) |
| "Low latency, simple task" | Prompt engineering first |

---

### STAGE 1: DATA PREPARATION — The Make-or-Break Stage

**Format: JSONL (JSON Lines)**

Every line is one training example. For chat-completions fine-tuning (GPT-4o, GPT-3.5):

```jsonl
{"messages": [{"role": "system", "content": "You are a medical documentation assistant."}, {"role": "user", "content": "Summarize this clinical note: Patient presents with..."}, {"role": "assistant", "content": "Clinical Summary: 45-year-old presenting with..."}]}
{"messages": [{"role": "system", "content": "You are a medical documentation assistant."}, {"role": "user", "content": "Summarize this clinical note: Follow-up visit for..."}, {"role": "assistant", "content": "Clinical Summary: Follow-up for..."}]}
```

> **EXAM TRAP #1:** The format is **JSONL** (one JSON object per line), NOT a JSON array. If you wrap everything in `[...]`, it fails validation.

> **EXAM TRAP #2:** Validation data must be **a separate file** — you cannot pass a train/test split ratio. You upload two `.jsonl` files.

**Data quality rules (Microsoft guidance):**
- Minimum: **10 examples** (absolute floor — performance will be poor)
- Recommended: **50–100+ examples** for meaningful improvement
- Sweet spot: **hundreds to a few thousand examples**
- Examples must cover the FULL range of inputs the model will see in production
- Balance your classes if doing classification
- UTF-8 encoding with BOM for validation files
- Max file size: **512 MB**

**Data quality > Data quantity.** One bad batch of examples with wrong answers causes regression. Garbage in, garbage out — permanently.

---

### STAGE 2: SELECTING A BASE MODEL

Microsoft Foundry currently supports fine-tuning for:

| Model Family | Notes |
|---|---|
| **GPT-4.1** | Complex tasks, language translation, domain adaptation, advanced code |
| **GPT-4.1-mini** | Faster iteration, lower cost, classification/sentiment/content moderation |
| **GPT-4o** | Multimodal (text + images) |
| **GPT-4o-mini** | Smaller/cheaper 4o variant |
| **Phi-4** | Microsoft's small model, highly efficient |
| **Meta Llama models** | Via model catalog (Llama 3.x variants) |

**Selection criteria for the exam:**

```
TASK COMPLEXITY?
├── Complex (translation, domain adaptation, code gen) → GPT-4.1
├── Focused (classification, sentiment, content mod) → GPT-4.1-mini
└── Need multimodal support → GPT-4o or GPT-4o-mini

BUDGET?
├── Cost-sensitive / high volume → Smaller model (mini variants, Phi-4)
└── Performance-critical → Larger model

DATA SIZE?
├── Small dataset (hundreds) → Smaller base model (simpler to adapt)
└── Large dataset (thousands+) → Can try larger model fine-tuning
```

> **EXAM TRAP #3:** Not all models in the catalog support fine-tuning. Fine-tuning is only available for models that explicitly support it — the Model Catalog shows a "Fine-tune" option in the model card.

---

### STAGE 3: FINE-TUNING CONFIGURATION

**Portal path:** Microsoft Foundry → Your Project → **Fine-tuning** (left nav) → Create fine-tuning job

**Key parameters:**

| Parameter | What it does | Typical range |
|---|---|---|
| **Training epochs** | How many times model sees the full dataset | 1–4 (default: auto) |
| **Batch size** | Examples per gradient update | auto (8–32) |
| **Learning rate multiplier** | Scales base LR; higher = faster adaptation but risk of overfitting | 0.1–1.0 (default: 1.0) |
| **Seed** | Reproducibility | Any integer |
| **Training type** | `standard` vs `globalstandard` (for global training) | Depends on model |

> **EXAM TRAP #4:** If you set learning rate multiplier too high (e.g., 3.0), the model **catastrophically forgets** its general abilities (catastrophic forgetting). Start at default or lower.

**SDK approach (Python):**
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview"
)

# Upload training file
with open("training_data.jsonl", "rb") as f:
    training_response = client.files.create(
        file=f,
        purpose="fine-tune"
    )
training_file_id = training_response.id

# Upload validation file (separate file, required)
with open("validation_data.jsonl", "rb") as f:
    validation_response = client.files.create(
        file=f,
        purpose="fine-tune"
    )

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_response.id,
    model="gpt-4o-mini",  # base model to fine-tune
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": 0.5
    },
    suffix="medical-docs-v1"  # appended to model name for easy ID
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")

# Check status
job_status = client.fine_tuning.jobs.retrieve(job.id)
# When status == "succeeded", job.fine_tuned_model has the deployed model name
```

**Monitoring the job:**
- Portal: Fine-tuning → job → live metrics (training loss, validation loss)
- SDK: `client.fine_tuning.jobs.list_events(job_id)` streams training events

---

### STAGE 4: EVALUATING THE FINE-TUNED MODEL

**Two evaluation dimensions:**

1. **Training metrics (during training):**
   - **Training loss** — should decrease steadily
   - **Validation loss** — if this rises while training loss falls → **overfitting**
   - Watch for a gap between train loss and val loss

2. **Post-training evaluation (after fine-tuning):**
   - **Use Azure AI Foundry Evaluation** to run systematic eval on a held-out test set
   - Compare fine-tuned model vs base model on the SAME test set
   - Key metrics depend on task:
     - Classification: accuracy, F1, precision, recall
     - Generation: coherence, fluency, groundedness, relevance
     - RAG + fine-tuning combined: groundedness (critical — does the answer stick to retrieved facts?)

> **EXAM TRAP #5:** Fine-tuning evaluation is NOT the same as RAG evaluation. For fine-tuned generative models, you care about **output quality on your task** — not retrieval relevance. Don't mix up the metric sets.

**The Deployment-After-Fine-Tuning flow:**
```
Fine-tuning job completes
        ↓
Model deployed automatically to Azure OpenAI resource
        ↓
fine_tuned_model name = "gpt-4o-mini:ft-<your-org>:<suffix>-<timestamp>"
        ↓
Use exactly like the base model — same API, same endpoint, different model name
        ↓
Monitor with Azure Monitor / Azure AI Foundry monitoring
```

---

### STAGE 5: RESPONSIBLE AI FOR LLMs

This appears in the exam as part of "evaluate fine-tuned models." Don't skip it.

**The 6 Microsoft RAI Principles:**
1. **Fairness** — No discriminatory outcomes
2. **Reliability & Safety** — Works as expected, fails safely
3. **Privacy & Security** — Protects user data
4. **Inclusiveness** — Works for diverse users
5. **Transparency** — Users know they're talking to AI
6. **Accountability** — Humans remain responsible

**Content Filtering in Azure AI Foundry:**

Content filters run on ALL deployed models in Foundry. They operate on:
- **Input** (user message) — block harmful prompts
- **Output** (model response) — block harmful responses

**Four harm categories with severity levels (Safe / Low / Medium / High):**
- Hate and fairness
- Sexual content
- Violence
- Self-harm

**Safety evaluation for fine-tuned models:**
- Azure AI Foundry provisions a separate fine-tuned GPT-4o model to run **adversarial attacks** against your fine-tuned model
- This tests if fine-tuning has degraded the model's safety behaviors
- Runs automatically as part of fine-tuning pipeline for supported models

> **EXAM TRAP #6:** Content filtering is ON by default and **cannot be fully disabled** for Azure OpenAI models. You can configure severity thresholds, but you cannot bypass the system entirely. For fine-tuned models, you still get content filtering on top.

**Groundedness (for RAG + fine-tuning):**
- Groundedness = does the model's response actually come from the retrieved context?
- Ungrounded responses = hallucination
- You can evaluate groundedness with Azure AI Foundry Evaluations

---

### THE DOMAIN 4 BIG PICTURE — Connecting All 4 Days

```
             THE LANGUAGE MODEL OPTIMIZATION DECISION

  ┌─────────────────────────────────────────────────────────────────────┐
  │              WHAT DO YOU NEED TO CUSTOMIZE?                         │
  └─────────────────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────────┐
            │                  │                       │
   BEHAVIOR/STYLE         KNOWLEDGE/FACTS         STARTING POINT
   (how it responds)      (what it knows)         (discover & evaluate)
            │                  │                       │
            ▼                  ▼                       ▼
       FINE-TUNING            RAG               MODEL CATALOG
       (Day 6)               (Day 5)               (Day 3)
            │                  │                       │
            └──────────────────┼───────────────────────┘
                               │
                               ▼
                      PROMPT FLOW (Day 4)
               ─────────────────────────────
               Orchestrates the WHOLE pipeline:
               - Calls RAG retrieval tools
               - Calls fine-tuned (or base) LLM
               - Evaluates output quality
               - Tests prompt variants
               - Deploys as managed API
```

**The workflow hierarchy:**
1. **Browse Model Catalog** → find the right base model
2. **Decide on optimization approach** → PE vs RAG vs Fine-tuning (or combine!)
3. **Build with Prompt Flow** → chain retrieval + LLM calls
4. **Evaluate** → run evaluations on your flow
5. **Deploy** → to an endpoint (online or batch)
6. **Monitor + RAI** → content filtering, groundedness, responsible AI

---

### COMBINING APPROACHES (Advanced — real exam scenarios)

You can and often SHOULD combine approaches:

**RAG + Fine-tuning:**
- Fine-tune for tone/format → RAG for fresh knowledge
- Example: Fine-tune on 500 customer service examples for your brand voice → RAG to fetch current product specs
- Fine-tuning doesn't make the model "know" your documents — RAG does that

**Prompt Engineering + RAG:**
- The system prompt structures how retrieved context is used
- Always start here before jumping to fine-tuning

> **EXAM TRAP #7:** "Should we fine-tune our model on our FAQ documents?" **NO.** FAQs change. Fine-tuning on documents that change = stale model. Use RAG for dynamic knowledge. Fine-tune only for stable behavioral patterns.

---

## ❓ SCENARIO QUESTIONS (20 min)

*Questions only — answers are in the answers file. Attempt them all.*

**Q1.** A data science team at a hospital wants their AI assistant to always respond in a specific clinical note format (SOAP format) and use proper medical terminology. The note format never changes. They have 300 annotated examples. What is the MOST appropriate optimization approach?

A) Prompt engineering — add the SOAP format to the system prompt
B) RAG — retrieve relevant clinical guidelines at query time
C) Fine-tuning — train the model to always use SOAP format
D) Model catalog — select a medical-domain model

**Q2.** You're fine-tuning GPT-4o-mini on 150 JSONL examples. During training, you notice training loss drops to 0.02 but validation loss is 0.45. What does this indicate and what should you do?

A) The model is underfitting; increase learning rate multiplier
B) The model is overfitting; reduce epochs or add more diverse training data
C) The model is performing well; deploy it as-is
D) The validation file is incorrectly formatted; re-upload it

**Q3.** You upload a training file to Azure AI Foundry for fine-tuning. The upload fails with a validation error. Your file has 200 examples in this format:
```json
[
  {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
  {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]}
]
```
What is the problem?

A) The file uses "messages" instead of "prompt"/"completion"
B) The file is formatted as a JSON array instead of JSONL (one object per line)
C) The file must include a "system" message in every example
D) The file encoding is wrong

**Q4.** You want to fine-tune a model to help with complex medical literature translation from English to Spanish. Which base model does Microsoft recommend starting with?

A) GPT-4.1-mini (for lower cost)
B) GPT-4.1 (for complex language tasks)
C) Phi-4 (small and efficient)
D) GPT-4o (for multimodal support)

**Q5 (Drag-and-drop style).** Order these fine-tuning steps correctly:
- A) Upload validation JSONL file
- B) Monitor training loss and validation loss curves
- C) Create fine-tuning job with hyperparameters
- D) Upload training JSONL file
- E) Deploy and test the fine-tuned model
- F) Evaluate on held-out test set using Azure AI Foundry Evaluations

*Correct order: __*

**Q6.** An application uses a fine-tuned GPT-4o model deployed in Azure AI Foundry. A user submits a message that triggers the violence content filter at "High" severity. What happens?

A) The fine-tuned model processes the request and returns a response; content filtering only applies to base models
B) The content filter blocks the input and returns an error before the model processes it
C) The fine-tuned model processes the request but the response is automatically redacted
D) An alert is sent to the administrator but the request is still processed

**Q7.** From Day 5 (RAG): Your RAG system is returning relevant documents but the final LLM answers don't actually use the retrieved content — they seem to ignore it and use the model's training knowledge instead. Which evaluation metric measures this problem?

A) Relevance
B) Coherence
C) Groundedness
D) Fluency

**Q8.** From Day 4 (Prompt Flow): You have a Prompt Flow deployed as a managed online endpoint. You want to run batch evaluation over 500 test cases. What is the correct approach?

A) Use the Foundry portal's "Chat" tab to manually test each case
B) Use `PromptFlowClient.run()` with a dataset JSONL file referencing your deployed endpoint
C) Re-deploy the flow as a batch endpoint and invoke it with a BatchDeployment job
D) Export the flow to AML and run it as an AML pipeline job

**Q9.** A company wants to fine-tune a model to be more conservative and safe in its responses to comply with financial regulations. After fine-tuning, they want to verify the model hasn't become MORE susceptible to generating harmful content. What Azure AI Foundry feature automatically addresses this?

A) Azure Content Safety resource
B) Safety evaluation during fine-tuning (adversarial attack testing)
C) Responsible AI dashboard
D) Model benchmark comparison in the catalog

**Q10 (Case Study).** Contoso Financial Services is building an AI advisor. They have:
- 1,000 past Q&A examples of advisors responding to clients (consistent formal tone + regulatory disclaimers always included)
- A SharePoint library with current regulations that changes monthly
- A requirement that all answers cite the specific regulation used

What is the BEST architecture?

A) Fine-tuning only — train on 1,000 examples including regulation text
B) RAG only — index all regulations and Q&A examples
C) Fine-tuning (on the 1,000 Q&A examples for tone + disclaimers) + RAG (for current regulations, citations)
D) Prompt engineering only — add all regulations to the system prompt

---

## 🛠️ HANDS-ON CHALLENGE (10 min)

**Task: Validate your fine-tuning data locally before uploading**

Write a Python script that:
1. Reads a local `training_data.jsonl` file
2. Validates each line has `{"messages": [...]}` structure with at least `user` and `assistant` turns
3. Counts total examples
4. Checks file size (must be under 512 MB)
5. Prints a "READY TO UPLOAD" or lists errors

```python
import json
import os

def validate_finetuning_file(filepath: str) -> dict:
    errors = []
    valid_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {i}: Invalid JSON")
                continue
            
            # Check structure
            if "messages" not in obj:
                errors.append(f"Line {i}: Missing 'messages' key")
                continue
            
            messages = obj["messages"]
            roles = [m.get("role") for m in messages]
            
            if "user" not in roles:
                errors.append(f"Line {i}: No 'user' turn found")
            if "assistant" not in roles:
                errors.append(f"Line {i}: No 'assistant' turn found")
            
            valid_count += 1
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if file_size_mb > 512:
        errors.append(f"File too large: {file_size_mb:.1f} MB (max 512 MB)")
    
    return {
        "valid_examples": valid_count,
        "errors": errors,
        "file_size_mb": round(file_size_mb, 2),
        "ready": len(errors) == 0 and valid_count >= 10
    }

# Run it
result = validate_finetuning_file("training_data.jsonl")
print(f"Valid examples: {result['valid_examples']}")
print(f"File size: {result['file_size_mb']} MB")
if result['ready']:
    print("✅ READY TO UPLOAD")
else:
    print("❌ ERRORS FOUND:")
    for e in result['errors']:
        print(f"  - {e}")
```

**Bonus challenge:** Add a check that no example exceeds 4,096 tokens (use `tiktoken`).

---

## 🎯 KEY TAKEAWAYS

1. **Fine-tuning = weight update** → changes permanent behavior. RAG = retrieval at query time → knowledge stays fresh. Never fine-tune on documents that change.

2. **Data format is JSONL** (one JSON object per line, not an array). UTF-8 with BOM for validation files. Separate training and validation files. Min 10 examples, sweet spot 50–1,000+.

3. **Overfitting signal:** training loss ↓ while validation loss ↑. Fix: reduce epochs, increase data variety, lower learning rate multiplier.

4. **Base model selection:** GPT-4.1 for complex tasks (translation, domain adaptation), GPT-4.1-mini for focused tasks (classification, content mod) or cost-sensitive scenarios.

5. **Content filtering applies ALWAYS** — even to fine-tuned models. Cannot be fully disabled. Safety evaluations run adversarial attacks to check fine-tuning hasn't degraded safety.

6. **The 4-component Domain 4 stack:** Model Catalog (choose) → Optimization decision (PE/RAG/FT) → Prompt Flow (orchestrate + evaluate) → Deploy + Monitor.

7. **Combining is normal:** Fine-tuning for stable behavior + RAG for dynamic knowledge is the recommended pattern for real enterprise applications.

---

## 🃏 ANKI CARD SUGGESTIONS

**Card 1:**
Front: What format is required for fine-tuning training data in Azure AI Foundry?
Back: JSONL — one JSON object per line. Each object: `{"messages": [{"role": "system/user/assistant", "content": "..."}]}`. NOT a JSON array.

**Card 2:**
Front: What does it mean when training loss decreases but validation loss increases during fine-tuning?
Back: Overfitting. The model memorized training data but can't generalize. Fix: fewer epochs, more diverse data, lower learning rate multiplier.

**Card 3:**
Front: When should you use fine-tuning vs RAG?
Back: Fine-tune for stable behaviors (tone, format, style, domain jargon). Use RAG for dynamic knowledge (docs that change, citations needed). Combine both for enterprise apps.

**Card 4:**
Front: Can content filtering be disabled for fine-tuned models in Azure AI Foundry?
Back: No. Content filtering applies to all models including fine-tuned ones. You can configure severity thresholds but cannot fully bypass it.

**Card 5:**
Front: What is "groundedness" in the context of RAG evaluation?
Back: Whether the model's response is actually supported by the retrieved context. Low groundedness = model is hallucinating instead of using retrieved docs.

**Card 6:**
Front: Microsoft's recommended base model for fine-tuning a complex task (language translation, domain adaptation)?
Back: GPT-4.1. For simpler focused tasks (classification, sentiment), use GPT-4.1-mini.

---

*Today is the last deep-content day. Tomorrow is practice exam + weak spot review. The gap is closed. 🏔️*
