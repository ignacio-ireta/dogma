# Model Selection Interview Bridge - Open Reasoning Models

*Short synthesis from the 2026 open-model efficiency notes into a consulting interview answer.*

Source notes:
- `AI-ML/13-Scaling-and-Alignment/Manik Wang Gemma Phi Qwen Tradeoffs 2026.md`
- `AI-ML/13-Scaling-and-Alignment/DeepSeek-V4 — DeepSeek AI 2026.md`
- `AI-ML/13-Scaling-and-Alignment/NVIDIA Nemotron 3 Super 2026.md`

## The Interview Point

The mature answer to "which LLM would you use?" is not a model name. It is an evaluation frame:

> I would choose the model as an operating point, not as a leaderboard winner: workload, latency, memory budget, prompt regime, context length, tool-use pattern, and accuracy all matter together.

Manik and Wang make this explicit by comparing open reasoning models across accuracy, latency, VRAM, tasks, and prompting regimes. DeepSeek-V4 and Nemotron 3 Super show why: modern models are no longer just dense Transformers with different parameter counts. They are systems shaped around long context, sparse activation, low precision, routing, and agentic throughput.

## Case Pattern

If the case is an enterprise assistant, ask four questions before naming a model:

1. **Workload shape:** Is the task reasoning-heavy, retrieval-heavy, coding-heavy, multilingual, or mostly summarization/classification?
2. **Context economics:** Does it truly need long context, or should retrieval and chunking carry most of the load?
3. **Serving constraints:** What latency, VRAM, cost per request, privacy, and deployment environment are acceptable?
4. **Prompt sensitivity:** Has the model been tested under the actual prompt style, not just a generic benchmark?

Then propose a small evaluation grid: two or three candidate models, one realistic prompt protocol, a fixed hardware target, representative test cases, and metrics that include both quality and cost.

## 60-Second Answer

> I would avoid picking an LLM from a single leaderboard. For an enterprise assistant, I would first define the operating surface: workload type, context length, latency target, privacy constraints, hardware budget, and acceptable error types. Then I would benchmark a small candidate set under the actual prompt regime. Papers like Manik and Wang show that accuracy, VRAM, latency, and prompting interact; DeepSeek-V4 and Nemotron show that modern models are architecture-and-systems tradeoffs, not just parameter counts. My default would be to choose the cheapest model that clears the quality bar on representative cases, then use retrieval, evaluation, and monitoring to keep it honest.

## Trap To Avoid

Do not say:

> "I would use the biggest/best model."

Say:

> "I would find the lowest-cost operating point that meets the business-quality threshold, and I would prove it on the workload."

That sounds like an engineer and a consultant at the same time.

Tags: #bridge #interview #llm-evaluation #model-selection
