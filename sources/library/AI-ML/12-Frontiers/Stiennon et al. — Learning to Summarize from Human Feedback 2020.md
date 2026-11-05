# Stiennon et al. — Learning to Summarize from Human Feedback (2020)

*The bridge paper between preference learning in RL and RLHF for language models. Before InstructGPT taught models to follow instructions, this paper taught them to write summaries humans actually preferred.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Learning to summarize from human feedback"** — Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, 2020.
- OpenAI
- [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
- NeurIPS 2020

## Key Ideas

1. **Optimize for what humans actually want, not just for reference overlap.**
   Summarization had been trained against reference summaries and evaluated with metrics like ROUGE. The problem is that ROUGE is only a proxy. A summary can score well on n-gram overlap and still be dull, misleading, or just worse than what a human would pick.

2. **Bring the Christiano preference-learning recipe into language.**
   The core pipeline is familiar now, but this was the paper that made it concrete for text:
   - sample multiple summaries for the same post,
   - ask humans which one is better,
   - train a **reward model** on those pairwise preferences,
   - fine-tune the summarization model with RL to maximize predicted human preference.

3. **Human preferences beat reference summaries.**
   This was the real shock: the RL-trained models were often preferred not just to supervised baselines, but even to the original human-written TL;DR references in the dataset. That is a huge conceptual shift. The dataset target is no longer the ceiling; human judgment at evaluation time is.

4. **Reward models can generalize better than the old metrics.**
   The learned reward model transferred beyond the exact training setup and correlated with what people judged to be good summaries more reliably than ROUGE. In other words: a learned preference model captured quality better than a fixed lexical metric.

5. **This is the rehearsal for RLHF on assistants.**
   InstructGPT did not emerge from nowhere in 2022. This paper is the dry run: reward modeling + RL on human preferences, but applied to summarization rather than open-ended instruction following.

## What the Pipeline Looks Like

- start with a pretrained language model
- fine-tune it on supervised summaries
- sample candidate summaries for the same post
- collect human comparisons between candidates
- train a reward model to predict which summary humans prefer
- use RL to optimize the policy against that reward model

That is already the RLHF skeleton. The only thing missing is the broader assistant framing.

## Why This Paper Matters So Much

Christiano et al. (2017) proved that human pairwise preferences could train a reward model for RL agents. This paper proved the same trick works for **language generation** at meaningful scale.

That is the hinge.

Without this paper, the jump from RL agents to ChatGPT-era post-training looks abrupt. With it, the lineage becomes clean:

- [[Christiano et al. — Deep RL from Human Preferences 2017]] — preference learning as a general recipe
- **Stiennon et al. (2020)** — preference learning works for language generation
- [[Ouyang InstructGPT RLHF 2022|Ouyang et al. — InstructGPT / RLHF (2022)]] — apply the same machinery to instruction-following assistants
- [[../13-Scaling-and-Alignment/Bai et al. — Constitutional AI 2022|Bai et al. — Constitutional AI (2022)]] — reduce reliance on direct human labels via AI feedback
- [[../13-Scaling-and-Alignment/Rafailov et al. — Direct Preference Optimization 2023|Rafailov et al. — DPO (2023)]] — keep the preference signal, simplify the optimization stack

## Lineage

- **Built on:** [[Christiano et al. — Deep RL from Human Preferences 2017]] and broader reward-model / preference-learning work
- **Leads to:** [[Ouyang InstructGPT RLHF 2022|InstructGPT / RLHF]], modern assistant post-training, and later alignment simplifications like [[../13-Scaling-and-Alignment/Rafailov et al. — Direct Preference Optimization 2023|DPO]]
- **Connected to:** reward modeling, pairwise comparison datasets, RL for sequence generation, and evaluation beyond ROUGE/BLEU-style proxies

## Why It Matters for the Lineage

This is the paper where alignment stops being just an RL safety idea and becomes a language-model training method.

It also quietly changes how to think about evaluation. If the metric is a bad proxy, train on the judgment you actually care about. That logic now appears everywhere:
- preference-tuned chat models,
- ranking systems,
- enterprise copilots evaluated by expert raters,
- human-in-the-loop workflows where “quality” is partly tacit.

## The Deeper Point

Language-model alignment did not begin with a chatbot.

It began when people admitted that our automatic metrics were not the thing we truly cared about, and then had the nerve to optimize for human taste directly.

That is a very modern AI story: not better loss functions in the abstract, but better proxies for what humans mean by *better*.

---

*Before RLHF became the standard recipe for assistants, this paper proved that human preference could beat reference text as the target itself.*
