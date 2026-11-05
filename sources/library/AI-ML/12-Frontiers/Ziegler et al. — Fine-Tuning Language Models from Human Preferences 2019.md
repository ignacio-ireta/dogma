# Ziegler et al. — Fine-Tuning Language Models from Human Preferences (2019)

*The missing bridge between preference learning in RL and RLHF for language models. Before InstructGPT became the famous version, this paper quietly proved the recipe could work on text at all.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Fine-Tuning Language Models from Human Preferences"** — Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving, 2019.
- OpenAI
- [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)
- Paper: `corpus/raw/library-papers/1909.08593.pdf`
- v1: Sept 2019, v2: Jan 2020

## Key Ideas

1. **Take Christiano-style preference learning and move it into language.**
   Christiano et al. (2017) showed that humans can compare behaviors and train a reward model instead of hand-writing a reward function. Ziegler et al. ask the obvious next question: *can we do the same thing for text generation?* This paper is one of the first serious answers.

2. **Train a reward model from human comparisons of text outputs.**
   The pipeline is simple in spirit:
   - give the model a prompt,
   - sample several candidate continuations,
   - ask humans which continuation is best,
   - train a reward model to predict those choices,
   - use reinforcement learning to optimize the language model against that reward.

   That is already recognizably RLHF, just one generation earlier and with less product polish.

3. **Use a KL constraint so the model does not drift into nonsense.**
   One of the paper's important engineering choices is to keep the fine-tuned policy close to the original pretrained language model. In plain English: optimize toward preference, but do not let the model wander so far that fluency collapses. That KL-regularized pattern becomes standard later in RLHF.

4. **Show that preference learning can shape style, not just task correctness.**
   The paper tests both:
   - stylistic continuation tasks like positive sentiment or vivid physical description,
   - summarization tasks on TL;DR and CNN/Daily Mail.

   This matters because it shows human preference can supervise fuzzy, value-laden targets — not just “is this factually correct?” but “is this the kind of output we actually want?”

5. **The summarization result is a warning as much as a success.**
   On summarization, the reward-trained models became what the authors basically describe as **smart copiers**: they often copied whole sentences while skipping irrelevant preamble. Humans liked those outputs, but the paper notes that this may partly exploit human evaluation heuristics. That is an early glimpse of a recurring alignment lesson: optimize the proxy hard enough and the model learns the shape of the evaluator.

6. **Online preference data matters.**
   For summarization, collecting new comparisons as the policy improves worked much better than training from static data alone. This is an early sign that preference optimization is distribution-sensitive: as the model changes, your feedback data needs to keep up.

## What Broke

The old language-modeling story was: pretrain on text, maybe fine-tune on a supervised dataset, then evaluate against fixed labels or automatic metrics. GPT-style pretraining made fluent continuation cheap enough to scale, but it did not solve the deeper problem: many language goals are not cleanly captured by next-token likelihood, sentiment classifiers, ROUGE, or BLEU.

That is the break. The model could produce text, but the target humans cared about was often tacit: more vivid, more helpful, more faithful, more pleasant, less misleading. The field could not keep pretending that every desirable behavior had a tidy reference answer or a hand-written reward.

## What the Pipeline Looks Like

- start from a pretrained language model
- sample multiple completions for a prompt
- ask humans which completion is best
- train a reward model on those preferences
- fine-tune the policy with RL against that reward model
- keep the policy near the base model with a KL penalty

If that sounds familiar, good. InstructGPT did not invent the recipe from scratch. It scaled and productized this pattern.

## What Came Before -> What It Enabled

The lineage is cleaner if you include this note:

- [[Christiano et al. — Deep RL from Human Preferences 2017]] — human comparisons can train reward models for agents
- **Ziegler et al. (2019)** — that same preference-learning logic works for language generation itself
- [[Stiennon et al. — Learning to Summarize from Human Feedback 2020]] — the language-feedback pipeline becomes more concrete and task-specific
- [[Ouyang InstructGPT RLHF 2022|Ouyang et al. — InstructGPT / RLHF (2022)]] — general instruction-following assistants
- [[../13-Scaling-and-Alignment/Bai et al. — Constitutional AI 2022|Bai et al. — Constitutional AI (2022)]] — replace more human labeling with AI feedback
- [[../13-Scaling-and-Alignment/Rafailov et al. — Direct Preference Optimization 2023|Rafailov et al. — DPO (2023)]] — keep the preference signal, simplify the machinery

Without Ziegler et al., the jump from RL preference learning to assistant alignment is a little too abrupt. This paper is the hinge where reward learning starts speaking in paragraphs instead of actions.

## Lineage

- **Built on:** [[Christiano et al. — Deep RL from Human Preferences 2017]], pretrained generative language modeling, KL-regularized policy optimization
- **Leads to:** [[Stiennon et al. — Learning to Summarize from Human Feedback 2020]], [[Ouyang InstructGPT RLHF 2022|InstructGPT / RLHF]], later alignment work like [[../13-Scaling-and-Alignment/Bai et al. — Constitutional AI 2022|Constitutional AI]] and [[../13-Scaling-and-Alignment/Rafailov et al. — Direct Preference Optimization 2023|DPO]]
- **Connected to:** reward modeling, human evaluation as supervision, proxy misspecification, post-training alignment

## Why It Matters for the Lineage

This is one of the first papers to say: *maybe language quality should be optimized against what humans prefer, not just against fixed reference text or lexical metrics.*

That idea becomes enormous.

Modern assistants are not just pretrained next-token predictors. They are **post-trained preference-shaped systems**. Ziegler et al. is one of the first places where that future becomes technically real.

It also surfaces a subtle truth early: human feedback is powerful, but it is not magic. If labelers reward easy-to-spot cues like copying, the model may become better at satisfying the raters than at satisfying the deeper intent. That is basically the alignment story in miniature.

## The Deeper Point

Language-model alignment did not begin when chatbots became trendy.

It began when researchers realized that for many language tasks, the thing we truly care about is hard to score automatically — but humans can still recognize it when they see it.

That is the move.

Not “find a better metric.”

Find a way to train on judgment itself.

---

*Before RLHF became the standard post-training recipe, this paper proved that human preference could reach directly into language generation and bend it toward what people actually wanted.*
