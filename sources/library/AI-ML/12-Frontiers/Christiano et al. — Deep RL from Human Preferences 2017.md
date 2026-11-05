# Christiano et al. — Deep RL from Human Preferences (2017)

*The missing hinge in the alignment story: before RLHF scaled to language models, this paper showed that human preferences could stand in for a hard-coded reward.*

**← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Deep reinforcement learning from human preferences"** — Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei, 2017.
- OpenAI + DeepMind
- [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Updated arXiv version: 2023, original work from 2017

## Key Ideas

1. **Sometimes the real problem is not optimization — it is specification.**
   In many real tasks, we do not know how to write the reward function we actually want. We can say “drive safely” or “be helpful,” but turning that into a clean scalar objective is ugly fast. Christiano et al. attack the reward-specification problem directly.

2. **Ask humans to compare behavior snippets instead of writing rewards.**
   Rather than forcing humans to assign scores, the system shows two short trajectory clips and asks which one is better. Humans are much better at *relative judgment* than at writing formal reward functions.

3. **Learn a reward model from those preferences.**
   A separate model is trained to predict which trajectory segment a human would prefer. That learned reward then becomes the training signal for the RL agent.

4. **Human supervision can be sparse.**
   The paper's most practical result is that you do *not* need to label everything. Less than 1% of the agent's interactions needed human feedback in their experiments. That is what made preference-based alignment feel operational instead of philosophical.

5. **This is the prototype of the modern alignment pipeline.**
   Preference data → reward model → policy optimization. That pattern later becomes RLHF for language models. InstructGPT did not appear from nowhere; it industrialized this recipe.

## What the Pipeline Looks Like

- agent interacts with environment
- sample short trajectory pairs
- human picks the better one
- train a reward predictor on those pairwise judgments
- optimize the policy against the learned reward
- repeat

The important move is subtle: **the human is not controlling the agent step by step**. The human is teaching the system what “better” looks like, and the reward model generalizes that signal.

## Lineage

- **Built on:** reinforcement learning, inverse/reward learning, pairwise preference modeling, and the broader problem of reward misspecification
- **Leads to:** [[Ouyang InstructGPT RLHF 2022|RLHF]] for language models, [[../13-Scaling-and-Alignment/Bai et al. — Constitutional AI 2022|Constitutional AI]] / RLAIF, and later simplifications like [[../13-Scaling-and-Alignment/Rafailov et al. — Direct Preference Optimization 2023|DPO]]
- **Connected to:** reward modeling, PPO-style policy optimization, AI safety's “outer alignment” problem, and enterprise feedback loops where direct objectives are messy

## Why It Matters for the Lineage

This is the paper that makes the later RLHF story click.

Without Christiano et al., RLHF can look like a weird OpenAI-specific trick from 2022. With this paper, the lineage becomes clean:

- **2017:** humans teach a reward model through pairwise comparisons
- **2022:** InstructGPT applies that pattern to language
- **2022:** Constitutional AI reduces the dependence on direct human labeling
- **2023:** DPO asks whether we can keep the preference signal while throwing away the RL machinery

So this note fills a real gap: it is the conceptual ancestor of modern post-training.

## The Deeper Point

Optimization is only as good as the thing you ask it to optimize.

This paper is one of the earliest modern demonstrations that when the reward is hard to write down, **human judgment itself can become the training signal**. That idea now sits underneath a ridiculous amount of frontier AI.

---

*Before models learned to follow instructions, someone had to teach machines that “better” could come from human taste instead of handwritten reward code.*