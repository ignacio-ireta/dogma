# Schulman et al. — Proximal Policy Optimization (2017)

*The workhorse optimizer behind early RLHF: not glamorous, just brutally useful.*

**← [[_MOC|Scaling & Alignment]] ← [[../_MOC|AI/ML Lineage]]**

---

## The Paper

**"Proximal Policy Optimization Algorithms"** — John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, 2017.
- OpenAI
- [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Paper: `corpus/raw/library-papers/1707.06347.pdf`
- Introduced the PPO family as a simpler, more practical alternative to TRPO

## Key Ideas

1. **Policy gradients were powerful, but too temperamental.**
   Before PPO, reinforcement learning had an annoying tradeoff. Simple policy-gradient methods were easy to implement but unstable. More principled methods like TRPO made safer updates, but at the cost of much more complicated optimization machinery.

2. **Do not let the policy move too far in one update.**
   PPO's core intuition is wonderfully pragmatic: if a policy changes too aggressively, training can collapse. So instead of taking whatever gradient step looks best locally, PPO constrains updates so the new policy stays reasonably close to the old one.

3. **The clipped objective is the famous trick.**
   PPO measures how much more or less probable an action becomes under the new policy compared with the old one. If that ratio gets too large, PPO clips the benefit. In plain English: *improvement is good, overconfidence is suspicious.*

4. **Multiple minibatch updates from the same data.**
   Standard policy-gradient methods often used each batch once and moved on. PPO made it practical to reuse the same rollout data across several optimization epochs, which improved sample efficiency without needing TRPO's heavier math.

5. **It became the default because it hit the sweet spot.**
   PPO was not the final word in RL theory. It became popular because it was good enough, stable enough, and simple enough that people could actually train things with it.

## The Mechanism That Matters

PPO optimizes a **surrogate objective** using the probability ratio

- new policy probability of an action
- divided by the old policy probability of that same action

and then **clips** that ratio when it moves outside a safe band.

That clipping is the whole vibe of the paper. It prevents the optimizer from "winning" by making one huge, brittle step that looks good on one batch and terrible everywhere else.

This is why PPO mattered so much later for RLHF. Once you have a reward model for language outputs, you still need a way to fine-tune the policy without it drifting into nonsense. PPO was the practical answer.

## Lineage

- **Built on:** policy gradients, actor-critic methods, and [[../12-Frontiers/Christiano et al. — Deep RL from Human Preferences 2017|preference-based RL]]; specifically responds to the complexity of trust-region approaches like TRPO
- **Leads to:** [[../12-Frontiers/Ouyang InstructGPT RLHF 2022|InstructGPT / RLHF]], where PPO becomes the optimization engine for preference-tuned language models
- **Responded to by:** [[Rafailov et al. — Direct Preference Optimization 2023|DPO]], which asks whether we can keep the preference signal while removing PPO entirely
- **Connected to:** KL regularization, reward-model optimization, safe policy updates, and the practical economics of post-training frontier models

## What Came Before → What It Enabled

Before PPO, the alignment lineage had a conceptual recipe but not a comfortable workhorse.

[[../12-Frontiers/Christiano et al. — Deep RL from Human Preferences 2017|Christiano et al.]] showed that human comparisons could train a reward model. TRPO showed one way to keep policy updates from wrecking the policy, but it required heavier second-order machinery. Simple policy-gradient updates were easier, but too willing to take destructive steps.

PPO made the reward-learning story operational. Once a reward model says "this output is better than that one," PPO gives the policy a controlled way to move toward the reward without drifting too far from the behavior that made it useful in the first place. That is why the later RLHF chain becomes legible:

- preference data supplies the target,
- a reward model translates preferences into scores,
- PPO turns those scores into policy updates,
- KL pressure keeps the new policy close to the old model.

This is the mechanical bridge between "humans prefer this behavior" and "the model now behaves more like that."

## Why It Matters for the Lineage

PPO is the missing mechanical link between **human preference data** and **aligned model behavior**.

Christiano et al. (2017) showed how to turn human comparisons into a reward signal. But that still leaves a hard question: *how do you actually optimize a policy against that reward without wrecking it?* PPO became the standard answer.

That makes the alignment lineage much cleaner:

- **Christiano et al. (2017):** learn a reward from preferences
- **PPO (2017):** optimize a policy against that learned reward, stably enough to be useful
- **InstructGPT / RLHF (2022):** scale that recipe to language models
- **DPO (2023):** simplify the recipe by removing the explicit RL stage

Without PPO, early RLHF is conceptually incomplete. With PPO, the pipeline becomes real.

## The Real Shift

PPO is one of those papers whose importance is easy to underestimate because the idea sounds modest.

It did not invent preference learning.
It did not invent language models.
It did not solve alignment.

What it did was give the field a **reliable training loop** for turning rewards into behavior. And for several years, that was enough to make modern assistant training feasible.

Not eternal. Not elegant in every setting. But absolutely pivotal.

---

*If Christiano et al. taught the model what humans prefer, PPO was the wrench that actually tightened the bolts.*
