# Learning Design Principles

General requirements for lessons, practice, feedback and the interface. This document describes
product behavior and an intended audience; it does not describe an individual. The filename is
retained for compatibility with existing documentation links.

See the [pedagogy-to-feature map](ARCHITECTURE.md#5-pedagogy--feature-map) for implementation.

## 1. Purpose

Make technical practice easy to start, clear to follow and measurable through demonstrated
understanding. Explain the system behind a technique, provide meaningful exercises and give
specific feedback about mistakes.

## 2. Audience and scope

- Learners developing or refreshing Python, mathematics, statistics and machine learning skills.
- Preparation for data science and AI engineering interviews, including coding, explanation and
  system design.
- A structured core path with optional advanced topics. Session lengths and the study calendar
  are course configuration, not assumptions about an individual learner's availability.

## 3. Design goals

| Goal | Product behavior |
|---|---|
| Explain structure | Open lessons with rules, relationships and a useful decision framework |
| Explain reasoning | Include a focused derivation or explanation in "Why it works" |
| Support deliberate practice | Use bounded sessions, a hint ladder and repeatable exercises |
| Connect ideas to applications | Frame problems as components of clearly described synthetic systems |
| Show meaningful progress | Track solving, retention and interview explanation rather than activity alone |

## 4. Usability requirements

| Requirement | Product behavior |
|---|---|
| Clear expectations | Show the session shape, time budget and completion criteria up front |
| Easy startup | Provide one concrete first action and a short-session option |
| Consistent notation | State conventions and label exceptions explicitly |
| Flexible session length | Offer low-energy mode with familiar practice and no new material |
| Readable presentation | Use calm layouts, semantic labels, visible focus and reduced-motion support |
| Constructive feedback | Name the kind of mistake and suggest a next step without judging the learner |
| Continuity | Resume from a transition note rather than requiring the learner to reconstruct context |

## 5. Teaching approach

1. **Explain, practise, reflect.** Introduce the system, work a meaningful problem, then use
   feedback to identify what to practise next.
2. **Retrieve before re-teaching.** Recall earlier material and make a prediction before reading
   an explanation.
3. **Introduce difficulty gradually.** Add spacing, interleaving and varied examples in manageable
   steps rather than changing every practice condition at once.
4. **Use errors as evidence.** Feedback should distinguish shape, arithmetic, algorithm and
   interpretation mistakes. Hints progress from concept to formula to code structure.
5. **Interleave after initial understanding.** Combine familiar exercises with new material once
   the basic method is established.
6. **Practise recall and explanation.** Repeat implementations and concise explanations, including
   trade-offs and failure modes.
7. **Use visual support where useful.** Include shape annotations, diagrams and comparison tables.
8. **Use neutral examples.** Scenarios must be synthetic, public or explicitly sourced; never
   depend on a learner's employment, medical history, private projects or personal experiences.

## 6. Session design (45-minute default)

These are the current course defaults. They describe a consistent practice structure, not a
claim that one schedule suits every learner.

```text
Learn session       Retrieve 5' → Predict 3' → Learn 12' → Practice 20' → Close 5'
Consolidate session Review cards 15' → timed re-solves 25' → Close 5'
Interview session   Mock block 40' (coding + theory or case) → debrief 5'
Minimum viable      Review 10' → one familiar re-solve 10'
```

Every session ends with a transition note recording the stopping point, next step and optional
blocker. The next session shows that note first. Notes are local learner data and are excluded
from published content.

## 7. Tutor contract

- Use concise, concrete and respectful language.
- Name the kind of difficulty and ground feedback in the available evidence.
- Refer to the grader for correctness. Do not reveal a full solution before the hint ladder is
  exhausted and the learner requests one.
- Ask useful questions without making the learner guess indefinitely. Explain plainly when a
  question has not helped.
- Keep examples and explanations independent of personal identity or background.

## 8. Success criteria

For each core topic, progress means a passing solution, a later timed solve without hints, and
an explanation of the method, trade-offs and failure modes. Retention and understanding matter
more than completing a list or maintaining a streak.
