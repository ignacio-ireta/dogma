# Mock Interviewer Guide — How to Run a Data Science Case Interview Mock

*Thank you for doing this! Here's everything you need.*

---

## The Format (55 minutes total)

You're simulating a first-round data science case interview. Stick to the timing as closely as possible — time pressure is part of the test.

| Block | Time | Your Role |
|-------|------|-----------|
| **Welcome** | 2 min | Introduce yourself casually. Ask: "Tell me about yourself — what brings you here?" |
| **Live Coding** | 15 min | Share the 2 coding problems. Watch him solve them in Python. Then ask the comprehension question verbally. |
| **Case: Setup** | 3 min | Read the scenario text from the interviewer guide. Let him take notes. |
| **Case: His Turn** | 25 min | He drives. You reveal info when he asks good questions. Challenge his choices. Drop the twist when indicated. |
| **Synthesis** | 3 min | Ask: "Can you summarize your recommendation?" |
| **Q&A** | 5 min | Your friend asks you questions. Improvise — you're a DS at a top consulting firm for this exercise. |
| **Debrief** | 10 min | Score using the rubric. Discuss together. |

---

## Rules

1. **He shares his screen** for the coding portion. He can use internet for syntax (e.g., pandas docs). No AI tools.
2. **For the case — don't give info he doesn't ask for.** If he doesn't ask about class balance, don't volunteer it. The whole point is testing whether he asks the right questions.
3. **Challenge his model choices.** "Why not a neural network?" "What if the client can't interpret that?" "How does [his model] actually work?" Push him to go deeper.
4. **Drop the twist** at the point indicated in the guide. Watch how he adapts.
5. **Use the scoring rubric** (SCORING_RUBRIC.md) — fill it out during or immediately after.
6. **Keep it conversational.** You're a friendly DS interviewer, not a professor giving an exam. Nod, react, ask follow-ups naturally.

---

## What a Strong Candidate Looks Like

- **Structures the problem** before diving into a solution (MECE thinking)
- **States a hypothesis** ("I believe X is the main driver. Here's how I'd test it.")
- **Names a model AND explains why** (with alternatives considered)
- **Connects technical choices to business impact** (not just model accuracy)
- **Quantifies with numbers** ("That would save approximately $X per year")
- **Thinks out loud** the entire time (no long silences)
- **Asks for clarification** when needed
- **Recovers gracefully** from mistakes or wrong turns

---

## Common Things to Watch For

- Does he reframe the problem in his own words, or just jump to modeling?
- Does he ask questions before solving, or assume he knows everything?
- When you challenge a model choice, does he defend it with reasoning or cave immediately?
- Does he translate model output to business action, or stop at "we'd get good accuracy"?
- Does he use numbers, or stay vague ("significant improvement")?
- Does he handle the twist by adapting, or does he freeze?

---

## Each Pack Contains

```
pack-X/
├── INTERVIEWER_GUIDE.md    ← Your script (he should NOT read this)
├── exhibits/               ← Charts/tables to show him at specific points
├── coding/
│   ├── problem_1.md        ← First coding task
│   ├── problem_2.md        ← Second coding task
│   ├── comprehension.md    ← Code to show + questions to ask
│   └── data/               ← CSV files for the coding tasks
└── coding_solutions.md     ← Expected solutions (for your reference)
```

**IMPORTANT:** The INTERVIEWER_GUIDE.md and coding_solutions.md are for YOUR eyes only. The candidate should NOT read these before the mock. Honor system — same as a real interview.

---

## Setup Checklist

- [ ] Pick a pack (A, B, or C) — coordinate so each friend picks a different one
- [ ] Read the INTERVIEWER_GUIDE.md thoroughly (10 min)
- [ ] Have the exhibits ready to screen-share at the right moments
- [ ] Print or have the SCORING_RUBRIC.md open
- [ ] Set a timer (visible to both of you)
- [ ] Zoom/Meet call with screen sharing enabled
- [ ] The candidate should have: Python environment, paper + pen, water. No notes, no reference docs.

---

*You're doing him a huge favor. Make it real — that's how he gets better. Good luck!* 🏔️
