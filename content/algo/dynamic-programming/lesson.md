---
title: Dynamic programming
minutes: 12
min_viable: "Read The system and the Worked example, then write the recurrence for p01 in words before any code."
---

## The system

A dynamic program answers a question by reusing answers to smaller questions. Define it by five
parts, in this order, before writing code:

| Part | Question it answers | Example: minimum coins for value $v$ |
|---|---|---|
| **State** | What does one table entry mean? | $best[v]$ = fewest coins that make value $v$ |
| **Base case** | What is known without recursing? | $best[0] = 0$ |
| **Transition** | How does a state depend on smaller states? | $best[v] = \min_{c \le v} best[v-c] + 1$ over coin types $c$ |
| **Order of evaluation** | In what order can states be computed, so each dependency is already final? | Increasing $v$ |
| **Answer location** | Which state is the answer? | $best[\text{amount}]$, or $-1$ if it is unreachable |

A problem is a DP when the state is small (a few integers), the transition only looks at
smaller states, and the same state is reached many ways. If any of the five parts is unclear,
the code will be wrong in a way that is hard to see. Say the five parts aloud first.

**Two ways to fill the table.**

| Style | How | Cost | Risk |
|---|---|---|---|
| Top-down memoisation | Recurse from the answer; cache each state the first time it is computed | Only reachable states; one call per state | Recursion depth can equal the number of states along a path |
| Bottom-up tabulation | Loop over states in an order that respects the dependencies | Every state; no recursion | The loop order must match the transition |

Both compute the same values. Choose bottom-up when the order is obvious, and top-down when only
a few states are reachable. Mention the recursion-depth limit whenever you choose top-down.

## Worked example

Coins $\{1, 3, 4\}$, amount 6. The table $best[v]$ for $v = 0..6$:

| $v$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| $best[v]$ | 0 | 1 | 2 | 1 | 1 | 2 | 2 |

For $v = 6$ the candidates are $best[5]+1 = 3$ (coin 1), $best[3]+1 = 2$ (coin 3) and
$best[2]+1 = 3$ (coin 4). The minimum is 2, from coin 3. Greedy, which takes the largest coin
first, gives $4 + 1 + 1 = 3$ coins. The table considers every coin at every value, so it finds 2.

**Reconstruction.** Store the coin that achieved each minimum, $pick[v]$. Start at $v = 6$:
$pick[6] = 3$, so use a 3 and move to $v = 3$. $pick[3] = 3$, so use another 3 and reach 0.
The answer is $\{3, 3\}$. The same idea gives an argmax: keep the choice that won the
$\max$ at each state.

## Why it works

Optimal substructure: the best solution for $v$ uses some first coin $c$, and its remainder must
be optimal for $v-c$, or swapping in a better remainder would improve $v$. Overlapping subproblems:
many paths reach the same $v$, so computing each state once costs states times work per state,
here $O(\text{amount} \times |\text{coins}|)$.

## Signal vs. noise

- **Signal:** write the five parts before the loops. The most common DP bug is a wrong state
  meaning, not a wrong loop.
- **Signal:** the order of evaluation is the part to check. Read the transition as "which entries
  it reads", and confirm each one is already final.
- **Noise for now:** a clever closed-form answer for a specific coin set, or micro-optimisations
  of the inner loop.

## Where it breaks

- **Greedy is not DP.** Taking the largest coin first is a plausible but wrong rule for some
  denominations, as the $\{1, 3, 4\}$ example shows.
- **Unreachable states.** Start them at a value that no real answer can reach, such as
  $amount + 1$, and check at the end. A value of 0 would be a real answer.
- **Top-down depth.** A memoised recursion from amount 50,000 needs about that many stack frames.
  Python's default limit is 1,000. Build the table in a loop, or make the recursion iterative.
- **Space.** The full table for LCS is $(n+1)(m+1)$ integers. Each row depends only on the one
  above, so two rows are enough for the length; the witness needs the full table or a
  different walk.
- **Off-by-one in prefix lengths.** Strings are indexed from 0, but table entries count
  characters. $a[i-1]$ pairs with $dp[i][\cdot]$.

## Interview angle

Say the state in one sentence, then the recurrence, the base case, the order, and the answer's
location. Then write the loop. Name the cost: states times work per state. If asked for the
choice, add a parent table and walk back. If asked to save memory, check which earlier rows the
transition reads, and keep only those.

## Connections

This topic uses hash maps for memoisation, and generalises the prefix-sum idea: each table entry
summarises a smaller prefix. The same five-part definition covers the longest common subsequence
(two prefix lengths), edit distance and sequence alignment. Larger state spaces need the same
discipline with more dimensions.
