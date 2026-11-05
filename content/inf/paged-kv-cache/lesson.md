---
title: "Paged KV cache"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 by hand for a 4-token block size first."
---

## The system

A serving system stores each sequence's keys and values in a KV cache. Paging replaces one long
reserved region per request with fixed-size **blocks** taken from a shared pool.

- The pool has $N$ physical blocks, ids $0 \ldots N-1$. Each block holds $B$ token slots, the
  **block size**.
- Each sequence keeps a **block table**: its physical block ids in logical order.
- Token position $p$ (0-based) lives in table index $\lfloor p / B \rfloor$ at offset $p \bmod B$.
- A sequence with $n$ tokens holds $\lceil n / B \rceil$ blocks. The table grows only when a token
  starts a new block, that is, when $p \bmod B = 0$.
- A finished sequence returns its blocks. The allocator always hands out the lowest free id.

Four rules follow:

1. **Allocate on demand.** Nothing is reserved for the maximum output length, which is unknown at
   the start.
2. **Waste is confined to the last block.** A sequence wastes at most $B - 1$ slots.
3. **Blocks are uniform.** Any free block fits any sequence, so there is no fragmentation between
   sequences.
4. **Failure is explicit.** When no block is free, the allocation raises a named error, and the
   scheduler decides what to do. The allocator never half-allocates.

## Worked example

Take a cache of $1024$ token slots and $B = 16$, so there are $64$ blocks. Each request may reach
$512$ tokens, but the average request ends at $180$.

**Reserve the maximum.** Each request reserves $512$ slots, which is 32 blocks. Two requests fit, and
each one leaves $332$ reserved slots empty on average.

**Page on demand.** A $180$-token request needs $\lceil 180/16 \rceil = 12$ blocks. Its last block has
$12 \cdot 16 - 180 = 12$ empty slots, below $B$. Five requests need $60$ blocks, which fits in $64$.

The batch size goes from 2 to 5 in the same memory. The same tokens are stored; only the empty
reservations went away.

Small trace, as in p01, with $B = 4$ and 8 blocks: sequence 0 takes 10 tokens and gets blocks
$[0, 1, 2]$. Its 11th token, at position 10, is offset 2 of block 2. When block 2 fills, the next token
takes the lowest free id.

## Why it works

Attention reads the key and value of every earlier position, so the kernel must find each token's
slot. The block table is an indirection, the same idea as a page table in virtual memory. Because the
kernel reads through the table, a sequence's blocks need not be adjacent. That is what makes one
uniform pool possible.

A reservation wastes $L_{\max} - n$ slots. A paged sequence wastes $\lceil n/B \rceil B - n$, which is
always between $0$ and $B - 1$. The block size sets the trade-off: a small $B$ wastes less per sequence
but makes tables longer, and a large $B$ does the opposite.

[paper: Kwon et al. 2023, arXiv:2309.06180] describes this block-table design for LLM serving. The
corpus note on DeepSeek-V4 names the same framework and says hybrid attention violates its
assumptions [corpus: library-papers/deepseek-v4-2026.pdf p.22].

## Signal vs. noise

- **Signal:** $\lceil n/B\rceil$, the offset rule $p \bmod B$, the lowest-id pool rule, and waste only in
  the last block.
- **Signal:** the batch-size effect. Compare reserved with used slots, not total capacity.
- **Noise for now:** GPU memory layout and kernel alignment. They change constants, not bookkeeping.
- **Noise for now:** copy-on-write prefix sharing. It is a separate topic; the tables make it possible.

## Where it breaks

- **Many tiny sequences.** A one-token sequence with $B = 16$ holds 15 empty slots. Short outputs waste
  more than the average suggests.
- **Off-by-one at boundaries.** Position $16$ with $B = 16$ is block 1, offset 0. A table that grows one
  token late produces offsets one too big.
- **Non-atomic allocation.** Taking blocks one by one and then failing leaks blocks. Check the free
  count first.
- **Returning the live table.** A caller that mutates the returned list corrupts the allocator.
- **Non-uniform layers.** When layers keep different window sizes or state, blocks are no longer
  interchangeable.

## Interview angle

**60-second answer.** Reserving a maximum-length cache per request wastes memory whenever requests end
early, and the output length is unknown at the start. Paging splits the cache into fixed blocks. Each
sequence keeps a block table and takes blocks as tokens arrive. Waste is limited to one partly filled
block per sequence, and blocks are interchangeable, so no fragmentation builds up between requests.
The same memory then holds more sequences, which is the batch size.

**Follow-ups.** *What is the cost?* An indirection in the attention kernel, and a block size that
trades per-sequence waste against table length. *When the pool is full?* The allocator raises a named
error; preemption, waiting or rejection belong to the scheduler.

## Connections

This topic uses the per-token byte count from [KV-cache size arithmetic](../kv-cache-size-arithmetic/)
and the dict bookkeeping of [hash maps and sets](../../algo/hash-maps-and-sets/). It supplies the memory
side of [continuous batching](../continuous-batching/), which decides which sequences get slots. Prefix
sharing builds on the same tables in the tier-C topic on copy-on-write KV block sharing.
