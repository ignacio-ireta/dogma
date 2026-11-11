from __future__ import annotations

END = "</w>"


def train_bpe_merges(word_counts, num_merges):  # noqa: PLR0912
    if type(word_counts) is not dict:
        raise TypeError("word_counts must be dict")
    if type(num_merges) is not int:
        raise TypeError("num_merges must be int")
    if num_merges < 0:
        raise ValueError("negative")
    for w, c in word_counts.items():
        if type(w) is not str:
            raise TypeError("key must be str")
        if type(c) is not int:
            raise TypeError("count must be int")
        if w == "" or any(ch.isspace() for ch in w) or END in w:
            raise ValueError("bad word")
        if c < 1:
            raise ValueError("count < 1")
    words = [([*w, END], c) for w, c in word_counts.items()]
    merges = []
    for _ in range(num_merges):
        counts = {}
        for syms, c in words:
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                counts[p] = counts.get(p, 0) + c
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)
        l, r = best
        new_words = []
        for syms, c in words:
            out, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == l and syms[i + 1] == r:
                    out.append(l + r)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            new_words.append((out, c))
        words = new_words
    return merges
