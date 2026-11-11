from __future__ import annotations

END = "</w>"


class BPETokenizer:
    def __init__(self, merges):
        if type(merges) is not list:
            raise TypeError("merges must be a list")
        self.rank = {}
        for i, m in enumerate(merges):
            if (
                type(m) is not tuple
                or len(m) != 2
                or type(m[0]) is not str
                or type(m[1]) is not str
            ):
                raise TypeError("bad merge")
            if m[0] == "" or m[1] == "":
                raise ValueError("empty symbol")
            if m in self.rank:
                raise ValueError("duplicate merge")
            self.rank[m] = i

    def encode(self, word):
        if type(word) is not str:
            raise TypeError("word must be str")
        if any(c.isspace() for c in word) or END in word:
            raise ValueError("bad word")
        syms = [*word, END]
        while True:
            best = None
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                if p in self.rank and (best is None or self.rank[p] < best[0]):
                    best = (self.rank[p], p)
            if best is None:
                return syms
            l, r = best[1]
            out, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == l and syms[i + 1] == r:
                    out.append(l + r)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out

    def decode(self, tokens):
        if type(tokens) is not list or any(type(t) is not str for t in tokens):
            raise TypeError("tokens must be list of str")
        text = "".join(tokens)
        if not text.endswith(END):
            raise ValueError("missing marker")
        word = text[: -len(END)]
        if END in word:
            raise ValueError("misplaced marker")
        return word
