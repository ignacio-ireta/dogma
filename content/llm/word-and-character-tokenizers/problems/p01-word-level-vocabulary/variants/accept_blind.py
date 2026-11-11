from __future__ import annotations

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class WordTokenizer:
    def __init__(self, corpus, min_count=1):
        if type(min_count) is not int:
            raise TypeError("min_count int")
        if min_count < 1:
            raise ValueError("min_count < 1")
        if type(corpus) is not list:
            raise TypeError("corpus list")
        for d in corpus:
            if type(d) is not str:
                raise TypeError("entry str")
        counts = {}
        for d in corpus:
            for w in d.split():
                if w in SPECIALS:
                    continue
                counts[w] = counts.get(w, 0) + 1
        kept = sorted((w for w in counts if counts[w] >= min_count), key=lambda w: (-counts[w], w))
        self.tokens = SPECIALS + kept
        self.ids = {w: i + 4 for i, w in enumerate(kept)}

    def encode(self, text):
        if type(text) is not str:
            raise TypeError("text str")
        return [self.ids.get(w, 1) for w in text.split()]

    def decode(self, ids):
        if type(ids) is not list:
            raise TypeError("ids list")
        for i in ids:
            if type(i) is not int:
                raise TypeError("id int")
            if not 0 <= i < len(self.tokens):
                raise ValueError("id range")
        return " ".join(self.tokens[i] for i in ids)
