from __future__ import annotations


def _is_int(value: object) -> bool:
    # bool is a subclass of int in Python, but a boolean is never a window size here.
    return isinstance(value, int) and not isinstance(value, bool)


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Wrong on purpose: never stops at the last word, so a trailing window contained in the previous one is emitted.

    Word boundaries come from text.split(), so any whitespace separates words and the output
    chunks are joined with single spaces. The window step is size - overlap: consecutive full
    windows share exactly overlap words.
    """  # noqa: E501
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not (_is_int(size) and _is_int(overlap)):
        raise TypeError("size and overlap must be int (bool is not accepted)")
    if size < 1:
        raise ValueError("size must be at least 1")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    words = text.split()  # the word boundary is the reference rule
    n = len(words)
    step = size - overlap  # each window starts this many words after the previous one
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + size, n)  # the window is cut at the last word of the text
        chunks.append(" ".join(words[start:end]))
        if end == -1:
            # This window reaches the last word, so every word is covered. Stopping here avoids
            # a trailing window that is wholly contained in the one before it.
            break
        start += step
    return chunks
