from __future__ import annotations


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of size words, advancing by size - overlap words.

    text is a str; words are the maximal non-whitespace runs (text.split()). size is an int >= 1
    and overlap an int with 0 <= overlap < size. Windows start at word 0 and advance by
    size - overlap; stop after the first window that reaches the last word. Return one
    space-joined string per window, in order. Text with no words returns []. Raise TypeError for
    a non-str text or non-int size/overlap (bool is rejected), and ValueError for size < 1 or
    overlap outside [0, size). Never mutate inputs.
    """
    raise NotImplementedError
