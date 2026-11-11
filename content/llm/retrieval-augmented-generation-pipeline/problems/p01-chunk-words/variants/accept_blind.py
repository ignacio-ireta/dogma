def chunk_words(text, size, overlap):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    for name, value in (("size", size), ("overlap", overlap)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int")
    if size < 1:
        raise ValueError("size must be >= 1")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")
    words = text.split()
    out = []
    step = size - overlap
    start = 0
    while start < len(words):
        window = words[start : start + size]
        out.append(" ".join(window))
        if start + size >= len(words):
            break
        start += step
    return out
