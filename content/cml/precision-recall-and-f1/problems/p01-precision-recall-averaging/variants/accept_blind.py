from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def _scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    return _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(2 * tp, 2 * tp + fp + fn)


def precision_recall_f1(  # noqa: PLR0912
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    average: str = "binary",
    pos_label: Any = 1,
) -> tuple[float, float, float]:
    if average not in ("binary", "micro", "macro", "weighted"):
        raise ValueError("unknown average")
    yt = list(y_true)
    yp = list(y_pred)
    if len(yt) != len(yp):
        raise ValueError("length mismatch")
    if not yt:
        raise ValueError("empty input")
    labels = set(yt) | set(yp)
    if average == "binary" and len(labels) > 2:
        raise ValueError("binary needs at most two labels")
    counts = {c: [0, 0, 0] for c in labels}
    for t, p in zip(yt, yp):  # noqa: B905
        if t == p:
            counts[t][0] += 1
        else:
            counts[p][1] += 1
            counts[t][2] += 1
    if average == "binary":
        tp, fp, fn = counts.get(pos_label, [0, 0, 0])
        return _scores(tp, fp, fn)
    if average == "micro":
        tp = sum(v[0] for v in counts.values())
        fp = sum(v[1] for v in counts.values())
        fn = sum(v[2] for v in counts.values())
        return _scores(tp, fp, fn)
    try:
        classes = sorted(labels)
    except TypeError:
        classes = list(labels)
    per = [(_scores(*counts[c]), counts[c][0] + counts[c][2]) for c in classes]
    if average == "macro":
        w = [1.0] * len(per)
    else:
        w = [float(s) for _, s in per]
    tot = sum(w)
    out = []
    for k in range(3):
        out.append(sum(wi * sc[k] for (sc, _), wi in zip(per, w)) / tot if tot else 0.0)  # noqa: B905
    return (float(out[0]), float(out[1]), float(out[2]))
