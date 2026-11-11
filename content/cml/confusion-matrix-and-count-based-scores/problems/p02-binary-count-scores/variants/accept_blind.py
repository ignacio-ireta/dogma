from __future__ import annotations

import math
import numbers


def _c(name, v):
    if not isinstance(v, numbers.Integral) or isinstance(v, bool):
        raise ValueError(name)
    if v < 0:
        raise ValueError(name)
    return int(v)


def _div(a, b):
    return a / b if b else 0.0


def binary_scores(tp, fp, fn, tn):
    tp, fp, fn, tn = _c("tp", tp), _c("fp", fp), _c("fn", fn), _c("tn", tn)
    total = tp + fp + fn + tn
    if total == 0:
        raise ValueError("all zero")
    pp, ap, pn, an = tp + fp, tp + fn, tn + fn, fp + tn
    den = pp * ap * pn * an
    mcc = (tp * tn - fp * fn) / math.sqrt(den) if den else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": _div(tp, pp),
        "recall": _div(tp, ap),
        "specificity": _div(tn, an),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "mcc": mcc,
    }
