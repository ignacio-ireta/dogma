"""mvslib — shared utilities for the Multivariate Statistics notebook series.

Small on purpose. It only holds things that genuinely recur across notebooks:
a fixed colour convention, a handful of plot helpers, cached dataset loaders,
and reproducibility/self-check helpers. Anything used once stays inline in its
notebook.

Typical notebook header:

    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    from mvslib import palette as P
    from mvslib import plots, data
    from mvslib.check import set_seed, assert_close, check
    RNG = set_seed(7)
"""

from . import palette, plots, data, check

__all__ = ["palette", "plots", "data", "check"]
