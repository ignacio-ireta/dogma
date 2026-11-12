"""Real offline datasets, seeded draws and realistic synthetic generators for content tests.

This package is for content tests only (``content/**/test_*.py``). Submissions may not import
it, and it never imports ``dogma`` or ``dogma_harness``. ``import dogma_data`` loads only numpy
and the standard library. pandas, sklearn, statsmodels, scipy and yaml load inside the
functions that need them.
"""

from dogma_data import synth
from dogma_data.draws import sample_rows, sample_window, sampleable
from dogma_data.real import (
    RealData,
    available,
    fingerprint,
    load,
    load_frame,
    load_series,
    load_text,
    retail_events,
    retail_invoices,
)
from dogma_data.seeds import rng

__all__ = [
    "RealData",
    "available",
    "fingerprint",
    "load",
    "load_frame",
    "load_series",
    "load_text",
    "retail_events",
    "retail_invoices",
    "rng",
    "sample_rows",
    "sample_window",
    "sampleable",
    "synth",
]
