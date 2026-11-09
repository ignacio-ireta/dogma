"""Deterministic, salted random generators keyed by a string.

``rng(key)`` hashes the key with sha256. The grader never sets ``DOGMA_DATA_SALT``, so the
draws are fixed. The exercise audit sets the salt to check that a test does not pass only for
one lucky data draw.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

SALT_ENV = "DOGMA_DATA_SALT"


def rng(key: str) -> np.random.Generator:
    """Return a fresh generator for ``key``; the same key (and salt) always gives the same draws."""
    if not isinstance(key, str):
        raise ValueError(f"rng key must be a str, got {type(key).__name__}")
    if key == "":
        raise ValueError("rng key must be a non-empty str")
    material = key
    salt = os.environ.get(SALT_ENV, "")
    if salt:
        material = f"{key}\x00{salt}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:16], "big")
    return np.random.default_rng(seed)
