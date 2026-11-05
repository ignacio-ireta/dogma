"""Tiny helpers for reproducibility and for self-checking computations.

These exist so that exercises can verify a hand/NumPy computation against a
library result *inside the notebook*, giving the learner immediate, deterministic
feedback ("your number matches scikit-learn") without revealing the method.
"""

from __future__ import annotations

import numpy as np


def set_seed(seed: int = 7) -> np.random.Generator:
    """Seed every source of randomness we use and return a NumPy Generator.

    Usage:  RNG = set_seed(7)
    Then draw with RNG (e.g. RNG.normal(...)). Also seeds the legacy global
    NumPy RNG and the stdlib ``random`` module so anything that ignores the
    Generator is still reproducible.
    """
    import random

    np.random.seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed)


def assert_close(a, b, tol: float = 1e-8, name: str = "value") -> None:
    """Assert two arrays/scalars agree within ``tol``; raise a helpful message.

    Used to confirm that a from-scratch formula and a library function produce
    the same answer on a small verification case.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise AssertionError(
            f"{name}: shapes differ {a.shape} vs {b.shape} — "
            "check you are comparing the same object."
        )
    diff = np.max(np.abs(a - b)) if a.size else 0.0
    if not np.allclose(a, b, atol=tol, rtol=0.0):
        raise AssertionError(
            f"{name}: not equal within tol={tol:g}. Max abs difference = {diff:.3g}.\n"
            f"  got     : {np.ravel(a)[:6]}{' ...' if a.size > 6 else ''}\n"
            f"  expected: {np.ravel(b)[:6]}{' ...' if b.size > 6 else ''}"
        )
    print(f"OK  {name}: matches within tol={tol:g} (max abs diff {diff:.2g}).")


def check(condition: bool, ok_msg: str = "OK", fail_msg: str = "Not yet") -> None:
    """Print a friendly pass/fail line for a boolean self-check."""
    print(f"OK  {ok_msg}" if condition else f"--  {fail_msg}")
    if not condition:
        raise AssertionError(fail_msg)


def selfcheck(attempted: bool, verify, blank_msg: str = None) -> None:
    """Run a self-check only once the student has actually attempted the TODO.

    This is what lets a *blank* notebook run top-to-bottom in a clean kernel
    (the delivered state) while still giving deterministic feedback once an
    answer is entered:

    - ``attempted`` False  -> print a reminder, do NOT raise (notebook stays runnable).
    - ``attempted`` True   -> call ``verify()`` (a zero-arg function holding the
      assert); a *wrong* answer then raises a real AssertionError.

    Example
    -------
        selfcheck(np.any(pc1_by_hand),
                  lambda: assert_close(pc1_by_hand, pc1_true, name="PC1"))
    """
    if not attempted:
        print("·  " + (blank_msg or
              "TODO not filled in yet — replace the placeholder(s) and re-run to self-check."))
        return
    verify()
