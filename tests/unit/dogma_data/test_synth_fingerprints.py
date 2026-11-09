"""Golden fingerprints for every synth generator at fixed knobs.

A change here means a generator changed output (a numpy or code change), which changes the
numbers that lessons quote. Fingerprints protect lesson numbers only; correctness tests never
depend on them. To update after an intended change, rerun the generators and re-pin, and bump
the version of every problem listed by ``grep -rl dogma_data content/``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from dogma_data import rng, synth
from dogma_data.seeds import SALT_ENV


@pytest.fixture(autouse=True)
def _unsalted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Golden values are for the unsalted stream; the exercise audit sets a salt on purpose.
    monkeypatch.delenv(SALT_ENV, raising=False)


def _digest(values: NDArray[np.generic]) -> str:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        payload = arr.astype("int64").tobytes()
    elif arr.dtype.kind in "US":
        payload = "\x1f".join(str(v) for v in arr.tolist()).encode("utf-8")
    elif np.issubdtype(arr.dtype, np.floating):
        payload = np.round(arr, 9).tobytes()
    else:
        payload = arr.astype(np.int64).tobytes()
    header = f"{arr.shape}|{arr.dtype}|".encode()
    return hashlib.sha256(header + payload).hexdigest()


def _event_log_digests() -> dict[str, str]:
    columns = synth.event_log(
        rng("fingerprint:event_log"),
        300,
        n_users=20,
        dirty_rate=0.1,
        duplicate_rate=0.05,
    )
    return {key: _digest(value) for key, value in columns.items()}


CASES: dict[str, Callable[[], NDArray[np.generic]]] = {
    "correlated_mixed_scale": lambda: synth.correlated_mixed_scale(
        rng("fingerprint:correlated_mixed_scale"), 50, 4
    ),
    "heavy_tailed_noise": lambda: synth.heavy_tailed_noise(
        rng("fingerprint:heavy_tailed_noise"),
        100,
        df=3.0,
        scale=2.0,
        spread=np.linspace(0.5, 2.0, 100),
    ),
    "imbalanced_labels": lambda: synth.imbalanced_labels(
        rng("fingerprint:imbalanced_labels"), 500, positive_rate=0.02, flip_rate=0.01
    ),
    "informative_scores": lambda: synth.informative_scores(
        rng("fingerprint:informative_scores"),
        np.arange(200) % 2,
        auc=0.85,
        decimals=3,
    ),
    "seasonal_ar_series": lambda: synth.seasonal_ar_series(
        rng("fingerprint:seasonal_ar_series"),
        120,
        period=12,
        amplitude=3.0,
        trend=0.01,
        phi=(0.6,),
        level_shifts=2,
        missing_rate=0.05,
        start=10.0,
    ),
    "latencies": lambda: synth.latencies(rng("fingerprint:latencies"), 200),
    "outlier_channel_activations": lambda: synth.outlier_channel_activations(
        rng("fingerprint:outlier_channel_activations"), 16, 8
    ),
    "peaked_probs": lambda: synth.peaked_probs(
        rng("fingerprint:peaked_probs"), np.arange(60) % 3, 3
    ),
}

PINNED: dict[str, str] = {
    "correlated_mixed_scale": "39def4da01649fe7712ce2ddd264ae4a4c6f6556efaea07e1ca8f89ef4e27b41",
    "heavy_tailed_noise": "d28160a1f8e7cb81b7037ea47bc1c0ab50b3b69f754d7d781617dbfff592e09b",
    "imbalanced_labels": "012d87a3f521d2777a56d2edbd9bf28c34e099a7653d4ff44756b339742bb2e8",
    "informative_scores": "8b527e84778dae90591d249d2b2c3eada43e5374e230b31d41daa128c7f52a8c",
    "seasonal_ar_series": "0983294916e8da0d83e85753462182dd7d8c4c67da4b3f65569e8485443841a9",
    "latencies": "dab97cf33bef4827ab9f438c84ab977702ff8833fb6ecbd89bb3b4a367a47ed1",
    "outlier_channel_activations": (
        "28adcd054bbd250add0988c448ec7553f7113d1e589dafffc1785c25dbe8d99f"
    ),
    "peaked_probs": "16a2690380f164009a4b5c9f045550122b6d9a65d00d212c2884c25d09529d14",
}

EVENT_LOG_PINNED: dict[str, str] = {
    "event_id": "0d6801a8467d419bc698d6362b6117294a26f9346bd564d5e89aff08935f18bf",
    "user_id": "1f769e0520b495c26d4ece7b0f9559bc4ff0313e1a2c8c0f1a719ad8318372e1",
    "ts": "efdf21e1dc779d0d34c5667b1ed6c58e1f2ade65f2f98c82f1276125a961c4fd",
    "amount": "c1fb484ba43f74e090e9fe4f84c9a7b3b689b992919e095a5b5276f9dd01e581",
    "category": "1a19db57612990aa90ca760c715f15c02b451728b8850c969e66edd116d275ca",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_generator_fingerprint(name: str) -> None:
    got = _digest(CASES[name]())
    assert got == PINNED[name], (
        f"{name} output changed (got {got}). If the change is intended, re-pin and bump the "
        "version of problems that use dogma_data (grep -rl dogma_data content/)."
    )


def test_event_log_fingerprint_per_column() -> None:
    assert _event_log_digests() == EVENT_LOG_PINNED
