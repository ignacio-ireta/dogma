"""``import dogma_data`` and ``dogma_data.synth`` must not load heavy or non-stdlib libraries."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("pandas", "scipy", "sklearn", "statsmodels", "yaml")
SNIPPET = f"""
import sys
import dogma_data
from dogma_data import synth
heavy = {HEAVY!r}
loaded = sorted(m for m in sys.modules if m.split(".")[0] in heavy)
print("LOADED:" + ",".join(loaded))
"""


def test_import_loads_no_heavy_libraries() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    line = next(x for x in result.stdout.splitlines() if x.startswith("LOADED:"))
    assert line == "LOADED:", f"import dogma_data loaded {line[len('LOADED:') :]}"
