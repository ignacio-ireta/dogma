from __future__ import annotations

import math


def _check_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class DynamicLossScaler:
    """Dynamic loss scale for fp16 mixed-precision training (pure Python, no tensors).

    After each backward pass, call ``update(found_inf)``. It returns True when the optimizer
    step should be applied. On overflow (``found_inf`` True) the step is skipped and ``scale`` is
    multiplied by ``backoff_factor``. After ``growth_interval`` consecutive good steps, ``scale``
    is multiplied by ``growth_factor``. ``good_steps`` counts good steps since the last change of
    ``scale``.

    Parameters
    ----------
    init_scale : finite float > 0, the starting scale.
    growth_factor : finite float > 1, multiplier applied after a run of good steps.
    backoff_factor : float strictly between 0 and 1, multiplier applied on overflow.
    growth_interval : int >= 1, number of consecutive good steps before growing.

    Raises
    ------
    ValueError
        If an argument is out of range or is a bool.
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        init_scale = _check_real("init_scale", init_scale)
        if not init_scale > 0:
            raise ValueError("init_scale must be greater than 0")
        growth_factor = _check_real("growth_factor", growth_factor)
        if not growth_factor > 1:
            raise ValueError("growth_factor must be greater than 1")
        if isinstance(backoff_factor, bool) or not isinstance(backoff_factor, (int, float)):
            raise ValueError("backoff_factor must be a real number")
        backoff_factor = float(backoff_factor)
        if not 0 < backoff_factor < 1:
            raise ValueError("backoff_factor must be strictly between 0 and 1")
        if isinstance(growth_interval, bool) or not isinstance(growth_interval, int):
            raise ValueError("growth_interval must be an int")
        if growth_interval < 1:
            raise ValueError("growth_interval must be at least 1")

        self.scale: float = init_scale
        self.good_steps: int = 0
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval

    def update(self, found_inf: bool) -> bool:
        """Apply one step's outcome. Return True if the optimizer step should run."""
        if found_inf:
            self.scale = self.scale * self._backoff_factor
            self.good_steps = 0
            return False
        self.good_steps += 1
        if self.good_steps >= self._growth_interval:
            self.scale = self.scale * self._growth_factor
            self.good_steps = 0
        return True
