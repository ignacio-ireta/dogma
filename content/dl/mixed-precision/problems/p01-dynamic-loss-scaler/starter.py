from __future__ import annotations


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
        raise NotImplementedError

    def update(self, found_inf: bool) -> bool:
        """Apply one step's outcome. Return True if the optimizer step should run."""
        raise NotImplementedError
