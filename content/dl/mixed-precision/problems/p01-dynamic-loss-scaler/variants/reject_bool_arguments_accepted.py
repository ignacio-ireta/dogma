from __future__ import annotations

import math


class DynamicLossScaler:
    """Wrong on purpose: accepts bool arguments as numbers."""

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        # Validate first. A bool is an int in Python, so it is rejected explicitly.
        for name, value in (
            ("init_scale", init_scale),
            ("growth_factor", growth_factor),
            ("backoff_factor", backoff_factor),
            ("growth_interval", growth_interval),
        ):
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, not a bool or other type")
        if not (math.isfinite(init_scale) and init_scale > 0):
            raise ValueError("init_scale must be a finite number greater than 0")
        if not (math.isfinite(growth_factor) and growth_factor > 1):
            raise ValueError("growth_factor must be greater than 1")
        if not 0 < backoff_factor < 1:
            raise ValueError("backoff_factor must be strictly between 0 and 1")
        if not isinstance(growth_interval, int) or growth_interval < 1:
            raise ValueError("growth_interval must be an int of at least 1")

        self.scale: float = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = growth_interval
        # Consecutive good steps since the last change of scale. Reset by both growth and backoff.
        self.good_steps: int = 0

    def update(self, found_inf: bool) -> bool:
        """Apply one step's outcome. Return True if the optimizer step should run."""
        if found_inf:
            # Overflow: the gradients are unusable, so skip the step and back off. Backing off
            # restarts the count, so a growth needs a full interval after the last overflow.
            self.scale *= self.backoff_factor
            self.good_steps = 0
            return False

        # A good step counts towards the next growth. The scale doubles only when the count
        # reaches growth_interval, and the count then restarts from zero.
        self.good_steps += 1
        if self.good_steps == self.growth_interval:
            self.scale *= self.growth_factor
            self.good_steps = 0
        return True
