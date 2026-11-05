"""Liveness for the web client (I-15): the app is up and which version answered."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
