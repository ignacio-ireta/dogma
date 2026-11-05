"""The local HTTP API (ARCHITECTURE §4.7, ENGINEERING_STANDARDS §8): build it with create_app."""

from dogma.api.app import create_app

__all__ = ["create_app"]
