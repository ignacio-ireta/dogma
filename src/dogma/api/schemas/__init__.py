"""Pydantic request and response models for the API (ENGINEERING_STANDARDS §8).

Every body is a model. Responses are mapped from domain objects in ``services/``, so the wire
format can stay stable while the domain code evolves.
"""
