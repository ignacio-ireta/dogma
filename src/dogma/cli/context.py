"""Plumbing shared by every command: settings from the Typer context, consoles, friendly errors.

A DogmaError or StorageError prints one line, ``error [CODE] message``, and exits 1. A CliError (a
usage problem the learner can fix) prints ``error: message`` and exits 2. Anything else is a bug and
keeps its traceback (ENGINEERING_STANDARDS §4.4).
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import typer
from rich.console import Console
from rich.markup import escape

from dogma.core.config import Settings
from dogma.core.errors import DogmaError
from dogma.db.errors import StorageError


class CliError(Exception):
    """A usage problem the learner can fix: a missing workspace file, a bad option value."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def settings_of(ctx: typer.Context) -> Settings:
    """The Settings the root command stored in the context (tests inject their own)."""
    settings = ctx.find_root().obj
    if not isinstance(settings, Settings):
        raise RuntimeError("the dogma CLI was started without Settings in its context")
    return settings


def out() -> Console:
    return Console(highlight=False)


def err() -> Console:
    return Console(stderr=True, highlight=False)


def friendly_errors[**P](command: Callable[P, None]) -> Callable[P, None]:
    """Turn domain and usage errors into one-line messages with the matching exit code."""

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except DogmaError as exc:
            # The code is escaped: Rich would read "[CONTENT_NOT_FOUND]" as a style tag.
            err().print(
                f"[bold red]error[/] {escape(f'[{exc.code.value}]')} {escape(exc.message)}",
                soft_wrap=True,
            )
            raise typer.Exit(1) from None
        except CliError as exc:
            err().print(f"[bold red]error[/] {escape(exc.message)}", soft_wrap=True)
            raise typer.Exit(2) from None
        except StorageError as exc:
            err().print(
                f"[bold red]error[/] {escape(f'[{exc.code}]')} {escape(exc.message)}",
                soft_wrap=True,
            )
            raise typer.Exit(1) from None

    return wrapper
