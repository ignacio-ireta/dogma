"""Authoring helpers for building the notebooks programmatically.

Notebooks are generated from the scripts in this folder so that their content
lives in readable Python triple-quoted strings (easy to write, easy to keep
consistent) instead of hand-edited .ipynb JSON. The generated .ipynb files are
the deliverable the learner works in. Re-running a generator OVERWRITES its notebook,
so do not regenerate after answers have been filled in.

Shared section banners and conventions are defined here so every notebook looks
and behaves the same.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parent.parent


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


# ---- shared section banners (markdown) -------------------------------------

def stage(emoji: str, title: str, subtitle: str = "") -> str:
    line = f"## {emoji} {title}"
    return line + (f"\n\n*{subtitle}*" if subtitle else "")


def retrieve(items: list[str], note: str = "") -> list:
    """Stage 1 — spaced retrieval. `items` are questions; answers live in the key."""
    body = stage("🔁", "Retrieve", "Warm-up from earlier notebooks. "
                 "Answer from memory — do not scroll up. Answers are in the key at the bottom.")
    qs = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(items))
    extra = f"\n\n{note}" if note else ""
    return [md(body + "\n\n" + qs + extra),
            md("**YOUR ANSWERS:**\n\n*(type here — words, numbers, or a line of Python)*")]


def predict(items: list[str]) -> list:
    """Stage 2 — generation / pre-test."""
    body = stage("🔮", "Predict", "Guess before you are taught. Wrong guesses are useful — "
                 "this is a generation exercise, not a test of worth. Predicting first makes "
                 "the explanation stick.")
    qs = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(items))
    return [md(body + "\n\n" + qs),
            md("**YOUR PREDICTIONS:**\n\n*(type here)*")]


def learn(title: str = "Learn the system") -> object:
    return md(stage("🧩", title, "The method as a system: question → data → objective → "
                    "assumptions → fit → diagnostics → interpretation → traps."))


def practice() -> object:
    return md(stage("🛠️", "Practice", "Interleaved: a little concept, a little hand-calc, a "
                    "little code, a little interpretation, a little choosing. One difficulty at a time."))


def close() -> object:
    return md(stage("✅", "Close", "Produce something exam-shaped, then fill the method card "
                    "from memory."))


def answer_key_banner() -> object:
    return md(
        "---\n\n"
        "# 🔑 Answer key\n\n"
        "> **Stop.** Everything below is the answer key. Attempt the exercises above first — "
        "the value is in the attempt, not the answer. Scroll down only to check.\n\n"
        "---")


# ---- method card template ---------------------------------------------------

METHOD_CARD_BLANK = """\
### 🃏 Method card — fill from memory

Fill each line in your own words. Check against the key only after attempting.

| Field | Your answer |
|---|---|
| **Method** |  |
| **Research question it answers** |  |
| **Supervised or unsupervised** |  |
| **Required input (data shape)** |  |
| **Mathematical objective** |  |
| **Main assumptions** |  |
| **Important output** |  |
| **Diagnostic checks** |  |
| **How to choose model size / complexity** |  |
| **Common trap** |  |
| **One-sentence interpretation** |  |
| **Closest competing method & why it differs** |  |
"""


def build(filename: str, cells: list) -> Path:
    """Write a notebook to the repo root and return its path."""
    nb = new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    out = REPO / filename
    nbf.write(nb, str(out))
    print(f"wrote {out.relative_to(REPO)}  ({len(cells)} cells)")
    return out


def header(title: str, est_time: str, objectives: list[str], prereq: str,
           mvp: str, extra: str = "") -> object:
    """The standard notebook header cell (stage 0)."""
    objs = "\n".join(f"- {o}" for o in objectives)
    body = f"""\
# {title}

**Estimated time:** {est_time}  ·  **Prerequisites:** {prereq}

**Minimum-viable-pass path (low-energy day):** {mvp}

### Learning objectives
By the end you should be able to:
{objs}

### How this notebook works
Five stages, every time: **🔁 Retrieve → 🔮 Predict → 🧩 Learn → 🛠️ Practice → ✅ Close**, then a
separated **🔑 Answer key**. Type answers in the *YOUR ANSWER* cells — plain words, numbers, or
Python are all fine. You never have to write LaTeX. `# TODO` code cells have escalating hints
(concept → formula → code) in the comments.
{extra}"""
    return md(body)


SETUP_CELL = """\
# Setup — run me first. Fixed seed everywhere makes results reproducible.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mvslib import palette as P
from mvslib import plots, data
from mvslib.check import set_seed, assert_close, check, selfcheck

RNG = set_seed(7)
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
P.show_legend(); plt.show()
"""
