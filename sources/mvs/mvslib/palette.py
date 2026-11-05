"""Consistent colour convention used across every notebook in the series.

The whole point is that a colour *means the same thing* in notebook 06 as it did
in notebook 01, so the learner never has to re-learn what orange is. Import these names
instead of hard-coding hex strings.

Convention
----------
- ORIGINAL   : an original, observed variable (a column of X)
- COMPONENT  : a derived component / factor / latent direction (PCA, FA)
- FIT        : a fitted / model / predicted quantity
- RESIDUAL   : a residual / error / unique variance
- HIGHLIGHT  : whatever the current cell wants to draw the eye to
- NEUTRAL    : grid lines, de-emphasised reference objects
GROUP_COLORS : a qualitative palette for categorical groups / classes / clusters
"""

from __future__ import annotations

ORIGINAL = "#1f77b4"   # steel blue   — original variables
COMPONENT = "#ff7f0e"  # orange       — components / factors / latent axes
FIT = "#2ca02c"        # green        — fitted / model / prediction
RESIDUAL = "#d62728"   # red          — residual / error / unique variance
HIGHLIGHT = "#9467bd"  # purple       — emphasis
NEUTRAL = "#7f7f7f"    # grey         — reference / grid

# Qualitative palette for groups (species, wine classes, clusters, ...).
GROUP_COLORS = [
    "#4c72b0",  # blue
    "#dd8452",  # orange
    "#55a868",  # green
    "#c44e52",  # red
    "#8172b3",  # purple
    "#937860",  # brown
    "#da8bc3",  # pink
    "#8c8c8c",  # grey
]

# Dict form, handy for ``color=PALETTE["component"]`` style access.
PALETTE = {
    "original": ORIGINAL,
    "component": COMPONENT,
    "fit": FIT,
    "residual": RESIDUAL,
    "highlight": HIGHLIGHT,
    "neutral": NEUTRAL,
}

_LEGEND_ITEMS = [
    (ORIGINAL, "original variable"),
    (COMPONENT, "component / factor"),
    (FIT, "fitted / model"),
    (RESIDUAL, "residual / error"),
    (HIGHLIGHT, "highlight"),
]


def group_color(i: int) -> str:
    """Colour for the i-th group, cycling if there are more groups than colours."""
    return GROUP_COLORS[i % len(GROUP_COLORS)]


def show_legend(ax=None):
    """Draw the shared colour convention as a small standalone legend.

    Call this once in a notebook's header so the reader can see, at a glance,
    what each colour will mean for the rest of the notebook.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.2, 1.4))
    handles = [Patch(facecolor=c, label=lab) for c, lab in _LEGEND_ITEMS]
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=9)
    ax.axis("off")
    ax.set_title("Colour convention used in every notebook", fontsize=10)
    if created:
        plt.tight_layout()
    return ax
