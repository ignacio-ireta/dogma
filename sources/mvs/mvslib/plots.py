"""Reusable plot helpers shared across notebooks.

Every helper (a) sets axis labels, and (b) accepts a ``question=`` string that
becomes the title — enforcing the rule that *every plot answers a named
question*. They use the shared colour convention from :mod:`mvslib.palette`.
"""

from __future__ import annotations

import numpy as np

from . import palette as P


def _ax(ax, figsize=(5, 4)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def cov_ellipse(mean, cov, ax=None, n_std=2.0, edgecolor=P.HIGHLIGHT,
                facecolor="none", lw=2, question=None, **kw):
    """Draw the covariance ellipse {x : (x-mean)' cov^-1 (x-mean) = n_std^2}.

    The ellipse axes are the eigenvectors of ``cov``; the semi-axis lengths are
    ``n_std * sqrt(eigenvalue)``. This is the picture behind Mahalanobis
    distance and the shape PCA reads off the data.
    """
    from matplotlib.patches import Ellipse

    ax = _ax(ax)
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell = Ellipse(xy=tuple(mean), width=width, height=height, angle=angle,
                  edgecolor=edgecolor, facecolor=facecolor, lw=lw, **kw)
    ax.add_patch(ell)
    if question:
        ax.set_title(question)
    return ell


def scree_plot(eigenvalues, ax=None, question=None, threshold=None):
    """Scree + cumulative-variance plot from a vector of eigenvalues.

    Bars = proportion of variance per component (orange = components).
    Line = cumulative proportion. Optional ``threshold`` draws a target line
    (e.g. 0.9) to support the cumulative-variance rule for choosing k.
    """
    ax = _ax(ax)
    ev = np.asarray(eigenvalues, dtype=float)
    ratio = ev / ev.sum()
    cum = np.cumsum(ratio)
    idx = np.arange(1, len(ev) + 1)
    ax.bar(idx, ratio, color=P.COMPONENT, alpha=0.85, label="per component")
    ax.plot(idx, cum, "-o", color=P.HIGHLIGHT, label="cumulative")
    if threshold is not None:
        ax.axhline(threshold, color=P.NEUTRAL, ls="--", lw=1,
                   label=f"{threshold:.0%} target")
    ax.set_xlabel("component number")
    ax.set_ylabel("proportion of variance")
    ax.set_xticks(idx)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    if question:
        ax.set_title(question)
    return ax


def biplot(scores, loadings, feature_names, ax=None, pc=(0, 1),
           groups=None, question=None, arrow_scale=None):
    """Biplot: sample scores (points) + variable loadings (orange arrows).

    Points are observations projected onto two components; arrows show how each
    original variable aligns with those components. Long arrows in the same
    direction as a cluster of points = variables that drive that region.
    """
    ax = _ax(ax, figsize=(6, 5))
    scores = np.asarray(scores, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    xs, ys = scores[:, pc[0]], scores[:, pc[1]]

    if groups is None:
        ax.scatter(xs, ys, s=16, color=P.ORIGINAL, alpha=0.6)
    else:
        groups = np.asarray(groups)
        for i, g in enumerate(np.unique(groups)):
            m = groups == g
            ax.scatter(xs[m], ys[m], s=18, color=P.group_color(i),
                       alpha=0.75, label=str(g))
        ax.legend(fontsize=8, title="group")

    if arrow_scale is None:
        span = max(np.abs(xs).max(), np.abs(ys).max())
        lmax = np.abs(loadings[:, list(pc)]).max()
        arrow_scale = 0.8 * span / lmax if lmax > 0 else 1.0
    for i, name in enumerate(feature_names):
        dx, dy = loadings[i, pc[0]] * arrow_scale, loadings[i, pc[1]] * arrow_scale
        ax.arrow(0, 0, dx, dy, color=P.COMPONENT, alpha=0.85,
                 head_width=0.02 * arrow_scale * 3, length_includes_head=True)
        ax.text(dx * 1.08, dy * 1.08, name, color=P.COMPONENT, fontsize=8,
                ha="center", va="center")

    ax.axhline(0, color=P.NEUTRAL, lw=0.6)
    ax.axvline(0, color=P.NEUTRAL, lw=0.6)
    ax.set_xlabel(f"component {pc[0] + 1}")
    ax.set_ylabel(f"component {pc[1] + 1}")
    if question:
        ax.set_title(question)
    return ax


def dendrogram_plot(Z, ax=None, labels=None, color_threshold=None,
                    question=None, truncate=None):
    """Wrapper around scipy's dendrogram with consistent styling.

    ``Z`` is a scipy linkage matrix. ``color_threshold`` is the height at which
    to cut for colouring clusters; the y-axis is the linkage distance at which
    clusters merge — read it bottom-up.
    """
    from scipy.cluster.hierarchy import dendrogram

    ax = _ax(ax, figsize=(7, 4))
    kw = dict(ax=ax, color_threshold=color_threshold, above_threshold_color=P.NEUTRAL)
    if labels is not None:
        kw["labels"] = list(labels)
    if truncate is not None:
        kw["truncate_mode"] = "lastp"
        kw["p"] = truncate
    dendrogram(Z, **kw)
    if color_threshold is not None:
        ax.axhline(color_threshold, color=P.RESIDUAL, ls="--", lw=1,
                   label=f"cut at {color_threshold:.2g}")
        ax.legend(fontsize=8)
    ax.set_xlabel("observations (or merged clusters)")
    ax.set_ylabel("merge distance")
    if question:
        ax.set_title(question)
    return ax


def decision_boundary(clf, X, y, ax=None, question=None, resolution=300,
                      class_names=None):
    """Filled decision regions of a fitted 2-D classifier + the data points.

    ``X`` must be 2 columns. ``y`` may be integer or string labels; points are
    coloured by the group palette. Use this to compare LDA / QDA / logistic
    boundaries on the same data.
    """
    from matplotlib.colors import ListedColormap

    ax = _ax(ax, figsize=(5.2, 4.6))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    to_idx = {c: i for i, c in enumerate(classes)}

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = clf.predict(grid)
    Z = np.array([to_idx[p] for p in pred]).reshape(xx.shape)

    cmap = ListedColormap([P.group_color(i) for i in range(len(classes))])
    ax.contourf(xx, yy, Z, levels=np.arange(len(classes) + 1) - 0.5,
                alpha=0.18, cmap=cmap)
    for i, c in enumerate(classes):
        m = y == c
        label = (class_names[i] if class_names is not None else str(c))
        ax.scatter(X[m, 0], X[m, 1], s=22, color=P.group_color(i),
                   edgecolor="k", linewidth=0.4, label=label)
    ax.legend(fontsize=8, title="group")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    if question:
        ax.set_title(question)
    return ax


def confusion(y_true, y_pred, ax=None, labels=None, question=None, normalize=False):
    """Confusion-matrix heatmap with cell counts.

    Rows = true class, columns = predicted class. The diagonal is correct
    predictions. ``normalize=True`` shows row-proportions (recall per class).
    """
    from sklearn.metrics import confusion_matrix

    ax = _ax(ax, figsize=(4.4, 4))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = cm.astype(float)
    if normalize:
        disp = disp / disp.sum(axis=1, keepdims=True)
    im = ax.imshow(disp, cmap="Blues")
    ticks = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(range(len(ticks)), ticks, rotation=45, ha="right")
    ax.set_yticks(range(len(ticks)), ticks)
    thresh = disp.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thresh else "black")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    if question:
        ax.set_title(question)
    return ax
