"""Matplotlib helpers for the HCR heatmap and lag-plot scripts."""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

DEFAULT_HEATMAP_ANNOTATION_FONTSIZE = 11

DEFAULT_HEATMAP_X_ROTATION = 90

def pvalue_stars(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""

def _decimals_from_fmt(fmt: str) -> int:
    match = re.search(r"\.(\d+)[fFgG]", fmt)
    return int(match.group(1)) if match else 3

def format_compact(value: float, *, decimals: int = 3, signed: bool = False) -> str:
    """Drop the leading zero: .25 instead of 0.25."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if signed:
        text = f"{value:+.{decimals}f}"
        if text.startswith("+0."):
            return "+" + text[2:]
        if text.startswith("-0."):
            return "-" + text[2:]
        return text
    text = f"{value:.{decimals}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text

def format_from_fmt(value: float, fmt: str) -> str:
    signed = fmt.strip().startswith("+")
    return format_compact(value, decimals=_decimals_from_fmt(fmt), signed=signed)

def format_p_value(p: float, decimals: int = 3) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if np.isclose(p, 1.0):
        return "1"
    cutoff = 10 ** (-decimals)
    if p < cutoff:
        return "<." + "0" * (decimals - 1) + "1"
    return format_compact(p, decimals=decimals)

def style_heatmap_axis(
    ax,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    *,
    xlabel: str = "",
    ylabel: str = "",
    show_ylabel: bool = True,
    rotation: int = DEFAULT_HEATMAP_X_ROTATION,
) -> None:
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=rotation, ha="center", va="top")
    ax.set_yticks(range(len(ylabels)))
    if show_ylabel:
        ax.set_yticklabels(ylabels)
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

def annotate_matrix_cells(
    ax,
    values: np.ndarray,
    *,
    fmt: str = ".2f",
    pvalues: np.ndarray | None = None,
    signed: bool = False,
    fontsize: int = DEFAULT_HEATMAP_ANNOTATION_FONTSIZE,
    value_fn: Callable[[float], str] | None = None,
    color_fn: Callable[[float], str] | None = None,
) -> None:
    decimals = _decimals_from_fmt(fmt)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                continue
            if value_fn is not None:
                text = value_fn(val)
            elif signed:
                text = format_compact(val, decimals=decimals, signed=True)
            else:
                text = format_from_fmt(val, fmt)
            if pvalues is not None:
                text += pvalue_stars(pvalues[i, j])
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color_fn(val) if color_fn else "black",
                fontsize=fontsize,
                clip_on=True,
                zorder=11,
            )

def pvalue_annotation_color(val: float, alpha: float = 0.05) -> str:
    if val < alpha:
        return "black"
    if alpha < 0.1 and val < 0.1:
        return "#1a1a1a"
    return "#4d4d4d"

def add_heatmap_cell_grid(ax, nrows: int, ncols: int, *, color: str = "0.78", linewidth: float = 0.9) -> None:
    for i in range(nrows):
        for j in range(ncols):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                    clip_on=False,
                    zorder=10,
                )
            )

def draw_heatmap(
    ax,
    values: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    *,
    cmap="coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm=None,
    annotate: bool = True,
    fmt: str = ".2f",
    annotation_fontsize: int = DEFAULT_HEATMAP_ANNOTATION_FONTSIZE,
    annot_formatter: Optional[Callable[[float], str]] = None,
    annot_color_fn: Optional[Callable[[float], str]] = None,
    xlabel: str = "",
    ylabel: str = "",
    show_ylabel: bool = True,
    grid: bool = True,
    rotation: int = DEFAULT_HEATMAP_X_ROTATION,
):
    ax.grid(False)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, aspect="equal")
    style_heatmap_axis(
        ax, xlabels, ylabels, xlabel=xlabel, ylabel=ylabel, show_ylabel=show_ylabel, rotation=rotation
    )
    if grid:
        add_heatmap_cell_grid(ax, values.shape[0], values.shape[1])
    if annotate:
        annotate_matrix_cells(
            ax,
            values,
            fmt=fmt,
            fontsize=annotation_fontsize,
            value_fn=annot_formatter,
            color_fn=annot_color_fn,
        )
    return im

def setup(context: str = "paper") -> None:
    sns.set_theme(context=context, style="whitegrid", palette="colorblind")
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 15,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
