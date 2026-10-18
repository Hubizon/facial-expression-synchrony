"""Matplotlib helpers for the analysis notebooks."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

DEFAULT_HEATMAP_ANNOTATION_FONTSIZE = 11
PVALUE_HEATMAP_ANNOTATION_FONTSIZE = 9
DEFAULT_HEATMAP_X_ROTATION = 90

# Set True to show figure / subplot titles in exports.
SHOW_TITLES = False

PERIOD_MONTH = {1: "Month 4", 2: "Month 7", 3: "Month 12"}


def month_label(period: int) -> str:
    return PERIOD_MONTH.get(period, f"Period {period}")


def comparison_month_label(period1: int, period2: int) -> str:
    return f"{month_label(period1)} vs {month_label(period2)}"


def pvalue_stars(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return ""


def ax_title(ax, text: str, **kwargs) -> None:
    if SHOW_TITLES:
        ax.set_title(text, **kwargs)


def fig_suptitle(fig, text: str, **kwargs) -> None:
    if SHOW_TITLES:
        fig.suptitle(text, **kwargs)


def period_panel_label(ax, period: int, text: str | None = None) -> None:
    if text is None:
        text = f"Period {period}" if SHOW_TITLES else month_label(period)
    ax.set_title(text, fontsize=10, fontweight="bold", pad=6)


def finish_figure(fig, show: bool = True):
    if show:
        plt.show()
        return None
    return fig


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
    cutoff = 10 ** (-decimals)
    if p < cutoff:
        return "<." + "0" * (decimals - 1) + "1"
    return format_compact(p, decimals=decimals)


def clean_au_labels(labels: Sequence[str]) -> list[str]:
    return [str(label).removesuffix("_r") for label in labels]


def create_shared_heatmap_axes(ncols: int, *, figsize=(10.5, 3.6), sharey: bool = True):
    fig, axes = plt.subplots(1, ncols, figsize=figsize, sharey=sharey, constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for ax in axes:
        ax.grid(False)
    return fig, axes


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


def finalize_shared_yaxis(axes, ylabels: Sequence[str], ylabel: str = "") -> None:
    axes = axes.ravel().tolist() if isinstance(axes, np.ndarray) else list(axes)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_yticks(range(len(ylabels)))
    axes[0].set_yticklabels(list(ylabels))
    axes[0].set_ylabel(ylabel)


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


def pvalue_discrete_cmap(alpha: float = 0.05):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    alpha_label = format_compact(alpha, decimals=2)
    marginal = 0.1
    sig_color, marginal_color, ns_color = "#D55E00", "#56B4E9", "#E8E8E8"

    if alpha < marginal:
        bounds = [0.0, alpha, marginal, 1.0]
        colors = [sig_color, marginal_color, ns_color]
        ticks = [alpha / 2, (alpha + marginal) / 2, (marginal + 1.0) / 2]
        labels = [f"p < {alpha_label}", f"{alpha_label} ≤ p < .1", "p ≥ .1"]
    else:
        bounds = [0.0, alpha, 1.0]
        colors = [sig_color, ns_color]
        ticks = [alpha / 2, (alpha + 1.0) / 2]
        labels = [f"p < {alpha_label}", f"p ≥ {alpha_label}"]

    return ListedColormap(colors), BoundaryNorm(bounds, len(colors)), ticks, labels


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


def locate_analysis_dir() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    for p in (here, cwd, cwd / "analysis", cwd / "dep-detection" / "analysis"):
        if p.is_dir() and (p / "plot_style.py").exists() and (p / "analysis_config.py").exists():
            return p.resolve()
    raise FileNotFoundError(f"Cannot locate analysis/. cwd={cwd}")


def bootstrap_notebook(context: str = "paper") -> tuple[Path, Path]:
    """Add analysis/ to sys.path, apply style, ensure PDF dir."""
    analysis_dir = locate_analysis_dir()
    analysis_str = str(analysis_dir)
    if analysis_str not in sys.path:
        sys.path.insert(0, analysis_str)
    setup(context=context)

    import analysis_config as cfg

    pdf_dir = ensure_pdf_dir(analysis_dir, cfg.PDF_SUBDIR)
    return analysis_dir, pdf_dir


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


def ensure_pdf_dir(base_dir: str | Path, subdir: str) -> Path:
    out = Path(base_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_pdf(fig: mpl.figure.Figure, name: str, pdf_dir: Path | None = None) -> Path:
    if pdf_dir is None:
        _, pdf_dir = bootstrap_notebook()
    path = pdf_dir / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    return path
