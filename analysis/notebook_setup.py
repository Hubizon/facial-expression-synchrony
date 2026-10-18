"""Notebook bootstrap. Call ``init()`` once at the top of each notebook."""

from __future__ import annotations

import sys
from pathlib import Path


def locate_analysis_dir() -> Path:
    """Find this folder (dep-detection/analysis/) from cwd or __file__."""
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    for path in (here, cwd, cwd / "analysis", cwd / "dep-detection" / "analysis"):
        if path.is_dir() and (path / "analysis_config.py").exists():
            return path.resolve()
    raise FileNotFoundError(
        f"Cannot locate analysis/. cwd={cwd}. "
        "Set the kernel working directory to dep-detection/analysis."
    )


def init(context: str = "paper") -> tuple[Path, Path]:
    """Add analysis/ to sys.path, apply plot style, create PDF output dir."""
    analysis_dir = locate_analysis_dir()
    analysis_str = str(analysis_dir)
    if analysis_str not in sys.path:
        sys.path.insert(0, analysis_str)

    from plot_style import bootstrap_notebook

    return bootstrap_notebook(context=context)
