"""Shared settings for the analysis notebooks."""

from __future__ import annotations

import os
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent


def _find_dependencies_root() -> Path:
    """Walk up from this file until we find the folder with questionnaire / manual data."""
    preferred = ("...",)
    for parent in _THIS_FILE.parents:
        if any((parent / name).exists() for name in preferred):
            return parent
    return _THIS_DIR.parents[2]


def _workspace_root() -> Path:
    if override := os.environ.get("DEPENDENCIES_ROOT"):
        return Path(override).expanduser().resolve()
    return _find_dependencies_root()


WORKSPACE_ROOT = _workspace_root()

BASE_PATH = WORKSPACE_ROOT / "..."
OPENFACE_PATH_ADD = WORKSPACE_ROOT / "..."
MANUAL_DIR = WORKSPACE_ROOT / "..."
QUESTIONNAIRE_DIR = WORKSPACE_ROOT / "..."

PDF_SUBDIR = "figures_pdf"

INFANT_CAMERA = "Camera 1"
ADULT_CAMERA = "Camera 2"
INFANT_CSV = "infant.csv"
ADULT_CSV = "adult.csv"

NUM_AUS_TO_CHOOSE = 6

FOCUS_AUS = [
    "AU12_r",
    "AU06_r",
    "AU07_r",
    "AU10_r",
    "AU17_r",
    "AU25_r",
]

SMILE_AUS = ["AU06_r", "AU12_r"]

ALPHA = 0.05
CONFIDENCE_THRESHOLD = 0.85

QUESTIONNAIRE_SCORE_COLS = ["EPDS_total", "STAI_X1", "STAI_X2"]
QUESTIONNAIRE_SCORE_LABELS = ["EPDS", "STAI X1", "STAI X2"]
