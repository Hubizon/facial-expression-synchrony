"""Load manual smile coding and align it with OpenFace infant AU data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_config as cfg
from dyad_analysis_utils import prepare_au_dataframe


def detect_csv_sep(file_path: Path) -> str:
    with open(file_path, encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return ";" if ";" in first_line else ","


def resolve_infant_csv(openface_dir: Path, diti_id: str, diti_norm: str) -> Path | None:
    for key in (diti_id, diti_norm):
        path = openface_dir / key / cfg.INFANT_CAMERA / cfg.INFANT_CSV
        if path.exists():
            return path
    return None


def load_manual_entries(manual_dir: Path | None = None) -> list[dict]:
    """Read all manual coding CSVs from *manual_dir*."""
    manual_dir = manual_dir or cfg.MANUAL_DIR
    entries = []
    for file_path in manual_dir.glob("*.csv"):
        match = re.search(r"diti_([0-9]+)_([0-9]+)", file_path.name)
        if not match:
            continue
        diti_raw = match.group(0)
        diti_norm = f"diti_{match.group(1)}_{match.group(2)}"
        df_man = pd.read_csv(file_path, sep=detect_csv_sep(file_path))
        df_man.columns = df_man.columns.str.strip()
        if len(df_man) == 0 or len(df_man.columns) <= 1:
            continue
        entries.append(
            {
                "diti_id": diti_raw,
                "diti_norm": diti_norm,
                "df": df_man,
            }
        )
    return entries


def add_manual_smile(df_of: pd.DataFrame, df_man: pd.DataFrame) -> pd.DataFrame:
    """Mark frames inside manual smile intervals as manual_smile=1."""
    df_of = df_of.copy()
    df_of["manual_smile"] = 0
    for _, row in df_man.iterrows():
        if "smile" not in str(row.get("category", "")).lower():
            continue
        mask = (df_of["time_ms"] >= row["onset_ms"]) & (df_of["time_ms"] <= row["offset_ms"])
        df_of.loc[mask, "manual_smile"] = 1
    return df_of


def load_synced_session(
    entry: dict,
    *,
    openface_dir: Path | None = None,
    au_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Return OpenFace infant data with manual_smile and composite AU score."""
    openface_dir = openface_dir or cfg.BASE_PATH
    au_columns = au_columns or cfg.SMILE_AUS

    of_csv = resolve_infant_csv(openface_dir, entry["diti_id"], entry["diti_norm"])
    if of_csv is None:
        return None

    df_raw = pd.read_csv(of_csv)
    df_raw.columns = df_raw.columns.str.strip()
    if "timestamp" not in df_raw.columns:
        return None

    df_prepared = prepare_au_dataframe(df_raw, au_columns)
    df_of = df_raw[["Frame", "timestamp"]].merge(df_prepared, on="Frame", how="inner")
    df_of["time_ms"] = pd.to_numeric(df_of["timestamp"], errors="coerce") * 1000.0
    df_of = df_of.dropna(subset=[*au_columns, "time_ms"]).reset_index(drop=True)
    df_of = add_manual_smile(df_of, entry["df"])
    if df_of["manual_smile"].nunique() < 2:
        return None

    au06 = pd.to_numeric(df_of[au_columns[0]], errors="coerce").fillna(0).values
    au12 = pd.to_numeric(df_of[au_columns[1]], errors="coerce").fillna(0).values
    df_of["composite"] = (au06 + au12) / 2.0
    return df_of
