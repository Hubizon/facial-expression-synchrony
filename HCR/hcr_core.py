"""Hierarchical Correlation Reconstruction (HCR) helpers for infant-adult AU dependence.

Loading, normalisation, Legendre-basis HCR coefficients, Holm-Bonferroni correction
and the moving-average detrending used by the figure scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import eval_legendre

FPS = 50
MIN_SEG_LEN = 300
M_DEFAULT = 5
SUCCESS_THRESHOLD = 0.85

# facial-expression-synchrony/HCR/ -> workspace root two levels up
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _WORKSPACE_ROOT / 'korelacje-transformer' / 'cut_short_fixed_results_openface2_22_01'


def find_data_root() -> Path:
    """KORELACJE_DATA_ROOT env var if set, otherwise the default dataset location."""
    env_root = os.environ.get('KORELACJE_DATA_ROOT')
    candidates = [Path(env_root)] if env_root else []
    candidates.append(_DEFAULT_DATA_ROOT)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = '\n'.join(str(c) for c in candidates)
    raise FileNotFoundError(
        'Unable to locate "cut_short_fixed_results_openface2_22_01".\n'
        f'Searched:\n{searched}\n'
        'Set KORELACJE_DATA_ROOT to point to the dataset.'
    )


# ── Loading ──────────────────────────────────────────────────────────────────

def load_all_valid_frames(infant_path, adult_path):
    """All frames where both trackers succeeded (>=0.85). Keeps frame col for lag indexing."""
    inf = pd.read_csv(infant_path)
    adu = pd.read_csv(adult_path)
    inf.columns = [c.strip() for c in inf.columns]
    adu.columns = [c.strip() for c in adu.columns]
    frame_col = 'frame' if 'frame' in inf.columns else 'Frame'
    au_cols = [c for c in inf.columns if c.startswith('AU') and c.endswith('_r')]
    keep = [frame_col, 'success', *au_cols]
    merged = pd.merge(inf[keep], adu[keep], on=frame_col,
                      how='inner', suffixes=('_infant', '_adult'))
    valid = merged[(merged.success_infant >= SUCCESS_THRESHOLD)
                   & (merged.success_adult >= SUCCESS_THRESHOLD)].copy()
    return valid.drop(columns=['success_infant', 'success_adult']).reset_index(drop=True)


def load_month_all_frames(month, root=None, min_frames=MIN_SEG_LEN):
    """Returns {video_name: DataFrame of all valid frames}. Per-video structure preserved."""
    root = Path(root) if root is not None else find_data_root()

    videos = {}
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if month != -1 and not (name.endswith(f'_{month}') or name.endswith(f'_0{month}')):
            continue
        inf_path = subdir / 'Kamera 1' / 'infant.csv'
        adu_path = subdir / 'Kamera 2' / 'adult.csv'
        if not inf_path.exists() or not adu_path.exists():
            continue
        try:
            df = load_all_valid_frames(inf_path, adu_path)
        except Exception as e:
            print(f'  [SKIP] {name}: {e}')
            continue
        if len(df) >= min_frames:
            videos[name] = df
    print(f'Month {month}: {len(videos)} videos with >= {min_frames} valid frames')
    return videos


# ── Normalisation ────────────────────────────────────────────────────────────

def empirical_normalize_with_noise(series, noise=1e-6, seed=None):
    """Rank-based CDF → uniform [0,1]. Tiny noise breaks ties (e.g. zero-inflated AUs).

    `seed` gives each call its own fixed generator, so a run can be repeated exactly.
    With seed=None the generator is unseeded and results vary between runs.
    """
    rng = np.random.default_rng(seed)
    noisy = series + rng.uniform(-noise, noise, len(series))
    n = len(noisy)
    ranks = pd.Series(noisy).rank(method='average')
    return pd.Series((ranks - 0.5) / n, index=series.index)


# ── Multiple-comparison correction ───────────────────────────────────────────

def holm_bonferroni_correct(values):
    """Return Holm-Bonferroni adjusted p-values, preserving NaNs and shape."""
    arr = np.asarray(values, dtype=float)
    flat = arr.ravel()
    corrected = np.full_like(flat, np.nan)
    mask = np.isfinite(flat)
    if mask.sum() == 0:
        return corrected.reshape(arr.shape)

    valid = flat[mask]
    order = np.argsort(valid)
    sorted_p = valid[order]
    m = sorted_p.size

    factors = m - np.arange(m)
    adjusted = sorted_p * factors
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.clip(adjusted, 0.0, 1.0)

    adj_valid = np.empty_like(valid)
    adj_valid[order] = adjusted
    corrected[mask] = adj_valid
    return corrected.reshape(arr.shape)


# ── HCR coefficients ─────────────────────────────────────────────────────────

def legendre_basis(x, m):
    """Orthonormal shifted Legendre polynomials on [0,1]. Returns (m+1, len(x))."""
    x = np.asarray(x, dtype=float)
    out = np.empty((m + 1, len(x)))
    for j in range(m + 1):
        out[j] = np.sqrt(2 * j + 1) * eval_legendre(j, 2 * x - 1)
    return out


def build_hcr_arrays(videos, au_cols, m, frame_col='frame', seed=None):
    """Per-month normalisation + precomputed Legendre basis arrays per video."""
    combined = pd.concat(
        [df[[frame_col, *au_cols]].assign(_video=name) for name, df in videos.items()],
        ignore_index=True
    )
    for col in au_cols:
        combined[col] = empirical_normalize_with_noise(combined[col], seed=seed)

    video_data = {}
    for vid, grp in combined.groupby('_video', sort=False):
        grp = grp.sort_values(frame_col)
        frm = grp[frame_col].values
        fmin, fmax = frm.min(), frm.max()
        rel = frm - fmin
        size = fmax - fmin + 1
        basis = {}
        for col in au_cols:
            b = np.full((m + 1, size), np.nan)
            b[:, rel] = legendre_basis(grp[col].values, m)
            basis[col] = b
        video_data[vid] = {'frames': rel, 'fmax': fmax - fmin, 'basis': basis}
    return video_data


def hcr_at_lag(video_data, col_y, col_z, lag, m):
    """
    Full (m+1)×(m+1) HCR coefficient matrix Ã at one lag, pooled across videos.
    Marginals removed: Ã_jk = A_jk - A_j0*A_0k  for j,k > 0.
    Returns None if no valid pairs found.
    """
    fy_parts, fz_parts = [], []
    for vd in video_data.values():
        frm = vd['frames']
        fmax = vd['fmax']
        by = vd['basis'][col_y]
        bz = vd['basis'][col_z]
        shifted = frm + lag
        ok = (shifted >= 0) & (shifted <= fmax)
        if ok.sum() < 2:
            continue
        fy = by[:, frm[ok]]
        fz = bz[:, shifted[ok]]
        valid = ~(np.any(np.isnan(fy), axis=0) | np.any(np.isnan(fz), axis=0))
        if valid.sum() >= 2:
            fy_parts.append(fy[:, valid])
            fz_parts.append(fz[:, valid])

    if not fy_parts:
        return None

    fy_all = np.concatenate(fy_parts, axis=1)  # (m+1, N)
    fz_all = np.concatenate(fz_parts, axis=1)
    N = fy_all.shape[1]

    A = (fy_all @ fz_all.T) / N           # (m+1, m+1)
    A[1:, 1:] -= np.outer(A[1:, 0], A[0, 1:])   # remove marginals
    return A


def dependency_indices(m):
    """Flat indices of the pure-dependency block j,k in {1..m} of an (m+1)x(m+1) matrix."""
    return [j * (m + 1) + k for j in range(1, m + 1) for k in range(1, m + 1)]


# ── Preprocessing ────────────────────────────────────────────────────────────

def ma_detrend_videos(videos, window=50):
    """
    Remove autocorrelation by subtracting a rolling mean (MA filter).
    Each AU column: residual = value - rolling_mean(window)
    Frames/structure preserved; only AU values changed.
    """
    au_cols = [c for c in next(iter(videos.values())).columns
               if c.startswith('AU') and c.endswith(('_r_infant', '_r_adult'))]
    result = {}
    for name, df in videos.items():
        d = df.copy()
        for col in au_cols:
            rm = d[col].rolling(window, min_periods=max(1, window // 4)).mean()
            d[col] = d[col] - rm
        result[name] = d.dropna(subset=au_cols).reset_index(drop=True)
    return result
