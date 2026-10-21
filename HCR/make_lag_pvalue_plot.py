"""Figure 3 (bottom): AD p-values of HCR coefficients over infant-adult lags.

Writes au12_au06_pairs_over_lags.png and .pdf to --out-dir.

Usage:
    python make_lag_pvalue_plot.py [--out-dir DIR] [--seed N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from statsmodels.stats.diagnostic import normal_ad

import hcr_core as hcr
from plot_style import setup

AUS = ['AU12', 'AU06']                              # the two AUs shown in the paper
LAGS_FRAMES = np.arange(-100, 101, 20)              # -2000 ... +2000 ms in 400 ms steps
MONTHS = [1, 2, 4]
DISPLAY_MONTHS = {1: 4, 2: 7, 4: 12}
M = 5
MA_WINDOW = 50

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / 'output' / 'plot_of_lags'


def pvalue_tseries(videos, aus, m, lags, seed=None):
    """AD p-value of HCR coefficients for every infant×adult AU pair and lag, Holm-corrected.

    Returns (pval_tseries[n, n, n_lags], lags_ms).
    """
    lags = np.array(list(lags), dtype=int)
    lags_ms = lags * 1000.0 / hcr.FPS
    inf_cols = [f'{au}_r_infant' for au in aus]
    adu_cols = [f'{au}_r_adult' for au in aus]
    all_cols = list(dict.fromkeys(inf_cols + adu_cols))
    n = len(aus)
    dep_idx = hcr.dependency_indices(m)

    vdata = hcr.build_hcr_arrays(videos, all_cols, m, seed=seed)

    pval_tseries = np.full((n, n, len(lags)), np.nan)
    for li, lag in enumerate(lags):
        for ri, rc in enumerate(inf_cols):
            for ci, cc in enumerate(adu_cols):
                A = hcr.hcr_at_lag(vdata, rc, cc, lag=int(lag), m=m)
                if A is not None:
                    vals = A.ravel()[dep_idx]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) >= 3:
                        _, pval = normal_ad(vals)
                        pval_tseries[ri, ci, li] = pval

    return hcr.holm_bonferroni_correct(pval_tseries), lags_ms


def smooth_positive(series, poly=2, min_window=5):
    """Savitzky-Golay smoothing in log10 space."""
    arr = np.asarray(series, dtype=float)
    n = arr.size
    if n < min_window:
        return arr

    arr = np.clip(arr, 1e-12, None)
    mask = np.isfinite(arr)
    if mask.sum() <= poly + 1:
        return arr

    x = np.arange(n)
    log_vals = np.log10(arr)
    if mask.sum() < n:
        log_vals = log_vals.copy()
        log_vals[~mask] = np.interp(x[~mask], x[mask], log_vals[mask])

    window = max(min_window, ((n // 3) * 2) + 1)
    window = min(window, n)
    if window % 2 == 0:
        window = max(3, window - 1)
    if window <= poly + 1:
        window = poly + 3 if (poly + 3) % 2 == 1 else poly + 2
    if window > n:
        window = n if n % 2 == 1 else max(3, n - 1)
    if window < 3:
        return arr

    poly_eff = min(poly, window - 2)
    smoothed_log = savgol_filter(log_vals, window_length=window, polyorder=poly_eff, mode='interp')
    return np.power(10.0, smoothed_log)


def make_pairs_figure(results, out_dir):
    """Three panels (Month 4, 7, 12) with the four AU12/AU06 pairs and their average."""
    setup(context='paper')
    plt.close('all')

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color', ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'])
    n_months = len(MONTHS)

    fig, axes = plt.subplots(1, n_months, figsize=(4.6 * n_months, 4.2), sharey=True, sharex=True)

    log_formatter = plt.matplotlib.ticker.FuncFormatter(
        lambda val, pos: '1' if np.isclose(val, 1.0) else f'{val:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+'))
    yticks_major = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

    legend_handles = []
    legend_labels = []

    def _add_legend_entry(handle, label):
        if label and label not in legend_labels:
            legend_handles.append(handle)
            legend_labels.append(label)

    for idx, month in enumerate(MONTHS):
        ax = axes[idx]
        res = results[month]
        lags_ms = res['lags_ms']
        smooth_series = res['smooth_combos']
        avg_smooth = res['avg_smooth']

        for combo_idx, (inf_label, adu_label) in enumerate(res['pair_labels']):
            series = smooth_series[combo_idx]
            base_color = color_cycle[combo_idx % len(color_cycle)]
            label = f'inf {inf_label} → adu {adu_label}' if idx == 0 else None
            line, = ax.plot(lags_ms, series, color=base_color, linewidth=1.8, label=label, alpha=0.85)
            if label:
                _add_legend_entry(line, label)

            mask = np.isfinite(series) & (series > 0)
            if mask.sum() >= 2:
                coeffs = np.polyfit(lags_ms[mask], np.log10(series[mask]), 1)
                trend_vals = np.power(10.0, np.polyval(coeffs, lags_ms))
                ax.plot(lags_ms, trend_vals, linestyle='--', color=base_color, linewidth=1.0, alpha=0.7)

        avg_line, = ax.plot(lags_ms, avg_smooth, color='black', linewidth=2.5,
                            label='Average (Savitzky-Golay)' if idx == 0 else None)
        if idx == 0:
            _add_legend_entry(avg_line, 'Avg of AU12 & AU06 → Avg of AU12 & AU06')

        ax.axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_yscale('log')
        ax.set_ylim(1e-6, 1.1)
        ax.set_xticks([-2000, -1000, 0, 1000, 2000])
        ax.set_yticks(yticks_major)
        ax.yaxis.set_major_formatter(log_formatter)
        ax.yaxis.set_minor_locator(plt.matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.tick_params(which='both', left=True, right=False, bottom=True, top=False)
        ax.set_title(f'Month {DISPLAY_MONTHS.get(month, month)}', fontsize=12, fontweight='bold')
        ax.grid(False)

    axes[0].set_ylabel('AD p-value', fontsize=11, fontweight='bold')

    fig.subplots_adjust(bottom=0.32, wspace=0.18)
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.25),
        ncol=min(len(legend_labels), 3),
        frameon=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'au12_au06_pairs_over_lags.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    setup(context='paper')
    results = {}
    for month in MONTHS:
        videos = hcr.load_month_all_frames(month=month)
        videos_ma = hcr.ma_detrend_videos(videos, window=MA_WINDOW)
        pvals, lags_ms = pvalue_tseries(videos_ma, AUS, M, LAGS_FRAMES, seed=args.seed)

        combos = pvals.reshape(-1, len(LAGS_FRAMES))
        pair_labels = [(AUS[i], AUS[j]) for i in range(len(AUS)) for j in range(len(AUS))]

        avg_raw = np.nanmean(combos, axis=0)
        results[month] = {
            'lags_ms': lags_ms,
            'combos': combos,
            'pair_labels': pair_labels,
            'smooth_combos': np.array([smooth_positive(s) for s in combos]),
            'avg_smooth': smooth_positive(avg_raw),
        }

    png_path = make_pairs_figure(results, args.out_dir)
    print(f'Saved pairwise figure to: {png_path}')


if __name__ == '__main__':
    main()
