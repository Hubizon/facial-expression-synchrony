"""Figure 3 (top): AD p-value heatmaps of HCR coefficients, months x lags.

Usage:
    python make_heatmap_grid.py [--out-dir DIR] [--seed N]

--seed is optional. Without it the rank-normalisation tie-breaking noise is drawn
from an unseeded generator, so the output varies between runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.diagnostic import normal_ad

import hcr_core as hcr
from plot_style import (
    setup,
    draw_heatmap,
    format_p_value,
    pvalue_annotation_color,
    DEFAULT_HEATMAP_ANNOTATION_FONTSIZE,
)

AUS = ['AU12', 'AU06', 'AU10', 'AU07', 'AU17', 'AU25']
LAGS_FRAMES = [-100, -50, 0, 50, 100]          # -2000, -1000, 0, 1000, 2000 ms at 50 fps
MONTHS = [1, 2, 4]                             # file-name month codes
DISPLAY_MONTHS = {1: 4, 2: 7, 4: 12}           # reported age in months
M = 5
MA_WINDOW = 50

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = HERE / 'output' / 'heatmaps'


def ms(lag_frames):
    return int(round(lag_frames * 1000 / hcr.FPS))


def make_grid(vdata_by_month, out_dir):
    setup(context='paper')

    inf_cols = [f'{au}_r_infant' for au in AUS]
    adu_cols = [f'{au}_r_adult' for au in AUS]
    n = len(AUS)
    dep_idx = hcr.dependency_indices(M)

    fig, axes = plt.subplots(
        len(MONTHS), len(LAGS_FRAMES),
        figsize=(3.5 * len(LAGS_FRAMES), 3.9 * len(MONTHS)),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    norm = mcolors.LogNorm(vmin=1e-4, vmax=1.0)
    cmap = matplotlib.colormaps['YlOrRd_r']
    mappable = None

    for ri_month, month in enumerate(MONTHS):
        for ci_lag, lag in enumerate(LAGS_FRAMES):
            ax = axes[ri_month, ci_lag]
            vdata = vdata_by_month[month]
            pval_raw = np.full((n, n), np.nan)
            for ri, rc in enumerate(inf_cols):
                for ci, cc in enumerate(adu_cols):
                    A = hcr.hcr_at_lag(vdata, rc, cc, lag=lag, m=M)
                    if A is not None:
                        vals = A.ravel()[dep_idx]
                        vals = vals[~np.isnan(vals)]
                        if len(vals) >= 3:
                            _, pval = normal_ad(vals)
                            pval_raw[ri, ci] = pval

            pval_mat = hcr.holm_bonferroni_correct(pval_raw)

            im = draw_heatmap(
                ax,
                pval_mat,
                AUS,
                AUS,
                cmap=cmap,
                norm=norm,
                annotate=True,
                fmt='.3f',
                annotation_fontsize=DEFAULT_HEATMAP_ANNOTATION_FONTSIZE,
                annot_formatter=format_p_value,
                annot_color_fn=lambda val: pvalue_annotation_color(val, alpha=0.05),
                xlabel='',
                ylabel='',
                show_ylabel=(ci_lag == 0),
                grid=True,
                rotation=90,
            )
            if mappable is None:
                mappable = im

            reported = DISPLAY_MONTHS.get(month, month)
            ax.set_title(f'Month {reported} | lag={ms(lag):+d} ms', fontsize=11, fontweight='bold', pad=6)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xlabel('Adult AU' if ri_month == len(MONTHS) - 1 and ci_lag == len(LAGS_FRAMES) // 2 else '')
            ax.set_ylabel('Infant AU' if ci_lag == 0 and ri_month == len(MONTHS) // 2 else '')

    cbar = fig.colorbar(mappable, ax=axes, shrink=0.72, pad=0.02)
    cbar.set_label('AD p-value (log scale)')
    cbar.ax.tick_params(labelsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'combined_5x3_extended.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    vdata_by_month = {}
    for month in MONTHS:
        videos = hcr.load_month_all_frames(month=month)
        videos_ma = hcr.ma_detrend_videos(videos, window=MA_WINDOW)
        all_cols = list(dict.fromkeys([f'{au}_r_infant' for au in AUS] + [f'{au}_r_adult' for au in AUS]))
        vdata_by_month[month] = hcr.build_hcr_arrays(videos_ma, all_cols, M, seed=args.seed)

    png_path = make_grid(vdata_by_month, args.out_dir)
    print(f'Saved 5x3 figure (lags: -2000, -1000, 0, 1000, 2000 ms): {png_path}')


if __name__ == '__main__':
    main()
