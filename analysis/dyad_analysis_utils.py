"""Helpers for the dyad-analysis notebooks."""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from scipy.stats import f as f_dist
from scipy.stats import pearsonr, ttest_1samp, ttest_rel, wilcoxon, shapiro
from statsmodels.stats.multitest import multipletests
import pingouin as pg

import analysis_config
import plot_style as _ps

SUCCESS_COLUMN = "success"
CONFIDENCE_COLUMN = "confidence"
FRAME_COLUMN_CANDIDATES = ["Frame", "frame"]


def prepare_au_dataframe(df: pd.DataFrame, au_columns: List[str]) -> pd.DataFrame:
    """Prepare AU dataframe by extracting relevant columns and handling invalid frames."""
    df = df.copy()
    frame_col = next((col for col in FRAME_COLUMN_CANDIDATES if col in df.columns), None)
    if frame_col is None:
        raise KeyError("Expected a frame column named 'Frame' or 'frame'.")
    if frame_col != "Frame":
        df = df.rename(columns={frame_col: "Frame"})

    missing_aus = [col for col in au_columns if col not in df.columns]
    if missing_aus:
        raise KeyError(f"Missing AU columns: {missing_aus}")

    if SUCCESS_COLUMN not in df.columns:
        raise KeyError(f"Expected column '{SUCCESS_COLUMN}' in dataframe")

    valid = df[SUCCESS_COLUMN] == 1
    confidence_threshold = analysis_config.CONFIDENCE_THRESHOLD
    if confidence_threshold > 0:
        if CONFIDENCE_COLUMN not in df.columns:
            raise KeyError(
                f"Expected column '{CONFIDENCE_COLUMN}' when CONFIDENCE_THRESHOLD > 0"
            )
        conf = pd.to_numeric(df[CONFIDENCE_COLUMN], errors="coerce").fillna(0)
        valid = valid & (conf >= confidence_threshold)

    subset = df[["Frame", SUCCESS_COLUMN] + au_columns].copy()
    subset.loc[~valid, au_columns] = np.nan
    subset = subset.drop(columns=[SUCCESS_COLUMN])
    return subset


def load_openface_data(base_path: Path):
    """Load OpenFace session folders grouped by meeting."""
    def session_sort_key(path: Path):
        parts = path.name.split("_")
        try:
            return (int(parts[1]), int(parts[2]))
        except (IndexError, ValueError):
            return (parts[1], parts[-1])

    input_paths = sorted(
        [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("diti")],
        key=session_sort_key,
    )
    
    input_paths_i = []
    input_paths_a = []
    missing_sessions = []

    for session_dir in input_paths:
        infant_path = session_dir / analysis_config.INFANT_CAMERA / analysis_config.INFANT_CSV
        adult_path = session_dir / analysis_config.ADULT_CAMERA / analysis_config.ADULT_CSV
        missing = []
        if not infant_path.exists():
            missing.append("infant")
        if not adult_path.exists():
            missing.append("adult")
        if missing:
            missing_sessions.append((session_dir.name, missing))
            continue
        input_paths_i.append(infant_path)
        input_paths_a.append(adult_path)

    if missing_sessions:
        print("Missing expected CSVs:")
        for session_name, missing in missing_sessions:
            print(f" - {session_name}: {', '.join(missing)}")

    meeting_groups = defaultdict(list)
    for infant_path, adult_path in zip(input_paths_i, input_paths_a):
        session_dir = infant_path.parents[1]
        parts = session_dir.name.split("_")
        meeting_id = "_".join(parts[:2])
        period = int(parts[-1])
        meeting_groups[meeting_id].append((infant_path, adult_path, period))

    for key, sessions in meeting_groups.items():
        sorted_sessions = sorted(sessions, key=lambda item: item[2])
        meeting_groups[key] = [(inf_path, ad_path) for inf_path, ad_path, _ in sorted_sessions]

    return dict(meeting_groups), input_paths


def discover_au_columns(df: pd.DataFrame) -> list[str]:
    """Return OpenFace intensity columns (names ending in ``_r``)."""
    return [col for col in df.columns if "AU" in col and col.endswith("_r")]


def load_prepared_meetings(
    meeting_groups: dict,
) -> tuple[dict, dict, list[str]]:
    """Read and prepare infant/adult AU dataframes for every meeting."""
    df_infant_arr: dict = {}
    df_adult_arr: dict = {}
    au_list: list[str] | None = None

    for meeting_id, sessions in meeting_groups.items():
        infant_dfs = []
        adult_dfs = []
        for infant_path, adult_path in sessions:
            raw_infant = pd.read_csv(infant_path)
            raw_adult = pd.read_csv(adult_path)
            if au_list is None:
                au_list = discover_au_columns(raw_infant)
            infant_dfs.append(prepare_au_dataframe(raw_infant, au_list))
            adult_dfs.append(prepare_au_dataframe(raw_adult, au_list))
        df_infant_arr[meeting_id] = infant_dfs
        df_adult_arr[meeting_id] = adult_dfs

    if au_list is None:
        raise ValueError("No sessions found in meeting_groups")
    return df_infant_arr, df_adult_arr, au_list


def organize_sessions_by_period(
    df_infant_arr: dict,
    df_adult_arr: dict,
    au_list: list[str],
    *,
    include_frame: bool = True,
) -> tuple[dict[int, list], list[dict]]:
    """Split prepared meetings into period buckets with aligned valid frames."""
    period_aggregated: dict[int, list] = {1: [], 2: [], 3: []}
    all_periods_data: list[dict] = []

    for meeting_id in df_infant_arr:
        for period_idx, (df_infant_raw, df_adult_raw) in enumerate(
            zip(df_infant_arr[meeting_id], df_adult_arr[meeting_id]), start=1
        ):
            valid = df_infant_raw[au_list].notna().all(axis=1) & df_adult_raw[au_list].notna().all(axis=1)
            if include_frame:
                df_infant = df_infant_raw.loc[valid, ["Frame", *au_list]].reset_index(drop=True)
                df_adult = df_adult_raw.loc[valid, ["Frame", *au_list]].reset_index(drop=True)
            else:
                df_infant = df_infant_raw.loc[valid, au_list].reset_index(drop=True)
                df_adult = df_adult_raw.loc[valid, au_list].reset_index(drop=True)
            if df_infant.empty or df_adult.empty:
                continue
            entry = {
                "infant": df_infant[au_list],
                "adult": df_adult[au_list],
                "meeting": meeting_id,
                "period": period_idx,
            }
            period_aggregated[period_idx].append(entry)
            all_periods_data.append(entry)

    return period_aggregated, all_periods_data


def fisher_z(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999, 0.999)))


def compute_session_correlation_matrix(
    infant_df: pd.DataFrame,
    adult_df: pd.DataFrame,
    au_list: list[str],
) -> pd.DataFrame:
    """Pearson-r matrix for one infant–adult session."""
    corr = pd.DataFrame(index=au_list, columns=au_list, dtype=float)
    for au_infant in au_list:
        for au_adult in au_list:
            x = infant_df[au_infant]
            y = adult_df[au_adult]
            valid = ~(x.isna() | y.isna())
            if valid.sum() >= 2:
                corr.loc[au_infant, au_adult] = pearsonr(x[valid], y[valid])[0]
            else:
                corr.loc[au_infant, au_adult] = np.nan
    return corr


def compute_averaged_correlation_matrix(
    data_list: list[dict],
    au_list: list[str],
) -> pd.DataFrame:
    """Fisher-z average of per-session Pearson correlation matrices."""
    matrices = [
        compute_session_correlation_matrix(d["infant"], d["adult"], au_list)
        for d in data_list
    ]
    if not matrices:
        return pd.DataFrame(index=au_list, columns=au_list, dtype=float)
    fisher_matrices = [np.arctanh(m.clip(-0.999, 0.999)) for m in matrices]
    avg_fisher = pd.concat(fisher_matrices).groupby(level=0).mean()
    return np.tanh(avg_fisher)


def compute_meeting_fisher_matrices(
    infant_dfs: list[pd.DataFrame],
    adult_dfs: list[pd.DataFrame],
    au_list: list[str],
) -> dict[int, pd.DataFrame]:
    """Fisher-z correlation matrices for each period of one meeting."""
    period_matrices: dict[int, pd.DataFrame] = {}

    for period_idx, (df_infant, df_adult) in enumerate(zip(infant_dfs, adult_dfs), start=1):
        valid = df_infant[au_list].notna().all(axis=1) & df_adult[au_list].notna().all(axis=1)
        infant_valid = df_infant.loc[valid, au_list].reset_index(drop=True)
        adult_valid = df_adult.loc[valid, au_list].reset_index(drop=True)
        corr_matrix = compute_session_correlation_matrix(infant_valid, adult_valid, au_list)
        period_matrices[period_idx] = corr_matrix.map(fisher_z)

    return period_matrices


def collect_fisher_values_per_au_pair(
    meeting_matrices: dict,
    au_list: list[str],
) -> dict[tuple[str, str], dict[int, list[float]]]:
    """Gather Fisher-Z values per AU pair across meetings and periods."""
    au_pair_values: dict[tuple[str, str], dict[int, list[float]]] = {}
    for au_infant in au_list:
        for au_adult in au_list:
            au_pair_values[(au_infant, au_adult)] = {1: [], 2: [], 3: []}

    for period_matrices in meeting_matrices.values():
        for period in (1, 2, 3):
            if period not in period_matrices:
                continue
            matrix = period_matrices[period]
            for au_infant in au_list:
                for au_adult in au_list:
                    val = matrix.loc[au_infant, au_adult]
                    if not np.isnan(val):
                        au_pair_values[(au_infant, au_adult)][period].append(val)
    return au_pair_values


def compute_pvalue_matrix(
    data_list: List[dict],
    au_list: List[str],
) -> pd.DataFrame:
    """One-sample t-test on Fisher-z per-meeting correlations (H0: mean z = 0)."""
    p_matrix = pd.DataFrame(index=au_list, columns=au_list, dtype=float)

    for au_infant in au_list:
        for au_adult in au_list:
            zs = []
            for meeting_data in data_list:
                x = meeting_data["infant"][au_infant]
                y = meeting_data["adult"][au_adult]
                valid = ~(x.isna() | y.isna())
                if valid.sum() >= 2:
                    r, _ = pearsonr(x[valid], y[valid])
                    zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
            if len(zs) >= 2:
                _, p = ttest_1samp(zs, 0.0)
                p_matrix.loc[au_infant, au_adult] = p
            else:
                p_matrix.loc[au_infant, au_adult] = np.nan

    return p_matrix


def _compute_au_scores(corr_matrix: pd.DataFrame, au_list: List[str]) -> Dict[str, float]:
    """Sum of absolute correlations involving an AU. The diagonal pair is counted once."""
    corr_abs = corr_matrix.abs()
    scores = {}
    for au in au_list:
        row_sum = corr_abs.loc[au].sum() if au in corr_abs.index else 0
        col_sum = corr_abs[au].sum() if au in corr_abs.columns else 0
        diag = 0
        if au in corr_abs.index and au in corr_abs.columns and pd.notna(corr_abs.loc[au, au]):
            diag = corr_abs.loc[au, au]
        scores[au] = row_sum + col_sum - diag
    return scores


def _rank_aus_from_scores(
    per_meeting_scores: List[Dict[str, float]],
    au_list: List[str],
    num_to_choose: int
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """Average rank across meetings. Rank 1 is the highest score."""
    from scipy.stats import rankdata

    rank_accumulator: Dict[str, List[float]] = {au: [] for au in au_list}

    for scores in per_meeting_scores:
        score_values = np.array([scores.get(au, 0.0) for au in au_list])
        # rankdata ranks ascending; negate so the highest score is rank 1
        ranks = rankdata(-score_values, method='average')
        for au, rank in zip(au_list, ranks):
            rank_accumulator[au].append(rank)

    avg_ranks = {au: np.mean(ranks) for au, ranks in rank_accumulator.items()}
    all_aus_sorted = sorted(avg_ranks.items(), key=lambda x: x[1])
    selected_aus = [au for au, _ in all_aus_sorted[:num_to_choose]]

    return selected_aus, all_aus_sorted


def select_best_aus_from_correlations(
    all_periods_data: List[dict],
    au_list: List[str],
    num_to_choose: int
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """Select AUs with the lowest average rank of absolute dyadic correlation."""
    per_meeting_scores = []

    for meeting_data in all_periods_data:
        corr = compute_session_correlation_matrix(
            meeting_data['infant'], meeting_data['adult'], au_list
        )
        per_meeting_scores.append(_compute_au_scores(corr, au_list))

    return _rank_aus_from_scores(per_meeting_scores, au_list, num_to_choose)


def holm_correct_matrix(p_matrix: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Holm-corrected p-values across all non-NaN cells of an AU-pair p-value matrix."""
    corrected = p_matrix.astype(float).copy()
    values = corrected.values
    mask = ~np.isnan(values)
    _, p_holm, _, _ = multipletests(values[mask], alpha=alpha, method="holm")
    values[mask] = p_holm
    corrected.loc[:, :] = values
    return corrected


def run_paired_test(values1, values2) -> float:
    """Wilcoxon signed-rank p-value for two equal-length lists of dyad-level values."""
    v1 = np.asarray(values1, dtype=float)
    v2 = np.asarray(values2, dtype=float)
    if len(v1) != len(v2):
        raise ValueError("Paired samples must have one value per dyad in both periods")
    valid = ~(np.isnan(v1) | np.isnan(v2))
    if valid.sum() == 0:
        return np.nan
    try:
        _, p_value = wilcoxon(v1[valid], v2[valid])
    except ValueError:
        p_value = 1.0
    return float(p_value)


def run_paired_comparisons(
    au_pair_values: dict,
    au_list: list[str],
    comparisons: list[tuple[int, int]],
) -> dict:
    """Wilcoxon signed-rank p-values for every AU pair and period comparison."""
    pvalue_matrices: dict = {}
    for p1, p2 in comparisons:
        pval_matrix = pd.DataFrame(index=au_list, columns=au_list, dtype=float)
        for au_infant in au_list:
            for au_adult in au_list:
                pval_matrix.loc[au_infant, au_adult] = run_paired_test(
                    au_pair_values[(au_infant, au_adult)][p1],
                    au_pair_values[(au_infant, au_adult)][p2],
                )
        pvalue_matrices[(p1, p2)] = pval_matrix
    return pvalue_matrices


def compute_cohens_d_matrix(
    au_pair_values: dict,
    period1: int,
    period2: int,
    au_list: list[str],
) -> pd.DataFrame:
    """Paired Cohen's d_avg between two periods; positive = higher in period2."""
    d_matrix = pd.DataFrame(index=au_list, columns=au_list, dtype=float)
    for au_infant in au_list:
        for au_adult in au_list:
            vals1 = np.asarray(au_pair_values[(au_infant, au_adult)][period1], dtype=float)
            vals2 = np.asarray(au_pair_values[(au_infant, au_adult)][period2], dtype=float)
            if len(vals1) != len(vals2):
                raise ValueError("Paired samples must have one value per dyad in both periods")
            valid_mask = ~(np.isnan(vals1) | np.isnan(vals2))
            if valid_mask.sum() < 2:
                d_matrix.loc[au_infant, au_adult] = np.nan
                continue
            d_matrix.loc[au_infant, au_adult] = pg.compute_effsize(
                vals2[valid_mask], vals1[valid_mask], paired=True, eftype="cohen"
            )
    return d_matrix


def compute_mean_abs_fisher(meeting_matrices: dict) -> pd.DataFrame:
    """Mean |Fisher z| across all AU pairs, per meeting and period."""
    results = []
    for meeting_id, period_matrices in meeting_matrices.items():
        row = {"meeting_id": meeting_id}
        for period in (1, 2, 3):
            matrix = period_matrices.get(period)
            if matrix is None:
                row[f"period_{period}"] = np.nan
                continue
            values = matrix.values.flatten()
            values = values[~np.isnan(values)]
            row[f"period_{period}"] = float(np.mean(np.abs(values))) if len(values) else np.nan
        results.append(row)
    return pd.DataFrame(results)


def pair_corr(x, y) -> float:
    """Pearson r. NaN when fewer than two paired observations remain."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan
    r, _ = pearsonr(x[mask], y[mask])
    return float(r)


def check_difference_normality(
    period_1_vals,
    period_2_vals,
    period_3_vals,
    alpha: float = 0.05,
) -> tuple[bool, pd.DataFrame]:
    """Shapiro-Wilk on pairwise difference scores (RM-ANOVA assumption)."""
    p1 = np.asarray(period_1_vals, dtype=float)
    p2 = np.asarray(period_2_vals, dtype=float)
    p3 = np.asarray(period_3_vals, dtype=float)
    rows = []
    all_normal = True
    for label, diff in [
        (_ps.comparison_month_label(1, 2), p1 - p2),
        (_ps.comparison_month_label(2, 3), p2 - p3),
        (_ps.comparison_month_label(1, 3), p1 - p3),
    ]:
        stat, p = shapiro(diff)
        ok = p >= alpha
        all_normal &= ok
        rows.append({"comparison": label, "W": stat, "p": p, "normal": ok})
    return all_normal, pd.DataFrame(rows)


def run_rm_anova_three_periods(
    period_1_vals,
    period_2_vals,
    period_3_vals,
    alpha: float = 0.05,
) -> dict:
    """Repeated-measures ANOVA. Uncorrected p-value when Greenhouse–Geisser epsilon >= 0.75."""
    p1 = np.asarray(period_1_vals, dtype=float)
    p2 = np.asarray(period_2_vals, dtype=float)
    p3 = np.asarray(period_3_vals, dtype=float)

    is_normal, norm_df = check_difference_normality(p1, p2, p3, alpha)

    rm_data = pd.DataFrame({
        "subject": np.tile(np.arange(len(p1)), 3),
        "period": np.repeat(["period_1", "period_2", "period_3"], len(p1)),
        "value": np.concatenate([p1, p2, p3]),
    })
    anova = pg.rm_anova(rm_data, dv="value", within="period", subject="subject", detailed=True)
    period_row = anova.loc[anova["Source"] == "period"].iloc[0]
    error_row = anova.loc[anova["Source"] == "Error"].iloc[0]

    F = float(period_row["F"])
    p_unc = float(period_row["p-unc"])
    df1 = int(period_row["DF"])
    df2 = int(error_row["DF"])
    eps = float(period_row["eps"]) if np.isfinite(period_row.get("eps", np.nan)) else np.nan
    if not np.isfinite(eps):
        eps = float(pg.epsilon(rm_data, dv="value", within="period", subject="subject", correction="gg"))

    p_gg = float(f_dist.sf(F, df1 * eps, df2 * eps))
    use_gg = eps < 0.75

    return {
        "test_name": "RM-ANOVA (Greenhouse-Geisser)" if use_gg else "RM-ANOVA",
        "statistic": F,
        "p_value": p_gg if use_gg else p_unc,
        "is_normal": is_normal,
        "normality": norm_df,
        "anova_table": anova,
        "eps": eps,
        "p_unc": p_unc,
        "p_gg": p_gg,
    }


def run_posthoc_period_tests(
    period_1_vals,
    period_2_vals,
    period_3_vals,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired t-tests between periods, with Holm correction across the three comparisons."""
    comparisons = [
        (_ps.comparison_month_label(1, 2), np.asarray(period_1_vals, dtype=float), np.asarray(period_2_vals, dtype=float)),
        (_ps.comparison_month_label(2, 3), np.asarray(period_2_vals, dtype=float), np.asarray(period_3_vals, dtype=float)),
        (_ps.comparison_month_label(1, 3), np.asarray(period_1_vals, dtype=float), np.asarray(period_3_vals, dtype=float)),
    ]
    rows = []
    for name, v1, v2 in comparisons:
        stat, p = ttest_rel(v1, v2)
        rows.append({"Comparison": name, "Statistic": stat, "p-value": p})

    df = pd.DataFrame(rows)
    reject, corrected, _, _ = multipletests(df["p-value"].values, alpha=alpha, method="holm")
    df["Corrected p-value (Holm)"] = corrected
    df["Significant (Holm)"] = reject
    return df


def merge_openface_paths(*paths: Path) -> dict:
    """Load meeting groups from several OpenFace roots (later paths override)."""
    meeting_groups: dict = {}
    for path in paths:
        if path.exists():
            groups, _ = load_openface_data(path)
            meeting_groups.update(groups)
    return meeting_groups


def plot_heatmaps(
    period_matrices: Dict[int, pd.DataFrame],
    focus_aus: List[str],
    title: str,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "coolwarm",
    xlabel: str = "AU",
    ylabel: str = "AU",
    cbar_label: str = "Value",
    fmt: str = ".2f",
    *,
    figsize: Tuple[float, float] = (10.5, 3.6),
    annotate: bool = True,
    pvalue_matrices: Optional[Dict[int, pd.DataFrame]] = None,
    annotation_fontsize: int = _ps.DEFAULT_HEATMAP_ANNOTATION_FONTSIZE,
    save_pdf_path: Optional[Path] = None,
    show: bool = True,
):
    """Heatmaps for each period."""
    fig, axes = _ps.create_shared_heatmap_axes(3, figsize=figsize, sharey=True)
    im = None

    for ax_idx, (ax, period) in enumerate(zip(axes, [1, 2, 3])):
        matrix = period_matrices[period]
        if focus_aus:
            matrix = matrix.reindex(index=focus_aus, columns=focus_aus)
        values = matrix.values.astype(float)
        xlabels = _ps.clean_au_labels(matrix.columns)
        ylabels = _ps.clean_au_labels(matrix.index)
        pvalues = None
        if pvalue_matrices is not None:
            pvalues = pvalue_matrices[period].reindex(
                index=focus_aus or matrix.index,
                columns=focus_aus or matrix.columns,
            ).values.astype(float)

        im = _ps.draw_heatmap(
            ax,
            values,
            xlabels,
            ylabels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annotate=False,
            fmt=fmt,
            annotation_fontsize=annotation_fontsize,
            xlabel=xlabel,
            ylabel=ylabel,
            show_ylabel=(ax_idx == 0),
        )
        if annotate:
            _ps.annotate_matrix_cells(
                ax,
                values,
                fmt=fmt,
                pvalues=pvalues,
                fontsize=annotation_fontsize,
            )
        _ps.period_panel_label(ax, period)

    ylabel = ylabel or "AU"
    _ps.finalize_shared_yaxis(axes, ylabels, ylabel)
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label(cbar_label)
    _ps.fig_suptitle(fig, title, y=1.02, fontweight="bold")
    if save_pdf_path is not None:
        fig.savefig(save_pdf_path, format="pdf")
    return _ps.finish_figure(fig, show=show)


def plot_pvalue_heatmaps(
    period_matrices: Dict[int, pd.DataFrame],
    focus_aus: List[str],
    title: str,
    alpha: float = 0.05,
    xlabel: str = "AU",
    ylabel: str = "AU",
    *,
    figsize: Tuple[float, float] = (10.5, 3.6),
    save_pdf_path: Optional[Path] = None,
    show: bool = True,
):
    """Discrete p-value heatmaps for each period (same layout as plot_heatmaps)."""
    fig, axes = _ps.create_shared_heatmap_axes(3, figsize=figsize, sharey=True)
    cmap, norm, cbar_ticks, cbar_labels = _ps.pvalue_discrete_cmap(alpha)
    im = None

    for ax_idx, (ax, period) in enumerate(zip(axes, [1, 2, 3])):
        matrix = period_matrices[period]
        if focus_aus:
            matrix = matrix.reindex(index=focus_aus, columns=focus_aus)
        values = matrix.values.astype(float)
        xlabels = _ps.clean_au_labels(matrix.columns)
        ylabels = _ps.clean_au_labels(matrix.index)

        im = _ps.draw_heatmap(
            ax,
            values,
            xlabels,
            ylabels,
            cmap=cmap,
            norm=norm,
            annotate=True,
            annotation_fontsize=_ps.PVALUE_HEATMAP_ANNOTATION_FONTSIZE,
            annot_formatter=_ps.format_p_value,
            annot_color_fn=lambda val, a=alpha: _ps.pvalue_annotation_color(val, a),
            xlabel=xlabel,
            ylabel=ylabel,
            show_ylabel=(ax_idx == 0),
        )
        _ps.period_panel_label(ax, period)

    _ps.finalize_shared_yaxis(axes, ylabels, ylabel or "AU")
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_labels)
    _ps.fig_suptitle(fig, title, y=1.02, fontweight="bold")
    if save_pdf_path is not None:
        fig.savefig(save_pdf_path, format="pdf")
    return _ps.finish_figure(fig, show=show)


def plot_comparison_pvalue_heatmaps(
    pvalue_matrices: dict,
    comparisons: list[tuple[int, int]],
    focus_aus: list[str],
    alpha: float = 0.05,
    test_label: str = "",
    *,
    save_path: Path | None = None,
    figsize: Tuple[float, float] = (10.5, 3.6),
    show: bool = True,
):
    """Discrete p-value heatmaps for pairwise month comparisons."""
    fig, axes = _ps.create_shared_heatmap_axes(len(comparisons), figsize=figsize, sharey=True)
    cmap, norm, cbar_ticks, cbar_labels = _ps.pvalue_discrete_cmap(alpha)
    im = None
    ylabels = _ps.clean_au_labels(focus_aus)

    for ax_idx, (ax, comp) in enumerate(zip(axes, comparisons)):
        matrix = pvalue_matrices[comp].reindex(index=focus_aus, columns=focus_aus)
        values = matrix.values.astype(float)
        xlabels = _ps.clean_au_labels(matrix.columns)

        im = _ps.draw_heatmap(
            ax,
            values,
            xlabels,
            ylabels,
            cmap=cmap,
            norm=norm,
            annotate=True,
            annotation_fontsize=_ps.PVALUE_HEATMAP_ANNOTATION_FONTSIZE,
            annot_formatter=_ps.format_p_value,
            annot_color_fn=lambda val, a=alpha: _ps.pvalue_annotation_color(val, a),
            xlabel="Adult AU",
            ylabel="Infant AU",
            show_ylabel=(ax_idx == 0),
        )
        _ps.period_panel_label(ax, comp[0], text=_ps.comparison_month_label(comp[0], comp[1]))

    _ps.finalize_shared_yaxis(axes, ylabels, "Infant AU")
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_labels)
    title = test_label or "Pairwise period comparisons"
    _ps.fig_suptitle(fig, title, y=1.02, fontweight="bold")
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
    return _ps.finish_figure(fig, show=show)


def plot_comparison_cohens_d_heatmaps(
    cohens_d_matrices: dict,
    pvalue_matrices: dict,
    comparisons: list[tuple[int, int]],
    focus_aus: list[str],
    alpha: float = 0.05,
    test_label: str = "",
    *,
    save_path: Path | None = None,
    figsize: Tuple[float, float] = (10.5, 3.6),
    show: bool = True,
):
    """Cohen's d heatmaps for pairwise period comparisons with significance stars."""
    fig, axes = _ps.create_shared_heatmap_axes(len(comparisons), figsize=figsize, sharey=True)
    all_vals = []
    for comp in comparisons:
        vals = cohens_d_matrices[comp].reindex(index=focus_aus, columns=focus_aus).values.flatten()
        all_vals.extend(vals[~np.isnan(vals)])
    vmax = max(abs(min(all_vals)), abs(max(all_vals))) if all_vals else 1.0
    if vmax == 0:
        vmax = 1.0

    im = None
    ylabels = _ps.clean_au_labels(focus_aus)
    for ax_idx, (ax, comp) in enumerate(zip(axes, comparisons)):
        matrix = cohens_d_matrices[comp].reindex(index=focus_aus, columns=focus_aus)
        pval_matrix = pvalue_matrices[comp].reindex(index=focus_aus, columns=focus_aus)
        values = matrix.values.astype(float)
        xlabels = _ps.clean_au_labels(matrix.columns)

        im = _ps.draw_heatmap(
            ax,
            values,
            xlabels,
            ylabels,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            annotate=False,
            xlabel="Adult AU",
            ylabel="Infant AU",
            show_ylabel=(ax_idx == 0),
        )
        _ps.annotate_matrix_cells(ax, values, fmt=".2f", pvalues=pval_matrix.values.astype(float))
        _ps.period_panel_label(ax, comp[0], text=_ps.comparison_month_label(comp[0], comp[1]))

    _ps.finalize_shared_yaxis(axes, ylabels, "Infant AU")
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Cohen's d")
    title = test_label or "Pairwise period comparisons"
    _ps.fig_suptitle(fig, f"{title} — Cohen's d", y=1.02, fontweight="bold")
    if save_path is not None:
        fig.savefig(save_path, format="pdf")
    return _ps.finish_figure(fig, show=show)
