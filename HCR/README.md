# HCR figures

Pip install dependencies, then generate the two Figure 3 plots (PNG and PDF) with a fixed seed:

```
pip install -r requirements.txt
python make_heatmap_grid.py --seed 42
python make_lag_pvalue_plot.py --seed 42
```

Output goes to `output/heatmaps/` and `output/plot_of_lags/`. The dataset is read from `KORELACJE_DATA_ROOT` if set.
