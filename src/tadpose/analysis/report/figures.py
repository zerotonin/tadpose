# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.report.figures                             ║
# ║  « report figures: significance as stars, no flow text »       ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Compact, caption-ready figures for the dataset report: the     ║
# ║  fingerprint seizure heatmap and the locomotion scalar panel,   ║
# ║  each with edited-vs-control significance drawn as stars.       ║
# ╚════════════════════════════════════════════════════════════════╝
"""Report figures with significance drawn as stars.

Each figure is edited-group vs its internal control; significance comes from the
lab Fisher-resampling test (``analysis.stats.compare_groups``) with BH-FDR and is
rendered as ``* / ** / ***`` (or ``ns``).  Returns the saved paths and the stats
frame so the caller can put the same numbers in the appendix.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from ...viz_constants import WONG, save_figure       # noqa: E402
from ..stats import compare_groups                   # noqa: E402

EPS = 1e-4


def stars(p: float) -> str:
    """p-value -> significance stars."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _ctrl_vs_gene(df: pd.DataFrame, value_col: str, group_col: str,
                  control: str, genes: list[str]) -> dict[str, float]:
    """BH-FDR corrected p for control vs each gene on one column."""
    sub = (df[[group_col, value_col]].rename(columns={value_col: "v"})
           .replace([np.inf, -np.inf], np.nan).dropna(subset=["v"]))
    try:
        res = compare_groups(sub, "v", group_col)
    except Exception:
        return {g: float("nan") for g in genes}
    out = {}
    for g in genes:
        m = (((res.groupA == control) & (res.groupB == g)) |
             ((res.groupA == g) & (res.groupB == control)))
        out[g] = float(res.loc[m, "p value corrected"].iloc[0]) if m.any() else float("nan")
    return out


def fingerprint_heatmap(fp: pd.DataFrame, output: Path, *, group_col: str,
                        control: str, genes: list[str], prototypes: list[int],
                        labeller=None) -> tuple[list[Path], pd.DataFrame]:
    """Genes x seizure-prototype log2 fold-change heatmap, stars where significant."""
    cl = [f"cluster_{i}" for i in prototypes]
    ctrl_mean = fp.loc[fp[group_col] == control, cl].mean()
    fc = np.vstack([np.log2((fp.loc[fp[group_col] == g, cl].mean() + EPS) / (ctrl_mean + EPS))
                    for g in genes])
    stat_rows, ann = [], np.empty_like(fc, dtype=object)
    for j, i in enumerate(prototypes):
        p = _ctrl_vs_gene(fp, f"cluster_{i}", group_col, control, genes)
        for gi, g in enumerate(genes):
            ann[gi, j] = stars(p[g])
            stat_rows.append({"gene": g, "prototype": i,
                              "label": labeller(i) if labeller else f"PM{i}",
                              "log2FC": round(float(fc[gi, j]), 3),
                              "p_corr": round(p[g], 4)})
    fig, ax = plt.subplots(figsize=(0.55 * len(prototypes) + 2, 0.6 * len(genes) + 1.6))
    vmax = np.nanmax(np.abs(fc)) or 1.0
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(prototypes)))
    ax.set_xticklabels([labeller(i) if labeller else f"PM{i}" for i in prototypes],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=9)
    for gi in range(len(genes)):
        for j in range(len(prototypes)):
            ax.text(j, gi, ann[gi, j], ha="center", va="center", fontsize=7,
                    color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log2 FC vs control")
    fig.tight_layout()
    return save_figure(fig, Path(output)), pd.DataFrame(stat_rows)


def kinematics_scalars(kin: pd.DataFrame, output: Path, *, metrics: list[str],
                       group_col: str, control: str, group_order: list[str],
                       colours: dict[str, str] | None = None,
                       ) -> tuple[list[Path], pd.DataFrame]:
    """Strip+box of locomotion scalars per group, stars for edited vs control."""
    genes = [g for g in group_order if g != control]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.7 * len(metrics), 3.8), squeeze=False)
    stat_rows = []
    for ax, metric in zip(axes[0], metrics):
        pvals = _ctrl_vs_gene(kin, metric, group_col, control, genes)
        col = kin[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        vmax = col.max() if col.size else 1.0
        # heavy-tailed metric (e.g. path length, rotation) -> asinh y-axis so the
        # bulk is readable without hiding the tail; linear (fractions) otherwise.
        pos = col[col > 0]
        if pos.size and np.nanpercentile(pos, 99) > 12 * max(float(np.nanmedian(pos)), 1e-9):
            lw = max(float(np.nanmedian(pos)), 1e-6)
            try:
                ax.set_yscale("asinh", linear_width=lw)
            except (ValueError, ImportError):
                ax.set_yscale("symlog", linthresh=lw)
        for i, g in enumerate(group_order):
            vals = kin.loc[kin[group_col] == g, metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            if not vals.size:
                continue
            col = (colours or {}).get(g, WONG["black"])
            ax.scatter(i + rng.uniform(-0.14, 0.14, vals.size), vals, s=7, color=col, alpha=0.55, linewidths=0)
            ax.boxplot(vals, positions=[i], widths=0.55, showfliers=False,
                       medianprops={"color": col}, boxprops={"color": col},
                       whiskerprops={"color": col}, capprops={"color": col})
            if g in pvals:
                s = stars(pvals[g])
                if s and s != "ns":
                    ax.text(i, vmax * 1.03, s, ha="center", va="bottom", fontsize=9)
                stat_rows.append({"metric": metric, "group": g,
                                  "p_corr": round(pvals[g], 4), "stars": s})
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(group_order, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, Path(output), csv_data={"table": kin[[group_col, *metrics]]}), \
        pd.DataFrame(stat_rows)
