# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.report.cli                                 ║
# ║  « choose experiment types, get report.md + report.pdf »      ║
# ╚════════════════════════════════════════════════════════════════╝
"""Build a general genetic dataset report for chosen experiment types.

Genes, the control group, and colours are derived from the database metadata
(each group's parsed gene + edited/control role), so the report is not specific
to any one gene set.

``python -m tadpose.analysis.report.cli --experiment-types 7 \\
    --fingerprints .../fingerprints_wide.csv \\
    --kinematics   .../kinematics_per_tadpole.csv \\
    --output-dir   .../report``
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ... import config
from ...viz_constants import WONG
from . import build, figures
from .data import gather

#: prototype behaviour categories that make up the seizure signature.
SEIZURE_CATS: tuple[str, ...] = ("CSC", "ECSC", "UTB", "IMP", "HB", "FLIP", "SAC")
KIN_METRICS = ["path_length_mm", "mobile_fraction", "periphery_fraction",
               "total_rotation_rad", "circling_fraction", "darting_fraction"]
#: gene colours are assigned from this palette in group order.
_PALETTE = [WONG[k] for k in ("vermilion", "blue", "orange", "bluish_green",
                              "reddish_purple", "sky_blue", "yellow")]
CONTROL = "control"


def _png(paths: list[Path]) -> Path:
    return next((p for p in paths if str(p).endswith(".png")), paths[0])


def _seizure_prototypes():
    try:
        from ...viz_constants import pm_label
        protos = [i for i in range(36) if str(pm_label(i)).split(".")[0] in SEIZURE_CATS]
        return (protos or list(range(12))), pm_label
    except Exception:
        return list(range(12)), None


def _label_map(groups):
    """DB-derived tadpole_group_id -> display label; control groups -> 'control'."""
    gid_to_label, gene_order = {}, []
    for _, r in groups.iterrows():
        if r["role"] == "internal control":
            gid_to_label[r["tadpole_group_id"]] = CONTROL
        elif r["role"] == "edited":
            gid_to_label[r["tadpole_group_id"]] = r["gene"]
            if r["gene"] not in gene_order:
                gene_order.append(r["gene"])
    return gid_to_label, gene_order


def _relabel(df, gid_to_label):
    """Add a DB-derived 'label' column via tadpole_group_id; drop unlabelled rows."""
    if df is None or "tadpole_group_id" not in df.columns:
        return None
    out = df.copy()
    out["label"] = out["tadpole_group_id"].map(gid_to_label)
    return out[out["label"].notna()]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-types", required=True, help="Comma-separated experiment_type_ids.")
    p.add_argument("--db", type=Path,
                   default=config.configured_path("db_path", "databases", "xenopus_DEE.sqlite3"))
    p.add_argument("--fingerprints", type=Path, default=None)
    p.add_argument("--kinematics", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    etypes = [int(e) for e in a.experiment_types.split(",")]
    data = gather(a.db, etypes, a.fingerprints, a.kinematics)

    gid_to_label, gene_order = _label_map(data.groups)
    order = [CONTROL, *gene_order]
    colours = {CONTROL: WONG["black"],
               **{g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(gene_order)}}
    has_control = CONTROL in gid_to_label.values()
    figs: dict[str, Path] = {}
    caps: dict[str, str] = {}
    appendix: dict[str, object] = {}
    if not has_control:
        data.notes.append("No internal control group in the selection; "
                          "fold-change / significance omitted.")

    fp = _relabel(data.fingerprints, gid_to_label)
    if fp is not None and has_control and gene_order:
        protos, labeller = _seizure_prototypes()
        genes = [g for g in gene_order if g in set(fp["label"])]
        paths, stats = figures.fingerprint_heatmap(
            fp, out / "fig_fingerprint", group_col="label", control=CONTROL,
            genes=genes, prototypes=protos, labeller=labeller)
        figs["fingerprint"] = _png(paths)
        caps["fingerprint"] = ("Behavioural fingerprint: log2 fold-change of seizure-associated "
                               "prototype abundance, edited group vs internal control. Stars: "
                               "Fisher resampling with BH-FDR (* p<0.05, ** p<0.01, *** p<0.001).")
        appendix["Fingerprint — per-prototype fold-change + significance"] = stats

    kin = _relabel(data.kinematics, gid_to_label)
    if kin is not None and has_control:
        metrics = [m for m in KIN_METRICS if m in kin.columns]
        korder = [g for g in order if g in set(kin["label"])]
        paths, kstats = figures.kinematics_scalars(
            kin, out / "fig_kinematics", metrics=metrics, group_col="label",
            control=CONTROL, group_order=korder, colours=colours)
        figs["kinematics"] = _png(paths)
        caps["kinematics"] = ("Classic locomotion metrics per group (strip + box, one animal per "
                              "point; asinh y-axis where the distribution is heavy-tailed). Stars: "
                              "edited vs internal control (Fisher resampling, BH-FDR).")
        appendix["Kinematics — per-metric significance"] = kstats
        pt = Path(a.kinematics).parent / "path_traces.png"
        if pt.exists():
            shutil.copy2(pt, out / "fig_path_traces.png")
            figs["path_traces"] = out / "fig_path_traces.png"
            caps["path_traces"] = ("Representative tail_base path traces per group (well outline "
                                   "shown), up to six animals per group.")

    md = build.build_markdown(data, figs, caps, appendix, out)
    md_path = out / "report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"wrote {md_path}")
    note = build.to_pdf(md_path, out / "report.pdf")
    print(f"PDF: {note}" if note else f"wrote {out / 'report.pdf'}")


if __name__ == "__main__":
    main()
