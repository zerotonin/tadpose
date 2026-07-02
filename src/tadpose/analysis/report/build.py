# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.report.build                               ║
# ║  « assemble the Markdown report and render it to PDF »         ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Tables + captioned figures + a statistics appendix, no flow    ║
# ║  text.  Markdown is authoritative; PDF is rendered via pandoc    ║
# ║  (xelatex).  If pandoc is absent the Markdown is still written.  ║
# ╚════════════════════════════════════════════════════════════════╝
"""Assemble the Markdown report and render it to PDF.

The report is tables and captioned figures only -- a one-glance overview.  PDF
rendering uses pandoc with the xelatex engine; when pandoc is unavailable the
Markdown (and figures) are still produced and a note is returned.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd


#: compact column headers so the PDF table columns do not run into each other.
SHORT_LABELS: dict[str, str] = {
    "experiment_type_id": "type", "tadpole_group_id": "group",
    "recording_date": "date", "fertilisation_date": "fert.",
    "development_stage": "stage", "n_animals": "n", "n_videos": "vids",
    "short_name": "name", "long_name": "description", "investigators": "who",
    "series_first": "first run", "series_last": "last run",
    "groups (non-empty)": "groups", "p_corr": "p(adj)", "log2FC": "log2 FC",
    "prototype": "PM", "n_groups": "n grp",
}


def df_to_md(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Render a DataFrame as a GitHub Markdown table (NaN -> em dash).

    Column names are shortened via SHORT_LABELS so the rendered PDF columns stay
    legible and do not overlap.
    """
    if df is None or df.empty:
        return "_none_\n"
    cols = columns or list(df.columns)
    head = "| " + " | ".join(SHORT_LABELS.get(str(c), str(c)) for c in cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, rule]
    for _, row in df[cols].iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:.3g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _rel(p: Path, base: Path) -> str:
    try:
        return str(Path(p).relative_to(base))
    except ValueError:
        return str(p)


def build_markdown(data, figures: dict[str, Path], captions: dict[str, str],
                   appendix: dict[str, pd.DataFrame], output_dir: Path) -> str:
    """Assemble the full report Markdown string."""
    out = Path(output_dir)
    exp = data.experiments
    names = ", ".join(exp["short_name"].astype(str))
    occupied = data.groups[data.groups["role"] != "empty"]
    md: list[str] = []
    md.append(f"# TadPose dataset report — {names}\n")

    # 1. summary (no flow text: a table)
    summ = pd.DataFrame([{
        "experiments": len(exp),
        "groups (non-empty)": occupied["tadpole_group_id"].nunique(),
        "animals": int(occupied["n_animals"].sum()),
        "videos": int(exp["n_videos"].sum()),
        "recording dates": " to ".join(sorted(set(occupied["recording_date"]))[::max(1, len(set(occupied['recording_date']))-1)]),
    }])
    md.append("## 1  Summary\n")
    md.append(df_to_md(summ))

    # 2. experiments
    md.append("\n## 2  Experiments\n")
    md.append(df_to_md(exp, ["experiment_type_id", "short_name", "long_name",
                             "protocol", "investigators", "n_videos"]))

    # 3. groups
    md.append("\n## 3  Groups\n")
    md.append(df_to_md(occupied, ["tadpole_group_id", "gene", "guide", "clutch",
                                  "stage", "recording_date", "n_animals", "n_videos", "role"]))
    n_empty = int(data.groups.loc[data.groups["role"] == "empty", "n_animals"].sum())
    if n_empty:
        md.append(f"\n_{n_empty} empty wells (no tadpole) excluded from analysis._\n")

    # 4. controls
    md.append("\n## 4  Controls\n")
    md.append("**Internal** (within-experiment, non-targeting 5MM guide, clutch-matched):\n\n")
    md.append(df_to_md(data.internal_controls, ["tadpole_group_id", "gene", "guide",
                                                "clutch", "n_animals"]))
    md.append("\n**Global** (cross-dataset reference cohorts):\n\n")
    md.append(df_to_md(data.global_controls))

    # 5. fingerprint  (empty alt text -> no redundant pandoc auto-caption)
    if "fingerprint" in figures:
        md.append("\n## 5  Behavioural fingerprint\n")
        md.append(f"![]({_rel(figures['fingerprint'], out)})\n")
        md.append(f"\n**Figure 1.** {captions.get('fingerprint','')}\n")

    # 6. kinematics
    if "kinematics" in figures:
        md.append("\n## 6  Classic locomotion kinematics\n")
        md.append(f"![]({_rel(figures['kinematics'], out)})\n")
        md.append(f"\n**Figure 2.** {captions.get('kinematics','')}\n")
    if "path_traces" in figures:
        md.append(f"\n![]({_rel(figures['path_traces'], out)})\n")
        md.append(f"\n**Figure 3.** {captions.get('path_traces','')}\n")

    # notes
    if data.notes:
        md.append("\n## Notes\n")
        for n in data.notes:
            md.append(f"- {n}\n")

    # appendix
    letters = iter("ABCDEFGH")
    md.append("\n---\n\n# Appendix — statistics\n")
    for name, tbl in appendix.items():
        md.append(f"\n## Appendix {next(letters)} — {name}\n")
        md.append(df_to_md(tbl))
    return "\n".join(md)


def to_pdf(md_path: Path, pdf_path: Path) -> str | None:
    """Render Markdown to PDF via pandoc+xelatex.  Returns a note if unavailable."""
    if shutil.which("pandoc") is None:
        return "pandoc not found; wrote Markdown only (install pandoc for PDF)."
    engine = "xelatex" if shutil.which("xelatex") else "pdflatex"
    cmd = ["pandoc", str(md_path), "-o", str(pdf_path),
           f"--pdf-engine={engine}", "-V", "geometry:margin=2cm"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(md_path).parent))
    if r.returncode != 0:
        return f"pandoc failed: {r.stderr.strip()[:400]}"
    return None
