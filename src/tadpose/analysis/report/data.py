# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.report.data                                ║
# ║  « gather dataset metadata + analysis tables for a report »    ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Given a set of experiment_type ids, pull the who/when/how,     ║
# ║  the groups and their internal 5MM controls, the cross-dataset  ║
# ║  global reference cohorts (PTZ / 4AP / WT), and the precomputed  ║
# ║  fingerprint + kinematics per-animal tables, filtered to them.  ║
# ╚════════════════════════════════════════════════════════════════╝
"""Gather dataset metadata + analysis tables for a dataset report.

The report selects whole ``experiment_type`` rows.  ``video.date_time`` is the
processing timestamp, so recording dates are parsed from the ``YYMMDD`` filename
prefix.  Controls are classified from the transgene string: a guide containing
``5MM`` / ``null`` is a mismatch/non-targeting **internal** control; the
seizure-positive (PTZ, 4AP) and negative (untreated / WT) cohorts in the OTHER
experiment types are the **global** reference controls.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: experiment_type short-name substrings that mark the global reference cohorts.
GLOBAL_CONTROL_MARKERS: dict[str, str] = {
    "PTZ": "seizure-positive (PTZ)",
    "4AP": "seizure-positive (4-AP)",
}
_MOTHER_CLUTCH: dict[int, str] = {10: "A", 11: "B", 3: "P"}
_DATE_RE = re.compile(r"(\d{2})(\d{2})(\d{2})")          # YYMMDD in the filename


def _connect(db_file: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_file)}?mode=ro", uri=True)


def _recording_date(filename: str) -> str:
    """Parse the YYMMDD prefix of a plate filename into an ISO date."""
    m = _DATE_RE.search(filename or "")
    if not m:
        return "unknown"
    yy, mm, dd = m.groups()
    return f"20{yy}-{mm}-{dd}"


def parse_transgene(transgene: str) -> dict[str, object]:
    """Split a transgene string into gene / guide / control role (best effort)."""
    s = (transgene or "").strip()
    low = s.lower()
    is_empty = "no tad" in low or s in ("", "NO TADOLE")
    is_control = ("5mm" in low or "null" in low) and not is_empty
    gene = "empty" if is_empty else None
    for key in ("neurod2", "eef1a2", "eelfa2", "eefla2", "ap2b3", "egfla2"):
        if key in low:
            gene = {"neurod2": "NeuroD2", "ap2b3": "Ap2b3"}.get(key, "Eef1a2")
            break
    if gene is None:
        gene = "WT/other"
    guide = "g15_5MM" if "5mm" in low else ("g20" if "g20" in low else
            ("null" if "null" in low else s.split(":")[-1].strip() or "-"))
    return {"gene": gene, "guide": guide, "is_control": is_control, "is_empty": is_empty}


@dataclass
class ReportData:
    """Everything a report needs, already resolved to tables."""
    experiment_type_ids: list[int]
    experiments: pd.DataFrame          # one row per experiment_type
    groups: pd.DataFrame               # one row per tadpole_group in the selection
    internal_controls: pd.DataFrame    # the within-selection 5MM/null groups
    global_controls: pd.DataFrame      # cross-dataset PTZ/4AP/WT reference cohorts
    fingerprints: pd.DataFrame | None = None       # per-animal, filtered
    kinematics: pd.DataFrame | None = None         # per-animal, filtered
    notes: list[str] = field(default_factory=list)


def _experiments_table(con, etypes) -> pd.DataFrame:
    ph = ",".join("?" * len(etypes))
    rows = con.execute(
        f"""select et.experiment_type_id, et.short_name, et.long_name, et.protocol,
                   group_concat(distinct inv.first_name || ' ' || inv.last_name),
                   min(es.experiment_date),
                   max(es.experiment_date), count(distinct v.video_id)
            from experiment_type et
            join experiment_series es on es.experiment_type_id = et.experiment_type_id
            left join investigator inv on inv.investigator_id = es.investigator_id
            left join video v on v.series_id = es.series_id
            where et.experiment_type_id in ({ph})
            group by et.experiment_type_id""", [int(e) for e in etypes]).fetchall()
    return pd.DataFrame(rows, columns=[
        "experiment_type_id", "short_name", "long_name", "protocol",
        "investigators", "series_first", "series_last", "n_videos"])


def _groups_table(con, etypes) -> pd.DataFrame:
    ph = ",".join("?" * len(etypes))
    rows = con.execute(
        f"""select tg.tadpole_group_id, tg.transgene, tg.mother_id, tg.development_stage,
                   tg.fertilisation_date, count(distinct t.trial_id),
                   count(distinct v.video_id), min(v.filename)
            from trial t join video v on t.video_id = v.video_id
            join experiment_series es on v.series_id = es.series_id
            join tadpole_group tg on t.tadpole_group_id = tg.tadpole_group_id
            where es.experiment_type_id in ({ph})
            group by tg.tadpole_group_id order by tg.tadpole_group_id""",
        [int(e) for e in etypes]).fetchall()
    df = pd.DataFrame(rows, columns=[
        "tadpole_group_id", "transgene", "mother_id", "stage", "fertilisation_date",
        "n_animals", "n_videos", "sample_file"])
    parsed = df["transgene"].apply(parse_transgene).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    df["clutch"] = df["mother_id"].map(_MOTHER_CLUTCH).fillna(df["mother_id"].astype(str))
    df["recording_date"] = df["sample_file"].apply(_recording_date)
    df["role"] = df.apply(
        lambda r: "empty" if r["is_empty"] else ("internal control" if r["is_control"] else "edited"),
        axis=1)
    return df.drop(columns=["sample_file"])


def _global_controls(con, exclude_types) -> pd.DataFrame:
    """Cross-dataset reference cohorts (PTZ / 4AP seizure-positive; WT negative)."""
    rows = con.execute(
        """select et.experiment_type_id, et.short_name,
                  count(distinct t.trial_id), count(distinct t.tadpole_group_id)
           from experiment_type et
           join experiment_series es on es.experiment_type_id = et.experiment_type_id
           join video v on v.series_id = es.series_id
           join trial t on t.video_id = v.video_id
           group by et.experiment_type_id""").fetchall()
    out = []
    for et, sn, n_tr, n_g in rows:
        if et in exclude_types:
            continue
        role = next((v for k, v in GLOBAL_CONTROL_MARKERS.items()
                     if k.lower() in (sn or "").lower()), None)
        if role:
            out.append({"experiment_type_id": et, "short_name": sn,
                        "role": role, "n_animals": n_tr, "n_groups": n_g})
    return pd.DataFrame(out, columns=[
        "experiment_type_id", "short_name", "role", "n_animals", "n_groups"])


def _load_analysis(path: Path | None, group_ids: list[int]) -> pd.DataFrame | None:
    """Load a per-animal analysis table and filter to the selection's groups."""
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    gcol = next((c for c in ("tadpole_group_id", "group_id") if c in df.columns), None)
    if gcol is not None:
        df = df[df[gcol].isin(group_ids)]
    return df


def gather(db_file: Path, experiment_type_ids: list[int],
           fingerprints_csv: Path | None = None,
           kinematics_csv: Path | None = None) -> ReportData:
    """Assemble every table the report needs for the selected experiment types."""
    etypes = [int(e) for e in experiment_type_ids]
    with _connect(db_file) as con:
        experiments = _experiments_table(con, etypes)
        groups = _groups_table(con, etypes)
        global_controls = _global_controls(con, exclude_types=set(etypes))
    internal = groups[groups["role"] == "internal control"].copy()
    gids = groups["tadpole_group_id"].tolist()
    data = ReportData(
        experiment_type_ids=etypes, experiments=experiments, groups=groups,
        internal_controls=internal, global_controls=global_controls,
        fingerprints=_load_analysis(fingerprints_csv, gids),
        kinematics=_load_analysis(kinematics_csv, gids))
    if data.fingerprints is None:
        data.notes.append("Fingerprint table not found for the selection; section omitted.")
    if data.kinematics is None:
        data.notes.append("Kinematics table not found for the selection; section omitted.")
    return data
