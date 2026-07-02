"""Unit tests for the report's pure helpers (no database, no pandoc)."""
from __future__ import annotations

import pandas as pd

from tadpose.analysis.report.build import df_to_md
from tadpose.analysis.report.data import parse_transgene
from tadpose.analysis.report.figures import stars


def test_parse_transgene_edited_vs_control():
    edited = parse_transgene("G: NeuroD2, SgRNA: g20")
    assert edited["gene"] == "NeuroD2" and not edited["is_control"]
    ctrl = parse_transgene("NeuroD2_g15_5MM")
    assert ctrl["is_control"] and ctrl["guide"] == "g15_5MM"
    assert parse_transgene("Ap2b3 g2")["gene"] == "Ap2b3"
    assert parse_transgene("NO TADOLE")["is_empty"]


def test_stars_thresholds():
    assert stars(0.0005) == "***"
    assert stars(0.005) == "**"
    assert stars(0.03) == "*"
    assert stars(0.2) == "ns"
    assert stars(float("nan")) == ""


def test_df_to_md_renders_nan_as_dash():
    df = pd.DataFrame({"a": [1, float("nan")], "b": ["x", "y"]})
    md = df_to_md(df)
    assert md.startswith("| a | b |")
    assert "—" in md          # NaN cell
    assert df_to_md(pd.DataFrame()) == "_none_\n"
