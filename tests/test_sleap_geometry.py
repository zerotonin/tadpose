"""Tests for tadpose.sleap_geometry."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tadpose.feature_extraction import WELL_DIAMETER_MM
from tadpose.sleap_geometry import (
    build_geometry_table,
    load_geometry_table,
    load_video_names,
    lookup_px_per_mm,
    save_geometry_table,
)
from tadpose.sleap_ingest import format_well_label, parse_well_label


@pytest.fixture
def geometry_json(tmp_path: Path) -> Path:
    path = tmp_path / "geometry_exp4.json"
    geometry = {"20": {"0": [60.0, 64.0, 54.6], "7": [61.0, 66.0, 52.0], "23": [59.0, 63.0, 55.0]}}
    path.write_text(json.dumps(geometry), encoding="utf-8")
    return path


@pytest.mark.parametrize("number", range(1, 25))
def test_format_well_label_inverts_parse(number: int) -> None:
    assert parse_well_label(format_well_label(number)) == number


def test_format_well_label_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        format_well_label(25)


def test_table_maps_crop_index_to_well_label(geometry_json: Path) -> None:
    table = build_geometry_table([geometry_json], {20: "240822cbnodrug_001_115005"})
    assert table.well.tolist() == ["A1", "B2", "D6"]
    assert table.loc[0, "px_per_mm"] == pytest.approx(2 * 54.6 / WELL_DIAMETER_MM)


def test_unknown_video_id_is_an_error(geometry_json: Path) -> None:
    with pytest.raises(KeyError):
        build_geometry_table([geometry_json], {})


def test_lookup_strips_the_sleap_prefix(geometry_json: Path, tmp_path: Path) -> None:
    table = build_geometry_table([geometry_json], {20: "240822cbnodrug_001_115005"})
    save_geometry_table(table, tmp_path / "geometry.csv")
    scales = lookup_px_per_mm(load_geometry_table(tmp_path / "geometry.csv"),
                              "cropped_240822cbnodrug_001_115005")
    assert set(scales) == {"A1", "B2", "D6"}
    with pytest.raises(KeyError, match="no ring geometry"):
        lookup_px_per_mm(table, "cropped_unknown_video")


def test_load_video_names_reads_db(tmp_path: Path) -> None:
    db = tmp_path / "mini.sqlite3"
    con = sqlite3.connect(db)
    con.execute("create table video (video_id integer, filename text)")
    con.execute("insert into video values (20, '240822cbnodrug_001_115005')")
    con.commit()
    con.close()
    assert load_video_names(db) == {20: "240822cbnodrug_001_115005"}
