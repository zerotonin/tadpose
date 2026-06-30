"""Tests for the spreadsheet -> plate-groups helper (metadata_ingest).

Covers the messy-sheet parsing dialects, the three-tier video matcher and its
ambiguity guard, the per-well layout, and the get-or-create idempotency that
keeps re-running a plan from duplicating Frog / TadpoleGroup / WellType rows.
"""
from __future__ import annotations

import pytest

from tadpose.database import DatabaseHandler
from tadpose.metadata_ingest import (
    PlateRecord,
    build_plate_layout,
    compact_wells,
    get_or_create_frog,
    get_or_create_tadpole_group,
    get_or_create_well_type,
    match_video,
    normalise_transgene,
    parse_well_range,
    write_meta_data_csv,
)


def _rec(key: str, videos: list[str], **kw) -> PlateRecord:
    base = dict(
        key=key, initials="CB", female_identifier="A", background_strain="WT",
        female_tank="", target_gene="NeuroD2", sgrna="g20", seq_folder="nGFP",
        fert_date_raw="", fert_date=None, exp_date=None,
        control_arenas="", test_arenas="1 to 24", drug_dose="", notes="",
        videos=videos,
    )
    base.update(kw)
    return PlateRecord(**base)


@pytest.mark.parametrize("text, expected", [
    ("1 to 24", list(range(1, 25))),
    ("1-24", list(range(1, 25))),
    ("1 to  24", list(range(1, 25))),                       # sheet's double space
    ("1,2 4-24", [1, 2] + list(range(4, 25))),              # mixed comma + space
    ("1 to 19, 21, 22", list(range(1, 20)) + [21, 22]),
    ("1 to 17", list(range(1, 18))),
    ("", []),
])
def test_parse_well_range_dialects(text, expected):
    assert parse_well_range(text) == expected


def test_compact_wells_roundtrips():
    # canonical forms: adjacent singletons collapse to a run (1,2 -> 1-2)
    for text in ("1-24", "1-2,4-24", "1-19,21-22", "1-17"):
        assert compact_wells(parse_well_range(text)) == text
    assert compact_wells(parse_well_range("1,2 4-24")) == "1-2,4-24"


def test_normalise_transgene():
    assert normalise_transgene("NeuroD2", "g20") == "NeuroD2 g20"
    assert normalise_transgene("no_xenopus_target", "g15_5MM") == "no_xenopus_target g15_5MM"
    assert normalise_transgene("null", "null") == "WT"
    assert normalise_transgene("", "") == "WT"


def test_match_video_three_tiers():
    records = [
        _rec("#1", ["241106Aap1_001_120823.mp4"]),             # clean
        _rec("#2", ["241106nd1_001_131251.mp4"]),              # sheet drops the 'A'
        _rec("#3", ["241106_A5m2_001_110633.mp4"]),            # time typo in sheet
    ]
    # exact normalised
    rec, how = match_video("241106Aap1_001_120823.mp4", records)
    assert (rec.key, how) == ("#1", "exact")
    # tail (index+time) rescues the missing-letter prefix
    rec, how = match_video("241106And1_001_131251.mp4", records)
    assert (rec.key, how) == ("#2", "tail")
    # prefix rescues the wrong-timestamp case
    rec, how = match_video("241106A5m2_001_110622.mp4", records)
    assert (rec.key, how) == ("#3", "prefix")


def test_match_video_ambiguous_is_refused():
    # same filename on two distinct rows -> no guess
    records = [
        _rec("#A", ["241107b5m1_001_111443.mp4"], female_identifier="A"),
        _rec("#B", ["241107b5m1_001_111443.mp4"], female_identifier="B"),
    ]
    rec, how = match_video("241107b5m1_001_111443.mp4", records)
    assert rec is None and how == "ambiguous"


def test_build_plate_layout_masks_empty_wells():
    layout = build_plate_layout([1, 2, 3], group_id=7, well_type_id=1, n_wells=24)
    assert layout["tadpole_type_ids"][:3] == [7, 7, 7]
    assert layout["tadpole_type_ids"][3:] == [None] * 21
    assert layout["well_type_ids"][0] == 1 and layout["well_type_ids"][5] is None


def test_meta_data_csv_blanks_empty_wells(tmp_path):
    out = tmp_path / "meta_data_table.csv"
    layout = build_plate_layout([1, 2], group_id=4, well_type_id=1)
    write_meta_data_csv(out, layout, investigator_id=3, experiment_type_id=7)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "well_number,investigator_id,experiment_type_id,well_type_ids,tadpole_type_ids"
    assert lines[1] == "0,3,7,1,4"
    assert lines[3] == "2,3,7,,"            # well 3 unoccupied -> blanks
    assert len(lines) == 25                  # header + 24 wells


def test_get_or_create_is_idempotent(tmp_path):
    handler = DatabaseHandler(f"sqlite:///{tmp_path / 'db.sqlite3'}")
    with handler as db:
        f1 = get_or_create_frog(db, "A", "WT", "")
        f2 = get_or_create_frog(db, "A", "WT", "")
        assert f1.frog_id == f2.frog_id
        g1 = get_or_create_tadpole_group(db, f1.frog_id, None, "NeuroD2 g20", "nGFP")
        g2 = get_or_create_tadpole_group(db, f1.frog_id, None, "NeuroD2 g20", "nGFP")
        assert g1.tadpole_group_id == g2.tadpole_group_id
        # a different transgene is a new group
        g3 = get_or_create_tadpole_group(db, f1.frog_id, None, "Eef1a2 g15", "nGFP")
        assert g3.tadpole_group_id != g1.tadpole_group_id
        w1 = get_or_create_well_type(db)
        w2 = get_or_create_well_type(db)
        assert w1.well_type_id == w2.well_type_id
