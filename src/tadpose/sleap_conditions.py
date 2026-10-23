# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — sleap_conditions                                      ║
# ║  « which plate, which drug, from the stem »                      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The Sept-2024 pharmacology videos name plate and condition in   ║
# ║  their stem (cropped_<yymmdd><tag>_001_<hhmmss>).  Baseline and  ║
# ║  drug recordings of one plate hold the same animals, so a well   ║
# ║  pairs baseline with drug.                                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Condition and plate of a SLEAP pharmacology video, from its stem."""

from __future__ import annotations

from tadpose.sleap_constants import SLEAP_CONDITION_PREFIXES, SLEAP_PLATE_PATTERN
from tadpose.sleap_geometry import SLEAP_STEM_PREFIX


def parse_video_tag(video_stem: str) -> str:
    """``cropped_2409124apfrogD_001_135831`` -> ``4apfrogd`` (date dropped, lower case)."""
    name = video_stem.removeprefix(SLEAP_STEM_PREFIX).split("_")[0]
    return name[6:].lower()


def parse_condition(video_stem: str) -> str:
    """Condition name (see ``SLEAP_CONDITION_PREFIXES``) of a video."""
    tag = parse_video_tag(video_stem)
    for prefix, condition in SLEAP_CONDITION_PREFIXES:
        if tag.startswith(prefix):
            return condition
    raise ValueError(f"no condition prefix matches video {video_stem!r} (tag {tag!r})")


def parse_plate(video_stem: str) -> str:
    """Plate id shared by the baseline and drug recordings of one plate."""
    tag = parse_video_tag(video_stem)
    match = SLEAP_PLATE_PATTERN.search(tag)
    if match is None:
        raise ValueError(f"no plate id in video {video_stem!r} (tag {tag!r})")
    return match.group(0)
