# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_cluster_naming.py
#  « every k=36 prototype gets a unique GROUP.index label »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from tadpose import viz_constants as vc


def test_all_36_clusters_have_a_unique_label():
    members = [c for ids in vc.THESIS_K36_GROUPS.values() for c in ids]
    assert sorted(members) == list(range(36))          # every raw id, once
    assert len(set(members)) == 36                      # no cluster in two groups
    labels = [vc.pm_label(c) for c in range(36)]
    assert len(set(labels)) == 36                       # labels unique


def test_label_format_and_prevalence_order():
    # First member of each group is index .1 (most prevalent within group).
    assert vc.pm_label(22) == "CSC.1"     # most frequent C-SC
    assert vc.pm_label(3) == "CSC.4"
    assert vc.pm_label(2) == "REST.1"     # the dominant at-rest cluster
    assert vc.pm_label(18) == "IMP.1"
    assert vc.pm_label(10) == "FLIP.1"
    assert vc.pm_label(19) == "SAC.1"     # 19 is a turn saccade, not the flip


def test_pm_category_and_fallback():
    assert vc.pm_category(15) == "csc_edge"
    assert vc.pm_category(0) == "undulatory_swimming"
    assert vc.pm_label(99) == "C99"        # unknown id falls back gracefully
    assert vc.pm_category(99) == "unclassified"


def test_thesis_map_builds_with_labels():
    from tadpose.analysis.cluster_map import thesis_k36_map

    cmap = thesis_k36_map()
    assert cmap.k == 36
    assert cmap.label(22) == "CSC.1"
    assert cmap.category(22) == "csc"
    # display ids are a 0..35 permutation
    assert sorted(cmap.display_id(c) for c in range(36)) == list(range(36))
