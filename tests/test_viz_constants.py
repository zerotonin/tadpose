# ─────────────────────────────────────────────────────────────────
#  TadPose — tests/test_viz_constants.py
#  « save_figure triple output, results tree, significance notation »
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import pandas as pd
import pytest

# matplotlib is required for save_figure; skip the whole module if the
# environment cannot import it (CI installs a working matplotlib).
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tadpose import viz_constants as vc
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"matplotlib unavailable: {exc}", allow_module_level=True)


def test_save_figure_writes_svg_and_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    saved = vc.save_figure(fig, tmp_path / "demo")
    plt.close(fig)
    suffixes = {p.suffix for p in saved}
    assert ".svg" in suffixes and ".png" in suffixes
    assert (tmp_path / "demo.svg").exists()
    assert (tmp_path / "demo.png").exists()


def test_save_figure_writes_csv_companion(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    vc.save_figure(fig, tmp_path / "demo", csv_data={"points": data})
    plt.close(fig)
    csv = tmp_path / "demo_points.csv"
    assert csv.exists()
    assert pd.read_csv(csv)["y"].tolist() == [0, 1]


def test_svg_keeps_text_editable(tmp_path):
    fig, ax = plt.subplots()
    ax.set_title("editable")
    vc.save_figure(fig, tmp_path / "t", formats=("svg",))
    plt.close(fig)
    svg = (tmp_path / "t.svg").read_text(encoding="utf-8")
    # svg.fonttype="none" emits <text> elements rather than vector paths.
    assert "<text" in svg


def test_make_results_tree_creates_dirs(tmp_path):
    dirs = vc.make_results_tree(tmp_path / "results")
    for path in dirs.values():
        assert path.is_dir()


@pytest.mark.parametrize("p,expected", [
    (0.0001, "★★★"), (0.005, "★★"), (0.03, "★"), (0.2, "n.s."),
])
def test_sig_stars(p, expected):
    assert vc.sig_stars(p) == expected
