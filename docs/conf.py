# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — docs/conf.py                                          ║
# ║  « Sphinx configuration: autodoc + Napoleon + RTD sidebar »      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Sphinx build configuration for the TadPose documentation."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

# Make the package importable for autodoc even without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ── Project information ──────────────────────────────────────────
project = "TadPose"
author = "Alexander R. H. Matthews, Caroline Beck, Bart R. H. Geurten"
copyright = "2026, University of Otago, Department of Zoology"

try:
    release = _pkg_version("tadpose")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# ── General configuration ────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Heavy / GPU / hardware deps are mocked so docs build on a cheap runner.
autodoc_mock_imports = [
    "deeplabcut",
    "cuml",
    "cupy",
    "torch",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML output ──────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_title = f"TadPose {version}"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
