# ╔════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.report                                     ║
# ║  « one-glance dataset reports (Markdown + PDF) »               ║
# ╠════════════════════════════════════════════════════════════════╣
# ║  Select experiment_types; get metadata + fingerprint +         ║
# ║  kinematics + a statistics appendix as report.md / report.pdf. ║
# ╚════════════════════════════════════════════════════════════════╝
"""One-glance dataset reports (Markdown + PDF).

``python -m tadpose.analysis.report.cli --experiment-types 7 --output-dir ...``
gathers the metadata, figures (significance as stars), and a statistics appendix
for the selected experiment types into report.md and report.pdf.
"""
from __future__ import annotations

from .data import ReportData, gather

__all__ = ["ReportData", "gather"]
