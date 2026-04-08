# ╔══════════════════════════════════════════════════════════════════╗
# ║  TadPose — analysis.cluster_map                                  ║
# ║  « giving 36 arbitrary numbers a name and an order »             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  K-means cluster IDs are arbitrary.  This module provides        ║
# ║  a mapping from raw cluster numbers to behavioural categories    ║
# ║  and a canonical display order for publication figures.          ║
# ║                                                                  ║
# ║  The mapping is stored as a JSON file so Alex and Bart can       ║
# ║  edit it without touching Python code.  A default mapping        ║
# ║  is built in for the k=36 clustering used in the thesis.         ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ┌──────────────────────────────────────────────────────────────┐
# │ Data structures  « what we know about each cluster »         │
# └──────────────────────────────────────────────────────────────┘

@dataclass
class ClusterInfo:
    """Metadata for a single behavioural prototype."""
    raw_id: int                          # original k-means label
    display_id: int                      # position in publication order
    category: str                        # behaviour category key (matches BEHAVIOUR_COLOURS)
    short_label: str = ""                # e.g. "CSC-3", "swim-1"
    description: str = ""                # free-text note


@dataclass
class ClusterMap:
    """Full mapping for one clustering solution.

    Attributes:
        k:        Number of clusters.
        entries:  List of ClusterInfo, one per cluster.
        _by_raw:  Lookup dict raw_id → ClusterInfo (built lazily).
    """
    k: int
    entries: list[ClusterInfo] = field(default_factory=list)
    _by_raw: dict[int, ClusterInfo] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_raw = {e.raw_id: e for e in self.entries}

    # ── lookup ───────────────────────────────────────────────

    def category(self, raw_id: int) -> str:
        """Return the behavioural category for a raw cluster ID."""
        return self._by_raw[raw_id].category

    def display_id(self, raw_id: int) -> int:
        """Return the publication display position for a raw ID."""
        return self._by_raw[raw_id].display_id

    def label(self, raw_id: int) -> str:
        """Short label for legends."""
        info = self._by_raw[raw_id]
        return info.short_label or f"C{info.display_id}"

    # ── reordering ───────────────────────────────────────────

    def display_order(self) -> list[int]:
        """Raw IDs sorted by display_id (publication order)."""
        return [e.raw_id for e in sorted(self.entries, key=lambda e: e.display_id)]

    def sort_by_display(self, data: NDArray, axis: int = 0) -> NDArray:
        """Reorder rows (or columns) of *data* from raw order to
        publication display order.

        Args:
            data: Array whose *axis* dimension has length k.
            axis: Axis to reorder.

        Returns:
            Reordered copy of the array.
        """
        order = self.display_order()
        return np.take(data, order, axis=axis)

    def sort_labels(self, labels: NDArray) -> NDArray:
        """Remap a label array from raw IDs to display IDs.

        Args:
            labels: (N,) array of raw cluster labels.

        Returns:
            (N,) array of display IDs.
        """
        mapping = np.zeros(self.k, dtype=int)
        for e in self.entries:
            mapping[e.raw_id] = e.display_id
        return mapping[labels]

    # ── persistence ──────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write the mapping to a JSON file."""
        path = Path(path)
        out = {
            "k": self.k,
            "clusters": [
                {
                    "raw_id": e.raw_id,
                    "display_id": e.display_id,
                    "category": e.category,
                    "short_label": e.short_label,
                    "description": e.description,
                }
                for e in sorted(self.entries, key=lambda e: e.display_id)
            ],
        }
        path.write_text(json.dumps(out, indent=2))

    @classmethod
    def load(cls, path: Path) -> ClusterMap:
        """Load a mapping from a JSON file."""
        path = Path(path)
        raw = json.loads(path.read_text())
        entries = [
            ClusterInfo(
                raw_id=c["raw_id"],
                display_id=c["display_id"],
                category=c["category"],
                short_label=c.get("short_label", ""),
                description=c.get("description", ""),
            )
            for c in raw["clusters"]
        ]
        return cls(k=raw["k"], entries=entries)


# ┌──────────────────────────────────────────────────────────────┐
# │ Convenience constructors  « for common cases »               │
# └──────────────────────────────────────────────────────────────┘

def identity_map(k: int, default_category: str = "unclassified") -> ClusterMap:
    """Create a trivial 1:1 mapping where display_id == raw_id.

    Useful as a starting point before manual annotation.
    """
    entries = [
        ClusterInfo(
            raw_id=i,
            display_id=i,
            category=default_category,
        )
        for i in range(k)
    ]
    return ClusterMap(k=k, entries=entries)


def map_from_dict(
    raw_to_category: dict[int, str],
    sort_key: Optional[str] = "category",
) -> ClusterMap:
    """Build a ClusterMap from a simple {raw_id: category} dict.

    Display IDs are assigned by sorting on *sort_key* (either
    'category' for grouping by behaviour, or 'raw_id' for original
    order).

    Args:
        raw_to_category: Mapping from raw cluster ID to category string.
        sort_key:        How to assign display order.

    Returns:
        ClusterMap with display IDs assigned.
    """
    items = list(raw_to_category.items())

    if sort_key == "category":
        items.sort(key=lambda x: (x[1], x[0]))
    else:
        items.sort(key=lambda x: x[0])

    entries = [
        ClusterInfo(raw_id=raw, display_id=i, category=cat)
        for i, (raw, cat) in enumerate(items)
    ]
    return ClusterMap(k=len(entries), entries=entries)
