"""Spatiotemporal index for accelerating vault queries.

Provides an R-tree spatial index (via libspatialindex/rtree) and a sorted-list
temporal index.  Both are read-optimized acceleration structures that can be
rebuilt from the authoritative SQLite file catalog at any time.
"""

from __future__ import annotations

import bisect
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

from rtree import index as rtree_index

if TYPE_CHECKING:
    from collections.abc import Iterable

    from voxelvault.models import FileRecord


# ---------------------------------------------------------------------------
# Spatial Index
# ---------------------------------------------------------------------------

class SpatialIndex:
    """R-tree backed spatial index for geographic bounding-box queries.

    Coordinates are ``(west, south, east, north)`` — the same order used by
    the rtree library's ``(minx, miny, maxx, maxy)`` convention.
    """

    def __init__(self, index_path: Path | None = None) -> None:
        self._id_to_file: dict[int, str] = {}
        self._file_to_id: dict[str, int] = {}
        self._next_id: int = 0
        self._index_path = index_path
        self._sidecar_path: Path | None = None

        props = rtree_index.Property()

        if index_path is not None:
            self._sidecar_path = index_path.with_suffix(".json")
            # Load mapping sidecar if it exists
            if self._sidecar_path.exists():
                self._load_sidecar()
            str_path = str(index_path)
            self._idx = rtree_index.Index(str_path, properties=props)
        else:
            self._idx = rtree_index.Index(properties=props)

    # -- sidecar persistence --------------------------------------------------

    def _load_sidecar(self) -> None:
        assert self._sidecar_path is not None
        data = json.loads(self._sidecar_path.read_text(encoding="utf-8"))
        self._id_to_file = {int(k): v for k, v in data["id_to_file"].items()}
        self._file_to_id = {v: int(k) for k, v in data["id_to_file"].items()}
        self._next_id = data["next_id"]

    def _save_sidecar(self) -> None:
        if self._sidecar_path is None:
            return
        data = {
            "id_to_file": {str(k): v for k, v in self._id_to_file.items()},
            "next_id": self._next_id,
        }
        self._sidecar_path.write_text(json.dumps(data), encoding="utf-8")

    # -- public API -----------------------------------------------------------

    def insert(self, file_id: str, bounds: tuple[float, float, float, float]) -> None:
        """Insert a file's bounding box.  *bounds* = ``(west, south, east, north)``."""
        if file_id in self._file_to_id:
            return  # already indexed
        int_id = self._next_id
        self._next_id += 1
        self._id_to_file[int_id] = file_id
        self._file_to_id[file_id] = int_id
        self._idx.insert(int_id, bounds)
        self._save_sidecar()

    def query(self, bounds: tuple[float, float, float, float]) -> list[str]:
        """Return file_ids whose bounding boxes intersect *bounds*."""
        hits = self._idx.intersection(bounds)
        return [self._id_to_file[h] for h in hits if h in self._id_to_file]

    def remove(self, file_id: str, bounds: tuple[float, float, float, float]) -> None:
        """Remove a file entry from the index."""
        int_id = self._file_to_id.pop(file_id, None)
        if int_id is None:
            return
        del self._id_to_file[int_id]
        self._idx.delete(int_id, bounds)
        self._save_sidecar()

    def count(self) -> int:
        """Return the number of indexed entries."""
        return len(self._id_to_file)

    def close(self) -> None:
        """Flush and close the underlying R-tree."""
        self._save_sidecar()
        self._idx.close()


# ---------------------------------------------------------------------------
# Temporal Index
# ---------------------------------------------------------------------------

class _TemporalEntry:
    """Lightweight container for a time interval keyed by *start*."""

    __slots__ = ("file_id", "start", "end")

    def __init__(self, file_id: str, start: datetime, end: datetime) -> None:
        self.file_id = file_id
        self.start = start
        self.end = end


class TemporalIndex:
    """Sorted-list temporal index using binary search for overlap queries.

    Entries are sorted by ``start``.  A query ``[qs, qe]`` returns every
    entry where ``start <= qe AND end >= qs``.
    """

    def __init__(self) -> None:
        self._entries: list[_TemporalEntry] = []
        self._starts: list[datetime] = []  # parallel list for bisect
        self._file_ids: set[str] = set()

    def insert(self, file_id: str, start: datetime, end: datetime) -> None:
        if file_id in self._file_ids:
            return
        self._file_ids.add(file_id)
        entry = _TemporalEntry(file_id, start, end)
        pos = bisect.bisect_right(self._starts, start)
        self._starts.insert(pos, start)
        self._entries.insert(pos, entry)

    def query(self, start: datetime, end: datetime) -> list[str]:
        """Return file_ids whose time ranges overlap ``[start, end]``."""
        if not self._entries:
            return []

        # Candidates must have entry.start <= end.
        right = bisect.bisect_right(self._starts, end)
        results: list[str] = []
        for i in range(right):
            if self._entries[i].end >= start:
                results.append(self._entries[i].file_id)
        return results

    def remove(self, file_id: str) -> None:
        if file_id not in self._file_ids:
            return
        self._file_ids.discard(file_id)
        for i, entry in enumerate(self._entries):
            if entry.file_id == file_id:
                del self._entries[i]
                del self._starts[i]
                return

    def count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Combined Catalog Index
# ---------------------------------------------------------------------------

class CatalogIndex:
    """Combined spatial + temporal index backed by :class:`SpatialIndex` and
    :class:`TemporalIndex`.

    Acts as a context manager so the underlying R-tree is properly flushed.
    """

    def __init__(self, index_dir: Path | None = None) -> None:
        spatial_path: Path | None = None
        if index_dir is not None:
            index_dir.mkdir(parents=True, exist_ok=True)
            spatial_path = index_dir / "spatial"
        self._spatial = SpatialIndex(index_path=spatial_path)
        self._temporal = TemporalIndex()

    # -- public API -----------------------------------------------------------

    def insert(self, record: FileRecord) -> None:
        ext = record.spatial_extent
        self._spatial.insert(record.file_id, (ext.west, ext.south, ext.east, ext.north))
        te = record.temporal_extent
        self._temporal.insert(record.file_id, te.start, te.end)

    def query(
        self,
        spatial_bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[datetime, datetime] | None = None,
    ) -> set[str]:
        """Return file_ids matching spatial AND/OR temporal criteria."""
        spatial_hits: set[str] | None = None
        temporal_hits: set[str] | None = None

        if spatial_bounds is not None:
            spatial_hits = set(self._spatial.query(spatial_bounds))
        if temporal_range is not None:
            temporal_hits = set(self._temporal.query(temporal_range[0], temporal_range[1]))

        if spatial_hits is not None and temporal_hits is not None:
            return spatial_hits & temporal_hits
        if spatial_hits is not None:
            return spatial_hits
        if temporal_hits is not None:
            return temporal_hits
        return set()

    def remove(self, record: FileRecord) -> None:
        ext = record.spatial_extent
        self._spatial.remove(record.file_id, (ext.west, ext.south, ext.east, ext.north))
        self._temporal.remove(record.file_id)

    def rebuild(self, records: Iterable[FileRecord]) -> None:
        """Rebuild the entire index from scratch."""
        # Close existing spatial index and create a fresh one
        old_path = self._spatial._index_path
        self._spatial.close()
        # Remove old index files if persistent
        if old_path is not None:
            for suffix in (".idx", ".dat", ".json"):
                p = old_path.with_suffix(suffix)
                if p.exists():
                    p.unlink()
        self._spatial = SpatialIndex(index_path=old_path)
        self._temporal = TemporalIndex()
        for record in records:
            self.insert(record)

    def close(self) -> None:
        self._spatial.close()

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
