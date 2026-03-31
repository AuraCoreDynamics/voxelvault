"""Tests for the spatiotemporal index module."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone


from voxelvault.index import CatalogIndex, SpatialIndex, TemporalIndex
from voxelvault.models import FileRecord, SpatialExtent, TemporalExtent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file_record(
    file_id: str | None = None,
    west: float = -10.0,
    south: float = -10.0,
    east: float = 10.0,
    north: float = 10.0,
    start: datetime | None = None,
    end: datetime | None = None,
) -> FileRecord:
    """Convenience factory for FileRecord instances."""
    if file_id is None:
        file_id = uuid.uuid4().hex[:12]
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    if end is None:
        end = start + timedelta(days=1)
    return FileRecord(
        file_id=file_id,
        cube_name="test_cube",
        relative_path=f"data/{file_id}.tif",
        spatial_extent=SpatialExtent(west=west, south=south, east=east, north=north, epsg=4326),
        temporal_extent=TemporalExtent(start=start, end=end),
        band_count=1,
        file_size_bytes=1024,
        created_at=datetime.now(timezone.utc),
        checksum="abc123",
    )


# ===========================================================================
# SpatialIndex tests
# ===========================================================================


class TestSpatialIndex:
    def test_insert_and_query_intersecting(self) -> None:
        idx = SpatialIndex()
        idx.insert("f1", (-10, -10, 10, 10))
        idx.insert("f2", (5, 5, 20, 20))
        # Query overlaps both
        result = idx.query((-5, -5, 15, 15))
        assert set(result) == {"f1", "f2"}

    def test_query_excludes_non_intersecting(self) -> None:
        idx = SpatialIndex()
        idx.insert("f1", (-10, -10, -5, -5))
        idx.insert("f2", (50, 50, 60, 60))
        result = idx.query((0, 0, 1, 1))
        assert result == []

    def test_remove(self) -> None:
        idx = SpatialIndex()
        idx.insert("f1", (-10, -10, 10, 10))
        assert idx.count() == 1
        idx.remove("f1", (-10, -10, 10, 10))
        assert idx.count() == 0
        assert idx.query((-10, -10, 10, 10)) == []

    def test_count(self) -> None:
        idx = SpatialIndex()
        assert idx.count() == 0
        idx.insert("a", (0, 0, 1, 1))
        idx.insert("b", (2, 2, 3, 3))
        assert idx.count() == 2

    def test_duplicate_insert_ignored(self) -> None:
        idx = SpatialIndex()
        idx.insert("f1", (0, 0, 1, 1))
        idx.insert("f1", (0, 0, 1, 1))
        assert idx.count() == 1

    def test_persistent_survives_close_reopen(self, tmp_path) -> None:
        idx_path = tmp_path / "spatial"
        # Create and populate
        idx = SpatialIndex(index_path=idx_path)
        idx.insert("f1", (-10, -10, 10, 10))
        idx.insert("f2", (20, 20, 30, 30))
        idx.close()

        # Reopen
        idx2 = SpatialIndex(index_path=idx_path)
        assert idx2.count() == 2
        result = idx2.query((-5, -5, 5, 5))
        assert result == ["f1"]
        idx2.close()

    def test_zero_area_bbox(self) -> None:
        """A point (zero-area box) can be inserted and found."""
        idx = SpatialIndex()
        idx.insert("point", (5.0, 5.0, 5.0, 5.0))
        result = idx.query((4, 4, 6, 6))
        assert result == ["point"]

    def test_empty_index_returns_empty(self) -> None:
        idx = SpatialIndex()
        assert idx.query((-180, -90, 180, 90)) == []


# ===========================================================================
# TemporalIndex tests
# ===========================================================================


class TestTemporalIndex:
    def test_insert_and_query_overlapping(self) -> None:
        idx = TemporalIndex()
        d1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        d2 = datetime(2024, 1, 10, tzinfo=timezone.utc)
        d3 = datetime(2024, 1, 15, tzinfo=timezone.utc)
        d4 = datetime(2024, 1, 20, tzinfo=timezone.utc)
        idx.insert("f1", d1, d2)
        idx.insert("f2", d3, d4)
        # Query overlaps f1 only
        result = idx.query(d1, datetime(2024, 1, 5, tzinfo=timezone.utc))
        assert set(result) == {"f1"}

    def test_query_excludes_non_overlapping(self) -> None:
        idx = TemporalIndex()
        idx.insert("f1", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 5, tzinfo=timezone.utc))
        # Query entirely after f1
        result = idx.query(datetime(2024, 6, 1, tzinfo=timezone.utc), datetime(2024, 6, 30, tzinfo=timezone.utc))
        assert result == []

    def test_remove(self) -> None:
        idx = TemporalIndex()
        idx.insert("f1", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 5, tzinfo=timezone.utc))
        assert idx.count() == 1
        idx.remove("f1")
        assert idx.count() == 0

    def test_instant_time_range(self) -> None:
        """Start == end should still match."""
        idx = TemporalIndex()
        t = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        idx.insert("instant", t, t)
        result = idx.query(t, t)
        assert result == ["instant"]

    def test_empty_index_returns_empty(self) -> None:
        idx = TemporalIndex()
        result = idx.query(datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc))
        assert result == []

    def test_count(self) -> None:
        idx = TemporalIndex()
        assert idx.count() == 0
        idx.insert("a", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))
        assert idx.count() == 1


# ===========================================================================
# CatalogIndex tests
# ===========================================================================


class TestCatalogIndex:
    def test_combined_spatial_temporal_narrows(self) -> None:
        with CatalogIndex() as cat:
            # f1: spatial overlap, temporal overlap  -> should match
            cat.insert(_make_file_record("f1", west=-10, south=-10, east=10, north=10,
                                         start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                                         end=datetime(2024, 1, 31, tzinfo=timezone.utc)))
            # f2: spatial overlap, temporal NO overlap -> excluded
            cat.insert(_make_file_record("f2", west=-10, south=-10, east=10, north=10,
                                         start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                                         end=datetime(2025, 1, 31, tzinfo=timezone.utc)))
            # f3: spatial NO overlap, temporal overlap -> excluded
            cat.insert(_make_file_record("f3", west=50, south=50, east=60, north=60,
                                         start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                                         end=datetime(2024, 1, 31, tzinfo=timezone.utc)))

            result = cat.query(
                spatial_bounds=(-5, -5, 5, 5),
                temporal_range=(datetime(2024, 1, 10, tzinfo=timezone.utc),
                                datetime(2024, 1, 20, tzinfo=timezone.utc)),
            )
            assert result == {"f1"}

    def test_spatial_only_query(self) -> None:
        with CatalogIndex() as cat:
            cat.insert(_make_file_record("f1", west=-10, south=-10, east=10, north=10))
            cat.insert(_make_file_record("f2", west=50, south=50, east=60, north=60))
            result = cat.query(spatial_bounds=(-5, -5, 5, 5))
            assert result == {"f1"}

    def test_temporal_only_query(self) -> None:
        with CatalogIndex() as cat:
            cat.insert(_make_file_record("f1", start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                                         end=datetime(2024, 1, 31, tzinfo=timezone.utc)))
            cat.insert(_make_file_record("f2", start=datetime(2025, 6, 1, tzinfo=timezone.utc),
                                         end=datetime(2025, 6, 30, tzinfo=timezone.utc)))
            result = cat.query(temporal_range=(datetime(2024, 1, 10, tzinfo=timezone.utc),
                                               datetime(2024, 1, 20, tzinfo=timezone.utc)))
            assert result == {"f1"}

    def test_no_filters_returns_empty(self) -> None:
        with CatalogIndex() as cat:
            cat.insert(_make_file_record("f1"))
            assert cat.query() == set()

    def test_rebuild_matches_incremental(self) -> None:
        records = [
            _make_file_record(f"r{i}", west=float(i), south=float(i), east=float(i + 1), north=float(i + 1),
                              start=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
                              end=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i + 1))
            for i in range(20)
        ]
        # Incremental
        with CatalogIndex() as cat_inc:
            for r in records:
                cat_inc.insert(r)
            inc_spatial = cat_inc.query(spatial_bounds=(5.5, 5.5, 15.5, 15.5))
            inc_temporal = cat_inc.query(temporal_range=(datetime(2024, 1, 6, tzinfo=timezone.utc),
                                                         datetime(2024, 1, 16, tzinfo=timezone.utc)))

        # Rebuild
        with CatalogIndex() as cat_reb:
            cat_reb.rebuild(records)
            reb_spatial = cat_reb.query(spatial_bounds=(5.5, 5.5, 15.5, 15.5))
            reb_temporal = cat_reb.query(temporal_range=(datetime(2024, 1, 6, tzinfo=timezone.utc),
                                                         datetime(2024, 1, 16, tzinfo=timezone.utc)))

        assert inc_spatial == reb_spatial
        assert inc_temporal == reb_temporal

    def test_remove_record(self) -> None:
        with CatalogIndex() as cat:
            rec = _make_file_record("f1", west=-10, south=-10, east=10, north=10)
            cat.insert(rec)
            assert cat.query(spatial_bounds=(-5, -5, 5, 5)) == {"f1"}
            cat.remove(rec)
            assert cat.query(spatial_bounds=(-5, -5, 5, 5)) == set()

    def test_persistent_catalog(self, tmp_path) -> None:
        idx_dir = tmp_path / "cat_idx"
        rec = _make_file_record("f1", west=-10, south=-10, east=10, north=10)
        with CatalogIndex(index_dir=idx_dir) as cat:
            cat.insert(rec)

        # Reopen — spatial index is persistent, temporal must be rebuilt
        with CatalogIndex(index_dir=idx_dir) as cat2:
            # Temporal was lost (memory-only) — only spatial persists
            result = cat2.query(spatial_bounds=(-5, -5, 5, 5))
            assert "f1" in result


# ===========================================================================
# Performance test
# ===========================================================================


class TestPerformance:
    def test_query_10000_entries_under_100ms(self) -> None:
        """Querying a 10,000-entry index must complete in < 100 ms."""
        idx = SpatialIndex()
        base_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t_idx = TemporalIndex()

        for i in range(10_000):
            x = (i % 100) * 1.0
            y = (i // 100) * 1.0
            fid = f"file_{i}"
            idx.insert(fid, (x, y, x + 1, y + 1))
            t_idx.insert(fid, base_start + timedelta(hours=i), base_start + timedelta(hours=i + 1))

        # Spatial query
        t0 = time.perf_counter()
        spatial_hits = idx.query((10, 10, 30, 30))
        spatial_elapsed = time.perf_counter() - t0
        assert spatial_elapsed < 0.1, f"Spatial query took {spatial_elapsed:.3f}s"
        assert len(spatial_hits) > 0

        # Temporal query
        t0 = time.perf_counter()
        temporal_hits = t_idx.query(base_start + timedelta(hours=500), base_start + timedelta(hours=1500))
        temporal_elapsed = time.perf_counter() - t0
        assert temporal_elapsed < 0.1, f"Temporal query took {temporal_elapsed:.3f}s"
        assert len(temporal_hits) > 0
