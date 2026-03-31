"""SQLite v2.db metadata sidecar for VoxelVault.

Manages cube schemas, file records, variable definitions, and custom metadata
in a single SQLite database using only the stdlib sqlite3 module.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Self

from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    FileRecord,
    GridDefinition,
    SpatialExtent,
    TemporalExtent,
    Variable,
)

_SCHEMA_VERSION = "1"

_DDL = """\
CREATE TABLE IF NOT EXISTS _meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cubes (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    grid_width INTEGER NOT NULL,
    grid_height INTEGER NOT NULL,
    grid_epsg INTEGER NOT NULL,
    grid_transform TEXT NOT NULL,
    temporal_resolution_seconds REAL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS variables (
    name TEXT PRIMARY KEY,
    unit TEXT NOT NULL,
    dtype TEXT NOT NULL,
    nodata REAL,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS bands (
    cube_name TEXT NOT NULL REFERENCES cubes(name),
    band_index INTEGER NOT NULL,
    variable_name TEXT NOT NULL REFERENCES variables(name),
    component TEXT NOT NULL DEFAULT 'scalar'
        CHECK(component IN ('scalar', 'real', 'imaginary')),
    PRIMARY KEY (cube_name, band_index)
);

CREATE TABLE IF NOT EXISTS files (
    file_id TEXT PRIMARY KEY,
    cube_name TEXT NOT NULL REFERENCES cubes(name),
    relative_path TEXT NOT NULL UNIQUE,
    west REAL NOT NULL,
    south REAL NOT NULL,
    east REAL NOT NULL,
    north REAL NOT NULL,
    epsg INTEGER NOT NULL,
    temporal_start TEXT NOT NULL,
    temporal_end TEXT NOT NULL,
    band_count INTEGER NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    checksum TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_spatial ON files(west, east, south, north);
CREATE INDEX IF NOT EXISTS idx_files_temporal ON files(temporal_start, temporal_end);
CREATE INDEX IF NOT EXISTS idx_files_cube ON files(cube_name);
"""


def _parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 string into a timezone-aware datetime."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class SchemaEngine:
    """Manages the VoxelVault v2.db SQLite metadata sidecar."""

    def __init__(self, db_path: Path | str) -> None:
        """Open or create the v2.db database at the given path."""
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all tables if they don't exist. Set WAL mode. Record schema version."""
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DDL)
        self._conn.execute(
            "INSERT OR IGNORE INTO _meta (key, value) VALUES (?, ?)",
            ("schema_version", _SCHEMA_VERSION),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Cube CRUD
    # ------------------------------------------------------------------

    def register_cube(self, descriptor: CubeDescriptor) -> None:
        """Insert a cube definition with its variables and band assignments.

        Registers any new variables that don't already exist.
        Raises ValueError if cube name already exists.
        """
        row = self._conn.execute("SELECT 1 FROM cubes WHERE name = ?", (descriptor.name,)).fetchone()
        if row is not None:
            raise ValueError(f"Cube {descriptor.name!r} already exists")

        temporal_seconds: float | None = None
        if descriptor.temporal_resolution is not None:
            temporal_seconds = descriptor.temporal_resolution.total_seconds()

        self._conn.execute(
            "INSERT INTO cubes (name, description, grid_width, grid_height, grid_epsg, "
            "grid_transform, temporal_resolution_seconds, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                descriptor.name,
                descriptor.description,
                descriptor.grid.width,
                descriptor.grid.height,
                descriptor.grid.epsg,
                json.dumps(list(descriptor.grid.transform)),
                temporal_seconds,
                json.dumps(descriptor.metadata),
            ),
        )

        for band in descriptor.bands:
            var = band.variable
            self._conn.execute(
                "INSERT OR IGNORE INTO variables (name, unit, dtype, nodata, description) "
                "VALUES (?, ?, ?, ?, ?)",
                (var.name, var.unit, var.dtype, var.nodata, var.description),
            )
            self._conn.execute(
                "INSERT INTO bands (cube_name, band_index, variable_name, component) "
                "VALUES (?, ?, ?, ?)",
                (descriptor.name, band.band_index, var.name, band.component),
            )

        self._conn.commit()

    def get_cube(self, name: str) -> CubeDescriptor | None:
        """Retrieve a cube descriptor by name. Returns None if not found."""
        cube_row = self._conn.execute("SELECT * FROM cubes WHERE name = ?", (name,)).fetchone()
        if cube_row is None:
            return None

        band_rows = self._conn.execute(
            "SELECT b.band_index, b.component, v.name, v.unit, v.dtype, v.nodata, v.description "
            "FROM bands b JOIN variables v ON b.variable_name = v.name "
            "WHERE b.cube_name = ? ORDER BY b.band_index",
            (name,),
        ).fetchall()

        bands = [
            BandDefinition(
                band_index=br["band_index"],
                variable=Variable(
                    name=br["name"],
                    unit=br["unit"],
                    dtype=br["dtype"],
                    nodata=br["nodata"],
                    description=br["description"],
                ),
                component=br["component"],
            )
            for br in band_rows
        ]

        transform_list: list[float] = json.loads(cube_row["grid_transform"])
        grid = GridDefinition(
            width=cube_row["grid_width"],
            height=cube_row["grid_height"],
            epsg=cube_row["grid_epsg"],
            transform=tuple(transform_list),
        )

        temporal_resolution: timedelta | None = None
        if cube_row["temporal_resolution_seconds"] is not None:
            temporal_resolution = timedelta(seconds=cube_row["temporal_resolution_seconds"])

        metadata: dict[str, str] = json.loads(cube_row["metadata"])

        return CubeDescriptor(
            name=cube_row["name"],
            bands=bands,
            grid=grid,
            temporal_resolution=temporal_resolution,
            description=cube_row["description"],
            metadata=metadata,
        )

    def list_cubes(self) -> list[str]:
        """Return all registered cube names."""
        rows = self._conn.execute("SELECT name FROM cubes ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    def delete_cube(self, name: str) -> None:
        """Delete a cube and its band assignments.

        Does NOT delete associated files.
        Raises ValueError if cube has associated file records.
        """
        file_count = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM files WHERE cube_name = ?", (name,)
        ).fetchone()["cnt"]
        if file_count > 0:
            raise ValueError(f"Cannot delete cube {name!r}: {file_count} file(s) still reference it")

        self._conn.execute("DELETE FROM bands WHERE cube_name = ?", (name,))
        self._conn.execute("DELETE FROM cubes WHERE name = ?", (name,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # File Record CRUD
    # ------------------------------------------------------------------

    def insert_file(self, record: FileRecord) -> None:
        """Insert a file record. Raises ValueError if file_id or relative_path already exists."""
        try:
            self._conn.execute(
                "INSERT INTO files (file_id, cube_name, relative_path, west, south, east, north, "
                "epsg, temporal_start, temporal_end, band_count, file_size_bytes, created_at, checksum) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.file_id,
                    record.cube_name,
                    record.relative_path,
                    record.spatial_extent.west,
                    record.spatial_extent.south,
                    record.spatial_extent.east,
                    record.spatial_extent.north,
                    record.spatial_extent.epsg,
                    record.temporal_extent.start.isoformat(),
                    record.temporal_extent.end.isoformat(),
                    record.band_count,
                    record.file_size_bytes,
                    record.created_at.isoformat(),
                    record.checksum,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            raise ValueError(str(exc)) from exc

    def get_file(self, file_id: str) -> FileRecord | None:
        """Retrieve a file record by ID."""
        row = self._conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_file_record(row)

    def query_files(
        self,
        cube_name: str | None = None,
        spatial_bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[datetime, datetime] | None = None,
    ) -> list[FileRecord]:
        """Query file records with optional spatial and temporal filters.

        spatial_bounds: (west, south, east, north)
        temporal_range: (start, end)
        """
        clauses: list[str] = []
        params: list[object] = []

        if cube_name is not None:
            clauses.append("cube_name = ?")
            params.append(cube_name)

        if spatial_bounds is not None:
            q_west, q_south, q_east, q_north = spatial_bounds
            clauses.append("west <= ? AND east >= ? AND south <= ? AND north >= ?")
            params.extend([q_east, q_west, q_north, q_south])

        if temporal_range is not None:
            q_start, q_end = temporal_range
            clauses.append("temporal_start <= ? AND temporal_end >= ?")
            params.extend([q_end.isoformat(), q_start.isoformat()])

        sql = "SELECT * FROM files"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY temporal_start"

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_file_record(r) for r in rows]

    def delete_file(self, file_id: str) -> None:
        """Delete a file record by ID."""
        self._conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        self._conn.commit()

    def file_count(self, cube_name: str | None = None) -> int:
        """Count file records, optionally filtered by cube."""
        if cube_name is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM files WHERE cube_name = ?", (cube_name,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM files").fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_file_record(row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            file_id=row["file_id"],
            cube_name=row["cube_name"],
            relative_path=row["relative_path"],
            spatial_extent=SpatialExtent(
                west=row["west"],
                south=row["south"],
                east=row["east"],
                north=row["north"],
                epsg=row["epsg"],
            ),
            temporal_extent=TemporalExtent(
                start=_parse_iso(row["temporal_start"]),
                end=_parse_iso(row["temporal_end"]),
            ),
            band_count=row["band_count"],
            file_size_bytes=row["file_size_bytes"],
            created_at=_parse_iso(row["created_at"]),
            checksum=row["checksum"],
        )
