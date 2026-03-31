"""Vault orchestrator — the unified public API for VoxelVault.

A vault is a directory containing:
- v2.db: SQLite metadata database
- vault.json: serialized VaultConfig
- data/: COG files organized by cube name
- index/: R-tree spatial index files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Self

from voxelvault.index import CatalogIndex
from voxelvault.models import CubeDescriptor, VaultConfig
from voxelvault.schema import SchemaEngine

if TYPE_CHECKING:
    from datetime import datetime

    import numpy as np

    from voxelvault.ingest import IngestResult
    from voxelvault.models import SpatialExtent, TemporalExtent
    from voxelvault.query import QueryResult


class Vault:
    """A serverless spatiotemporal raster vault.

    A vault is a directory containing:
    - v2.db: SQLite metadata database
    - vault.json: serialized VaultConfig
    - data/: COG files organized by cube name
    - index/: R-tree spatial index files
    """

    def __init__(
        self,
        path: Path,
        config: VaultConfig,
        schema: SchemaEngine,
        catalog: CatalogIndex,
    ) -> None:
        self._path = path
        self._config = config
        self._schema = schema
        self._catalog = catalog

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, path: Path | str, config: VaultConfig | None = None) -> Vault:
        """Create a new vault at the given path.

        Creates the directory structure, initializes v2.db, writes vault.json,
        and returns an open Vault instance.

        Raises:
            FileExistsError: If path already contains a vault (v2.db exists).
        """
        path = Path(path)
        db_path = path / "v2.db"

        if db_path.exists():
            raise FileExistsError(f"Vault already exists at {path} (v2.db found)")

        if config is None:
            config = VaultConfig()

        # Create directory structure
        path.mkdir(parents=True, exist_ok=True)
        (path / "data").mkdir(exist_ok=True)
        (path / "index").mkdir(exist_ok=True)

        # Write vault.json
        (path / "vault.json").write_text(
            json.dumps(config.model_dump(), indent=2),
            encoding="utf-8",
        )

        # Initialize schema engine
        schema = SchemaEngine(db_path)
        schema.initialize()

        # Create catalog index
        catalog = CatalogIndex(index_dir=path / "index")

        return cls(path=path, config=config, schema=schema, catalog=catalog)

    @classmethod
    def open(cls, path: Path | str) -> Vault:
        """Open an existing vault.

        Loads the schema engine, reads vault.json, rebuilds the spatial index
        from the file catalog, and returns an open Vault instance.

        Raises:
            FileNotFoundError: If v2.db doesn't exist at path.
        """
        path = Path(path)
        db_path = path / "v2.db"

        if not db_path.exists():
            raise FileNotFoundError(f"No vault found at {path} (v2.db missing)")

        # Read vault.json
        config_path = path / "vault.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = VaultConfig(**raw)

        # Open schema engine
        schema = SchemaEngine(db_path)

        # Rebuild catalog index from all file records
        catalog = CatalogIndex(index_dir=path / "index")
        all_files = schema.query_files()
        catalog.rebuild(all_files)

        return cls(path=path, config=config, schema=schema, catalog=catalog)

    # ------------------------------------------------------------------
    # Cube management
    # ------------------------------------------------------------------

    def register_cube(self, descriptor: CubeDescriptor) -> None:
        """Register a new raster cube schema in this vault.

        Args:
            descriptor: The cube descriptor defining the cube's schema.

        Raises:
            ValueError: If a cube with the same name already exists.
        """
        self._schema.register_cube(descriptor)
        # Ensure the data subdirectory for this cube exists
        (self._path / "data" / descriptor.name).mkdir(parents=True, exist_ok=True)

    def list_cubes(self) -> list[str]:
        """Return the names of all registered cubes."""
        return self._schema.list_cubes()

    def get_cube(self, name: str) -> CubeDescriptor | None:
        """Retrieve a cube descriptor by name.

        Args:
            name: The cube name.

        Returns:
            The CubeDescriptor, or None if not found.
        """
        return self._schema.get_cube(name)

    # ------------------------------------------------------------------
    # Ingestion convenience methods
    # ------------------------------------------------------------------

    def ingest(
        self,
        cube_name: str,
        data: np.ndarray,
        temporal_extent: TemporalExtent,
        spatial_extent: SpatialExtent | None = None,
    ) -> IngestResult:
        """Ingest a numpy array into the vault as a COG.

        Args:
            cube_name: Name of the target cube.
            data: Raster data array (bands, height, width) or (height, width).
            temporal_extent: Time range for this data.
            spatial_extent: Optional spatial extent override.

        Returns:
            IngestResult with file metadata.
        """
        from voxelvault.ingest import ingest_array

        return ingest_array(self, cube_name, data, temporal_extent, spatial_extent)

    def ingest_file(
        self,
        cube_name: str,
        source_path: Path | str,
        temporal_extent: TemporalExtent,
    ) -> IngestResult:
        """Ingest an existing GeoTIFF/COG file into the vault.

        Args:
            cube_name: Name of the target cube.
            source_path: Path to the source GeoTIFF file.
            temporal_extent: Time range for this data.

        Returns:
            IngestResult with file metadata.
        """
        from voxelvault.ingest import ingest_file

        return ingest_file(self, cube_name, source_path, temporal_extent)

    # ------------------------------------------------------------------
    # Query convenience methods
    # ------------------------------------------------------------------

    def query(
        self,
        cube_name: str,
        spatial_bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[datetime, datetime] | None = None,
        variables: list[str] | None = None,
    ) -> QueryResult:
        """Query a raster cube for matching data.

        Args:
            cube_name: Name of the cube to query.
            spatial_bounds: Optional (west, south, east, north) filter.
            temporal_range: Optional (start, end) datetime filter.
            variables: Optional list of variable names to select.

        Returns:
            QueryResult with data and provenance.
        """
        from voxelvault.query import query_cube

        return query_cube(self, cube_name, spatial_bounds, temporal_range, variables)

    def query_single(
        self,
        cube_name: str,
        spatial_bounds: tuple[float, float, float, float] | None = None,
        temporal_range: tuple[datetime, datetime] | None = None,
        variables: list[str] | None = None,
    ) -> QueryResult:
        """Query a raster cube expecting exactly one matching file.

        Args:
            cube_name: Name of the cube to query.
            spatial_bounds: Optional (west, south, east, north) filter.
            temporal_range: Optional (start, end) datetime filter.
            variables: Optional list of variable names to select.

        Returns:
            QueryResult with data and provenance.

        Raises:
            ValueError: If zero or multiple files match.
        """
        from voxelvault.query import query_single

        return query_single(self, cube_name, spatial_bounds, temporal_range, variables)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the vault — flush index, close database."""
        self._catalog.close()
        self._schema.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Root directory of the vault."""
        return self._path

    @property
    def config(self) -> VaultConfig:
        """Vault configuration."""
        return self._config

    @property
    def file_count(self) -> int:
        """Total number of cataloged files across all cubes."""
        return self._schema.file_count()

    def cube_file_count(self, cube_name: str) -> int:
        """Number of cataloged files for a specific cube."""
        return self._schema.file_count(cube_name=cube_name)
