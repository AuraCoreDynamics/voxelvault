# VoxelVault

**Serverless spatiotemporal raster engine** — map N-dimensional sensor tensors to Cloud Optimized GeoTIFFs.

## Core Concept

> **Space = Pixels · Time = Files · Variables = Bands**

Any dataset with a spatiotemporal component becomes a *raster cube* that can be queried, sliced, and served — no running server required. Supports both **Cloud Optimized GeoTIFF** and **JPEG 2000 lossless** storage backends.

| Dimension | Mapping |
|-----------|---------|
| **X, Y** (space) | Pixels within each GeoTIFF |
| **Z** (variables / complex values) | Bands within a file |
| **T** (time) | Individual COG file instances |

## Installation

```bash
# Core (models + vault logic, no I/O)
pip install voxelvault

# With all optional extras (recommended)
pip install "voxelvault[all]"

# Individual extras
pip install "voxelvault[storage]"   # rasterio COG I/O
pip install "voxelvault[index]"     # R-tree spatial index
pip install "voxelvault[geo]"       # pyproj CRS support
pip install "voxelvault[cli]"       # click CLI
pip install "voxelvault[remote]"    # fsspec cloud storage (S3, GCS, Azure)

# Development
pip install "voxelvault[all,dev]"
```

## Quickstart

```python
from datetime import datetime, timezone
import numpy as np
from voxelvault import (
    Vault, CubeDescriptor, GridDefinition,
    BandDefinition, Variable, TemporalExtent,
)

# 1. Create a vault
with Vault.create("./my_vault") as vault:
    # 2. Define and register a cube
    cube = CubeDescriptor(
        name="temperature",
        bands=[BandDefinition(band_index=1, variable=Variable(name="temp", unit="K", dtype="float32"))],
        grid=GridDefinition(width=256, height=256, epsg=4326, transform=(0.01, 0, -180, 0, -0.01, 90)),
    )
    vault.register_cube(cube)

    # 3. Ingest data
    data = np.random.default_rng(0).random((1, 256, 256), dtype=np.float32)
    extent = TemporalExtent(start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                            end=datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc))
    vault.ingest("temperature", data, extent)

    # 4. Query
    result = vault.query("temperature")
    print(result.data.shape)  # (1, 256, 256)
```

## CLI Usage

VoxelVault ships with a `vv` command-line tool.

```bash
# Create a new vault (GeoTIFF/deflate by default)
vv create ./my_vault --compression deflate --tile-size 256

# Create a JPEG 2000 lossless vault
vv create ./my_vault --format jp2k

# Show vault info
vv info ./my_vault
vv info ./my_vault --json

# Ingest a GeoTIFF into a cube
vv ingest ./my_vault temperature input.tif \
    --start 2024-01-01 --end 2024-01-01

# Query a cube
vv query ./my_vault temperature
vv query ./my_vault temperature --json
vv query ./my_vault temperature -o result.tif \
    --bounds -180 -90 180 90 --start 2024-01-01 --end 2024-12-31
```

## Architecture

```
my_vault/
├── v2.db           # SQLite metadata sidecar (cubes, files, provenance)
├── vault.json      # VaultConfig (compression, tile size, EPSG)
├── data/
│   └── cube_name/  # One COG per time step
│       ├── 20240101T000000Z_a1b2c3d4.tif
│       └── 20240102T000000Z_e5f6g7h8.tif
└── index/          # R-tree spatial + temporal index files
```

**How it works:**

1. **COG files** store raster data — one file per time step per cube, tiled and compressed.
2. **SQLite sidecar** (`v2.db`) stores cube schemas, file records, spatial extents, and checksums.
3. **R-tree index** enables fast spatial and temporal lookups across potentially thousands of files.
4. **Vault orchestrator** ties everything together behind a clean Python API.

## Key Features

- **Multi-INT Fusion** — query with a `target_grid` to reproject and resample disparate sensor grids into a common spatial reference on the fly.
- **Incremental Indexing** — `Vault.open()` only indexes new files instead of rebuilding from scratch, making reopens O(new) instead of O(all).
- **Memory-Safe Checksumming** — large files are checksummed in streaming 64 KB chunks, avoiding out-of-memory errors on multi-GB COGs.
- **Concurrent Access Locking** — file-based advisory locks prevent index corruption when multiple processes open the same vault simultaneously.
- **Cloud Storage Foundation** — `fsspec`-backed path resolution and GDAL `/vsis3/`/`/vsigs/` translation for S3, GCS, and Azure Blob URIs (install `voxelvault[remote]`).

## Multi-INT Fusion Example

```python
from voxelvault import Vault, GridDefinition

# Define a common output grid (e.g. 100m UTM)
target = GridDefinition(
    width=512, height=512, epsg=32633,
    transform=(100.0, 0, 300000, 0, -100.0, 6200000),
)

with Vault.open("./my_vault") as vault:
    # All source data is warped to the target grid before stacking
    result = vault.query("sar_cube", target_grid=target)
    print(result.data.shape)  # (time, bands, 512, 512)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `pydantic>=2.0` | Immutable domain models with validation |
| `numpy>=1.24` | Array data representation |
| `rasterio>=1.3` | COG read/write via GDAL *(optional: storage)* |
| `rtree>=1.0` | R-tree spatial indexing *(optional: index)* |
| `pyproj>=3.4` | CRS transformations *(optional: geo)* |
| `click>=8.0` | CLI framework *(optional: cli)* |
| `fsspec>=2023.1` | Cloud storage abstraction *(optional: remote)* |

## Documentation

📖 **[Full User Guide](docs/user-guide.md)** — comprehensive documentation covering installation, core concepts, Python API, CLI reference, recipes, architecture deep dive, and troubleshooting.

## License

MIT
