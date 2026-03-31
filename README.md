# VoxelVault

**Serverless spatiotemporal raster engine** — map N-dimensional sensor tensors to Cloud Optimized GeoTIFFs.

## Core Concept

> **Space = Pixels · Time = Files · Variables = Bands**

Any dataset with a spatiotemporal component becomes a *raster cube* that can be queried, sliced, and served — no running server required.

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
# Create a new vault
vv create ./my_vault --compression deflate --tile-size 256

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

## Dependencies

| Package | Purpose |
|---------|---------|
| `pydantic>=2.0` | Immutable domain models with validation |
| `numpy>=1.24` | Array data representation |
| `rasterio>=1.3` | COG read/write via GDAL *(optional: storage)* |
| `rtree>=1.0` | R-tree spatial indexing *(optional: index)* |
| `pyproj>=3.4` | CRS transformations *(optional: geo)* |
| `click>=8.0` | CLI framework *(optional: cli)* |

## Documentation

📖 **[Full User Guide](docs/user-guide.md)** — comprehensive documentation covering installation, core concepts, Python API, CLI reference, recipes, architecture deep dive, and troubleshooting.

## License

MIT
