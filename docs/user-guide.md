# VoxelVault User Guide

> **Serverless spatiotemporal raster engine** — map N-dimensional sensor tensors to Cloud Optimized GeoTIFFs.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Core Concepts](#3-core-concepts)
4. [Getting Started](#4-getting-started)
5. [Working with Vaults](#5-working-with-vaults)
6. [Defining Cubes](#6-defining-cubes)
7. [Ingesting Data](#7-ingesting-data)
8. [Querying Data](#8-querying-data)
9. [CLI Reference](#9-cli-reference)
10. [Storage Layer](#10-storage-layer)
11. [Recipes & Patterns](#11-recipes--patterns)
12. [API Reference](#12-api-reference)
13. [Architecture Deep Dive](#13-architecture-deep-dive)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Introduction

VoxelVault is a Python library and CLI tool for managing georeferenced raster data at scale. It treats any N-dimensional dataset with a spatiotemporal component as a **raster cube** — a collection of Cloud Optimized GeoTIFF (COG) files indexed by space and time.

**What VoxelVault does:**

- Stores sensor data (SAR, EO, thermal, atmospheric, seismic, etc.) as standard GeoTIFF files
- Indexes thousands of files for sub-second spatial and temporal queries
- Provides a unified Python API and CLI for create → ingest → query workflows
- Requires no running server, no database process, no cloud infrastructure

**What VoxelVault does NOT do:**

- It is not a database — it's a file-based catalog with an index
- It does not stream data over HTTP (though COG format supports future HTTP range requests)
- It does not handle multi-process concurrent writes (single-writer model)
- It does not reproject data — all files in a cube share the same CRS

---

## 2. Installation

### Requirements

- Python 3.11 or later
- A platform supported by [rasterio](https://rasterio.readthedocs.io/) (Windows, macOS, Linux)

### Install Options

```bash
# Core only — models and vault logic, no I/O
# Useful for metadata-only workflows or environments without GDAL
pip install voxelvault

# Full install (recommended for most users)
pip install "voxelvault[all]"

# Individual optional groups
pip install "voxelvault[storage]"   # rasterio — COG read/write
pip install "voxelvault[index]"     # rtree — spatial indexing
pip install "voxelvault[geo]"       # pyproj — CRS transformations
pip install "voxelvault[cli]"       # click — vv command-line tool

# Development (includes pytest, ruff, coverage)
pip install "voxelvault[all,dev]"
```

### Editable Install (for development)

```bash
git clone <repo-url>
cd voxelvault
pip install -e ".[all,dev]"
```

### Verify Installation

```bash
python -c "import voxelvault; print(voxelvault.__version__)"
# 0.1.0

vv --help
# Usage: vv [OPTIONS] COMMAND [ARGS]...
```

---

## 3. Core Concepts

### The Raster Cube Model

VoxelVault's central insight is that any spatiotemporal dataset can be mapped to a collection of GeoTIFF files using three dimensions:

| Dimension | Maps To | Example |
|-----------|---------|---------|
| **Space (X, Y)** | Pixels in each GeoTIFF | 256×256 grid at 0.01° resolution |
| **Variables (Z)** | Bands within a file | Band 1 = temperature, Band 2 = humidity |
| **Time (T)** | Individual COG file instances | One file per day, hour, or observation |

This is called a **raster cube**. Each cube has:

- A fixed **spatial grid** (dimensions, CRS, and affine transform)
- A fixed **band layout** (which variables map to which bands)
- A growing collection of **time slices** (COG files)

### Key Entities

```
Vault                    ← A directory containing one or more cubes
├── CubeDescriptor       ← Schema definition (grid + bands + metadata)
│   ├── GridDefinition   ← Spatial grid (width, height, CRS, transform)
│   └── BandDefinition[] ← Band-to-variable mapping
│       └── Variable     ← Name, unit, dtype, nodata
├── VaultConfig          ← Storage format, codec, tiling, EPSG defaults
│   └── StorageConfig   ← Format (geotiff/jp2k), codec, tile settings
└── FileRecord[]         ← Catalog of ingested COG files
    ├── SpatialExtent    ← Geographic bounding box
    └── TemporalExtent   ← Time range
```

### Vault Directory Structure

When you create a vault, VoxelVault creates this structure on disk:

```
my_vault/
├── v2.db              ← SQLite metadata catalog (cube schemas, file records)
├── vault.json         ← Serialized VaultConfig
├── data/              ← COG files, organized by cube name
│   └── temperature/
│       ├── 20240101T000000Z_a1b2c3d4.tif
│       ├── 20240102T000000Z_e5f6g7h8.tif
│       └── ...
└── index/             ← R-tree spatial index files
    ├── spatial.idx
    └── spatial.dat
```

The entire vault is self-contained — you can copy, move, or archive the directory as a unit.

---

## 4. Getting Started

### Minimal Example: Create → Ingest → Query

```python
from datetime import datetime, timezone
import numpy as np
from voxelvault import (
    Vault,
    CubeDescriptor,
    GridDefinition,
    BandDefinition,
    Variable,
    TemporalExtent,
)

# --- Step 1: Create a vault ---
vault = Vault.create("./my_vault")

# --- Step 2: Define and register a cube ---
cube = CubeDescriptor(
    name="temperature",
    bands=[
        BandDefinition(
            band_index=1,
            variable=Variable(name="temp_k", unit="K", dtype="float32"),
        )
    ],
    grid=GridDefinition(
        width=256,
        height=256,
        epsg=4326,
        transform=(0.01, 0.0, -180.0, 0.0, -0.01, 90.0),
    ),
)
vault.register_cube(cube)

# --- Step 3: Ingest some data ---
rng = np.random.default_rng(42)
data = rng.random((1, 256, 256), dtype=np.float32) * 50 + 250  # ~250–300 K

time_slice = TemporalExtent(
    start=datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc),
    end=datetime(2024, 6, 15, 23, 59, 59, tzinfo=timezone.utc),
)
result = vault.ingest("temperature", data, time_slice)
print(f"Ingested: {result.file_id} ({result.file_size_bytes} bytes)")

# --- Step 4: Query it back ---
qr = vault.query("temperature")
print(f"Shape: {qr.data.shape}")        # (1, 256, 256)
print(f"Source files: {len(qr.source_files)}")  # 1

# --- Step 5: Clean up ---
vault.close()
```

### Using the Context Manager

The `Vault` class supports Python's `with` statement, which automatically closes the vault (flushing the index and closing the database) when the block exits:

```python
with Vault.create("./my_vault") as vault:
    # ... register cubes, ingest data, query ...
    pass  # vault.close() called automatically
```

For existing vaults:

```python
with Vault.open("./my_vault") as vault:
    result = vault.query("temperature")
    print(result.data.shape)
```

---

## 5. Working with Vaults

### Creating a Vault

```python
from voxelvault import Vault, VaultConfig, StorageConfig

# With default config (GeoTIFF/deflate, 256px tiles, EPSG 4326)
vault = Vault.create("./my_vault")

# GeoTIFF with custom codec
config = VaultConfig(
    compression="zstd",        # zstd, deflate, lzw, or none
    compression_level=3,       # 0–9 (higher = smaller files, slower)
    tile_size=512,             # Internal tile size in pixels
    overview_levels=[2, 4, 8], # Downsampling levels for overviews
    default_epsg=32618,        # UTM Zone 18N
)
vault = Vault.create("./my_vault", config=config)

# JPEG 2000 lossless vault
jp2k_config = VaultConfig(
    storage=StorageConfig(format="jp2k", codec="jp2k_lossless"),
)
vault = Vault.create("./my_jp2k_vault", config=jp2k_config)
```

**Raises `FileExistsError`** if a vault already exists at that path (i.e., `v2.db` is found).

### Opening an Existing Vault

```python
vault = Vault.open("./my_vault")
```

When a vault is opened:
1. `vault.json` is read to restore the `VaultConfig`
2. The SQLite schema engine connects to `v2.db`
3. The spatial/temporal index is **rebuilt from the file catalog** (the index is a cache, not a source of truth)

**Raises `FileNotFoundError`** if no vault exists at that path.

### Inspecting a Vault

```python
with Vault.open("./my_vault") as vault:
    # Basic info
    print(vault.path)            # Path to vault root
    print(vault.config)          # VaultConfig
    print(vault.file_count)      # Total files across all cubes

    # Cube management
    print(vault.list_cubes())    # ['temperature', 'wind_speed']

    cube = vault.get_cube("temperature")
    if cube:
        print(cube.band_count)   # 1
        print(cube.variables)    # [Variable(name='temp_k', ...)]
        print(cube.grid.resolution)  # (0.01, -0.01)

    # Per-cube file count
    n = vault.cube_file_count("temperature")
    print(f"Temperature has {n} time slices")
```

### Closing a Vault

Always close your vault when done. This flushes the spatial index to disk and closes the SQLite connection:

```python
vault.close()
```

Or use the context manager (recommended):

```python
with Vault.open("./my_vault") as vault:
    ...  # close() called automatically
```

---

## 6. Defining Cubes

A **cube** is a schema that defines what data looks like in the vault. You must register a cube before ingesting data into it.

### Variables

A `Variable` describes a single measured quantity:

```python
from voxelvault import Variable

temp = Variable(name="temperature", unit="K", dtype="float32")
wind = Variable(name="wind_speed", unit="m/s", dtype="float32", nodata=-9999.0)
sigma = Variable(name="sigma0_vv", unit="dB", dtype="float32",
                 description="VV-polarized backscatter coefficient")
```

**Constraints:**
- `name` must be a valid Python identifier (letters, digits, underscores; cannot start with a digit)
- `dtype` must be a valid NumPy dtype string (`float32`, `float64`, `int16`, `uint16`, `uint8`, `complex64`, etc.)

### Band Definitions

A `BandDefinition` maps a variable to a 1-based GeoTIFF band index:

```python
from voxelvault import BandDefinition

bands = [
    BandDefinition(band_index=1, variable=temp),
    BandDefinition(band_index=2, variable=wind),
]
```

Band indices are **1-based** to match GDAL/rasterio convention. Band index 0 raises a `ValidationError`.

For complex-valued data, use the `component` field:

```python
complex_bands = [
    BandDefinition(band_index=1, variable=sigma, component="real"),
    BandDefinition(band_index=2, variable=sigma, component="imaginary"),
]
```

### Grid Definitions

A `GridDefinition` specifies the spatial grid that all files in the cube share:

```python
from voxelvault import GridDefinition

# A 1024×1024 grid covering the contiguous US at ~1km resolution
grid = GridDefinition(
    width=1024,
    height=1024,
    epsg=4326,
    transform=(0.05, 0.0, -125.0, 0.0, -0.05, 50.0),
    #          x_res  x_skew x_origin y_skew y_res   y_origin
)

print(grid.resolution)  # (0.05, -0.05)
print(grid.bounds)       # SpatialExtent(west=-125.0, south=-1.2, east=-73.8, north=50.0)
```

The `transform` tuple contains 6 affine coefficients matching the [rasterio/GDAL convention](https://rasterio.readthedocs.io/en/latest/topics/migrating-to-v1.html#affine-affine-vs-gdal-style-geotransforms):

```
(x_pixel_size, x_rotation, x_origin, y_rotation, y_pixel_size, y_origin)
```

For north-up images, `y_pixel_size` is negative.

### Putting It Together: CubeDescriptor

```python
from datetime import timedelta
from voxelvault import CubeDescriptor

cube = CubeDescriptor(
    name="weather",
    bands=[
        BandDefinition(band_index=1, variable=Variable(name="temp", unit="K", dtype="float32")),
        BandDefinition(band_index=2, variable=Variable(name="precip", unit="mm", dtype="float32")),
        BandDefinition(band_index=3, variable=Variable(name="humidity", unit="%", dtype="float32")),
    ],
    grid=GridDefinition(
        width=512, height=512, epsg=4326,
        transform=(0.1, 0.0, -180.0, 0.0, -0.1, 90.0),
    ),
    temporal_resolution=timedelta(hours=6),
    description="Global weather reanalysis — 6-hourly intervals",
    metadata={"source": "ERA5", "processing_level": "L3"},
)

# Register it
with Vault.create("./weather_vault") as vault:
    vault.register_cube(cube)
    print(vault.list_cubes())  # ['weather']
```

### Immutability

All VoxelVault models are **frozen** (immutable). You cannot modify them after creation:

```python
var = Variable(name="temp", unit="K", dtype="float32")
var.unit = "C"  # ❌ Raises ValidationError — frozen model
```

To "modify" a model, create a new one:

```python
var_celsius = Variable(name="temp", unit="C", dtype="float32")
```

---

## 7. Ingesting Data

Ingestion writes data into a vault as COG files and catalogs them in the metadata database and spatial index.

### Ingesting NumPy Arrays

The most common workflow — you have data in memory and want to store it:

```python
import numpy as np
from datetime import datetime, timezone
from voxelvault import Vault, TemporalExtent

with Vault.open("./weather_vault") as vault:
    # Generate or load your data
    # Shape must be (bands, height, width) matching the cube's band_count and grid
    data = np.random.default_rng(42).random((3, 512, 512), dtype=np.float32)

    # Define the time range this data covers
    t = TemporalExtent(
        start=datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, 5, 59, 59, tzinfo=timezone.utc),
    )

    result = vault.ingest("weather", data, t)
    print(f"File ID:  {result.file_id}")
    print(f"Path:     {result.relative_path}")
    print(f"Size:     {result.file_size_bytes} bytes")
    print(f"Checksum: {result.checksum}")
    print(f"Elapsed:  {result.elapsed_seconds:.2f}s")
```

**What happens during ingestion:**

1. Validates that the cube exists and the data shape matches `(band_count, height, width)`
2. Writes a Cloud Optimized GeoTIFF to `data/{cube_name}/{timestamp}_{uuid}.tif`
3. Builds overviews and internal tiling per the vault's compression config
4. Computes a SHA-256 checksum of the written file
5. Inserts a `FileRecord` into the SQLite catalog
6. Inserts the file's spatial/temporal extent into the R-tree index
7. Returns an `IngestResult`

**Atomicity:** If any step after the COG write fails (e.g., database insert), the COG file is automatically deleted.

### Ingesting Existing GeoTIFF Files

If you already have a GeoTIFF on disk:

```python
with Vault.open("./weather_vault") as vault:
    t = TemporalExtent(
        start=datetime(2024, 7, 2, 0, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 7, 2, 5, 59, 59, tzinfo=timezone.utc),
    )
    result = vault.ingest_file("weather", "./external_data/weather_20240702.tif", t)
```

The file is copied into the vault's `data/` directory. The original file is not modified.

### Single-Band Data

For single-band cubes, you can pass a 2D array — it's automatically reshaped to `(1, height, width)`:

```python
data_2d = np.random.default_rng(42).random((256, 256), dtype=np.float32)
result = vault.ingest("temperature", data_2d, time_extent)
```

### Batch Ingestion

For ingesting many time slices, loop over your data:

```python
import numpy as np
from datetime import datetime, timedelta, timezone
from voxelvault import Vault, TemporalExtent

with Vault.open("./weather_vault") as vault:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)

    for day in range(365):
        start = base_time + timedelta(days=day)
        end = start + timedelta(hours=23, minutes=59, seconds=59)
        t = TemporalExtent(start=start, end=end)

        data = rng.random((3, 512, 512), dtype=np.float32)
        vault.ingest("weather", data, t)

    print(f"Ingested {vault.cube_file_count('weather')} files")  # 365
```

### Error Handling

```python
# Band count mismatch
data_wrong = np.random.default_rng(0).random((5, 512, 512), dtype=np.float32)
vault.ingest("weather", data_wrong, t)
# ValueError: Data has 5 bands but cube 'weather' expects 3

# Cube doesn't exist
vault.ingest("nonexistent", data, t)
# ValueError: Cube 'nonexistent' not found in vault
```

---

## 8. Querying Data

Queries find and read data from the vault, with optional spatial, temporal, and variable filters.

### Basic Query (All Data)

```python
with Vault.open("./weather_vault") as vault:
    result = vault.query("weather")

    print(result.data.shape)          # (time, bands, height, width) if multiple files
    print(result.cube.name)           # 'weather'
    print(result.spatial_extent)      # SpatialExtent of the query
    print(result.temporal_extent)     # TemporalExtent spanning matched files
    print(len(result.source_files))   # Number of COG files that contributed
    print(result.variables)           # [Variable(name='temp', ...), ...]
```

### Temporal Filtering

Return only files within a time range:

```python
from datetime import datetime, timezone

result = vault.query(
    "weather",
    temporal_range=(
        datetime(2024, 3, 1, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
    ),
)
print(f"March 2024: {len(result.source_files)} files")
```

### Spatial Filtering

Return only the pixels within a geographic bounding box:

```python
# Query a subset: bounds = (west, south, east, north)
result = vault.query(
    "weather",
    spatial_bounds=(-100.0, 25.0, -80.0, 50.0),  # Central/Eastern US
)
print(result.data.shape)  # Smaller than full grid — only the matching window
```

### Variable Filtering

Select specific bands by variable name:

```python
# Only get temperature (band 1) — not precip or humidity
result = vault.query("weather", variables=["temp"])
print(result.data.shape)  # (time, 1, height, width) — single band
```

### Combining Filters

All filters can be combined:

```python
result = vault.query(
    "weather",
    spatial_bounds=(-100.0, 25.0, -80.0, 50.0),
    temporal_range=(
        datetime(2024, 6, 1, tzinfo=timezone.utc),
        datetime(2024, 6, 30, tzinfo=timezone.utc),
    ),
    variables=["temp", "precip"],
)
print(f"Shape: {result.data.shape}")
print(f"Files: {len(result.source_files)}")
```

### Single-File Query

When you expect exactly one file (e.g., a specific date):

```python
result = vault.query_single(
    "weather",
    temporal_range=(
        datetime(2024, 6, 15, tzinfo=timezone.utc),
        datetime(2024, 6, 15, 23, 59, 59, tzinfo=timezone.utc),
    ),
)
print(result.data.shape)  # (bands, height, width) — no time axis
```

**Raises `ValueError`** if zero or more than one file matches.

### QueryResult Structure

```python
@dataclass
class QueryResult:
    data: np.ndarray             # The raster data
    cube: CubeDescriptor         # Schema of the queried cube
    spatial_extent: SpatialExtent # Spatial coverage of the result
    temporal_extent: TemporalExtent # Time range of matched files
    source_files: list[FileRecord]  # Provenance — which files contributed

    @property
    def variables(self) -> list[Variable]:
        """Variables present in the result."""
```

**Data shape conventions:**
- Single file: `(bands, height, width)`
- Multiple files: `(time, bands, height, width)` — stacked along a new axis 0

### Provenance Tracking

Every query result tracks which source files contributed:

```python
result = vault.query("weather")
for f in result.source_files:
    print(f"  {f.file_id[:8]}  {f.relative_path}  {f.checksum[:16]}")
```

---

## 9. CLI Reference

VoxelVault includes a `vv` command-line tool (requires the `cli` optional dependency).

### `vv create`

Create a new vault.

```bash
vv create PATH [OPTIONS]

Options:
  --format [geotiff|jp2k]                    Storage format (default: geotiff)
  --compression [deflate|lzw|zstd|none|jp2k_lossless]
                                             Compression codec (default depends on format)
  --tile-size INTEGER                        Tile size in pixels (default: 256)
  --epsg INTEGER                             Default EPSG code (default: 4326)
```

**Examples:**

```bash
# Default settings (GeoTIFF/deflate)
vv create ./my_vault

# GeoTIFF with zstd and large tiles
vv create ./my_vault --compression zstd --tile-size 512

# JPEG 2000 lossless vault
vv create ./my_vault --format jp2k

# UTM Zone 18N
vv create ./my_vault --epsg 32618
```

### `vv info`

Show vault information.

```bash
vv info VAULT_PATH [OPTIONS]

Options:
  --json  Output as JSON
```

**Examples:**

```bash
vv info ./my_vault
# VoxelVault: ./my_vault
# Compression: deflate (level 6)
# Tile size: 256
# Cubes: 2
#   temperature: 365 files
#   wind_speed: 365 files
# Total files: 730

vv info ./my_vault --json
# {"config": {...}, "cubes": [...], "file_count": 730}
```

### `vv ingest`

Ingest a GeoTIFF file into a vault cube.

```bash
vv ingest VAULT_PATH CUBE_NAME SOURCE_FILE [OPTIONS]

Options:
  --start TEXT  Temporal start (ISO 8601) [required]
  --end TEXT    Temporal end (ISO 8601) [required]
```

**Examples:**

```bash
vv ingest ./my_vault temperature ./data/temp_20240101.tif \
    --start 2024-01-01 --end 2024-01-01

vv ingest ./my_vault temperature ./data/temp_20240615.tif \
    --start "2024-06-15T00:00:00" --end "2024-06-15T23:59:59"
```

> **Note:** Cube schemas must be registered via the Python API before ingesting via CLI. A future version may add a `vv register` command.

### `vv query`

Query a vault cube.

```bash
vv query VAULT_PATH CUBE_NAME [OPTIONS]

Options:
  --bounds FLOAT FLOAT FLOAT FLOAT  Spatial bounds: west south east north
  --start TEXT                       Temporal start (ISO 8601)
  --end TEXT                         Temporal end (ISO 8601)
  -o, --output PATH                  Output GeoTIFF path
  --json                             Output metadata as JSON
```

**Examples:**

```bash
# Summary of all data
vv query ./my_vault temperature

# Export a spatial/temporal subset to a GeoTIFF
vv query ./my_vault temperature \
    --bounds -100 25 -80 50 \
    --start 2024-06-01 --end 2024-06-30 \
    -o june_subset.tif

# Get metadata as JSON
vv query ./my_vault temperature --json

# Combined: export + JSON metadata
vv query ./my_vault temperature \
    --bounds -180 -90 180 90 \
    -o full_export.tif --json
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | User error (vault not found, cube doesn't exist, bad arguments) |
| 2 | Internal error (unexpected exception) |

---

## 10. Storage Layer

The storage module (`voxelvault.storage`) provides a pluggable backend architecture for raster I/O. Two built-in backends are available:

| Backend | Format | Codecs | Best For |
|---------|--------|--------|----------|
| **GeoTIFF/COG** | `.tif` | deflate, lzw, zstd, none | Float data, complex SAR, COG-based workflows |
| **JPEG 2000** | `.jp2` | jp2k_lossless | Integer imagery where compression ratio matters |

### Storage Backends

```python
from voxelvault.storage import get_backend, GeoTiffBackend, JP2KBackend

# Get a backend by format name
geotiff = get_backend("geotiff")
jp2k = get_backend("jp2k")

# Check capabilities
caps = jp2k.capabilities()
print(caps.supports_lossless)       # True
print(caps.supports_overviews)      # False (uses wavelet resolution levels)
print(caps.supports_complex_dtypes) # False
print(caps.supported_dtypes)        # {'uint8', 'uint16', 'int16'}
```

### Writing a COG (GeoTIFF)

```python
import numpy as np
from pathlib import Path
from rasterio.transform import Affine
from voxelvault.storage import write_cog

data = np.random.default_rng(42).random((3, 512, 512), dtype=np.float32)

path = write_cog(
    data=data,
    path="./output.tif",
    crs=4326,                      # EPSG code or rasterio.CRS object
    transform=Affine(0.01, 0, -180, 0, -0.01, 90),
    compression="deflate",
    compression_level=6,
    tile_size=256,
    overview_levels=[2, 4, 8, 16],
)
print(f"Written to {path}")
```

### Writing JP2K Lossless

```python
from voxelvault.models import StorageConfig
from voxelvault.storage import JP2KBackend

backend = JP2KBackend()
data = np.random.default_rng(42).integers(0, 10000, (3, 512, 512), dtype=np.uint16)

config = StorageConfig(format="jp2k", codec="jp2k_lossless")
path = backend.write(
    data=data,
    path="./output.jp2",
    crs=4326,
    transform=Affine(0.01, 0, -180, 0, -0.01, 90),
    config=config,
)
```

### Reading a Window

Windowed reads work identically for both GeoTIFF and JP2K files:

```python
from rasterio.windows import Window
from voxelvault.storage import read_window

# By pixel coordinates
data, profile = read_window("./output.tif", window=Window(0, 0, 128, 128))
print(data.shape)  # (3, 128, 128)

# By geographic bounds (west, south, east, north)
data, profile = read_window("./output.tif", bounds=(-179, 89, -178, 90))
print(data.shape)  # (3, ~100, ~100)

# Specific bands only
data, profile = read_window("./output.tif", bands=[1, 3])
print(data.shape)  # (2, 512, 512)

# Works the same for JP2K
data, profile = read_window("./output.jp2", bounds=(-179, 89, -178, 90))
```

### Reading Metadata

```python
from voxelvault.storage import read_metadata

meta = read_metadata("./output.tif")
print(f"Size: {meta.width}×{meta.height}, {meta.band_count} bands")
print(f"CRS: EPSG:{meta.crs_epsg}")
print(f"Tiled: {meta.is_tiled}")
print(f"Compression: {meta.compression}")
print(f"Format: {meta.storage_format}")  # "geotiff" or "jp2k"
print(f"File size: {meta.file_size_bytes / 1024:.0f} KB")
```

### Supported Data Types by Backend

| NumPy dtype | GeoTIFF/COG | JP2K Lossless | Use case |
|-------------|:-----------:|:-------------:|----------|
| `float32` | ✅ | ❌ | Most sensor data (temperature, reflectance) |
| `float64` | ✅ | ❌ | High-precision measurements |
| `int16` | ✅ | ✅ | Elevation, scaled reflectance |
| `uint16` | ✅ | ✅ | Raw digital numbers |
| `uint8` | ✅ | ✅ | Classification maps, masks |
| `int32` | ✅ | ❌ | Extended range integers |
| `uint32` | ✅ | ❌ | Extended range unsigned integers |
| `complex64` | ✅ | ❌ | SAR complex data (I/Q) |

### When to Choose GeoTIFF vs JP2K

| Criterion | GeoTIFF/COG | JP2K Lossless |
|-----------|-------------|---------------|
| **Compression ratio** | Good (zstd ≈ 60-65%) | Better (≈ 45-55% for integer data) |
| **Write speed** | Fast (zstd) to moderate (deflate) | Competitive with zstd |
| **Read speed** | Fastest | Slightly slower |
| **Windowed reads** | Fastest (tiled COG) | Slightly slower |
| **Overview support** | External overviews (fast zoom) | Internal wavelet levels only |
| **Float support** | Yes | No |
| **Complex support** | Yes | No |
| **Ecosystem support** | Universal (QGIS, GDAL, web) | Good (GDAL, limited web) |

**Recommendation:** Use GeoTIFF/COG as the default. Use JP2K lossless when you need maximum compression for integer-only imagery and all consumers can read JP2K.

### JP2K Driver Notes

- VoxelVault uses the **JP2OpenJPEG** GDAL driver (based on the OpenJPEG library)
- Lossless mode uses the **reversible 5/3 wavelet** transform (DWT)
- Activated via `QUALITY='100'` + `REVERSIBLE='YES'` creation options
- The Kakadu driver (JP2KAK) is not used — it requires a commercial license
- `int32` and `uint32` write successfully but may fail on read with some OpenJPEG versions — these dtypes are excluded from the supported set
- JP2K does not build GeoTIFF-style external overviews; instead, it uses wavelet resolution levels (controlled by the `RESOLUTIONS` creation option)

---

## 11. Recipes & Patterns

### Multi-Sensor Fusion

Register multiple cubes in the same vault for different sensor types:

```python
with Vault.create("./fusion_vault") as vault:
    # SAR data — complex backscatter
    sar_cube = CubeDescriptor(
        name="sar_vv",
        bands=[
            BandDefinition(band_index=1,
                variable=Variable(name="sigma0_vv", unit="dB", dtype="float32")),
        ],
        grid=GridDefinition(width=1024, height=1024, epsg=32618,
            transform=(10.0, 0, 500000, 0, -10.0, 4500000)),
    )

    # Optical data — multispectral
    optical_cube = CubeDescriptor(
        name="optical_msi",
        bands=[
            BandDefinition(band_index=1,
                variable=Variable(name="red", unit="reflectance", dtype="uint16")),
            BandDefinition(band_index=2,
                variable=Variable(name="green", unit="reflectance", dtype="uint16")),
            BandDefinition(band_index=3,
                variable=Variable(name="blue", unit="reflectance", dtype="uint16")),
            BandDefinition(band_index=4,
                variable=Variable(name="nir", unit="reflectance", dtype="uint16")),
        ],
        grid=GridDefinition(width=1024, height=1024, epsg=32618,
            transform=(10.0, 0, 500000, 0, -10.0, 4500000)),
    )

    vault.register_cube(sar_cube)
    vault.register_cube(optical_cube)
```

### Time-Series Analysis

Ingest daily data and query monthly subsets:

```python
from datetime import datetime, timedelta, timezone
import numpy as np

with Vault.open("./weather_vault") as vault:
    # Query all of March 2024
    result = vault.query(
        "temperature",
        temporal_range=(
            datetime(2024, 3, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
        ),
    )

    # result.data shape: (31, 1, height, width) — one per day
    # Compute monthly mean
    monthly_mean = result.data.mean(axis=0)  # (1, height, width)
    print(f"Mean temperature: {monthly_mean.mean():.1f} K")
```

### Vault Portability

Vaults are fully self-contained directories. To share or archive:

```bash
# Archive
tar -czf weather_vault.tar.gz ./weather_vault/

# Restore
tar -xzf weather_vault.tar.gz
python -c "
from voxelvault import Vault
with Vault.open('./weather_vault') as v:
    print(f'{v.file_count} files recovered')
"
```

The spatial index is rebuilt automatically on `Vault.open()` from the SQLite catalog, so even if the `index/` directory is missing or corrupted, the vault recovers.

### JSON Serialization

All Pydantic models support JSON serialization:

```python
# Serialize
cube = vault.get_cube("temperature")
json_str = cube.model_dump_json(indent=2)
print(json_str)

# Deserialize
from voxelvault import CubeDescriptor
restored = CubeDescriptor.model_validate_json(json_str)
assert restored == cube
```

> **Important:** Always use `model_dump(mode="json")` or `model_dump_json()` — not `model_dump()` alone — when you need JSON-safe output. The `timedelta` and `datetime` fields require the `mode="json"` flag for proper serialization.

---

## 12. API Reference

### Domain Models (`voxelvault.models`)

All models are frozen Pydantic v2 `BaseModel` instances.

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `Variable` | Measured quantity | `name`, `unit`, `dtype`, `nodata`, `description` |
| `BandDefinition` | Variable → band mapping | `band_index` (1-based), `variable`, `component` |
| `SpatialExtent` | Geographic bounding box | `west`, `south`, `east`, `north`, `epsg` |
| `TemporalExtent` | Time range | `start`, `end` (both timezone-aware) |
| `GridDefinition` | Spatial grid | `width`, `height`, `epsg`, `transform` |
| `CubeDescriptor` | Cube schema | `name`, `bands`, `grid`, `temporal_resolution`, `metadata` |
| `VaultConfig` | Vault settings | `storage` (StorageConfig), `default_epsg` |
| `StorageConfig` | Storage backend config | `format`, `codec`, `codec_level`, `tile_size`, `overview_levels` |
| `FileRecord` | Ingested file metadata | `file_id`, `cube_name`, `relative_path`, `spatial_extent`, `temporal_extent`, `checksum` |

### Vault Class (`voxelvault.vault.Vault`)

| Method / Property | Signature | Description |
|-------------------|-----------|-------------|
| `create()` | `(path, config=None) → Vault` | Create new vault |
| `open()` | `(path) → Vault` | Open existing vault |
| `register_cube()` | `(descriptor) → None` | Register a cube schema |
| `get_cube()` | `(name) → CubeDescriptor \| None` | Get cube by name |
| `list_cubes()` | `() → list[str]` | List all cube names |
| `ingest()` | `(cube_name, data, temporal_extent, spatial_extent=None) → IngestResult` | Ingest numpy array |
| `ingest_file()` | `(cube_name, source_path, temporal_extent) → IngestResult` | Ingest existing GeoTIFF |
| `query()` | `(cube_name, spatial_bounds=None, temporal_range=None, variables=None) → QueryResult` | Query cube |
| `query_single()` | `(cube_name, ...) → QueryResult` | Query expecting 1 file |
| `close()` | `() → None` | Close vault |
| `path` | `→ Path` | Vault root directory |
| `config` | `→ VaultConfig` | Vault configuration |
| `file_count` | `→ int` | Total cataloged files |
| `cube_file_count()` | `(cube_name) → int` | Files in a cube |

### Result Types

| Type | Fields |
|------|--------|
| `IngestResult` | `file_id`, `relative_path`, `file_size_bytes`, `checksum`, `elapsed_seconds` |
| `QueryResult` | `data` (ndarray), `cube`, `spatial_extent`, `temporal_extent`, `source_files` |

### Storage Functions (`voxelvault.storage`)

| Function / Class | Description |
|------------------|-------------|
| `write_cog()` | Write numpy array as COG (backward-compat wrapper) |
| `read_window()` | Read spatial subset from any rasterio-supported raster |
| `read_metadata()` | Read metadata without loading pixels |
| `get_backend(format)` | Get a `RasterStorageBackend` by format name |
| `GeoTiffBackend` | COG backend with full dtype + overview support |
| `JP2KBackend` | JPEG 2000 lossless backend (integer dtypes only) |
| `RasterStorageBackend` | Abstract base class for custom backends |
| `BackendCapabilities` | Dataclass describing backend support matrix |
| `RasterMetadata` | Format-agnostic file metadata (aliased as `COGMetadata`) |

---

## 13. Architecture Deep Dive

### The Four Layers

```
┌──────────────────────────────────────────────────┐
│                  CLI (cli.py)                     │  ← Thin Click wrapper
├──────────────────────────────────────────────────┤
│              Vault Orchestrator                   │  ← Public API
│    vault.py  ·  ingest.py  ·  query.py           │
├──────────────────────────────────────────────────┤
│              Service Layer                        │  ← Internal engines
│    schema.py (SQLite)  ·  index.py (R-tree)      │
├──────────────────────────────────────────────────┤
│              Storage Layer                        │  ← File I/O
│    storage.py (pluggable backends)                │
│    GeoTiffBackend · JP2KBackend                   │
├──────────────────────────────────────────────────┤
│              Domain Models                        │  ← Type foundation
│    models.py (Pydantic v2)                        │
└──────────────────────────────────────────────────┘
```

Each layer depends only on the layers below it. The CLI is a thin wrapper; the Vault orchestrator composes the service and storage layers; the service layer knows nothing about files; the storage layer knows nothing about schemas.

### Why COG? (Not Zarr, Not HDF5)

Cloud Optimized GeoTIFFs were chosen because:

1. **Universal compatibility** — every GIS tool reads GeoTIFF
2. **Internal tiling** — enables efficient spatial subsetting without loading the full file
3. **Future-proof** — HTTP range requests for cloud-native access
4. **No metadata complexity** — unlike HDF5/Zarr, there's no chunking metadata to manage

**Trade-off:** No built-in multi-dimensional chunking along time. VoxelVault's model is "Time = Files" — the filesystem IS the time dimension.

### Why SQLite? (Not PostgreSQL, Not Parquet)

1. **Serverless** — no process to manage
2. **Single-file** — vault is fully self-contained
3. **WAL mode** — good concurrent read performance
4. **Portable** — copy the directory, done

**Trade-off:** No built-in spatial indexing (unlike PostGIS). Mitigated by the R-tree layer.

### Why R-tree? (Not S2, Not H3)

1. **Rectangular extents** — raster data is rectangular; hexagonal grids add complexity without benefit
2. **Disk-persistent** — survives close/reopen cycles
3. **O(log N)** — proven query complexity
4. **Minimal dependencies** — just `rtree` wrapping libspatialindex

**Trade-off:** No geodesic accuracy for antimeridian-crossing bounding boxes. Acceptable for v2.

### Data Flow

```
Ingest:  numpy array → write_cog() → .tif on disk → FileRecord in v2.db → entry in R-tree

Query:   filter params → R-tree lookup → file_ids → FileRecords from v2.db
         → read_window() for each file → np.stack() → QueryResult
```

---

## 14. Troubleshooting

### Common Errors

**`FileExistsError: Vault already exists at ./path (v2.db found)`**

A vault already exists. Either use `Vault.open()` or choose a different path.

**`FileNotFoundError: No vault found at ./path (v2.db missing)`**

No vault at this path. Create one with `Vault.create()` first.

**`ValueError: Cube 'name' not found in vault`**

Register the cube before ingesting or querying:
```python
vault.register_cube(descriptor)
```

**`ValueError: Data has N bands but cube 'name' expects M`**

Your data array's first dimension (or single band if 2D) doesn't match the cube's band count. Reshape your data to match.

**`ValidationError: ... is not a valid Python identifier`**

Variable names must be valid Python identifiers: letters, digits, underscores; cannot start with a digit or contain spaces.

**`ValidationError: ... datetimes must be timezone-aware`**

Always use timezone-aware datetimes:
```python
# ❌ Wrong
datetime(2024, 1, 1)

# ✅ Correct
datetime(2024, 1, 1, tzinfo=timezone.utc)
```

**`ImportError: No module named 'rasterio'`**

Install the storage extra: `pip install "voxelvault[storage]"` or `pip install "voxelvault[all]"`

### Performance Tips

1. **Use small tile sizes (256) for random access** — better for spatial subsetting
2. **Use large tile sizes (512+) for sequential reading** — less overhead per tile
3. **Choose zstd compression** for best compression ratio with fast decompression
4. **Keep grids moderate** — 10K×10K grids work fine; 100K×100K may strain memory
5. **The spatial index rebuilds on open** — for vaults with 100K+ files, the first `Vault.open()` may take a few seconds

### Getting Help

```bash
# CLI help
vv --help
vv create --help
vv query --help

# Python help
python -c "from voxelvault import Vault; help(Vault)"
python -c "from voxelvault import Vault; help(Vault.query)"
```
