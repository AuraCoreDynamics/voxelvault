"""Benchmark scaffolding for comparing storage backends.

Compares write time, read time, file size, and windowed read time between
GeoTIFF codecs (deflate, lzw, zstd) and JP2K lossless on representative
sample rasters.

Run with: pytest tests/test_benchmark.py -v -s
Mark:     @pytest.mark.benchmark (skipped by default in CI)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from voxelvault.models import StorageConfig
from voxelvault.storage import GeoTiffBackend, JP2KBackend, RasterStorageBackend

from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window

RNG = np.random.default_rng(42)
SAMPLE_CRS = CRS.from_epsg(4326)
SAMPLE_TRANSFORM = Affine(0.01, 0.0, -10.0, 0.0, -0.01, 50.0)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    label: str
    write_seconds: float
    read_seconds: float
    windowed_read_seconds: float
    file_size_bytes: int
    lossless: bool


def _benchmark_backend(
    backend: RasterStorageBackend,
    config: StorageConfig,
    data: np.ndarray,
    path: Path,
    label: str,
) -> BenchmarkResult:
    """Run write/read/windowed-read benchmarks for a backend."""
    # Write
    t0 = time.perf_counter()
    out = backend.write(data, path, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, config=config)
    write_time = time.perf_counter() - t0

    file_size = out.stat().st_size

    # Full read
    t0 = time.perf_counter()
    read_data, _ = backend.read_window(out)
    read_time = time.perf_counter() - t0

    lossless = np.array_equal(read_data, data)

    # Windowed read (center 128x128)
    h, w = data.shape[1], data.shape[2]
    win = Window(w // 4, h // 4, w // 2, h // 2)
    t0 = time.perf_counter()
    backend.read_window(out, window=win)
    windowed_time = time.perf_counter() - t0

    return BenchmarkResult(
        label=label,
        write_seconds=write_time,
        read_seconds=read_time,
        windowed_read_seconds=windowed_time,
        file_size_bytes=file_size,
        lossless=lossless,
    )


def _format_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a table."""
    lines = [
        f"{'Backend':<25s} {'Write (s)':<12s} {'Read (s)':<12s} "
        f"{'Win Read (s)':<14s} {'Size (KB)':<12s} {'Lossless':<10s}",
        "-" * 85,
    ]
    for r in results:
        lines.append(
            f"{r.label:<25s} {r.write_seconds:<12.4f} {r.read_seconds:<12.4f} "
            f"{r.windowed_read_seconds:<14.4f} {r.file_size_bytes / 1024:<12.1f} "
            f"{'Yes' if r.lossless else 'No':<10s}"
        )
    return "\n".join(lines)


@pytest.mark.benchmark
class TestStorageBenchmarks:
    """Compare storage backends on representative data.

    These tests always pass — they exist to produce timing/size output.
    Run with ``pytest tests/test_benchmark.py -v -s`` to see results.
    """

    def test_uint16_comparison(self, tmp_path: Path) -> None:
        """Compare GeoTIFF codecs vs JP2K on uint16 data."""
        data = RNG.integers(0, 10000, (3, 512, 512), dtype=np.uint16)

        configs = [
            ("GeoTIFF/deflate", GeoTiffBackend(),
             StorageConfig(format="geotiff", codec="deflate")),
            ("GeoTIFF/lzw", GeoTiffBackend(),
             StorageConfig(format="geotiff", codec="lzw")),
            ("GeoTIFF/zstd", GeoTiffBackend(),
             StorageConfig(format="geotiff", codec="zstd")),
            ("GeoTIFF/none", GeoTiffBackend(),
             StorageConfig(format="geotiff", codec="none")),
            ("JP2K/lossless", JP2KBackend(),
             StorageConfig(format="jp2k", codec="jp2k_lossless")),
        ]

        results: list[BenchmarkResult] = []
        for label, backend, config in configs:
            ext = backend.file_extension
            path = tmp_path / f"bench_{label.replace('/', '_')}{ext}"
            result = _benchmark_backend(backend, config, data, path, label)
            results.append(result)

        print("\n\n=== uint16 3-band 512x512 Benchmark ===")
        print(_format_results(results))

        # All should be lossless for integer data
        for r in results:
            assert r.lossless, f"{r.label} was not lossless"

    def test_int16_comparison(self, tmp_path: Path) -> None:
        """Compare on int16 data (e.g. SAR)."""
        data = RNG.integers(-5000, 5000, (1, 512, 512), dtype=np.int16)

        configs = [
            ("GeoTIFF/deflate", GeoTiffBackend(),
             StorageConfig(format="geotiff", codec="deflate")),
            ("GeoTIFF/zstd", GeoTiffBackend(),
             StorageConfig(format="geotiff", codec="zstd")),
            ("JP2K/lossless", JP2KBackend(),
             StorageConfig(format="jp2k", codec="jp2k_lossless")),
        ]

        results: list[BenchmarkResult] = []
        for label, backend, config in configs:
            ext = backend.file_extension
            path = tmp_path / f"bench_{label.replace('/', '_')}{ext}"
            result = _benchmark_backend(backend, config, data, path, label)
            results.append(result)

        print("\n\n=== int16 1-band 512x512 Benchmark ===")
        print(_format_results(results))

        for r in results:
            assert r.lossless, f"{r.label} was not lossless"

    def test_float32_geotiff_only(self, tmp_path: Path) -> None:
        """Float32: only GeoTIFF supports it — JP2K must reject."""
        data = RNG.standard_normal((3, 512, 512)).astype(np.float32)

        configs = [
            ("GeoTIFF/deflate", GeoTiffBackend(),
             StorageConfig(format="geotiff", codec="deflate")),
            ("GeoTIFF/zstd", GeoTiffBackend(),
             StorageConfig(format="geotiff", codec="zstd")),
        ]

        results: list[BenchmarkResult] = []
        for label, backend, config in configs:
            path = tmp_path / f"bench_{label.replace('/', '_')}.tif"
            result = _benchmark_backend(backend, config, data, path, label)
            results.append(result)

        print("\n\n=== float32 3-band 512x512 Benchmark (GeoTIFF only) ===")
        print(_format_results(results))

        # Verify JP2K rejects float32
        jp2k = JP2KBackend()
        with pytest.raises(ValueError, match="not supported"):
            jp2k.validate_write(data)
