"""Tests for the VoxelVault COG storage backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window

from voxelvault.storage import COGMetadata, read_metadata, read_window, write_cog

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# A simple WGS-84 affine: 0.01° pixels starting at lon=-10, lat=50
SAMPLE_CRS = CRS.from_epsg(4326)
SAMPLE_TRANSFORM = Affine(0.01, 0.0, -10.0, 0.0, -0.01, 50.0)


def _make_array(dtype: str, bands: int = 3, height: int = 512, width: int = 512) -> np.ndarray:
    """Create a random 3D array with the given dtype."""
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        real = RNG.standard_normal((bands, height, width)).astype("float32")
        imag = RNG.standard_normal((bands, height, width)).astype("float32")
        return (real + 1j * imag).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.floating):
        return RNG.standard_normal((bands, height, width)).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(np.dtype(dtype))
        return RNG.integers(info.min, info.max, size=(bands, height, width), dtype=dtype)
    # uint8
    return RNG.integers(0, 255, size=(bands, height, width), dtype=dtype)


@pytest.fixture()
def sample_f32(tmp_path: Path) -> Path:
    """Write a float32 3-band COG and return its path."""
    data = _make_array("float32")
    return write_cog(data, tmp_path / "f32.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)


# ---------------------------------------------------------------------------
# T2.4 — write_cog tests
# ---------------------------------------------------------------------------


class TestWriteCOG:
    """Tests for write_cog()."""

    def test_produces_valid_geotiff(self, tmp_path: Path) -> None:
        data = _make_array("float32")
        out = write_cog(data, tmp_path / "out.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        with rasterio.open(out) as ds:
            assert ds.driver == "GTiff"

    def test_correct_crs(self, tmp_path: Path) -> None:
        data = _make_array("float32")
        out = write_cog(data, tmp_path / "crs.tif", crs=4326, transform=SAMPLE_TRANSFORM)
        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() == 4326

    def test_correct_transform(self, tmp_path: Path) -> None:
        data = _make_array("float32")
        out = write_cog(data, tmp_path / "xf.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        with rasterio.open(out) as ds:
            assert ds.transform == SAMPLE_TRANSFORM

    def test_correct_band_count_and_shape(self, tmp_path: Path) -> None:
        data = _make_array("float32", bands=5)
        out = write_cog(data, tmp_path / "bands.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        with rasterio.open(out) as ds:
            assert ds.count == 5
            assert ds.width == 512
            assert ds.height == 512

    def test_correct_dtype(self, tmp_path: Path) -> None:
        data = _make_array("int16")
        out = write_cog(data, tmp_path / "dtype.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "int16"

    def test_is_tiled(self, tmp_path: Path) -> None:
        data = _make_array("float32")
        ts = 256
        out = write_cog(data, tmp_path / "tile.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, tile_size=ts)
        with rasterio.open(out) as ds:
            for bh, bw in ds.block_shapes:
                assert bh == ts
                assert bw == ts

    def test_has_overviews(self, tmp_path: Path) -> None:
        data = _make_array("float32")
        levels = [2, 4]
        out = write_cog(
            data, tmp_path / "ovr.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, overview_levels=levels
        )
        with rasterio.open(out) as ds:
            assert ds.overviews(1) == levels

    def test_overwrite_false_raises(self, tmp_path: Path) -> None:
        data = _make_array("float32", bands=1, height=64, width=64)
        p = tmp_path / "exist.tif"
        write_cog(data, p, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        with pytest.raises(FileExistsError):
            write_cog(data, p, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, overwrite=False)

    def test_overwrite_true_succeeds(self, tmp_path: Path) -> None:
        data = _make_array("float32", bands=1, height=64, width=64)
        p = tmp_path / "ow.tif"
        write_cog(data, p, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        out = write_cog(data, p, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, overwrite=True)
        assert out.exists()

    def test_2d_input_single_band(self, tmp_path: Path) -> None:
        data = RNG.standard_normal((256, 256)).astype("float32")
        out = write_cog(data, tmp_path / "2d.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        with rasterio.open(out) as ds:
            assert ds.count == 1
            read_back = ds.read(1)
            np.testing.assert_array_equal(read_back, data)

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int16", "uint16", "uint8"])
    def test_supported_dtypes(self, tmp_path: Path, dtype: str) -> None:
        data = _make_array(dtype, bands=1, height=256, width=256)
        out = write_cog(data, tmp_path / f"{dtype}.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == dtype
            read_back = ds.read()
            np.testing.assert_array_equal(read_back, data)

    def test_complex64_write_read(self, tmp_path: Path) -> None:
        """complex64 should be writable and readable even without overviews."""
        data = _make_array("complex64", bands=1, height=256, width=256)
        out = write_cog(data, tmp_path / "cplx.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "complex64"
            read_back = ds.read()
            np.testing.assert_array_equal(read_back, data)

    @pytest.mark.parametrize("comp", ["deflate", "lzw", "zstd"])
    def test_compression_options(self, tmp_path: Path, comp: str) -> None:
        data = _make_array("float32", bands=1, height=256, width=256)
        out = write_cog(data, tmp_path / f"{comp}.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, compression=comp)
        with rasterio.open(out) as ds:
            assert ds.compression.name.lower() == comp

    def test_nodata_set(self, tmp_path: Path) -> None:
        data = _make_array("float32", bands=1, height=64, width=64)
        out = write_cog(data, tmp_path / "nd.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, nodata=-9999.0)
        with rasterio.open(out) as ds:
            assert ds.nodata == -9999.0

    def test_returns_path(self, tmp_path: Path) -> None:
        data = _make_array("float32", bands=1, height=64, width=64)
        out = write_cog(data, tmp_path / "ret.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM)
        assert isinstance(out, Path)
        assert out.exists()


# ---------------------------------------------------------------------------
# T2.4 — read_window tests
# ---------------------------------------------------------------------------


class TestReadWindow:
    """Tests for read_window()."""

    def test_pixel_window_returns_correct_subset(self, sample_f32: Path) -> None:
        win = Window(col_off=10, row_off=20, width=100, height=50)
        data, profile = read_window(sample_f32, window=win)
        assert data.shape == (3, 50, 100)
        assert profile["width"] == 100
        assert profile["height"] == 50

    def test_geographic_bounds(self, sample_f32: Path) -> None:
        # The sample covers lon [-10, -4.88], lat [44.88, 50]
        # Request a sub-region
        west, south, east, north = -9.0, 49.0, -8.0, 50.0
        data, profile = read_window(sample_f32, bounds=(west, south, east, north))
        assert data.shape[0] == 3
        assert data.shape[1] > 0
        assert data.shape[2] > 0
        # Profile transform origin should be near the requested bounds
        assert abs(profile["transform"].c - west) < 0.01
        assert abs(profile["transform"].f - north) < 0.01

    def test_band_selection(self, sample_f32: Path) -> None:
        data, profile = read_window(sample_f32, bands=[2])
        assert data.shape[0] == 1
        assert profile["count"] == 1

    def test_band_selection_multiple(self, sample_f32: Path) -> None:
        data, profile = read_window(sample_f32, bands=[1, 3])
        assert data.shape[0] == 2

    def test_full_read_no_window(self, sample_f32: Path) -> None:
        data, profile = read_window(sample_f32)
        assert data.shape == (3, 512, 512)

    def test_bounds_beyond_extent_clipped(self, sample_f32: Path) -> None:
        """Bounds extending beyond the file should be clipped to available data."""
        # File covers approx lon [-10, -4.88], lat [44.88, 50]
        data, profile = read_window(sample_f32, bounds=(-20.0, 40.0, -4.0, 55.0))
        assert data.shape[0] == 3
        # Should be clipped to the file extent, so shape ≤ (3, 512, 512)
        assert data.shape[1] <= 512
        assert data.shape[2] <= 512

    def test_profile_has_crs(self, sample_f32: Path) -> None:
        _, profile = read_window(sample_f32)
        assert profile["crs"].to_epsg() == 4326


# ---------------------------------------------------------------------------
# T2.4 — read_metadata tests
# ---------------------------------------------------------------------------


class TestReadMetadata:
    """Tests for read_metadata()."""

    def test_returns_cogmetadata(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert isinstance(meta, COGMetadata)

    def test_correct_dimensions(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert meta.width == 512
        assert meta.height == 512
        assert meta.band_count == 3

    def test_correct_dtype(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert meta.dtype == "float32"

    def test_correct_crs(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert meta.crs_epsg == 4326

    def test_correct_transform(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert meta.transform == tuple(SAMPLE_TRANSFORM)

    def test_correct_bounds(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        west, south, east, north = meta.bounds
        assert west == pytest.approx(-10.0)
        assert north == pytest.approx(50.0)

    def test_compression(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert meta.compression is not None
        assert meta.compression.lower() == "deflate"

    def test_is_tiled(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert meta.is_tiled is True

    def test_file_size(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert meta.file_size_bytes > 0
        assert meta.file_size_bytes == sample_f32.stat().st_size

    def test_nodata(self, tmp_path: Path) -> None:
        data = _make_array("float32", bands=1, height=64, width=64)
        p = write_cog(data, tmp_path / "nd.tif", crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, nodata=-9999.0)
        meta = read_metadata(p)
        assert meta.nodata == -9999.0

    def test_path_stored(self, sample_f32: Path) -> None:
        meta = read_metadata(sample_f32)
        assert meta.path == sample_f32
