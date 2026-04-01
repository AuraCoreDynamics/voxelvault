"""Tests for JPEG 2000 lossless storage backend.

Covers:
- Lossless round-trip for supported integer dtypes (uint8, uint16, int16)
- Rejection of unsupported dtypes (float, complex, int32, uint32)
- Windowed reads
- Config validation
- Integration with the vault lifecycle
- Backend capability reporting
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window

from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    GridDefinition,
    StorageConfig,
    TemporalExtent,
    Variable,
    VaultConfig,
)
from voxelvault.storage import (
    BackendCapabilities,
    GeoTiffBackend,
    JP2KBackend,
    RasterMetadata,
    get_backend,
)
from voxelvault.vault import Vault

RNG = np.random.default_rng(42)

SAMPLE_CRS = CRS.from_epsg(4326)
SAMPLE_TRANSFORM = Affine(0.01, 0.0, -10.0, 0.0, -0.01, 50.0)

JP2K_CONFIG = StorageConfig(format="jp2k", codec="jp2k_lossless")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_int_array(dtype: str, bands: int = 3, height: int = 64, width: int = 64) -> np.ndarray:
    info = np.iinfo(np.dtype(dtype))
    lo = max(info.min, 0)
    hi = min(info.max, 10000)
    return RNG.integers(lo, hi, size=(bands, height, width), dtype=dtype)


def _make_cube(
    name: str = "jp2k_cube",
    dtype: str = "uint16",
    width: int = 64,
    height: int = 64,
) -> CubeDescriptor:
    grid = GridDefinition(
        width=width, height=height, epsg=4326,
        transform=(0.01, 0.0, -180.0, 0.0, -0.01, 90.0),
    )
    bands = [
        BandDefinition(band_index=1, variable=Variable(name="red", unit="dn", dtype=dtype)),
        BandDefinition(band_index=2, variable=Variable(name="green", unit="dn", dtype=dtype)),
        BandDefinition(band_index=3, variable=Variable(name="blue", unit="dn", dtype=dtype)),
    ]
    return CubeDescriptor(name=name, bands=bands, grid=grid, temporal_resolution=timedelta(days=1))


# ---------------------------------------------------------------------------
# Backend capability tests
# ---------------------------------------------------------------------------


class TestJP2KCapabilities:
    def test_capabilities_type(self):
        backend = JP2KBackend()
        caps = backend.capabilities()
        assert isinstance(caps, BackendCapabilities)

    def test_supports_lossless(self):
        caps = JP2KBackend().capabilities()
        assert caps.supports_lossless is True

    def test_no_overviews(self):
        caps = JP2KBackend().capabilities()
        assert caps.supports_overviews is False

    def test_supports_windowed_reads(self):
        caps = JP2KBackend().capabilities()
        assert caps.supports_windowed_reads is True

    def test_no_complex_dtypes(self):
        caps = JP2KBackend().capabilities()
        assert caps.supports_complex_dtypes is False

    def test_supported_dtypes(self):
        caps = JP2KBackend().capabilities()
        assert "uint8" in caps.supported_dtypes
        assert "uint16" in caps.supported_dtypes
        assert "int16" in caps.supported_dtypes
        assert "float32" not in caps.supported_dtypes
        assert "complex64" not in caps.supported_dtypes

    def test_file_extension(self):
        assert JP2KBackend().file_extension == ".jp2"


class TestGeoTiffCapabilities:
    def test_supports_all_common_dtypes(self):
        caps = GeoTiffBackend().capabilities()
        for dt in ("uint8", "uint16", "int16", "float32", "float64", "complex64"):
            assert dt in caps.supported_dtypes

    def test_supports_complex(self):
        assert GeoTiffBackend().capabilities().supports_complex_dtypes is True

    def test_supports_overviews(self):
        assert GeoTiffBackend().capabilities().supports_overviews is True

    def test_file_extension(self):
        assert GeoTiffBackend().file_extension == ".tif"


# ---------------------------------------------------------------------------
# Backend registry tests
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    def test_get_geotiff(self):
        b = get_backend("geotiff")
        assert isinstance(b, GeoTiffBackend)

    def test_get_jp2k(self):
        b = get_backend("jp2k")
        assert isinstance(b, JP2KBackend)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="No storage backend"):
            get_backend("hdf5")


# ---------------------------------------------------------------------------
# JP2K write + lossless round-trip tests
# ---------------------------------------------------------------------------


class TestJP2KLosslessRoundTrip:
    """Verify exact (bit-for-bit) round-trip for supported integer dtypes."""

    @pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16"])
    def test_lossless_integer_roundtrip(self, tmp_path: Path, dtype: str) -> None:
        backend = JP2KBackend()
        data = _make_int_array(dtype, bands=3, height=64, width=64)
        out = backend.write(data, tmp_path / f"{dtype}.jp2", crs=SAMPLE_CRS,
                            transform=SAMPLE_TRANSFORM, config=JP2K_CONFIG)
        read_data, profile = backend.read_window(out)
        np.testing.assert_array_equal(read_data, data)
        assert profile["dtype"] == dtype

    def test_single_band_roundtrip(self, tmp_path: Path) -> None:
        backend = JP2KBackend()
        data = _make_int_array("uint16", bands=1)
        out = backend.write(data, tmp_path / "single.jp2", crs=SAMPLE_CRS,
                            transform=SAMPLE_TRANSFORM, config=JP2K_CONFIG)
        read_data, _ = backend.read_window(out)
        np.testing.assert_array_equal(read_data, data)

    def test_2d_input(self, tmp_path: Path) -> None:
        backend = JP2KBackend()
        data = RNG.integers(0, 10000, (64, 64), dtype=np.uint16)
        out = backend.write(data, tmp_path / "2d.jp2", crs=SAMPLE_CRS,
                            transform=SAMPLE_TRANSFORM, config=JP2K_CONFIG)
        read_data, _ = backend.read_window(out)
        np.testing.assert_array_equal(read_data[0], data)

    def test_nodata_preserved(self, tmp_path: Path) -> None:
        backend = JP2KBackend()
        data = _make_int_array("uint16", bands=1)
        out = backend.write(data, tmp_path / "nd.jp2", crs=SAMPLE_CRS,
                            transform=SAMPLE_TRANSFORM, nodata=0, config=JP2K_CONFIG)
        _, profile = backend.read_window(out)
        assert profile["nodata"] == 0.0


# ---------------------------------------------------------------------------
# JP2K dtype rejection tests
# ---------------------------------------------------------------------------


class TestJP2KDtypeRejection:
    """Unsupported dtype/codec combinations must fail explicitly."""

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_reject_float(self, dtype: str) -> None:
        backend = JP2KBackend()
        data = RNG.standard_normal((1, 32, 32)).astype(dtype)
        with pytest.raises(ValueError, match="not supported"):
            backend.validate_write(data)

    def test_reject_complex64(self) -> None:
        backend = JP2KBackend()
        data = (RNG.standard_normal((1, 32, 32)) + 1j * RNG.standard_normal((1, 32, 32))).astype("complex64")
        with pytest.raises(ValueError, match="Complex dtype"):
            backend.validate_write(data)

    def test_reject_int32(self) -> None:
        backend = JP2KBackend()
        data = RNG.integers(0, 1000, (1, 32, 32), dtype=np.int32)
        with pytest.raises(ValueError, match="not supported"):
            backend.validate_write(data)


# ---------------------------------------------------------------------------
# JP2K windowed read tests
# ---------------------------------------------------------------------------


class TestJP2KWindowedReads:
    @pytest.fixture()
    def jp2_file(self, tmp_path: Path) -> tuple[Path, np.ndarray]:
        backend = JP2KBackend()
        data = _make_int_array("uint16", bands=3, height=64, width=64)
        out = backend.write(data, tmp_path / "windowed.jp2", crs=SAMPLE_CRS,
                            transform=SAMPLE_TRANSFORM, config=JP2K_CONFIG)
        return out, data

    def test_pixel_window(self, jp2_file: tuple[Path, np.ndarray]) -> None:
        path, data = jp2_file
        backend = JP2KBackend()
        win = Window(10, 10, 20, 20)
        read_data, profile = backend.read_window(path, window=win)
        assert read_data.shape == (3, 20, 20)
        np.testing.assert_array_equal(read_data, data[:, 10:30, 10:30])

    def test_geographic_bounds(self, jp2_file: tuple[Path, np.ndarray]) -> None:
        path, _ = jp2_file
        backend = JP2KBackend()
        bounds = (-9.5, 49.5, -9.0, 50.0)
        read_data, profile = backend.read_window(path, bounds=bounds)
        assert read_data.shape[0] == 3
        assert read_data.shape[1] > 0
        assert read_data.shape[2] > 0

    def test_band_selection(self, jp2_file: tuple[Path, np.ndarray]) -> None:
        path, data = jp2_file
        backend = JP2KBackend()
        read_data, _ = backend.read_window(path, bands=[2])
        assert read_data.shape[0] == 1
        np.testing.assert_array_equal(read_data[0], data[1])


# ---------------------------------------------------------------------------
# JP2K metadata tests
# ---------------------------------------------------------------------------


class TestJP2KMetadata:
    def test_metadata_format(self, tmp_path: Path) -> None:
        backend = JP2KBackend()
        data = _make_int_array("uint16", bands=3)
        out = backend.write(data, tmp_path / "meta.jp2", crs=SAMPLE_CRS,
                            transform=SAMPLE_TRANSFORM, config=JP2K_CONFIG)
        meta = backend.read_metadata(out)
        assert isinstance(meta, RasterMetadata)
        assert meta.storage_format == "jp2k"
        assert meta.width == 64
        assert meta.height == 64
        assert meta.band_count == 3
        assert meta.dtype == "uint16"
        assert meta.crs_epsg == 4326
        assert meta.file_size_bytes > 0

    def test_overwrite_false_raises(self, tmp_path: Path) -> None:
        backend = JP2KBackend()
        data = _make_int_array("uint16", bands=1, height=32, width=32)
        p = tmp_path / "exist.jp2"
        backend.write(data, p, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, config=JP2K_CONFIG)
        with pytest.raises(FileExistsError):
            backend.write(data, p, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, config=JP2K_CONFIG)

    def test_overwrite_true_succeeds(self, tmp_path: Path) -> None:
        backend = JP2KBackend()
        data = _make_int_array("uint16", bands=1, height=32, width=32)
        p = tmp_path / "ow.jp2"
        backend.write(data, p, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM, config=JP2K_CONFIG)
        out = backend.write(data, p, crs=SAMPLE_CRS, transform=SAMPLE_TRANSFORM,
                            config=JP2K_CONFIG, overwrite=True)
        assert out.exists()


# ---------------------------------------------------------------------------
# JP2K vault lifecycle integration
# ---------------------------------------------------------------------------


class TestJP2KVaultLifecycle:
    """End-to-end: create JP2K vault → register cube → ingest → query → verify."""

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        vault_dir = tmp_path / "jp2k_vault"
        config = VaultConfig(storage=StorageConfig(format="jp2k", codec="jp2k_lossless"))
        cube = _make_cube()
        data = RNG.integers(0, 10000, (3, 64, 64), dtype=np.uint16)
        te = TemporalExtent(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with Vault.create(vault_dir, config=config) as vault:
            vault.register_cube(cube)
            result = vault.ingest(cube.name, data, te)

            assert result.file_size_bytes > 0
            assert result.relative_path.endswith(".jp2")

            qr = vault.query(cube.name)
            np.testing.assert_array_equal(qr.data, data)
            assert len(qr.source_files) == 1

    def test_reopen_jp2k_vault(self, tmp_path: Path) -> None:
        vault_dir = tmp_path / "jp2k_vault"
        config = VaultConfig(storage=StorageConfig(format="jp2k", codec="jp2k_lossless"))
        cube = _make_cube()
        data = RNG.integers(0, 10000, (3, 64, 64), dtype=np.uint16)
        te = TemporalExtent(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with Vault.create(vault_dir, config=config) as vault:
            vault.register_cube(cube)
            vault.ingest(cube.name, data, te)

        with Vault.open(vault_dir) as vault:
            assert vault.config.storage.format == "jp2k"
            qr = vault.query(cube.name)
            np.testing.assert_array_equal(qr.data, data)

    def test_multi_temporal_jp2k(self, tmp_path: Path) -> None:
        vault_dir = tmp_path / "jp2k_vault"
        config = VaultConfig(storage=StorageConfig(format="jp2k", codec="jp2k_lossless"))
        cube = _make_cube()

        slices = []
        for day in range(1, 4):
            arr = RNG.integers(0, 10000, (3, 64, 64), dtype=np.uint16)
            te = TemporalExtent(
                start=datetime(2024, 1, day, tzinfo=timezone.utc),
                end=datetime(2024, 1, day, 23, 59, 59, tzinfo=timezone.utc),
            )
            slices.append((arr, te))

        with Vault.create(vault_dir, config=config) as vault:
            vault.register_cube(cube)
            for arr, te in slices:
                vault.ingest(cube.name, arr, te)

            qr = vault.query(cube.name)
            assert len(qr.source_files) == 3
            assert qr.data.shape == (3, 3, 64, 64)

    def test_reject_float_ingest_jp2k(self, tmp_path: Path) -> None:
        """Attempting to ingest float data into a JP2K vault raises ValueError."""
        vault_dir = tmp_path / "jp2k_vault"
        config = VaultConfig(storage=StorageConfig(format="jp2k", codec="jp2k_lossless"))
        grid = GridDefinition(
            width=32, height=32, epsg=4326,
            transform=(0.01, 0.0, -180.0, 0.0, -0.01, 90.0),
        )
        cube = CubeDescriptor(
            name="float_cube",
            bands=[BandDefinition(band_index=1, variable=Variable(name="val", unit="dn", dtype="float32"))],
            grid=grid,
        )
        data = RNG.random((1, 32, 32), dtype=np.float32)
        te = TemporalExtent(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
        )

        with Vault.create(vault_dir, config=config) as vault:
            vault.register_cube(cube)
            with pytest.raises(ValueError, match="not supported"):
                vault.ingest("float_cube", data, te)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_jp2k_lossless_config_valid(self):
        sc = StorageConfig(format="jp2k", codec="jp2k_lossless")
        assert sc.format == "jp2k"

    def test_jp2k_with_geotiff_codec_invalid(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StorageConfig(format="jp2k", codec="deflate")

    def test_geotiff_with_jp2k_codec_invalid(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StorageConfig(format="geotiff", codec="jp2k_lossless")
