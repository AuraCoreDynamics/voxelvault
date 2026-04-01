"""Tests for the _remote module (cloud storage helpers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxelvault._remote import is_remote_uri, rasterio_open_path, resolve_path


class TestIsRemoteUri:
    def test_s3_uri(self):
        assert is_remote_uri("s3://bucket/key") is True

    def test_gs_uri(self):
        assert is_remote_uri("gs://bucket/key") is True

    def test_az_uri(self):
        assert is_remote_uri("az://container/blob") is True

    def test_local_path(self):
        assert is_remote_uri("/tmp/vault") is False
        assert is_remote_uri(Path("/tmp/vault")) is False

    def test_windows_path(self):
        assert is_remote_uri("C:\\Users\\vault") is False

    def test_file_uri_is_not_remote(self):
        assert is_remote_uri("file:///tmp/vault") is False


class TestResolvePath:
    def test_local_path_returns_path(self):
        result = resolve_path("/tmp/vault")
        assert isinstance(result, Path)
        assert str(result).replace("\\", "/") == "/tmp/vault"

    def test_remote_uri_without_fsspec_raises(self, monkeypatch):
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fsspec":
                raise ImportError("no fsspec")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="fsspec"):
            resolve_path("s3://bucket/vault")


class TestRasterioOpenPath:
    def test_s3_to_vsis3(self):
        assert rasterio_open_path("s3://bucket/key") == "/vsis3/bucket/key"

    def test_gs_to_vsigs(self):
        assert rasterio_open_path("gs://bucket/key") == "/vsigs/bucket/key"

    def test_az_to_vsiaz(self):
        assert rasterio_open_path("az://container/blob") == "/vsiaz/container/blob"

    def test_abfs_to_vsiaz(self):
        assert rasterio_open_path("abfs://container/blob") == "/vsiaz/container/blob"

    def test_local_path_unchanged(self):
        assert rasterio_open_path("/tmp/file.tif") == "/tmp/file.tif"

    def test_windows_path_unchanged(self):
        assert rasterio_open_path("C:\\data\\file.tif") == "C:\\data\\file.tif"
