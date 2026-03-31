"""Tests for the Vault orchestrator."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    GridDefinition,
    Variable,
    VaultConfig,
)
from voxelvault.vault import Vault


def _make_cube(name: str = "test_cube", width: int = 64, height: int = 64) -> CubeDescriptor:
    """Helper to build a small CubeDescriptor for tests."""
    grid = GridDefinition(
        width=width,
        height=height,
        epsg=4326,
        transform=(0.01, 0.0, -180.0, 0.0, -0.01, 90.0),
    )
    bands = [
        BandDefinition(
            band_index=1,
            variable=Variable(name="red", unit="reflectance", dtype="uint16"),
        ),
        BandDefinition(
            band_index=2,
            variable=Variable(name="green", unit="reflectance", dtype="uint16"),
        ),
        BandDefinition(
            band_index=3,
            variable=Variable(name="blue", unit="reflectance", dtype="uint16"),
        ),
    ]
    return CubeDescriptor(
        name=name,
        bands=bands,
        grid=grid,
        temporal_resolution=timedelta(days=1),
        description="Test RGB cube",
    )


# ------------------------------------------------------------------
# T5.5: Vault lifecycle tests
# ------------------------------------------------------------------


class TestVaultCreate:
    """Tests for Vault.create()."""

    def test_create_directory_structure(self, tmp_path):
        """Vault.create() creates v2.db, vault.json, data/, and index/ dirs."""
        vault_dir = tmp_path / "vault"
        vault = Vault.create(vault_dir)
        vault.close()

        assert (vault_dir / "v2.db").exists()
        assert (vault_dir / "vault.json").exists()
        assert (vault_dir / "data").is_dir()
        assert (vault_dir / "index").is_dir()

    def test_create_writes_config(self, tmp_path):
        """vault.json contains the serialized VaultConfig."""
        vault_dir = tmp_path / "vault"
        config = VaultConfig(compression="lzw", tile_size=512)
        vault = Vault.create(vault_dir, config=config)
        vault.close()

        raw = json.loads((vault_dir / "vault.json").read_text())
        assert raw["compression"] == "lzw"
        assert raw["tile_size"] == 512

    def test_create_existing_raises(self, tmp_path):
        """Creating a vault where v2.db already exists raises FileExistsError."""
        vault_dir = tmp_path / "vault"
        vault = Vault.create(vault_dir)
        vault.close()

        with pytest.raises(FileExistsError):
            Vault.create(vault_dir)

    def test_create_default_config(self, tmp_path):
        """Vault.create() without config uses VaultConfig defaults."""
        vault_dir = tmp_path / "vault"
        vault = Vault.create(vault_dir)
        assert vault.config == VaultConfig()
        vault.close()


class TestVaultOpen:
    """Tests for Vault.open()."""

    def test_open_existing(self, tmp_path):
        """Vault.open() on a created vault succeeds and loads config."""
        vault_dir = tmp_path / "vault"
        config = VaultConfig(compression="zstd")
        v = Vault.create(vault_dir, config=config)
        v.close()

        v2 = Vault.open(vault_dir)
        assert v2.config.compression == "zstd"
        v2.close()

    def test_open_nonexistent_raises(self, tmp_path):
        """Vault.open() on a path without v2.db raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Vault.open(tmp_path / "no_such_vault")

    def test_open_preserves_cubes(self, tmp_path):
        """Cubes registered before close are visible after re-open."""
        vault_dir = tmp_path / "vault"
        cube = _make_cube()

        v = Vault.create(vault_dir)
        v.register_cube(cube)
        v.close()

        v2 = Vault.open(vault_dir)
        assert v2.list_cubes() == [cube.name]
        v2.close()


class TestVaultContextManager:
    """Tests for the context manager protocol."""

    def test_context_manager(self, tmp_path):
        """Vault supports with-statement usage."""
        vault_dir = tmp_path / "vault"
        with Vault.create(vault_dir) as v:
            assert v.path == vault_dir

    def test_context_manager_closes(self, tmp_path):
        """After exiting context, the vault's schema engine is closed."""
        vault_dir = tmp_path / "vault"
        with Vault.create(vault_dir) as v:
            v.register_cube(_make_cube())
        # Re-opening should work — proves it was closed properly
        with Vault.open(vault_dir) as v2:
            assert v2.list_cubes() == ["test_cube"]


class TestVaultCubeManagement:
    """Tests for register_cube, get_cube, list_cubes."""

    def test_register_get_roundtrip(self, tmp_path):
        """register_cube → get_cube returns the same descriptor."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            got = v.get_cube(cube.name)
            assert got is not None
            assert got.name == cube.name
            assert got.band_count == 3

    def test_get_cube_not_found(self, tmp_path):
        """get_cube returns None for an unregistered name."""
        with Vault.create(tmp_path / "vault") as v:
            assert v.get_cube("nonexistent") is None

    def test_list_cubes(self, tmp_path):
        """list_cubes returns all registered cube names."""
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(_make_cube("alpha"))
            v.register_cube(_make_cube("beta"))
            names = v.list_cubes()
            assert sorted(names) == ["alpha", "beta"]

    def test_register_duplicate_raises(self, tmp_path):
        """Registering a cube with the same name raises ValueError."""
        cube = _make_cube()
        with Vault.create(tmp_path / "vault") as v:
            v.register_cube(cube)
            with pytest.raises(ValueError):
                v.register_cube(cube)

    def test_file_count_empty(self, tmp_path):
        """file_count is 0 for a fresh vault."""
        with Vault.create(tmp_path / "vault") as v:
            assert v.file_count == 0

    def test_path_property(self, tmp_path):
        """path property returns the vault root."""
        vault_dir = tmp_path / "vault"
        with Vault.create(vault_dir) as v:
            assert v.path == vault_dir
