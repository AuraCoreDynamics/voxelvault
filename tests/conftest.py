"""Shared fixtures for VoxelVault tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from voxelvault.models import (
    BandDefinition,
    CubeDescriptor,
    GridDefinition,
    Variable,
)


@pytest.fixture()
def sample_variable() -> Variable:
    return Variable(name="temperature", unit="K", dtype="float32")


@pytest.fixture()
def sample_grid() -> GridDefinition:
    return GridDefinition(
        width=256,
        height=256,
        epsg=4326,
        transform=(0.01, 0.0, -180.0, 0.0, -0.01, 90.0),
    )


@pytest.fixture()
def sample_cube_descriptor(sample_grid: GridDefinition) -> CubeDescriptor:
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
        name="rgb_cube",
        bands=bands,
        grid=sample_grid,
        temporal_resolution=timedelta(days=1),
        description="An RGB raster cube",
    )


@pytest.fixture()
def tmp_vault_dir(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return vault_dir
