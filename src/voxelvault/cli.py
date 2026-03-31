"""VoxelVault CLI — command-line interface for vault operations.

Provides commands to create vaults, inspect them, ingest GeoTIFF files,
and query raster cubes.
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import click


def _handle_errors(func):
    """Decorator implementing the standard CLI error-handling pattern."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileExistsError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from exc
        except FileNotFoundError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from exc
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from exc
        except SystemExit:
            raise
        except Exception as exc:
            click.echo(f"Internal error: {exc}", err=True)
            raise SystemExit(2) from exc

    return wrapper


@click.group()
def cli():
    """VoxelVault — Serverless spatiotemporal raster engine."""


@cli.command()
@click.argument("path")
@click.option("--compression", default="deflate",
              type=click.Choice(["deflate", "lzw", "zstd", "none"]))
@click.option("--tile-size", default=256, type=int)
@click.option("--epsg", default=4326, type=int)
@_handle_errors
def create(path: str, compression: str, tile_size: int, epsg: int) -> None:
    """Create a new VoxelVault at PATH."""
    from voxelvault.models import VaultConfig
    from voxelvault.vault import Vault

    config = VaultConfig(
        compression=compression,
        tile_size=tile_size,
        default_epsg=epsg,
    )
    vault = Vault.create(Path(path), config=config)
    vault.close()
    click.echo(f"Created vault at {path}")


@cli.command()
@click.argument("vault_path")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@_handle_errors
def info(vault_path: str, json_output: bool) -> None:
    """Show vault information — cubes, file counts, disk usage."""
    from voxelvault.vault import Vault

    with Vault.open(Path(vault_path)) as vault:
        cubes = vault.list_cubes()
        total_files = vault.file_count

        if json_output:
            data = {
                "path": str(vault.path),
                "config": vault.config.model_dump(mode="json"),
                "total_files": total_files,
                "cubes": [],
            }
            for name in cubes:
                cube = vault.get_cube(name)
                cube_info = {
                    "name": name,
                    "file_count": vault.cube_file_count(name),
                }
                if cube is not None:
                    cube_info["descriptor"] = cube.model_dump(mode="json")
                data["cubes"].append(cube_info)
            click.echo(json.dumps(data, indent=2))
        else:
            click.echo(f"Vault: {vault.path}")
            click.echo(f"Compression: {vault.config.compression}")
            click.echo(f"Tile size: {vault.config.tile_size}")
            click.echo(f"Default EPSG: {vault.config.default_epsg}")
            click.echo(f"Total files: {total_files}")
            if cubes:
                click.echo(f"Cubes ({len(cubes)}):")
                for name in cubes:
                    fc = vault.cube_file_count(name)
                    cube = vault.get_cube(name)
                    bands = cube.band_count if cube else "?"
                    click.echo(f"  {name}: {fc} files, {bands} bands")
            else:
                click.echo("Cubes: (none)")


@cli.command()
@click.argument("vault_path")
@click.argument("cube_name")
@click.argument("source_file")
@click.option("--start", required=True, type=click.DateTime(),
              help="Temporal start (ISO 8601).")
@click.option("--end", required=True, type=click.DateTime(),
              help="Temporal end (ISO 8601).")
@_handle_errors
def ingest(vault_path: str, cube_name: str, source_file: str,
           start, end) -> None:
    """Ingest a GeoTIFF file into a vault cube."""
    from voxelvault.models import TemporalExtent
    from voxelvault.vault import Vault

    # click.DateTime() returns naive datetimes — add UTC
    start_utc = start.replace(tzinfo=timezone.utc)
    end_utc = end.replace(tzinfo=timezone.utc)
    temporal = TemporalExtent(start=start_utc, end=end_utc)

    with Vault.open(Path(vault_path)) as vault:
        result = vault.ingest_file(cube_name, Path(source_file), temporal)

    click.echo(f"Ingested {source_file}")
    click.echo(f"  File ID: {result.file_id}")
    click.echo(f"  Size: {result.file_size_bytes} bytes")
    click.echo(f"  Checksum: {result.checksum[:16]}...")
    click.echo(f"  Elapsed: {result.elapsed_seconds:.2f}s")


@cli.command()
@click.argument("vault_path")
@click.argument("cube_name")
@click.option("--bounds", nargs=4, type=float,
              help="Spatial bounds: west south east north.")
@click.option("--start", type=click.DateTime(),
              help="Temporal start (ISO 8601).")
@click.option("--end", type=click.DateTime(),
              help="Temporal end (ISO 8601).")
@click.option("--output", "-o", type=click.Path(),
              help="Output GeoTIFF path.")
@click.option("--json", "json_output", is_flag=True,
              help="Output metadata as JSON.")
@_handle_errors
def query(vault_path: str, cube_name: str, bounds, start, end,
          output, json_output: bool) -> None:
    """Query a vault cube — find and optionally export matching data."""
    from voxelvault.vault import Vault

    spatial_bounds = tuple(bounds) if bounds else None
    temporal_range = None
    if start is not None and end is not None:
        start_utc = start.replace(tzinfo=timezone.utc)
        end_utc = end.replace(tzinfo=timezone.utc)
        temporal_range = (start_utc, end_utc)

    with Vault.open(Path(vault_path)) as vault:
        result = vault.query(cube_name, spatial_bounds=spatial_bounds,
                             temporal_range=temporal_range)

        if output:
            from rasterio.crs import CRS
            from rasterio.transform import Affine

            from voxelvault.storage import write_cog

            cube = vault.get_cube(cube_name)
            crs = CRS.from_epsg(cube.grid.epsg) if cube else CRS.from_epsg(4326)
            transform = Affine(
                (result.spatial_extent.east - result.spatial_extent.west) / result.data.shape[-1],
                0.0,
                result.spatial_extent.west,
                0.0,
                -(result.spatial_extent.north - result.spatial_extent.south) / result.data.shape[-2],
                result.spatial_extent.north,
            )
            write_data = result.data
            if write_data.ndim == 4:
                # (time, bands, h, w) → merge time into bands
                t, b, h, w = write_data.shape
                write_data = write_data.reshape(t * b, h, w)
            write_cog(
                data=write_data,
                path=Path(output),
                crs=crs,
                transform=transform,
                overwrite=True,
            )
            click.echo(f"Wrote {output}")

        if json_output:
            meta = {
                "cube": result.cube.model_dump(mode="json"),
                "spatial_extent": result.spatial_extent.model_dump(mode="json"),
                "temporal_extent": result.temporal_extent.model_dump(mode="json"),
                "shape": list(result.data.shape),
                "dtype": str(result.data.dtype),
                "source_files": [f.model_dump(mode="json") for f in result.source_files],
            }
            click.echo(json.dumps(meta, indent=2))
        elif not output:
            click.echo(f"Query result for cube '{cube_name}':")
            click.echo(f"  Shape: {result.data.shape}")
            click.echo(f"  Dtype: {result.data.dtype}")
            click.echo(f"  Source files: {len(result.source_files)}")
            click.echo(f"  Spatial: {result.spatial_extent.west:.4f}, "
                        f"{result.spatial_extent.south:.4f}, "
                        f"{result.spatial_extent.east:.4f}, "
                        f"{result.spatial_extent.north:.4f}")
            click.echo(f"  Temporal: {result.temporal_extent.start.isoformat()} → "
                        f"{result.temporal_extent.end.isoformat()}")
