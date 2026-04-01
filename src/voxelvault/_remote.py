"""Remote / cloud storage helpers for VoxelVault.

Provides a thin abstraction over local and remote file paths via
``fsspec``.  When *fsspec* is installed (``pip install voxelvault[remote]``),
vault paths may use URIs such as ``s3://bucket/vault`` or
``gs://bucket/vault``.  When *fsspec* is not installed, only local
filesystem paths are supported (the default).

This module deliberately avoids hard-importing fsspec so the core
package remains functional without it.
"""

from __future__ import annotations

from pathlib import Path


def is_remote_uri(path: str | Path) -> bool:
    """Return True if *path* looks like a remote URI (e.g. ``s3://...``)."""
    s = str(path)
    return "://" in s and not s.startswith("file://")


def resolve_path(path: str | Path) -> Path:
    """Resolve a vault path, validating remote URI availability.

    For local paths, simply returns a ``pathlib.Path``.  For remote URIs,
    verifies that *fsspec* is installed and returns the path unchanged
    (as a ``Path`` object for API consistency — callers should convert
    back to ``str`` when passing to fsspec/rasterio).

    Raises:
        ImportError: If *path* is a remote URI and fsspec is not installed.
    """
    if is_remote_uri(path):
        try:
            import fsspec as _  # noqa: F401
        except ImportError:
            raise ImportError(
                f"Remote URI {str(path)!r} requires fsspec. "
                "Install it with: pip install 'voxelvault[remote]'"
            ) from None
    return Path(path) if not is_remote_uri(path) else Path(str(path))


def open_file(path: str | Path, mode: str = "rb"):
    """Open a local or remote file.

    If *path* is a remote URI and ``fsspec`` is available, uses
    ``fsspec.open`` to return a file-like object.  Otherwise, falls
    back to the stdlib ``open``.

    Returns:
        A context manager yielding a file-like object.
    """
    if is_remote_uri(path):
        import fsspec

        return fsspec.open(str(path), mode=mode)
    return open(str(path), mode=mode)


def rasterio_open_path(path: str | Path) -> str:
    """Return a string suitable for ``rasterio.open()``.

    For S3 URIs, converts ``s3://bucket/key`` into rasterio's GDAL
    virtual filesystem path ``/vsis3/bucket/key`` so that reads work
    without fsspec.  For other URIs, returns the string as-is (rasterio
    handles ``gs://`` and ``https://`` natively in recent versions).
    For local paths, returns the string representation.
    """
    s = str(path)
    if s.startswith("s3://"):
        return "/vsis3/" + s[5:]
    if s.startswith("gs://"):
        return "/vsigs/" + s[5:]
    if s.startswith("az://") or s.startswith("abfs://"):
        prefix = "az://" if s.startswith("az://") else "abfs://"
        return "/vsiaz/" + s[len(prefix):]
    return s
