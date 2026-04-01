"""Local CI-parity checks for the release pipeline.

These tests intentionally mirror the non-publish GitHub workflow gates so
release failures are caught client-side before a GitHub release is cut.
The pytest suite itself already exercises the runtime/unit/integration tests,
so these parity checks extend local validation to the remaining workflow steps
and keep the workflow contract under test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "publish.yaml"


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_publish_workflow_contains_expected_quality_gates() -> None:
    """Keep the local parity checks aligned with the publish workflow."""
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "validate:" in workflow
    assert "build-n-publish:" in workflow
    assert "needs: validate" in workflow
    assert 'python-version: "3.11"' in workflow
    assert 'python -m pip install -e ".[all,dev]"' in workflow
    assert "python -m ruff check src tests" in workflow
    assert "python -m pytest tests/ -q --tb=short" in workflow
    assert "python -m build --sdist --wheel" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow


def test_client_release_gate_runs_ruff() -> None:
    """Run the same lint command that the publish workflow executes."""
    result = _run([sys.executable, "-m", "ruff", "check", "src", "tests"], cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_client_release_gate_builds_package(tmp_path: Path) -> None:
    """Build the sdist and wheel locally, matching the publish workflow."""
    dist_dir = tmp_path / "dist"
    result = _run(
        [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(dist_dir)],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    artifacts = sorted(path.name for path in dist_dir.iterdir())
    assert any(name.endswith(".whl") for name in artifacts), artifacts
    assert any(name.endswith(".tar.gz") for name in artifacts), artifacts

    shutil.rmtree(dist_dir, ignore_errors=True)
