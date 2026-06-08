"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def resource_dir() -> Path:
    """Return the path to the test resources directory."""
    return Path(__file__).parent / "resources"
