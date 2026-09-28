"""Deterministic project signal extraction."""

from github2cv_ml.signals.extractor import (
    dependency_manifest_paths,
    extract_repository_signals,
)

__all__ = ["dependency_manifest_paths", "extract_repository_signals"]
