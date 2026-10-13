"""Validate machine-readable Repo Analyzer benchmark tables in CI."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments" / "repo_analyzer" / "2026-10-12-v8"


def parse_strict_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        assert reader.fieldnames == columns, path
        rows = list(reader)
    assert rows, path
    for row in rows:
        assert None not in row, f"Extra columns in {path}: {row}"
        assert all(value is not None and value.strip() for value in row.values()), path
    return rows


def test_structural_summary_has_all_ten_configurations():
    rows = parse_strict_csv(
        REPORT / "data" / "structural_summary.csv",
        ["model_id", "strict_structural_pass_of_14", "fence_normalized_pass_of_14"],
    )
    assert len(rows) == 10
    assert len({row["model_id"] for row in rows}) == len(rows)
    assert {"gemma3-4b-q4", "phi4mini-q4"} <= {row["model_id"] for row in rows}
    for row in rows:
        strict = int(row["strict_structural_pass_of_14"])
        normalized = int(row["fence_normalized_pass_of_14"])
        assert 0 <= strict <= normalized <= 14


def test_semantic_summary_is_parseable():
    rows = parse_strict_csv(
        REPORT / "data" / "semantic_summary.csv",
        [
            "model_id", "model", "grounded_precision_pct", "grounded_recall_pct",
            "grounded_f1_pct", "gold_grounded_of_34", "claims_code",
            "unparseable_of_7", "t4_peak_gib",
        ],
    )
    assert len(rows) == 6
    assert len({row["model_id"] for row in rows}) == len(rows)
    for row in rows:
        for field in ("grounded_precision_pct", "grounded_recall_pct", "grounded_f1_pct"):
            assert 0 <= float(row[field]) <= 100
        assert 0 <= int(row["gold_grounded_of_34"]) <= 34
        assert 0 <= int(row["claims_code"]) <= 35
        assert 0 <= int(row["unparseable_of_7"]) <= 7
        assert float(row["t4_peak_gib"]) > 0


def test_vram_summary_preserves_missing_measurement():
    path = REPORT / "data" / "vram_summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        assert reader.fieldnames == [
            "model_id", "t4_gpu_process_total_peak_gib",
            "context_tokens", "full_gpu_offload",
        ]
        rows = list(reader)
    assert len(rows) == 10
    assert len({row["model_id"] for row in rows}) == len(rows)
    for row in rows:
        assert None not in row
        assert row["context_tokens"] == "8192"
        assert row["full_gpu_offload"] in {"true", "false"}
        if row["full_gpu_offload"] == "true":
            assert float(row["t4_gpu_process_total_peak_gib"]) > 0
        else:
            assert row["t4_gpu_process_total_peak_gib"] == ""
