#!/usr/bin/env python3
"""Verify the immutable Colab v8 source ZIP against the published tables."""

import argparse
import ast
import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
RUN = "20261012T160658Z"
ARCHIVE_SHA256 = "4ad3f16f080a1500d74f717a53504b15142c6881af48fdb9073168fc7c46a6a3"
ORIGINAL_HASHES = {
    f"snapshots_{RUN}.json": "1d6de61a342a16f9dd53ab2df04370ccfb271c3e62566a2b7028a3ecaeb52aff",
    f"predictions_{RUN}.json": "5e1366bc82e8aab9dd56f93305c703134e16dde4c8d8d35bd2a92e267ebd087c",
    f"metadata_{RUN}.json": "ab66f9345ec058fdb2598b728bfdb1817223109b622d3848469352c0b022894e",
    f"results_{RUN}.csv": "359893c701beb39a9bfbe69238d3b4f92116d1b6ecac3a357eb4fdc1af0a7c66",
    f"vram_summary_{RUN}.csv": "292b7f68154dda914bfe446c6b19da9ae632cb55f473da9011f44953e8e878fe",
}
CATEGORIES = ["technology", "architecture", "testing", "deployment", "project_type"]


def notebook_scorer():
    """Use the published notebook's scorer without executing Colab cells."""
    notebook_path = ROOT / "notebooks" / "colab_benchmark_v8.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            module = ast.parse("".join(cell["source"]))
        except SyntaxError:  # IPython magic is not valid Python.
            continue
        functions = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "score"
        ]
        if not functions:
            continue
        code = compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            str(notebook_path),
            "exec",
        )
        namespace = {
            "json": json,
            "OUTPUT_SCHEMA": {
                "properties": {
                    "observations": {
                        "items": {"properties": {"category": {"enum": CATEGORIES}}}
                    }
                }
            },
        }
        # The function is checked into this repository; no notebook setup cells are run.
        exec(code, namespace)
        return namespace["score"]
    raise RuntimeError("The published notebook does not contain score()")


def remove_fence(raw):
    stripped = raw.strip()
    match = re.fullmatch(r"\x60\x60\x60(?:json)?\s*\n(.*?)\n\x60\x60\x60", stripped, re.S | re.I)
    return match.group(1).strip() if match else stripped


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, strict=True))


def verify_archive(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 differs from the published benchmark")
    with ZipFile(archive) as zipped:
        for filename, expected_hash in ORIGINAL_HASHES.items():
            if hashlib.sha256(zipped.read(filename)).hexdigest() != expected_hash:
                raise ValueError(f"Incorrect SHA-256: {filename}")
        cases = {
            c["case_id"]: c
            for c in json.loads(zipped.read(f"snapshots_{RUN}.json"))
        }
        predictions = json.loads(zipped.read(f"predictions_{RUN}.json"))
        results = list(csv.DictReader(
            io.TextIOWrapper(zipped.open(f"results_{RUN}.csv"), encoding="utf-8-sig")
        ))
        original_vram = list(csv.DictReader(
            io.TextIOWrapper(zipped.open(f"vram_summary_{RUN}.csv"), encoding="utf-8-sig")
        ))
        metadata = json.loads(zipped.read(f"metadata_{RUN}.json"))

    if metadata["run_id"] != RUN or len(cases) != 14 or len(results) != 196:
        raise ValueError("Unexpected run ID, case count or result count")
    score = notebook_scorer()
    counted = defaultdict(lambda: {"strict": 0, "normalized": 0, "total": 0})
    for prediction in predictions:
        if ":" in prediction["model"]:
            continue  # Groq/OpenRouter do not belong to the local-model comparison.
        if prediction["case_id"] not in cases:
            raise ValueError("Prediction references an unknown case")
        counts = counted[prediction["model"]]
        counts["total"] += 1
        for metric, raw in (
            ("strict", prediction["raw"]),
            ("normalized", remove_fence(prediction["raw"])),
        ):
            result = score(raw, cases[prediction["case_id"]])
            if all(result[k] for k in ("json_valid", "schema_valid", "locators_valid")):
                counts[metric] += 1

    structural = load_csv(ROOT / "data" / "structural_summary.csv")
    if len(structural) != 10:
        raise ValueError("Expected 10 structural summary rows")
    for row in structural:
        model = row["model_id"]
        observed = counted[model]
        if model == "phi4mini-q4":
            failures = [r for r in results if r["model_id"] == model]
            if len(failures) != 14 or any(
                r["status"] != "model_load_or_run_failed" for r in failures
            ):
                raise ValueError("Phi-4-mini error count differs from the archive")
        elif observed["total"] != 14:
            raise ValueError(f"Expected 14 raw predictions for {model}")
        if int(row["strict_structural_pass_of_14"]) != observed["strict"]:
            raise ValueError(f"Strict structural mismatch for {model}")
        if int(row["fence_normalized_pass_of_14"]) != observed["normalized"]:
            raise ValueError(f"Normalized structural mismatch for {model}")

    source_vram = {r["model_id"]: r for r in original_vram}
    for row in load_csv(ROOT / "data" / "vram_summary.csv"):
        model = row["model_id"]
        if model == "phi4mini-q4":
            if row["t4_gpu_process_total_peak_gib"]:
                raise ValueError("Failed model must not have VRAM measurements")
            continue
        original = float(source_vram[model]["gpu_process_total_peak_gib"])
        published = float(row["t4_gpu_process_total_peak_gib"])
        if abs(original - published) > 0.0005:
            raise ValueError(f"VRAM mismatch for {model}")
    print(
        f"Verified original archive, {len(predictions)} predictions, "
        f"{len(results)} attempts and both published summaries"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Immutable benchmark ZIP")
    verify_archive(parser.parse_args().archive)


if __name__ == "__main__":
    main()
