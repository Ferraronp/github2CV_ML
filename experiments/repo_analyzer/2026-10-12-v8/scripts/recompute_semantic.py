#!/usr/bin/env python3
"""Recalculate grounded precision/recall/F1 from preserved annotations (no API)."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    gold = read_csv(ROOT / "annotations" / "gold_facts_ai.csv")
    annotations = read_csv(ROOT / "annotations" / "claims_test_ai_annotated.csv")
    test_gold = [r for r in gold if r["split"] == "test"]
    test_ids = {r["gold_id"] for r in test_gold}
    code_cases = {r["case_id"] for r in test_gold}
    if len(test_gold) != 34 or len(code_cases) != 7:
        raise ValueError("Unexpected v8 gold subset: expected 34 facts in 7 code cases")
    models = defaultdict(list)
    for claim in annotations:
        if claim["split"] == "test" and claim["case_id"] in code_cases:
            models[claim["model"]].append(claim)
    print("model,claims,grounded_claims,covered_gold,grounded_precision,grounded_recall,grounded_f1")
    for model, claims in sorted(models.items()):
        grounded = [r for r in claims if r["claim_label"] == "supported" and r["evidence_label"] == "direct"]
        covered = {r["matched_gold_id"] for r in grounded if r["matched_gold_id"] in test_ids}
        precision = len(grounded) / len(claims) if claims else float("nan")
        recall = len(covered) / len(test_gold)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"{model},{len(claims)},{len(grounded)},{len(covered)},{precision:.5f},{recall:.5f},{f1:.5f}")

if __name__ == "__main__":
    main()
