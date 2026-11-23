"""Persist the complete MVP vertical slice as inspectable JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from github2cv_ml.agent.cli import render_report
from github2cv_ml.cv_demo import render_resume_preview
from github2cv_ml.pipeline import PipelineResult, normalize_repository
from github2cv_ml.profile_demo import render_profile_preview


def write_pipeline_artifacts(result: PipelineResult, output_dir: Path) -> dict[str, str]:
    """Write structured outputs only into an empty destination.

    This deliberately refuses to overwrite an earlier run: a failed retry
    must not leave stale resume.json that a backend could mistake for success.
    Output files must be treated as sensitive, like repository source excerpts.
    """
    if not isinstance(result, PipelineResult):
        raise TypeError("result must be PipelineResult")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Pipeline output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def save(relative: str, content: str) -> None:
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        paths[relative] = relative

    def save_model(relative: str, obj) -> None:
        save(relative, obj.model_dump_json(indent=2) + "\n")

    for index, snapshot in enumerate(result.snapshots, start=1):
        key = normalize_repository(
            f"{snapshot.repository.owner}/{snapshot.repository.name}"
        ).replace("/", "_")
        save_model(f"snapshots/{index:02d}_{key}.json", snapshot)
    for index, analysis in enumerate(result.analyses, start=1):
        key = normalize_repository(
            f"{analysis.repository.owner}/{analysis.repository.name}"
        ).replace("/", "_")
        folder = f"analyses/{index:02d}_{key}"
        save_model(f"{folder}/analysis.json", analysis)
        save(f"{folder}/report.md", render_report(analysis))

    if result.profile is not None:
        save_model("profile/candidate_profile.json", result.profile)
        save(
            "profile/profile_preview.md",
            render_profile_preview(result.profile, result.analyses),
        )
    if result.resume is not None:
        save_model("cv/resume.json", result.resume)
        save("cv/resume_preview.md", render_resume_preview(result.resume))
    if result.profile_trace:
        save(
            "profile/llm_profile_trace.json",
            json.dumps(result.profile_trace, ensure_ascii=False, indent=2) + "\n",
        )
    if result.profile_error_responses:
        save(
            "profile/llm_profile_error.json",
            json.dumps({
                "problems": [
                    p.model_dump(mode="json") for p in result.problems
                    if p.stage == "profile"
                ],
                "responses": result.profile_error_responses,
            }, ensure_ascii=False, indent=2) + "\n",
        )
    if result.cv_trace:
        save(
            "cv/cv_writer_trace.json",
            json.dumps(result.cv_trace, ensure_ascii=False, indent=2) + "\n",
        )
    if result.correction_events:
        save(
            "model_output_corrections.json",
            json.dumps(result.correction_events, ensure_ascii=False, indent=2) + "\n",
        )

    # Flat evidence-ID index makes provenance easy to resolve in the backend
    # without repeating GitHub queries or searching nested analysis files.
    save(
        "provenance/evidence_index.json",
        json.dumps({
            identifier: evidence.model_dump(mode="json")
            for identifier, evidence in sorted(result.evidence_index.items())
        }, ensure_ascii=False, indent=2) + "\n",
    )
    # Store the exact typed result and a lightweight lookup manifest, even if
    # the workflow failed at an earlier stage.
    save_model("pipeline_result.json", result)
    manifest = {
        "schema_version": "1.0",
        "status": result.status,
        "requested_repositories": result.request.repositories,
        "problems": [p.model_dump(mode="json") for p in result.problems],
        "warnings": result.warnings,
        "has_resume": result.resume is not None,
        "evidence_count": len(result.evidence_index),
        "claim_count": len(result.profile.claims) if result.profile else 0,
        "correction_count": len(result.correction_events),
        "artifacts": sorted([*paths, "pipeline_manifest.json"]),
    }
    save(
        "pipeline_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return paths
