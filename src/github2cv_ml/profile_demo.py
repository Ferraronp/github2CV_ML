"""Live LLM demo of Candidate Profile Builder on cached RepoAnalysis artifacts.

Reads analysis.json or smoke ZIP files. Never fetches GitHub source files.
Outputs a structured CandidateProfile, human-readable preview and a trace of
real model proposals to make the LLM's contribution observable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zipfile import BadZipFile, ZipFile

from github2cv_ml.agent.runner import LocalOpenAIModel, RepoAnalysis
from github2cv_ml.domain.profile import CandidateIdentity, CandidateProfile, ClaimKind
from github2cv_ml.profile_builder import ProfileModel, build_profile_from_analyses

MAX_ARTIFACT_BYTES = 8_000_000


def load_analysis(path: Path) -> RepoAnalysis:
    """Read one bounded JSON artifact, optionally from a Colab smoke ZIP."""
    if path.suffix.lower() == ".zip":
        try:
            with ZipFile(path) as archive:
                info = archive.getinfo("analysis.json")
                if info.file_size > MAX_ARTIFACT_BYTES:
                    raise ValueError(f"Analysis artifact too large: {path}")
                with archive.open(info) as stream:
                    raw = stream.read(MAX_ARTIFACT_BYTES + 1)
        except (BadZipFile, KeyError) as exc:
            raise ValueError(f"ZIP lacks a valid analysis.json: {path}") from exc
    else:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError(f"Analysis artifact too large: {path}")
        raw = path.read_bytes()
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Analysis artifact too large: {path}")
    return RepoAnalysis.model_validate_json(raw)


class TracingProfileModel:
    """Record actual LLM proposals, not simulated or hardcoded skill rules."""

    def __init__(self, model: ProfileModel) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        payload = json.loads(messages[-1]["content"])
        started = time.monotonic()
        answer = self.model.complete(messages, tools=tools, json_mode=json_mode)
        elapsed = round(time.monotonic() - started, 3)
        content = answer.get("content", "") if isinstance(answer, dict) else ""
        proposed_skills: list[str] = []
        paragraph = None
        try:
            data = json.loads(content)
            proposed_skills = [
                str(item["name"]) for item in data.get("skills", [])
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
            project = data.get("project")
            if isinstance(project, dict):
                paragraph = project.get("paragraph")
        except (ValueError, TypeError, AttributeError):
            pass
        self.calls.append({
            "repository": payload["repository"],
            "evidence_count": len(payload["observations"]),
            "duration_seconds": elapsed,
            "llm_proposed_skills": proposed_skills,
            "llm_proposed_project_paragraph": paragraph,
        })
        return answer


def _render_source_link(url: str | None, path: str | None) -> str:
    if not url:
        return path or "Источник"
    parsed = urlparse(url)
    # Never blindly place untrusted URLs inside a Markdown link.
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        return path or "Источник"
    label = (path or "GitHub").replace("[", "").replace("]", "")
    return f"[{label}]({url})"


def render_profile_preview(
    profile: CandidateProfile, analyses: list[RepoAnalysis]
) -> str:
    """Readable demonstration, not a finalized ResumeDocument/CV."""
    sources = {item.id: item for analysis in analyses for item in analysis.evidence}
    skills = [claim for claim in profile.claims if claim.kind == ClaimKind.SKILL]
    projects = [claim for claim in profile.claims if claim.kind == ClaimKind.PROJECT]
    lines = [
        "# Черновик GitHub-профиля кандидата",
        "",
        f"GitHub: `{profile.candidate.github_login}`",
        "",
        "## Навыки (LLM)",
        "",
        ", ".join(f"`{claim.text}`" for claim in skills)
        if skills else "Подтверждённых навыков не извлечено.",
        "",
        "## Проекты (LLM)",
        "",
    ]
    for claim in projects:
        lines.extend([f"### {claim.text.split(':', 1)[0]}", "", claim.text.split(": ", 1)[-1], ""])
        for identifier in claim.evidence_ids:
            item = sources.get(identifier)
            if item is None:
                raise ValueError(f"Missing source evidence for profile: {identifier}")
            lines.append(
                f"- {identifier}: {_render_source_link(item.source.url, item.source.path)}"
            )
        lines.append("")
    lines.extend([
        "## Ограничения",
        "",
        "- Навыки и абзацы предложены реальной LLM, а не взяты из словаря.",
        "- Evidence проверены по идентификаторам, но смысл технических утверждений не проверен.",
        "- Описания репозиториев не доказывают личное авторство кандидата.",
        "- Исходники не скачиваются повторно; использованы сохранённые результаты анализа.",
        "",
    ])
    return "\n".join(lines)


def generate_profile_artifacts(
    inputs: list[Path],
    *,
    candidate: CandidateIdentity,
    model: ProfileModel,
    output_dir: Path,
    max_skills: int = 20,
) -> CandidateProfile:
    """Run model on cached/fresh analysis artifacts and save three demo outputs."""
    if not inputs:
        raise ValueError("At least one analysis artifact is required")
    analyses = [load_analysis(path) for path in inputs]
    observed = TracingProfileModel(model)
    profile = build_profile_from_analyses(
        analyses, candidate=candidate, model=observed, max_skills=max_skills
    )
    preview = render_profile_preview(profile, analyses)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_profile.json").write_text(
        profile.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "profile_preview.md").write_text(preview, encoding="utf-8")
    (output_dir / "llm_profile_trace.json").write_text(
        json.dumps(observed.calls, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a live LLM candidate profile from existing analysis artifacts"
    )
    parser.add_argument(
        "--analysis", action="append", type=Path, required=True,
        help="Repeat for each analysis.json or Repo Analyzer smoke ZIP",
    )
    parser.add_argument("--github-login", required=True)
    parser.add_argument("--display-name")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen-coder")
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/profile_demo"))
    parser.add_argument("--allow-remote-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        identity = CandidateIdentity(
            github_login=args.github_login, display_name=args.display_name
        )
        model = LocalOpenAIModel(
            base_url=args.base_url, model=args.model,
            api_key=os.environ.get("LLM_API_KEY", "not-needed"),
            allow_remote=args.allow_remote_model,
        )
        profile = generate_profile_artifacts(
            args.analysis, candidate=identity, model=model,
            output_dir=args.output, max_skills=args.max_skills,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"LLM profile generated: {len(profile.claims)} claims from "
        f"{len(args.analysis)} cached analysis artifact(s)"
    )
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
