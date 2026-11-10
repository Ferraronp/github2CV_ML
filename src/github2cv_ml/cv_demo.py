"""CLI and human-readable preview for the evidence-backed CV Writer.

Input: CandidateProfile JSON, or a profile JSON inside a previous Colab ZIP.
Output: ResumeDocument JSON, simple Markdown preview, model-call trace.
No repository source files, PDF rendering or GitHub fetches.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from github2cv_ml.agent.runner import LocalOpenAIModel
from github2cv_ml.cv_writer import WriterModel, write_resume
from github2cv_ml.domain.profile import CandidateProfile
from github2cv_ml.domain.resume import ResumeDocument

MAX_PROFILE_BYTES = 4_000_000


def load_profile(path: Path) -> CandidateProfile:
    """Read a bounded profile from JSON or the previous smoke ZIP."""
    if path.suffix.lower() == ".zip":
        try:
            with ZipFile(path) as bundle:
                # Support both the full pipeline ZIP and a plain profile ZIP.
                target = next(
                    (name for name in ("profile/candidate_profile.json",
                                       "candidate_profile.json")
                     if name in bundle.namelist()), None
                )
                if target is None:
                    raise ValueError(f"ZIP lacks candidate_profile.json: {path}")
                info = bundle.getinfo(target)
                if info.file_size > MAX_PROFILE_BYTES:
                    raise ValueError(f"Candidate profile too large: {path}")
                with bundle.open(info) as stream:
                    raw = stream.read(MAX_PROFILE_BYTES + 1)
        except BadZipFile as exc:
            raise ValueError(f"Invalid ZIP: {path}") from exc
    else:
        if path.stat().st_size > MAX_PROFILE_BYTES:
            raise ValueError(f"Candidate profile too large: {path}")
        raw = path.read_bytes()
    if len(raw) > MAX_PROFILE_BYTES:
        raise ValueError(f"Candidate profile too large: {path}")
    return CandidateProfile.model_validate_json(raw)


class TracingWriterModel:
    """Capture what the live LLM returned; no prefilled or hardcoded answers."""

    def __init__(self, model: WriterModel) -> None:
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
        start = time.monotonic()
        answer = self.model.complete(messages, tools=tools, json_mode=json_mode)
        elapsed = round(time.monotonic() - start, 3)
        raw = answer.get("content") if isinstance(answer, dict) else None
        try:
            proposed = json.loads(raw) if isinstance(raw, str) else None
        except (TypeError, ValueError):
            proposed = None
        self.calls.append({
            "stage": "cv_overview",
            "input_claim_ids": [item["id"] for item in payload["claims"]],
            "duration_seconds": elapsed,
            "llm_proposed_overview": proposed,
        })
        return answer


def _escape_markdown(text: str) -> str:
    """Show untrusted claim text as content, not Markdown instructions/links."""
    special = "\\\\`*_{}[]()#+-.!|>"
    return "".join("\\" + char if char in special else char for char in text)


def render_resume_preview(resume: ResumeDocument) -> str:
    """One-paragraph-per-project preview; UI/PDF rendering is external."""
    lines = ["# " + _escape_markdown(resume.candidate_name), ""]
    if resume.headline is not None:
        lines += [_escape_markdown(resume.headline.text), ""]
    if resume.summary is not None:
        lines += ["## О себе", "", _escape_markdown(resume.summary.text), ""]
    for section in resume.sections:
        lines += ["## " + _escape_markdown(section.title), ""]
        if section.title == "Ключевые навыки":
            lines += [
                ", ".join(_escape_markdown(item.text) for item in section.items),
                "",
            ]
        elif section.title == "Проекты":
            for item in section.items:
                # Profile Builder stores "owner/repo: paragraph". Keep each
                # project as a single narrative paragraph, not achievement bullets.
                project_name, separator, paragraph = item.text.partition(": ")
                if separator:
                    lines += ["### " + _escape_markdown(project_name), "",
                              _escape_markdown(paragraph), ""]
                else:
                    lines += [_escape_markdown(item.text), ""]
        else:
            lines += [
                *("- " + _escape_markdown(item.text) for item in section.items),
                "",
            ]
    if not resume.sections and resume.summary is None:
        lines += ["Недостаточно подтверждённых данных для формирования резюме.", ""]
    return "\n".join(lines)


def generate_resume_artifacts(
    source: Path,
    *,
    model: WriterModel,
    output_dir: Path,
    max_skills: int = 20,
    max_projects: int | None = None,
) -> ResumeDocument:
    profile = load_profile(source)
    tracing_model = TracingWriterModel(model)
    resume = write_resume(
        profile, model=tracing_model,
        max_skills=max_skills, max_projects=max_projects,
    )
    # Atomic-at-application-level: no output files are written if the model,
    # validation, or rendering fails. Existing files are overwritten on success.
    preview = render_resume_preview(resume)
    if tracing_model.calls:
        # Compare raw LLM references with the validated/recovered references.
        call = tracing_model.calls[-1]
        raw = call.get("llm_proposed_overview")
        raw = raw if isinstance(raw, dict) else {}
        for key in ("summary", "headline"):
            proposed = raw.get(key)
            original_ids = proposed.get("claim_ids", []) if isinstance(proposed, dict) else []
            resulting = getattr(resume, key)
            if resulting is None:
                continue
            call[f"final_{key}_claim_ids"] = resulting.claim_ids
            call[f"recovered_{key}_claim_ids"] = [
                identifier for identifier in resulting.claim_ids
                if identifier not in original_ids
            ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resume.json").write_text(
        resume.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "resume_preview.md").write_text(preview, encoding="utf-8")
    (output_dir / "cv_writer_trace.json").write_text(
        json.dumps(tracing_model.calls, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resume


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an evidence-backed ResumeDocument from CandidateProfile"
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cv_demo"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen-3.5")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument("--max-projects", type=int)
    parser.add_argument("--allow-remote-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        model = LocalOpenAIModel(
            base_url=args.base_url, model=args.model,
            api_key=os.environ.get("LLM_API_KEY", "not-needed"),
            allow_remote=args.allow_remote_model,
            temperature=args.temperature,
        )
        resume = generate_resume_artifacts(
            args.profile, model=model, output_dir=args.output,
            max_skills=args.max_skills, max_projects=args.max_projects,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"ResumeDocument generated: {len(resume.sections)} sections; "
        f"output: {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
