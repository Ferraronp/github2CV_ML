"""CLI demo tests: cached profile JSON/ZIP, Markdown and live-LLM boundary."""

import json
from datetime import datetime, timezone
from zipfile import ZipFile

import pytest

from github2cv_ml.cv_demo import (
    MAX_PROFILE_BYTES,
    generate_resume_artifacts,
    load_profile,
    render_resume_preview,
)
from github2cv_ml.domain.profile import (
    CandidateClaim, CandidateIdentity, CandidateProfile, ClaimKind,
)
from github2cv_ml.domain.resume import ResumeDocument

NOW = datetime(2026, 11, 10, tzinfo=timezone.utc)


def _profile():
    return CandidateProfile(
        candidate=CandidateIdentity(
            github_login="developer", display_name="Имя Фамилия",
        ),
        claims=[
            CandidateClaim(
                id="s1", kind=ClaimKind.SKILL, text="Python",
                evidence_ids=["e1"],
            ),
            CandidateClaim(
                id="s2", kind=ClaimKind.SKILL, text="FastAPI",
                evidence_ids=["e2"],
            ),
            CandidateClaim(
                id="p1", kind=ClaimKind.PROJECT,
                text="example/api: REST API реализовано на FastAPI. "
                     "Сервис работает с PostgreSQL.",
                evidence_ids=["e2", "e3"],
            ),
            CandidateClaim(
                id="p2", kind=ClaimKind.PROJECT,
                text="example/worker: Выполняет фоновые задачи с очередью.",
                evidence_ids=["e4"],
            ),
            CandidateClaim(
                id="a1", kind=ClaimKind.ACHIEVEMENT,
                text="Есть автоматизированные тесты.", evidence_ids=["e5"],
            ),
        ],
        generated_at=NOW,
    )


class FakeModel:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer or {
            "headline": {"text": "API и фоновые задачи", "claim_ids": ["p1", "p2"]},
            "summary": {
                "text": "В репозиториях представлены API и фоновые задачи.",
                "claim_ids": ["p1", "p2"],
            },
        }

    def complete(self, messages, *, tools=None, json_mode=False):
        assert json_mode is True
        assert tools is None
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        return {"content": json.dumps(self.answer, ensure_ascii=False)}


def test_cli_generates_resume_from_cached_json_and_zip(tmp_path):
    profile = _profile()
    path = tmp_path / "candidate_profile.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    archive = tmp_path / "colab_results.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("profile/candidate_profile.json", profile.model_dump_json())

    assert load_profile(path) == profile
    assert load_profile(archive) == profile

    model = FakeModel()
    destination = tmp_path / "cv_results"
    actual = generate_resume_artifacts(
        archive, model=model, output_dir=destination,
    )
    assert isinstance(actual, ResumeDocument)
    assert len(model.calls) == 1
    assert model.calls[0]["task"] == "cv_overview"

    stored = ResumeDocument.model_validate_json(
        (destination / "resume.json").read_text("utf-8")
    )
    assert stored == actual
    assert actual.candidate_name == "Имя Фамилия"
    assert [section.title for section in actual.sections] == [
        "Ключевые навыки", "Проекты", "Задачи и достижения",
    ]
    trace = json.loads((destination / "cv_writer_trace.json").read_text("utf-8"))
    assert trace[0]["stage"] == "cv_overview"
    assert trace[0]["duration_seconds"] >= 0
    assert trace[0]["input_claim_ids"] == ["s1", "s2", "p1", "p2", "a1"]
    assert trace[0]["llm_proposed_overview"]["summary"]["claim_ids"] == ["p1", "p2"]
    assert "observations" not in json.dumps(trace)
    preview = (destination / "resume_preview.md").read_text("utf-8")
    assert "## О себе" in preview
    assert "Python, FastAPI" in preview
    assert "### example/api" in preview
    assert "### example/worker" in preview
    assert "Сервис работает с PostgreSQL" in preview
    assert "- Есть автоматизированные тесты" in preview


def test_render_preview_does_not_turn_untrusted_markdown_into_links():
    # Rendering is kept separate from file and model paths.
    from github2cv_ml.domain.resume import ResumeItem, ResumeSection

    example = ResumeDocument(
        candidate_name="Test [external](https://evil.example)",
        summary=ResumeItem(
            text="Репозиторий с [ссылкой](https://evil.example)",
            claim_ids=["s1"],
        ),
        sections=[ResumeSection(
            title="Ключевые навыки",
            items=[ResumeItem(text="C++", claim_ids=["s1"])],
        )],
        generated_at=NOW,
    )
    preview = render_resume_preview(example)
    assert r"\[external\]\(https://evil" in preview
    assert r"\[ссылкой\]\(https://evil" in preview


def test_zip_supports_root_profile_path_and_rejects_invalid_inputs(tmp_path):
    p = _profile()
    archive = tmp_path / "plain.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("candidate_profile.json", p.model_dump_json())
    assert load_profile(archive) == p

    missing = tmp_path / "missing.zip"
    with ZipFile(missing, "w") as z:
        z.writestr("analysis.json", "{}")
    with pytest.raises(ValueError, match="lacks candidate_profile"):
        load_profile(missing)

    malformed = tmp_path / "malformed.zip"
    malformed.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Invalid ZIP"):
        load_profile(malformed)

    huge = tmp_path / "huge.json"
    huge.write_bytes(b"x" * (MAX_PROFILE_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        load_profile(huge)
    zipped_huge = tmp_path / "huge.zip"
    with ZipFile(zipped_huge, "w") as z:
        z.writestr("candidate_profile.json", b"x" * (MAX_PROFILE_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        load_profile(zipped_huge)


def test_model_failure_does_not_write_partial_resume_artifacts(tmp_path):
    source = tmp_path / "profile.json"
    source.write_text(_profile().model_dump_json(), encoding="utf-8")
    result_dir = tmp_path / "result"
    with pytest.raises(ValueError, match="Unknown CandidateProfile claim id"):
        generate_resume_artifacts(
            source, model=FakeModel({
                "summary": {"text": "Неизвестный факт.", "claim_ids": ["invented"]}
            }), output_dir=result_dir,
        )
    assert not result_dir.exists()


def test_empty_profile_can_be_previewed_without_model_call(tmp_path):
    empty = CandidateProfile(
        candidate=CandidateIdentity(github_login="github-only"),
        claims=[], generated_at=NOW,
    )
    source = tmp_path / "profile.json"
    source.write_text(empty.model_dump_json(), encoding="utf-8")
    model = FakeModel()
    result = generate_resume_artifacts(
        source, model=model, output_dir=tmp_path / "output",
    )
    assert result.sections == []
    assert result.summary is None
    assert model.calls == []
    assert "Недостаточно подтверждённых данных" in (
        tmp_path / "output" / "resume_preview.md"
    ).read_text("utf-8")
    assert json.loads(
        (tmp_path / "output" / "cv_writer_trace.json").read_text("utf-8")
    ) == []
