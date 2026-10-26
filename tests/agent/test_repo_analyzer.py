"""Tests for bounded local Repo Analyzer orchestration (no GPU or network)."""

import json

import pytest

from github2cv_ml.agent.runner import LocalOpenAIModel, RepoAnalyzer
from github2cv_ml.agent.source_tools import RepositoryBrowser, allowed_source_path
from github2cv_ml.domain.repository import (
    RepoSnapshot,
    RepoTreeEntry,
    RepoTreeEntryKind,
)

SHA = "a" * 40
CODE = "\n".join(
    ["from typing import Any", "def dense_search(query: str) -> Any:",
     "    return retrieve(query)", "DISTINCTIVE_SOURCE_TEXT_NOT_FOR_EXPLORER"]
)


class FakeGitHub:
    def __init__(self):
        self.refs = []

    def get_file(self, owner, name, path, ref):
        self.refs.append((path, ref))
        return CODE if path == "src/search.py" else None


class FakeModel:
    def __init__(self):
        self.calls = []
        self.tools = iter([
            ("list_files", {"path": "src"}),
            ("read_file", {"path": "src/search.py", "start_line": 1, "end_line": 4}),
            ("finish_analysis", {}),
        ])

    def complete(self, messages, *, tools=None, json_mode=False):
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        system = messages[0]["content"]
        if "Write a short project summary" in system:
            return {"content": json.dumps({
                "project_summary": "A demonstration source search project.",
                "project_type": "Library",
            })}
        if json_mode:
            return {"content": json.dumps({
                "file_summary": "Defines a dense-search function.",
                "observations": [
                    {
                        "claim": "The module delegates dense search to retrieve().",
                        "category": "architecture", "source_path": "src/search.py",
                        "line_start": 2, "line_end": 3,
                    },
                    {
                        "claim": "This path is invented and should be rejected.",
                        "category": "technology", "source_path": "src/missing.py",
                        "line_start": 2, "line_end": 3,
                    },
                ],
            })}
        name, arguments = next(self.tools)
        return {"tool_calls": [{"type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments),
        }}]}


def make_snapshot():
    return RepoSnapshot.model_validate({
        "repository": {
            "owner": "example", "name": "repo",
            "url": "https://github.com/example/repo", "default_branch": "main",
        },
        "captured_at": "2026-10-01T00:00:00Z",
        "file_tree": [
            {"path": "src/search.py", "kind": "file", "size": len(CODE)},
            {"path": ".env", "kind": "file", "size": 10},
            {"path": "src/private.key", "kind": "file", "size": 123},
        ],
    })


def test_explore_then_extract_in_isolated_context():
    github = FakeGitHub()
    browser = RepositoryBrowser(github, make_snapshot(), SHA)
    model = FakeModel()
    result = RepoAnalyzer(model, browser).run()

    assert result.status == "completed"
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.observation.startswith("The module delegates")
    assert evidence.source.commit_sha == SHA
    assert evidence.source.line_start == 2
    assert evidence.source.line_end == 3
    assert f"/blob/{SHA}/src/search.py#L2-L3" in evidence.source.url
    assert github.refs == [("src/search.py", SHA)]
    assert result.file_summaries[0]["rejected"] == 1
    assert result.verification_level == "locator_only"

    explorer_calls = [c for c in model.calls if c["tools"] is not None]
    extractor_calls = [
        c for c in model.calls
        if c["json_mode"] and "Extract up to five" in c["messages"][0]["content"]
    ]
    assert len(explorer_calls) == 3
    assert len(extractor_calls) == 1
    assert "DISTINCTIVE_SOURCE_TEXT_NOT_FOR_EXPLORER" in json.dumps(extractor_calls)
    assert "DISTINCTIVE_SOURCE_TEXT_NOT_FOR_EXPLORER" not in json.dumps(explorer_calls)


def test_browser_enforces_safe_paths_ranges_and_fetch_budget():
    github = FakeGitHub()
    browser = RepositoryBrowser(github, make_snapshot(), SHA, max_unique_files=1)
    assert browser.list_files()["total"] == 1
    assert browser.read_file("src/search.py", 2, 3)["numbered_text"].startswith("2: def")
    with pytest.raises(ValueError):
        browser.read_file(".env", 1, 2)
    with pytest.raises(ValueError):
        browser.read_file("src/search.py", 2, 200)
    with pytest.raises(ValueError):
        browser.read_file("src/search.py", 7, 8)
    assert not allowed_source_path("../outside.py")
    assert not allowed_source_path("src/credentials.json")
    assert browser.search_code("dense_search")["matches"][0]["line"] == 2
    assert github.refs == [("src/search.py", SHA)]


def test_model_rejects_remote_endpoint_by_default():
    with pytest.raises(ValueError, match="Remote model"):
        LocalOpenAIModel(base_url="https://llm.example.com/v1")
    model = LocalOpenAIModel(base_url="http://127.0.0.1:8080/v1")
    assert model.model == "qwen-coder"


def test_collector_can_pin_tree_and_manifest_reads():
    from github2cv_ml.collectors.github_collector import GitHubRepositoryCollector

    class StubClient:
        def __init__(self):
            self.refs = []

        def get_repository(self, owner, name):
            return {"default_branch": "main", "owner": {"login": owner}, "name": name}

        def get_languages(self, owner, name):
            return {"Python": 12}

        def get_readme(self, owner, name):
            return None

        def get_tree(self, owner, name, ref):
            self.refs.append(("tree", ref))
            return {"tree": [
                {"path": "requirements.txt", "type": "blob", "size": 6},
            ], "truncated": False}

        def get_file(self, owner, name, path, ref):
            self.refs.append(("file", ref))
            return "pytest"

    client = StubClient()
    snap = GitHubRepositoryCollector(client=client).collect("example/repo", commit_sha=SHA)
    assert snap.signals.dependencies[0].name == "pytest"
    assert client.refs == [("tree", SHA), ("file", SHA)]
