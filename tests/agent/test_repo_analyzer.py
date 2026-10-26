"""Tests for bounded local Repo Analyzer orchestration (no GPU or network)."""

import json

import pytest

from github2cv_ml.agent.runner import LocalOpenAIModel, RepoAnalyzer
from github2cv_ml.agent.source_tools import RepositoryBrowser, allowed_source_path
from github2cv_ml.domain.repository import RepoSnapshot, RepoTreeEntry, RepoTreeEntryKind

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


def test_read_file_rejects_oversized_prompt_fragment():
    browser = RepositoryBrowser(
        FakeGitHub(), make_snapshot(), SHA, max_excerpt_chars=12
    )
    with pytest.raises(ValueError, match="character budget"):
        browser.read_file("src/search.py", 1, 4)


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

def test_http_adapter_uses_openai_tool_protocol(monkeypatch):
    from io import BytesIO

    import github2cv_ml.agent.runner as runner

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        message = {"tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "list_files", "arguments": "{}"},
        }]}
        return BytesIO(json.dumps({"choices": [{"message": message}]}).encode())

    monkeypatch.setattr(runner, "urlopen", fake_urlopen)
    model = LocalOpenAIModel(base_url="http://127.0.0.1:8080/v1", model="qwen-coder")
    prompt = [{"role": "user", "content": "inspect"}]
    response = model.complete(prompt, tools=[{"type": "function", "function": {
        "name": "list_files", "parameters": {"type": "object", "properties": {}},
    }}])
    assert response["tool_calls"][0]["function"]["name"] == "list_files"
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["model"] == "qwen-coder"
    assert requests[0]["messages"] == prompt


@pytest.mark.parametrize(
    "blocked_path",
    [
        ".npmrc", ".pypirc", ".netrc", "pip.conf",
        "src/.npmrc", "config/.pypirc", "home/.netrc", "config/pip.conf",
        "src/.env.production", "src/credentials.py", "src/secret_settings.py",
        "src/config.yaml", "src/random.json", "src/unknown_extension",
    ],
)
def test_secret_or_config_paths_never_reach_github_client(blocked_path):
    snapshot = make_snapshot()
    snapshot.file_tree.append(
        RepoTreeEntry(path=blocked_path, kind=RepoTreeEntryKind.FILE, size=24)
    )
    github = FakeGitHub()
    browser = RepositoryBrowser(github, snapshot, SHA)
    assert not allowed_source_path(blocked_path)
    assert blocked_path not in browser.list_files()["files"]
    with pytest.raises(ValueError):
        browser.read_file(blocked_path, 1, 1)
    browser.search_code("password")
    assert all(path != blocked_path for path, _ in github.refs)


@pytest.mark.parametrize("bad_path", [None, 42, [], {}])
def test_bad_list_files_path_is_a_recoverable_value_error(bad_path):
    browser = RepositoryBrowser(FakeGitHub(), make_snapshot(), SHA)
    with pytest.raises(ValueError, match="must be a string"):
        browser.list_files(bad_path)


@pytest.mark.parametrize("bad_path", [None, 42, [], {}])
def test_bad_read_file_path_is_a_recoverable_value_error(bad_path):
    browser = RepositoryBrowser(FakeGitHub(), make_snapshot(), SHA)
    with pytest.raises(ValueError, match="must be a string"):
        browser.read_file(bad_path, 1, 1)


def test_invalid_tool_arguments_are_recorded_and_explorer_can_continue():
    model = FakeModel()
    model.tools = iter([
        ("list_files", {"path": None}),
        ("list_files", {"path": 42}),
        ("finish_analysis", {}),
    ])
    result = RepoAnalyzer(model, RepositoryBrowser(FakeGitHub(), make_snapshot(), SHA)).run()
    assert result.status == "completed"
    assert [a["status"] for a in result.tool_trace] == ["error", "error", "ok"]
    assert all("must be a string" in a["error"] for a in result.tool_trace[:2])


def test_explorer_endpoint_failure_preserves_partial_evidence():
    class FailingExplorer(FakeModel):
        def __init__(self):
            super().__init__()
            self.explorer_calls = 0

        def complete(self, messages, *, tools=None, json_mode=False):
            if tools is not None:
                self.explorer_calls += 1
                if self.explorer_calls == 3:
                    raise RuntimeError("llama-server unavailable")
            return super().complete(messages, tools=tools, json_mode=json_mode)

    result = RepoAnalyzer(
        FailingExplorer(), RepositoryBrowser(FakeGitHub(), make_snapshot(), SHA)
    ).run()
    assert result.status == "incomplete"
    assert len(result.evidence) == 1
    assert result.file_summaries[0]["accepted"] == 1
    assert result.tool_trace[-1]["status"] == "error"
    assert "model call failed" in " ".join(result.warnings).lower()


@pytest.mark.parametrize(
    "malformed",
    [
        {"tool_calls": [None]},
        {"tool_calls": ["bad"]},
        {"tool_calls": {"wrong": "shape"}},
        {"tool_calls": [{"function": "bad"}]},
        {"tool_calls": [{"function": {"name": None, "arguments": "{}"}}]},
        ["not a response object"],
    ],
)
def test_malformed_tool_calls_return_incomplete_result(malformed):
    class MalformedExplorer:
        def complete(self, messages, *, tools=None, json_mode=False):
            if tools is not None:
                return malformed
            return {"content": '{"project_summary": "Fallback", "project_type": null}'}

    result = RepoAnalyzer(
        MalformedExplorer(), RepositoryBrowser(FakeGitHub(), make_snapshot(), SHA)
    ).run()
    assert result.status == "incomplete"
    assert result.tool_trace[-1]["status"] == "error"
    assert result.warnings
    assert result.evidence == []


def test_cli_writes_incomplete_json_and_markdown_when_model_is_down(tmp_path, monkeypatch):
    import github2cv_ml.agent.cli as cli

    class StubClient:
        def __init__(self, token=None):
            self.token = token

        def get_repository(self, owner, name):
            return {"default_branch": "main"}

        def get_branch_head_sha(self, owner, name, branch):
            return SHA

    class StubCollector:
        def __init__(self, client):
            pass

        def collect(self, repository, *, commit_sha):
            return make_snapshot()

    class OfflineModel:
        def __init__(self, **kwargs):
            pass

        def complete(self, messages, *, tools=None, json_mode=False):
            raise RuntimeError("llama-server is unavailable")

    monkeypatch.setattr(cli, "GitHubApiClient", StubClient)
    monkeypatch.setattr(cli, "GitHubRepositoryCollector", StubCollector)
    monkeypatch.setattr(cli, "LocalOpenAIModel", OfflineModel)

    out = tmp_path / "incomplete-analysis"
    assert cli.main(["example/repo", "--output", str(out)]) == 0
    saved = json.loads((out / "analysis.json").read_text())
    report = (out / "report.md").read_text()
    assert saved["status"] == "incomplete"
    assert saved["tool_trace"][0]["status"] == "error"
    assert "model call failed" in " ".join(saved["warnings"]).lower()
    assert "Status: `incomplete`" in report


def test_no_tool_call_records_model_content_for_diagnostics():
    class PlainTextExplorer:
        def complete(self, messages, *, tools=None, json_mode=False):
            if tools is not None:
                return {"content": "I'll inspect the files first, but no tool call."}
            return {"content": '{"project_summary": "README only", "project_type": null}'}

    result = RepoAnalyzer(
        PlainTextExplorer(), RepositoryBrowser(FakeGitHub(), make_snapshot(), SHA)
    ).run()
    assert result.status == "model_stopped_without_tool"
    assert not result.evidence
    assert result.tool_trace[0]["error"] == "Model returned no tool_calls"
    assert "inspect the files" in result.tool_trace[0]["response_preview"]
    assert "tool_choice=required" in " ".join(result.warnings)
