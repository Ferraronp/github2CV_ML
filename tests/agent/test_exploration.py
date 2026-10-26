"""Deterministic tests for exploratory planning, pagination and coverage."""

import json

import pytest

from github2cv_ml.agent.cli import render_report
from github2cv_ml.agent.planning import build_exploration_plan
from github2cv_ml.agent.runner import RepoAnalyzer
from github2cv_ml.agent.source_tools import RepositoryBrowser
from github2cv_ml.domain.repository import RepoSnapshot

SHA = "a" * 40


class FakeClient:
    def __init__(self, content=None):
        self.content = content or {}
        self.refs = []

    def get_file(self, owner, name, path, ref):
        self.refs.append((path, ref))
        return self.content.get(path, "dense_search(query)\nrerank(results)\n")


def snapshot(paths):
    return RepoSnapshot.model_validate({
        "repository": {
            "owner": "example", "name": "repo",
            "url": "https://github.com/example/repo", "default_branch": "main",
        },
        "captured_at": "2026-10-26T00:00:00Z",
        "file_tree": [
            {"path": path, "kind": "file", "size": 64} for path in paths
        ],
    })


def test_plan_covers_distinct_candidate_roles_before_extra_files():
    paths = [
        "src/ovp_rag/retrieval.py",
        "src/ovp_rag/pipeline.py",
        "src/ovp_rag/generation.py",
        "src/ovp_rag/main.py",
        "src/ovp_rag/database.py",
        "src/ovp_rag/helpers.py",
        "src/ovp_rag/another_helper.py",
        "evaluation/evaluate_retrieval.py",
    ]
    result = build_exploration_plan(paths, limit=7)
    assert len(result) == 7
    assert [entry["candidate_role"] for entry in result] == [
        "pipeline", "retrieval", "generation", "infrastructure",
        "entrypoint", "core", "evaluation",
    ]
    assert result[0]["path"] == "src/ovp_rag/pipeline.py"
    assert build_exploration_plan(paths, limit=7) == result
    assert build_exploration_plan([], limit=7) == []


def test_list_files_pagination_has_no_hidden_first_80_truncation():
    paths = [f"src/module_{i:03d}.py" for i in range(101)]
    browser = RepositoryBrowser(FakeClient(), snapshot(paths), SHA)
    first = browser.list_files("src", offset=0, limit=40)
    second = browser.list_files("src", offset=first["next_offset"], limit=40)
    last = browser.list_files("src", offset=second["next_offset"], limit=40)
    assert [len(page["files"]) for page in (first, second, last)] == [40, 40, 21]
    assert last["next_offset"] is None
    assert first["total"] == 101
    assert set(first["files"] + second["files"] + last["files"]) == set(paths)
    assert browser.list_files("src", offset=200)["files"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offset": -1}, {"offset": "1"}, {"offset": True},
        {"limit": 0}, {"limit": 100}, {"limit": None},
        {"path": "../src"}, {"path": None},
    ],
)
def test_list_files_rejects_untrusted_paging_args(kwargs):
    browser = RepositoryBrowser(FakeClient(), snapshot(["src/file.py"]), SHA)
    with pytest.raises(ValueError):
        browser.list_files(**kwargs)


def test_search_code_can_page_by_prefix_without_repeating_scan():
    paths = [
        "evaluation/nope.py", "src/a.py", "src/b.py",
        "src/c.py", "src/d.py",
    ]
    client = FakeClient()
    browser = RepositoryBrowser(
        client, snapshot(paths), SHA,
        max_search_files=2, max_unique_files=8, max_search_unique_files=5,
    )
    first = browser.search_code("dense", path="src", offset=0)
    second = browser.search_code("dense", path="src", offset=first["next_offset"])
    assert first["next_offset"] == 2
    assert second["next_offset"] is None
    assert first["total_candidate_files"] == 4
    assert first["scanned_files"] == second["scanned_files"] == 2
    assert len(client.refs) == 4
    assert all(path.startswith("src/") and ref == SHA for path, ref in client.refs)
    assert browser.search_code("dense", path="evaluation")["matches"][0]["path"] == (
        "evaluation/nope.py"
    )


def test_search_budget_preserves_capacity_for_read_file():
    paths = [f"src/file_{i:03d}.py" for i in range(8)]
    client = FakeClient()
    browser = RepositoryBrowser(
        client, snapshot(paths), SHA,
        max_search_files=3, max_search_unique_files=4, max_unique_files=6,
    )
    assert browser.search_code("missing", offset=0)["next_offset"] == 3
    second = browser.search_code("missing", offset=3)
    assert second["search_budget_exhausted"] is True
    assert second["next_offset"] is None
    assert browser.read_file("src/file_007.py", 1, 2)["end_line"] == 2
    assert len(client.refs) == 5


class SequenceModel:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.explorer_states = []

    def complete(self, messages, *, tools=None, json_mode=False):
        if tools is not None:
            state = json.loads(messages[-1]["content"])
            self.explorer_states.append(state)
            tool, args = next(self.actions)
            return {"tool_calls": [{"function": {
                "name": tool, "arguments": json.dumps(args),
            }}]}
        if "Extract up to five" in messages[0]["content"]:
            chunk = json.loads(messages[-1]["content"])
            return {"content": json.dumps({
                "file_summary": "Implements a simple retrieval function.",
                "observations": [{
                    "claim": "The file calls dense_search and rerank.",
                    "category": "architecture",
                    "source_path": chunk["source_path"],
                    "line_start": 1, "line_end": 2,
                }],
            })}
        return {"content": json.dumps({
            "project_summary": "Example retrieval project",
            "project_type": "service",
        })}


def test_explorer_receives_plan_and_reports_partial_coverage():
    paths = ["src/retrieval.py", "src/pipeline.py", "src/generation.py"]
    client = FakeClient()
    model = SequenceModel([
        ("list_files", {"path": "src", "limit": 2}),
        ("list_files", {"path": "src", "offset": 2, "limit": 2}),
        ("read_file", {"path": "src/retrieval.py", "start_line": 1, "end_line": 2}),
        ("finish_analysis", {}),
    ])
    result = RepoAnalyzer(model, RepositoryBrowser(client, snapshot(paths), SHA)).run()
    assert result.status == "completed"
    assert result.coverage["planned_files"] == 3
    assert result.coverage["planned_files_read"] == 1
    assert result.coverage["planned_coverage_ratio"] == 0.333
    assert len(result.coverage["unvisited_planned_files"]) == 2
    assert len(result.evidence) == 1
    assert len(model.explorer_states[0]["exploration_plan"]) == 3
    assert model.explorer_states[-1]["visited_planned_files"] == ["src/retrieval.py"]
    assert "project-wide coverage is unknown" in " ".join(result.warnings)
    report = render_report(result)
    assert "Selected candidate files inspected: 1/3" in report
    assert "Planned files not inspected:" in report
    assert "src/pipeline.py" in report


def test_empty_source_tree_yields_no_fake_coverage_percentage():
    browser = RepositoryBrowser(FakeClient(), snapshot([]), SHA)
    model = SequenceModel([("finish_analysis", {})])
    result = RepoAnalyzer(model, browser).run()
    assert result.coverage["planned_files"] == 0
    assert result.coverage["planned_coverage_ratio"] is None
    assert not result.coverage["unvisited_planned_files"]


def snapshot_with_sizes(sizes):
    return RepoSnapshot.model_validate({
        "repository": {
            "owner": "example", "name": "repo",
            "url": "https://github.com/example/repo", "default_branch": "main",
        },
        "captured_at": "2026-10-26T00:00:00Z",
        "file_tree": [
            {"path": path, "kind": "file", "size": size}
            for path, size in sizes.items()
        ],
    })


def test_planner_skips_unreadable_candidates_and_selects_replacement():
    # pipeline.py would win by path score but cannot be fetched (150 KB).
    snap = snapshot_with_sizes({
        "src/pipeline.py": 150_000,
        "src/workflow.py": 63,
        "src/retrieval.py": 64,
        "src/generation.py": None,
        "src/empty.py": 0,
    })
    client = FakeClient()
    browser = RepositoryBrowser(client, snap, SHA, max_file_bytes=80_000)
    assert set(browser.source_paths()) == {"src/workflow.py", "src/retrieval.py"}
    plan = build_exploration_plan(browser.source_paths(), limit=6)
    assert [item["path"] for item in plan] == [
        "src/workflow.py", "src/retrieval.py",
    ]
    model = SequenceModel([
        ("read_file", {"path": "src/workflow.py", "start_line": 1, "end_line": 2}),
        ("read_file", {"path": "src/retrieval.py", "start_line": 1, "end_line": 2}),
        ("finish_analysis", {}),
    ])
    result = RepoAnalyzer(model, browser).run()
    assert result.coverage["planned_files"] == 2
    assert result.coverage["planned_files_read"] == 2
    assert result.coverage["planned_coverage_ratio"] == 1.0
    assert all("pipeline.py" not in path for path, _ in client.refs)


def test_oversized_files_do_not_spend_search_quota_or_hide_later_pages():
    snap = snapshot_with_sizes({
        "src/a.py": 100_000,
        "src/b.py": 100_000,
        "src/c.py": 64,
    })
    client = FakeClient()
    browser = RepositoryBrowser(
        client, snap, SHA,
        max_file_bytes=80_000,
        max_search_files=2,
        max_search_unique_files=2,
        max_unique_files=3,
    )
    first = browser.search_code("dense", offset=0)
    assert first["considered_files"] == 2
    assert first["scanned_files"] == 0
    assert first["next_offset"] == 2
    assert not first["search_budget_exhausted"]
    assert browser._search_attempted == set()
    assert client.refs == []

    second = browser.search_code("dense", offset=first["next_offset"])
    assert second["matches"] == [{"path": "src/c.py", "line": 1}]
    assert second["scanned_files"] == 1
    assert second["next_offset"] is None
    assert not second["search_budget_exhausted"]
    assert browser._search_attempted == {"src/c.py"}
    assert client.refs == [("src/c.py", SHA)]


def test_unknown_size_and_empty_file_advance_offset_without_search_fetch():
    snap = snapshot_with_sizes({
        "src/a.py": None, "src/b.py": 0, "src/c.py": 64,
    })
    client = FakeClient()
    browser = RepositoryBrowser(
        client, snap, SHA, max_search_files=2, max_search_unique_files=1,
    )
    first = browser.search_code("dense")
    second = browser.search_code("dense", offset=first["next_offset"])
    assert first["scanned_files"] == 0
    assert second["matches"][0]["path"] == "src/c.py"
    assert browser._search_attempted == {"src/c.py"}


def test_default_exploration_plan_does_not_exceed_read_budget():
    paths = [
        "src/pipeline.py", "src/retrieval.py", "src/generation.py",
        "src/main.py", "src/database.py", "src/algorithm.py",
        "evaluation/evaluate.py",
    ]
    browser = RepositoryBrowser(FakeClient(), snapshot(paths), SHA)
    normal = RepoAnalyzer(SequenceModel([("finish_analysis", {})]), browser)
    result = normal.run()
    assert normal.max_read_calls == 6
    assert len(result.exploration_plan) == normal.max_read_calls
    assert result.coverage["planned_files"] == 6
    assert len({x["candidate_role"] for x in result.exploration_plan}) == 6

    budgeted = RepoAnalyzer(
        SequenceModel([("finish_analysis", {})]), browser, max_read_calls=2
    ).run()
    assert len(budgeted.exploration_plan) == 2
    assert budgeted.coverage["planned_files"] == 2


def test_readable_file_failed_github_fetch_still_spends_search_attempt():
    class MissingClient(FakeClient):
        def get_file(self, owner, name, path, ref):
            self.refs.append((path, ref))
            return None

    client = MissingClient()
    browser = RepositoryBrowser(
        client, snapshot(["src/a.py", "src/b.py"]), SHA,
        max_search_files=2, max_search_unique_files=1,
    )
    result = browser.search_code("dense")
    assert client.refs == [("src/a.py", SHA)]
    assert browser._search_attempted == {"src/a.py"}
    assert result["search_budget_exhausted"] is True
    assert result["next_offset"] is None


def test_read_budget_exhaustion_stops_without_repeated_tool_errors():
    paths = ["src/pipeline.py", "src/retrieval.py", "src/generation.py", "src/config.py"]
    browser = RepositoryBrowser(FakeClient(), snapshot(paths), SHA)
    model = SequenceModel([
        ("list_files", {"path": "src"}),
        ("read_file", {"path": "src/pipeline.py", "start_line": 1, "end_line": 2}),
        ("read_file", {"path": "src/retrieval.py", "start_line": 1, "end_line": 2}),
        ("read_file", {"path": "src/generation.py", "start_line": 1, "end_line": 2}),
        ("read_file", {"path": "src/config.py", "start_line": 1, "end_line": 2}),
        # Deliberately no finish_analysis: budget must end the loop directly.
    ])
    result = RepoAnalyzer(
        model, browser, max_steps=10, max_read_calls=4
    ).run()
    assert result.status == "read_budget_exhausted"
    assert result.coverage["planned_coverage_ratio"] == 1.0
    assert result.coverage["planned_files_read"] == 4
    assert len(result.file_summaries) == 4
    assert len(model.explorer_states) == 5
    assert len(result.tool_trace) == 5
    assert all(action["status"] == "ok" for action in result.tool_trace)
    assert any("Source read-call budget exhausted (4/4)" in w for w in result.warnings)


def test_last_step_exhausted_read_is_not_misreported_as_step_budget():
    browser = RepositoryBrowser(FakeClient(), snapshot(["src/retrieval.py"]), SHA)
    model = SequenceModel([
        ("read_file", {"path": "src/retrieval.py", "start_line": 1, "end_line": 2}),
    ])
    result = RepoAnalyzer(
        model, browser, max_steps=1, max_read_calls=1
    ).run()
    assert result.status == "read_budget_exhausted"
    assert len(result.file_summaries) == 1
    assert len(result.tool_trace) == 1


def test_explicit_finish_still_reports_completed_when_reads_remain():
    browser = RepositoryBrowser(FakeClient(), snapshot(["src/retrieval.py"]), SHA)
    model = SequenceModel([("finish_analysis", {})])
    result = RepoAnalyzer(
        model, browser, max_steps=5, max_read_calls=1
    ).run()
    assert result.status == "completed"


def test_backend_plan_prefers_business_logic_over_routers_and_tests():
    # Regresses the external FastAPI backend smoke: the old heuristic chose
    # main, router, settings and tests while missing auth/service and user/repo.
    paths = [
        "src/main.py",
        "src/auth/router.py", "src/auth/service.py",
        "src/user/router.py", "src/user/service.py", "src/user/repo.py",
        "src/common/security.py", "src/common/db/models.py",
        "src/common/db/session.py", "src/common/redis/session.py",
        "src/errors.py", "src/settings.py", "tests/test_auth.py",
    ]
    chosen = build_exploration_plan(paths, limit=7)
    selected = {item["path"] for item in chosen}
    assert len(chosen) == 7
    assert {
        "src/auth/service.py", "src/user/service.py", "src/user/repo.py",
        "src/common/security.py", "src/common/db/session.py", "src/main.py",
    } <= selected
    assert "src/auth/router.py" not in selected
    assert "tests/test_auth.py" not in selected
    assert "src/settings.py" not in selected
    assert build_exploration_plan(paths, limit=7) == chosen


def test_duplicate_read_immediately_recovers_to_next_planned_module():
    paths = [
        "src/main.py", "src/auth/service.py", "src/common/security.py",
    ]
    repeated = {"path": "src/main.py", "start_line": 1, "end_line": 2}
    model = SequenceModel([
        ("read_file", repeated),
        ("read_file", repeated),
        ("finish_analysis", {}),
    ])
    result = RepoAnalyzer(
        model, RepositoryBrowser(FakeClient(), snapshot(paths), SHA),
        max_steps=100, max_read_calls=3,
    ).run()
    assert result.status == "completed"
    assert len(model.explorer_states) == 3
    assert [entry["status"] for entry in result.tool_trace] == [
        "ok", "error", "ok", "ok",
    ]
    assert result.tool_trace[2]["origin"] == "automatic_recovery"
    assert result.tool_trace[2]["recovery_reason"] == "duplicate_source"
    assert result.tool_trace[2]["arguments"]["path"] == "src/auth/service.py"
    assert {item["path"] for item in result.file_summaries} == {
        "src/main.py", "src/auth/service.py",
    }
    assert len(result.evidence) == 2


def test_no_unvisited_planned_file_stops_after_one_duplicate():
    repeated = {"path": "src/retrieval.py", "start_line": 1, "end_line": 2}
    model = SequenceModel([
        ("read_file", repeated),
        ("read_file", repeated),
    ])
    result = RepoAnalyzer(
        model, RepositoryBrowser(FakeClient(), snapshot(["src/retrieval.py"]), SHA),
        max_steps=100, max_read_calls=10,
    ).run()
    assert result.status == "stalled"
    assert len(model.explorer_states) == 2
    assert len(result.tool_trace) == 2
    assert len(result.evidence) == 1
    assert "no readable unvisited planned file" in " ".join(result.warnings)


def test_different_requested_ranges_do_not_reextract_same_source_lines():
    # File has only two lines. Requested 1:120 is clamped to 1:2, so
    # a later 1:2 request should reuse evidence without extractor work.
    model = SequenceModel([
        ("read_file", {"path": "src/retrieval.py", "start_line": 1, "end_line": 120}),
        ("read_file", {"path": "src/retrieval.py", "start_line": 1, "end_line": 2}),
        ("finish_analysis", {}),
    ])
    client = FakeClient()
    result = RepoAnalyzer(
        model, RepositoryBrowser(client, snapshot(["src/retrieval.py"]), SHA),
        max_steps=20, max_read_calls=3,
    ).run()
    assert result.status == "stalled"
    assert len(model.explorer_states) == 2
    assert len(result.file_summaries) == 1
    assert len(result.evidence) == 1
    assert result.tool_trace[1]["status"] == "error"
    assert "Source lines already inspected" in result.tool_trace[1]["error"]
    assert len(client.refs) == 1


def test_perpetually_repeated_tool_call_cannot_consume_100_steps():
    paths = ["src/pipeline.py", "src/retrieval.py", "src/generation.py"]
    repeated = {"path": "src/pipeline.py", "start_line": 1, "end_line": 2}
    model = SequenceModel([("read_file", repeated)] * 100)
    result = RepoAnalyzer(
        model, RepositoryBrowser(FakeClient(), snapshot(paths), SHA),
        max_steps=100, max_read_calls=3,
    ).run()
    assert result.status == "read_budget_exhausted"
    assert len(model.explorer_states) == 3
    assert len(result.file_summaries) == 3
    assert {entry["path"] for entry in result.file_summaries} == set(paths)
    assert [entry["status"] for entry in result.tool_trace] == [
        "ok", "error", "ok", "error", "ok",
    ]


def test_successful_but_unproductive_navigation_recovers_after_three_steps():
    paths = ["src/pipeline.py", "src/retrieval.py"]
    model = SequenceModel([
        ("list_files", {"path": "src", "limit": 1}),
        ("list_files", {"path": "src", "limit": 2}),
        ("search_code", {"query": "missing", "path": "src"}),
        ("finish_analysis", {}),
    ])
    result = RepoAnalyzer(
        model, RepositoryBrowser(FakeClient(), snapshot(paths), SHA),
        max_steps=20, max_read_calls=2,
    ).run()
    assert result.status == "completed"
    assert len(model.explorer_states) == 4
    assert len(result.file_summaries) == 1
    assert result.tool_trace[3]["origin"] == "automatic_recovery"
    assert result.tool_trace[3]["recovery_reason"] == "no_source_progress"
    assert result.tool_trace[3]["arguments"]["path"] == "src/pipeline.py"


def test_recovery_shrinks_range_when_120_lines_exceed_character_limit():
    path = "src/pipeline.py"
    # Fits the file byte limit, but its first 120 lines exceed 12K chars.
    content = ("dense_search(query) " + "x" * 200 + "\n") * 90
    model = SequenceModel([
        ("list_files", {"limit": 1}),
        ("list_files", {"limit": 2}),
        ("search_code", {"query": "missing"}),
        ("finish_analysis", {}),
    ])
    browser = RepositoryBrowser(FakeClient({path: content}), snapshot([path]), SHA)
    result = RepoAnalyzer(model, browser, max_steps=20, max_read_calls=1).run()
    assert result.status == "read_budget_exhausted"
    assert len(result.file_summaries) == 1
    recovered = result.tool_trace[3]
    assert recovered["origin"] == "automatic_recovery"
    assert 1 <= recovered["arguments"]["end_line"] < 120
    assert len(result.evidence) == 1


def test_unreadable_recovery_candidate_does_not_block_next_planned_file():
    paths = ["src/pipeline.py", "src/retrieval.py"]
    model = SequenceModel([
        ("list_files", {"limit": 1}),
        ("list_files", {"limit": 2}),
        ("search_code", {"query": "missing"}),
        ("finish_analysis", {}),
    ])
    browser = RepositoryBrowser(
        FakeClient({"src/pipeline.py": "binary\x00code"}),
        snapshot(paths), SHA,
    )
    result = RepoAnalyzer(model, browser, max_steps=10, max_read_calls=2).run()
    assert result.status == "completed"
    assert [entry["path"] for entry in result.file_summaries] == ["src/retrieval.py"]
    assert "Recovery skipped src/pipeline.py" in " ".join(result.warnings)
