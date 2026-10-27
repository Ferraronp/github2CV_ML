"""Deterministic tests for exploratory planning, pagination and coverage."""

import json

import pytest

from github2cv_ml.agent.cli import render_report
from github2cv_ml.agent.planning import build_exploration_plan
from github2cv_ml.agent.runner import RepoAnalyzer, _range_already_analyzed
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


def test_fully_covered_plan_stops_with_distinct_status_after_duplicate():
    repeated = {"path": "src/retrieval.py", "start_line": 1, "end_line": 2}
    model = SequenceModel([
        ("read_file", repeated),
        ("read_file", repeated),
    ])
    result = RepoAnalyzer(
        model, RepositoryBrowser(FakeClient(), snapshot(["src/retrieval.py"]), SHA),
        max_steps=100, max_read_calls=10,
    ).run()
    assert result.status == "plan_covered"
    assert len(model.explorer_states) == 2
    assert len(result.tool_trace) == 2
    assert len(result.evidence) == 1
    assert result.coverage["planned_coverage_ratio"] == 1.0
    assert "Whole-repository coverage is unknown" in " ".join(result.warnings)


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
    assert result.status == "plan_covered"
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


def test_migrations_are_not_treated_as_runtime_pipeline():
    paths = [
        "migrations/versions/d92f_github_report_pipeline.py",
        "src/auth/service.py", "src/projects/repo.py",
        "src/common/db/session.py", "src/main.py",
    ]
    chosen = build_exploration_plan(paths, limit=4)
    assert len(chosen) == 4
    assert all(not item["path"].startswith("migrations/") for item in chosen)
    assert {item["candidate_role"] for item in chosen} == {
        "service", "data_access", "infrastructure", "entrypoint",
    }
    expanded = build_exploration_plan(paths, limit=5)
    migration = expanded[-1]
    assert migration["candidate_role"] == "migration"


def test_deep_recovery_reads_unvisited_later_lines_in_planned_module():
    path = "src/projects/service.py"
    content = "dense_search(query)\n" * 250
    initial = {"path": path, "start_line": 1, "end_line": 120}
    model = SequenceModel([
        ("read_file", initial),
        ("read_file", initial),
        ("finish_analysis", {}),
    ])
    browser = RepositoryBrowser(
        FakeClient({path: content}), snapshot([path]), SHA
    )
    result = RepoAnalyzer(
        model, browser, max_steps=10, max_read_calls=3
    ).run()
    assert result.status == "completed"
    assert len(result.file_summaries) == 2
    assert result.file_summaries[0]["total_lines"] == 250
    assert result.file_summaries[0]["line_start"] == 1
    assert result.file_summaries[0]["line_end"] == 120
    assert result.file_summaries[1]["line_start"] == 121
    assert result.file_summaries[1]["line_end"] == 240
    assert result.tool_trace[2]["origin"] == "automatic_recovery"
    assert len(model.explorer_states) == 3
    assert len(browser._attempted) == 1


def test_unvisited_planned_file_still_reports_stalled_when_unreadable():
    paths = ["src/pipeline.py", "src/retrieval.py"]
    model = SequenceModel([
        ("read_file", {"path": "src/pipeline.py", "start_line": 1, "end_line": 2}),
        ("read_file", {"path": "src/pipeline.py", "start_line": 1, "end_line": 2}),
    ])
    browser = RepositoryBrowser(
        FakeClient({"src/retrieval.py": "broken\x00binary"}), snapshot(paths), SHA
    )
    result = RepoAnalyzer(model, browser, max_steps=10, max_read_calls=4).run()
    assert result.status == "stalled"
    assert result.coverage["planned_coverage_ratio"] == 0.5
    assert result.coverage["unvisited_planned_files"] == ["src/retrieval.py"]


def test_smoke_trace_with_all_planned_reads_and_repeated_listing_is_not_stalled():
    # Real Qwen3.5 smoke: seven successful reads, one list_files, then two
    # repeated identical listings before the controller recovered.
    paths = [
        "src/pipeline.py", "src/retrieval.py", "src/generation.py",
        "src/auth/service.py", "src/projects/repo.py",
        "src/common/security.py", "src/main.py",
    ]
    listing = {"path": "src", "limit": 50}
    model = SequenceModel([
        *[("read_file", {"path": path, "start_line": 1, "end_line": 2})
          for path in paths],
        ("list_files", listing),
        ("list_files", listing),
        ("list_files", listing),
    ])
    result = RepoAnalyzer(
        model, RepositoryBrowser(FakeClient(), snapshot(paths), SHA),
        max_steps=12, max_read_calls=8,
    ).run()
    assert result.status == "plan_covered"
    assert result.coverage["planned_coverage_ratio"] == 1.0
    assert len(result.file_summaries) == 7
    assert len(model.explorer_states) == 10
    assert len(result.tool_trace) == 10
    assert sum(x["status"] == "error" for x in result.tool_trace) == 2
    assert "whole-repository coverage is unknown" in " ".join(
        result.warnings
    ).lower()


@pytest.mark.parametrize(
    ("previous", "requested", "covered"),
    [
        ([(1, 60), (61, 120)], (1, 120), True),
        ([(61, 120), (1, 60)], (1, 120), True),
        ([(1, 50), (40, 90)], (1, 90), True),
        ([(1, 60), (62, 120)], (1, 120), False),
        ([(1, 60), (61, 120)], (1, 121), False),
        ([(1, 60), (61, 90)], (1, 90), True),
        ([], (1, 120), False),
    ],
)
def test_analyzed_range_union_respects_adjacency_overlap_and_gaps(
    previous, requested, covered
):
    assert _range_already_analyzed(*requested, previous) is covered


class CountingSequenceModel(SequenceModel):
    def __init__(self, actions):
        super().__init__(actions)
        self.extract_calls = 0

    def complete(self, messages, *, tools=None, json_mode=False):
        if tools is None and "Extract up to five" in messages[0]["content"]:
            self.extract_calls += 1
        return super().complete(messages, tools=tools, json_mode=json_mode)


def test_union_covered_read_is_rejected_without_extractor_or_read_budget():
    pipeline = "src/pipeline.py"
    retrieval = "src/retrieval.py"
    model = CountingSequenceModel([
        ("read_file", {"path": pipeline, "start_line": 1, "end_line": 60}),
        ("read_file", {"path": pipeline, "start_line": 61, "end_line": 120}),
        ("read_file", {"path": pipeline, "start_line": 1, "end_line": 120}),
        ("finish_analysis", {}),
    ])
    client = FakeClient({pipeline: "dense_search(query)\n" * 120})
    result = RepoAnalyzer(
        model, RepositoryBrowser(client, snapshot([pipeline, retrieval]), SHA),
        max_steps=10, max_read_calls=4,
    ).run()

    assert result.status == "completed"
    assert model.extract_calls == 3  # 2 original fragments + recovery on retrieval
    assert [(entry["path"], entry["line_start"], entry["line_end"])
            for entry in result.file_summaries] == [
        (pipeline, 1, 60), (pipeline, 61, 120), (retrieval, 1, 2),
    ]
    assert [entry["status"] for entry in result.tool_trace] == [
        "ok", "ok", "error", "ok", "ok",
    ]
    assert "Source lines already inspected" in result.tool_trace[2]["error"]
    assert result.tool_trace[3]["origin"] == "automatic_recovery"
    assert result.tool_trace[3]["arguments"]["path"] == retrieval
    assert len(client.refs) == 2  # No second fetch of cached pipeline


def test_union_covered_request_with_eof_clamp_is_rejected():
    path = "src/pipeline.py"
    model = CountingSequenceModel([
        ("read_file", {"path": path, "start_line": 1, "end_line": 60}),
        ("read_file", {"path": path, "start_line": 41, "end_line": 120}),
        ("read_file", {"path": path, "start_line": 1, "end_line": 120}),
    ])
    result = RepoAnalyzer(
        model, RepositoryBrowser(
            FakeClient({path: "dense_search(query)\n" * 90}), snapshot([path]), SHA
        ),
        max_steps=10, max_read_calls=4,
    ).run()

    assert result.status == "plan_covered"
    assert model.extract_calls == 2
    assert len(result.file_summaries) == 2
    assert result.file_summaries[1]["line_end"] == 90
    assert [entry["status"] for entry in result.tool_trace] == [
        "ok", "ok", "error",
    ]
    assert "Source lines already inspected" in result.tool_trace[2]["error"]


def test_union_dedup_does_not_reject_an_actual_one_line_gap():
    path = "src/pipeline.py"
    model = CountingSequenceModel([
        ("read_file", {"path": path, "start_line": 1, "end_line": 60}),
        ("read_file", {"path": path, "start_line": 62, "end_line": 120}),
        ("read_file", {"path": path, "start_line": 1, "end_line": 120}),
        ("finish_analysis", {}),
    ])
    result = RepoAnalyzer(
        model, RepositoryBrowser(
            FakeClient({path: "dense_search(query)\n" * 120}), snapshot([path]), SHA
        ),
        max_steps=10, max_read_calls=4,
    ).run()

    assert result.status == "completed"
    assert model.extract_calls == 3
    assert len(result.file_summaries) == 3
    assert all(item["status"] == "ok" for item in result.tool_trace)


def test_whole_file_experiment_reads_250_lines_in_one_extraction():
    path = "src/projects/service.py"
    content = "async def something(): pass\n" * 250
    client = FakeClient({path: content})
    browser = RepositoryBrowser(client, snapshot([path]), SHA)

    fragment = browser.read_for_extraction(
        path, start_line=1, end_line=120, prefer_whole_file=True
    )
    assert fragment["start_line"] == 1
    assert fragment["end_line"] == fragment["total_lines"] == 250
    assert fragment["numbered_text"].startswith("1: async def")
    assert "250: async def" in fragment["numbered_text"]
    assert len(fragment["numbered_text"]) <= browser.max_excerpt_chars
    assert client.refs == [(path, SHA)]


def test_whole_file_experiment_falls_back_when_source_is_too_long():
    path = "src/projects/service.py"
    content = ("long_method_with_parameters_" + "x" * 50 + "\n") * 250
    browser = RepositoryBrowser(FakeClient({path: content}), snapshot([path]), SHA)

    fragment = browser.read_for_extraction(
        path, prefer_whole_file=True
    )
    assert fragment["total_lines"] == 250
    assert fragment["end_line"] == 120
    assert len(fragment["numbered_text"]) <= browser.max_excerpt_chars
    assert "121:" not in fragment["numbered_text"]


def test_whole_file_experiment_preserves_explicit_ranges_and_bounds():
    path = "src/projects/service.py"
    browser = RepositoryBrowser(
        FakeClient({path: "some_code()\n" * 250}), snapshot([path]), SHA
    )
    fragment = browser.read_for_extraction(
        path, start_line=61, end_line=120, prefer_whole_file=True
    )
    assert (fragment["start_line"], fragment["end_line"]) == (61, 120)
    small = browser.read_for_extraction(
        path, start_line=1, end_line=20, prefer_whole_file=True
    )
    assert (small["start_line"], small["end_line"]) == (1, 20)
    with pytest.raises(ValueError, match="Invalid range"):
        browser.read_for_extraction(
            path, start_line=1, end_line=250, prefer_whole_file=True
        )


@pytest.mark.parametrize("prefer_whole", [False, True])
def test_experiment_switch_changes_extracted_range_not_retrieval_budget(
    prefer_whole,
):
    path = "src/projects/service.py"
    content = "dense_search(query)\n" * 250
    client = FakeClient({path: content})
    model = SequenceModel([
        ("read_file", {"path": path, "start_line": 1, "end_line": 120}),
        ("finish_analysis", {}),
    ])
    result = RepoAnalyzer(
        model,
        RepositoryBrowser(client, snapshot([path]), SHA),
        prefer_whole_files=prefer_whole,
        max_steps=3,
        max_read_calls=2,
    ).run()
    assert result.status == "completed"
    assert len(result.file_summaries) == 1
    fragment_summary = result.file_summaries[0]
    assert fragment_summary["line_start"] == 1
    assert fragment_summary["total_lines"] == 250
    assert fragment_summary["line_end"] == (250 if prefer_whole else 120)
    assert fragment_summary["whole_file"] is prefer_whole
    assert result.coverage["source_read_strategy"] == (
        "adaptive_whole_file" if prefer_whole else "bounded_ranges"
    )
    assert len(result.evidence) == 1
    assert client.refs == [(path, SHA)]


def test_experiment_falls_back_without_changing_read_call_budget():
    path = "src/projects/service.py"
    content = ("dense_search(query) # " + "x" * 65 + "\n") * 250
    model = SequenceModel([
        ("read_file", {"path": path}),
        ("finish_analysis", {}),
    ])
    result = RepoAnalyzer(
        model,
        RepositoryBrowser(FakeClient({path: content}), snapshot([path]), SHA),
        prefer_whole_files=True, max_steps=3, max_read_calls=2,
    ).run()
    assert result.status == "completed"
    assert result.file_summaries[0]["total_lines"] == 250
    assert result.file_summaries[0]["line_end"] == 120
    assert result.file_summaries[0]["whole_file"] is False
    assert len(result.evidence) == 1
