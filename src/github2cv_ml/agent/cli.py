"""Run the bounded Repo Analyzer using a local llama-server."""

import argparse
import os
from pathlib import Path

from github2cv_ml.agent.runner import LocalOpenAIModel, RepoAnalyzer
from github2cv_ml.agent.source_tools import RepositoryBrowser
from github2cv_ml.collectors.github_client import GitHubApiClient
from github2cv_ml.collectors.github_collector import (
    GitHubRepositoryCollector,
    parse_repository_reference,
)


def render_report(result) -> str:
    lines = [
        f"# {result.repository.owner}/{result.repository.name}",
        "",
        f"Commit: `{result.commit_sha}`",
        f"Status: `{result.status}`",
        f"Evidence verification: `{result.verification_level}`",
        "",
        "## Project",
        "",
        result.project_summary,
        "",
        f"Project type (provisional): {result.project_type or 'Not established'}",
        "",
        "## Observations",
        "",
    ]
    if not result.evidence:
        lines.extend(["No validated source locators were extracted.", ""])
    for evidence in result.evidence:
        source = evidence.source
        claim = evidence.observation.replace("\n", " ").strip()
        lines.extend([
            f"- **{claim}**",
            f"  - {source.path}:{source.line_start}-{source.line_end} "
            f"([view source]({source.url}))",
        ])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a frozen GitHub repo with a local LLM")
    parser.add_argument("repository", help="GitHub owner/repo or https://github.com/owner/repo")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen-coder")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-reads", type=int, default=6)
    parser.add_argument("--allow-remote-model", action="store_true",
                        help="Explicitly allow sending source code outside the local machine")
    options = parser.parse_args(argv)
    owner, name = parse_repository_reference(options.repository)
    client = GitHubApiClient(token=os.environ.get("GITHUB_TOKEN"))
    metadata = client.get_repository(owner, name)
    branch = metadata.get("default_branch")
    if not isinstance(branch, str) or not branch:
        parser.error("GitHub did not provide the default branch")
    sha = client.get_branch_head_sha(owner, name, branch)
    snapshot = GitHubRepositoryCollector(client=client).collect(
        f"{owner}/{name}", commit_sha=sha
    )
    browser = RepositoryBrowser(client=client, snapshot=snapshot, commit_sha=sha)
    model = LocalOpenAIModel(
        base_url=options.base_url,
        model=options.model,
        api_key=os.environ.get("LLM_API_KEY", "not-needed"),
        allow_remote=options.allow_remote_model,
    )
    analysis = RepoAnalyzer(
        model=model,
        browser=browser,
        max_steps=options.max_steps,
        max_read_calls=options.max_reads,
    ).run()
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "analysis.json").write_text(
        analysis.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (options.output / "report.md").write_text(render_report(analysis), encoding="utf-8")
    print(f"Analyzed: {analysis.repository.url} @ {analysis.commit_sha}")
    print(f"Status: {analysis.status}; evidence: {len(analysis.evidence)}")
    print(f"Results: {options.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
