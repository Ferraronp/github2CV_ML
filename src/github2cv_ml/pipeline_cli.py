"""One-command production-shaped smoke entrypoint (no HTTP server)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from github2cv_ml.agent.runner import LocalOpenAIModel
from github2cv_ml.collectors.github_client import GitHubApiClient
from github2cv_ml.domain.profile import CandidateIdentity
from github2cv_ml.pipeline import PipelineRequest, run_pipeline
from github2cv_ml.pipeline_artifacts import write_pipeline_artifacts
from github2cv_ml.profile_demo import load_analysis

MAX_REQUEST_BYTES = 64_000


def _read_request(path: Path) -> PipelineRequest:
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("Pipeline request JSON exceeds 64 KB")
    return PipelineRequest.model_validate_json(path.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end GitHub -> RepoEvidence -> CandidateProfile -> ResumeDocument"
    )
    parser.add_argument(
        "--request", type=Path,
        help="PipelineRequest JSON file for backend-compatible invocation",
    )
    parser.add_argument(
        "--repo", action="append", default=[],
        help="GitHub owner/repo or URL; repeat for multiple repositories",
    )
    parser.add_argument(
        "--analysis", action="append", type=Path, default=[],
        help="Existing RepoAnalysis JSON or single-repo smoke ZIP; repeat",
    )
    parser.add_argument("--github-login")
    parser.add_argument("--display-name")
    parser.add_argument("--max-steps", type=int, default=28)
    parser.add_argument("--max-reads", type=int, default=12)
    parser.add_argument("--min-planned-reads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/mvp"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="qwen-3.5")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--allow-remote-model", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.request is not None:
            if args.repo or args.github_login or args.display_name:
                raise ValueError(
                    "Choose either --request or --repo/--github-login arguments"
                )
            request = _read_request(args.request)
        else:
            if not args.github_login:
                raise ValueError("--github-login is required without --request")
            request = PipelineRequest(
                candidate=CandidateIdentity(
                    github_login=args.github_login, display_name=args.display_name
                ),
                repositories=args.repo,
                max_steps=args.max_steps,
                max_reads=args.max_reads,
                min_planned_reads=args.min_planned_reads,
            )
        cached = [load_analysis(path) for path in args.analysis]
        model = LocalOpenAIModel(
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
            api_key=os.environ.get("LLM_API_KEY", "not-needed"),
            allow_remote=args.allow_remote_model,
        )
        client = (
            GitHubApiClient(token=os.environ.get("GITHUB_TOKEN"))
            if request.repositories else None
        )
        result = run_pipeline(
            request, github_client=client, model=model,
            cached_analyses=cached,
        )
        write_pipeline_artifacts(result, args.output)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    print(f"Pipeline status: {result.status}")
    print(f"Snapshots: {len(result.snapshots)}; analyses: {len(result.analyses)}")
    print(f"Evidence: {len(result.evidence_index)}")
    print(f"Claims: {len(result.profile.claims) if result.profile else 0}")
    print(f"ResumeDocument: {'ready' if result.resume else 'not generated'}")
    for problem in result.problems:
        print(
            f"[{problem.stage}/{problem.code}] "
            f"{problem.repository or '-'}: {problem.message}"
        )
    print(f"Artifacts: {args.output.resolve()}")
    return 0 if result.resume is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
