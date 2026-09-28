# github2CV_ML

Minimal Python foundation for the ML side of github2CV.

## Local setup

Requirements: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Domain contracts

The pipeline uses explicit Pydantic contracts instead of passing raw GitHub API responses between components:

```text
GitHub -> RepoSnapshot -> RepoEvidence -> CandidateProfile -> ResumeDocument
```

`RepoEvidence` keeps the repository source locator for an observation. `CandidateProfile` claims reference evidence IDs, and `ResumeDocument` items reference claim IDs so generated CV text remains traceable to repository evidence.

Valid JSON examples for the contracts live in `examples/domain_contracts.json`.

## GitHub repository collector

`GitHubRepositoryCollector` accepts either `owner/repo` or a GitHub repository URL and returns a validated `RepoSnapshot` with repository metadata, languages, topics, README contents, the recursive file tree, activity signals, and deterministic project signals extracted before any LLM call.

```python
from github2cv_ml.collectors import GitHubRepositoryCollector

collector = GitHubRepositoryCollector()
snapshot = collector.collect("octocat/hello-world")
```

For private repositories, pass a GitHub token that has access to the repository:

```python
collector = GitHubRepositoryCollector(token="...")
snapshot = collector.collect("owner/private-repo")
```

Empty repositories are represented with an empty language map and file tree, and without README contents. Missing or inaccessible repositories raise explicit collector errors instead of leaking raw GitHub API responses downstream.

## Deterministic project signals

Before any LLM analysis, the collector recognizes common dependency manifests and repository conventions. `RepoSnapshot.signals` contains:

- dependencies from `pyproject.toml`, `requirements*.txt`, `package.json`, and `go.mod`;
- recognized manifest paths;
- Dockerfile, Compose, and GitHub Actions workflow signals;
- common top-level layouts such as `src`, `tests`, `frontend`, `backend`, `cmd`, `internal`, and `pkg`.

Unsupported files and malformed manifests are ignored rather than breaking collection. The source path is kept on every extracted dependency/tooling/structure signal so later evidence generation can remain traceable.

## Quality checks

```bash
ruff check .
python -m pytest
```

CI runs the same lint and test checks for pull requests and pushes to `main`.

Development work is tracked in GitHub issues and merged through reviewed squash pull requests.
