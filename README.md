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

## Candidate Profile Builder (Issue #5)

Profile synthesis is deterministic and makes **no** GitHub or LLM calls. It
accepts normalized repository evidence from any number of repositories
(including backend-cached analyses) and returns the existing
`CandidateProfile` domain contract:

```python
from github2cv_ml.domain import CandidateIdentity, RepoEvidence
from github2cv_ml.profile_builder import (
    build_candidate_profile,
    build_profile_from_analyses,
)

identity = CandidateIdentity(github_login="developer")  # supplied by backend

# Input A: evidence from one or multiple repository analyses.
profile = build_candidate_profile(
    [RepoEvidence.model_validate(payload) for payload in evidence_payloads],
    candidate=identity,
)

# Input B: previously serialized or fresh RepoAnalysis objects.
# Backend chooses fresh vs cached per repository before passing it here.
profile = build_profile_from_analyses(analyses, candidate=identity)
```

The result contains up to **20 canonical skill-tag claims** and one
evidence-backed project claim **per repository**, ordered by diversity of
observations. Project claim text is a conservative draft paragraph; the
CV Writer (#6) will produce the final short summary, tag list and polished
**one-paragraph-per-project** presentation.

Each claim references existing `RepoEvidence.id` values; keep the original
evidence records available downstream to resolve their repository and source
URLs. A skill appearing ten times in one repository counts only once for
ranking; multiple distinct repositories count as broader evidence. All
outputs have stable claim IDs independent of input order.

Important limits: this is vocabulary-based tagging and prioritization, **not
semantic verification**. An LLM extractor's incorrect statement may still
produce a tag; source locators are not sufficient to assert personal authorship.
No skill is inferred from a file extension, and a project's existence does not
mean the candidate personally implemented it. Backend-provided identity and
declared role must stay separate from repository observations.

For safety, if the backend mixes two commits of the same repository or if
different evidence records reuse the same ID, the builder rejects the input
instead of silently attributing claims to incorrect source material. Neither
raw repository files nor an S3 storage layer are required for profile synthesis.
