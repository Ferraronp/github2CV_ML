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

The Profile Builder is a **model-driven** synthesis stage. It does not
use a hardcoded technology or skill vocabulary: an LLM examines existing
`RepoEvidence` observations and proposes skill tags plus a concise
paragraph for each repository. The Python layer validates provenance,
normalizes repeated names, and ranks skills by independent repository
support.

```python
from github2cv_ml.agent.runner import LocalOpenAIModel, RepoAnalysis
from github2cv_ml.domain import CandidateIdentity, RepoEvidence
from github2cv_ml.profile_builder import (
    build_candidate_profile,
    build_profile_from_analyses,
)

identity = CandidateIdentity(github_login="developer")  # supplied by backend
model = LocalOpenAIModel()  # local OpenAI-compatible llama-server

# Input A: all evidence items from several repositories.
profile = build_candidate_profile(
    [RepoEvidence.model_validate(payload) for payload in evidence_payloads],
    candidate=identity,
    model=model,
)

# Input B: previously serialized/cached or freshly computed analyses.
# Backend selects cached versus new analysis before calling this stage.
analyses = [RepoAnalysis.model_validate(payload) for payload in analysis_payloads]
profile = build_profile_from_analyses(
    analyses,
    candidate=identity,
    model=model,
)
```

Each repository is sent to the LLM in a separate bounded JSON-mode call,
which returns its own proposed skill tags and one project paragraph, all
referencing the supplied evidence IDs. Up to 20 skills are retained in the
final `CandidateProfile`; duplicate names and repeated mentions within a
repository do not inflate their rank. Project claims are ordered by diversity
of source observation categories. The final short *About me* text and
user-facing paragraphs belong to the CV Writer (#6).

The original `RepoEvidence.id` records must remain available downstream to
resolve source locators. The Profile Builder makes **no GitHub fetches** and
does not store repository source files. A backend can therefore combine fresh
and previously cached repository analyses.

**Limits and caveats:** This component checks that every model-generated
skill/project claim references real evidence from the **same repository**, not
whether its technical meaning is true. The underlying Repo Analyzer currently
performs locator-only validation; semantic grounding is tracked in #29.
A tag or project description is not proof of a candidate's personal
contribution. The model should not infer individual authorship. Conflicting
evidence IDs and mixed repository revisions are rejected rather than
silently attributed to incorrect sources. Oversized per-repository inputs
fail with an explicit size-limit error; no silent truncation occurs.
