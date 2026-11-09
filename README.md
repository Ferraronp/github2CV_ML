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

## Живое демо: LLM генерирует навыки и описания проектов

Для демонстрации работы **модели**, а не жёстких правил, используйте
[Colab-ноутбук](notebooks/colab_repo_analyzer_toolcalling_smoke.ipynb) из
ветки `feat/candidate-profile-builder`. Он выполняет две последовательные ML-стадии:
сначала Repo Analyzer получает evidence через GitHub, затем Candidate Profile
Builder **повторно не читает GitHub**, а передаёт готовые наблюдения локальной
Qwen и получает от неё навыки-теги и русские абзацы о проектах.

Также можно показать работу именно с кэшем — из готовых файлов, без запуска
Repo Analyzer:

```bash
# Требуется уже запущенный локальный llama-server
python -m github2cv_ml.profile_demo \
  --analysis ./repo_analyzer_smoke_whole.zip \
  --github-login demo_developer \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen-coder \
  --output ./results/profile_demo

# Для двух разных репозиториев добавьте ещё один --analysis:
#   --analysis ./another_repo/analysis.json
```

Поддерживаются `analysis.json` и ZIP-архивы предыдущих Colab smoke-прогонов.
На выходе появляются три **реально созданных**, а не подготовленных заранее,
артефакта:

- `candidate_profile.json` — валидный доменный контракт с claims и evidence IDs;
- `profile_preview.md` — навыки-теги, связные абзацы проектов и ссылки на код;
- `llm_profile_trace.json` — отдельная запись **каждого вызова модели**:
  репозиторий, число входных evidence, время, предложенные навыки и абзац.

Дополнительные `--analysis` должны относиться к различным репозиториям либо
к одинаковой версии одного репозитория. При конфликте evidence/ревизий
генерация явно завершается ошибкой. Все исходные файлы по-прежнему
остаются в GitHub; демо потребляет только сохранённые результаты анализа.

**Оговорка:** код проверяет существование evidence ID, а не истинность
утверждения. Реальная Qwen может выдать неточный навык или неверную формулировку;
результат — черновик для проверки, а не готовое резюме. Автоматические CI-тесты
используют имитацию LLM; реальный GPU-прогон нового Profile Builder пока не
подтверждён.

### Colab: официальный готовый llama.cpp, без компиляции

В старом Colab ссылка на сторонний архив `ai-dock/b8660` завершалась ошибкой
`wget exit status 4`. Попытка заменить её компиляцией `llama.cpp` через
CUDA/CMake оказалась слишком медленной для демонстрации.

Поэтому в PR #30 перенесён **способ установки из предоставленного
`Aiva_Character998_WebUI_AB.ipynb`**: скачивание готового CUDA-бинарника
из **официального** релиза
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp/releases/tag/b11896)
`b11896`. Используется архив
`llama-b11896-bin-ubuntu-cuda-12.8-x64.tar.gz`, закреплённый SHA-256 и
локальный кэш; при отсутствии CUDA-библиотек автоматически скачивается
официальный `cudart-llama-b11896-bin-ubuntu-cuda-12.8-x64.tar.gz` с отдельной
проверкой SHA-256.

Сохраняются проверки Linux x64, glibc >= 2.39, T4/SM75, динамических
библиотек (`ldd`), `llama-server --version` и обнаружения CUDA/GPU
(`--list-devices`). Ни CMake, ни `nvcc`, ни сборка из исходников не нужны.
Из Aiva-ноутбука не копировались части про LoRA, Drive и Web UI.

После проверки готового бинарника сервер запускается с `--jinja`,
`--reasoning off`, Flash Attention и Q8 KV cache на localhost.
**CI проверяет код ноутбука статически, но реальный GPU-запуск нового
официального бинарника в Colab ещё не подтверждён.**


### Настройка модели для демонстрации

В Colab PR #30 используется **Qwen3.5-9B Q4_K_M** из
`unsloth/Qwen3.5-9B-GGUF` вместо Qwen2.5-Coder-7B.
Параметр `MODEL_TEMPERATURE = 0.5` в первой ячейке задаёт температуру
Explorer, Extractor, Profile Builder и проверки tool calling через
`LocalOpenAIModel`. Можно вернуть `0.1` для сравнения результатов.
Отключение thinking на стороне `llama-server` (`--reasoning off`) сохранено.
Температура влияет на вариативность ответов, но сама по себе не
исправляет нарушения протокола `tool_calls`.

В отдельных CLI запускается тот же режим через аргумент
`--temperature 0.5` (необязательный, по умолчанию `0.1`).
