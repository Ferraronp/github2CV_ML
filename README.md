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
which returns its proposed skill tags and one project paragraph, all
referencing the supplied evidence IDs. After per-repo extraction, **another
bounded LLM call compares tag names across repositories**, proposing only
well-justified exact synonyms (for example, Postgres/PostgreSQL). There is
no fixed skill dictionary. A merge must use existing names, must contain a
reason, and cannot reuse a name in another group; uncertain names stay
separate. These checks validate the **merge format**, not its semantic truth.

Up to 20 skills are retained in the final `CandidateProfile`; duplicate
names and aliases are merged. Crucially, ranking is by **distinct
repositories**, whereas each skill's `evidence_ids` is the **complete
deduplicated union** of all LLM-cited sources across those repositories.
The same skill mentioned several times in one repo does not inflate the
score, but its evidence IDs are never dropped. Project claims are ordered
by diversity of observations. The final short *About me* text and polished
CV sections belong to CV Writer (#6).

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
  для репозиториев — число evidence, навыки и абзацы; для reconciliation —
  какие теги сравнивались и какие объединения предложила LLM.

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

## CV Writer — задача #6

Третья ML-стадия преобразует готовый `CandidateProfile` в существующий
доменный контракт `ResumeDocument` (JSON, без PDF/HTML).
**Репозитории и исходники ей не передаются.**

Структура рассчитана на согласованный формат резюме в духе hh.ru:

- короткое **«О себе»** (2–3 предложения) и необязательный технический заголовок;
- **до 20 навыков-тегов**;
- **каждый проект — отдельный связный абзац**;
- необязательная секция «Задачи и достижения» — отдельные пункты из
  существующих profile claims типов `achievement` / `responsibility`.
  Если таких claims нет, пустые достижения не придумываются.

LLM в JSON-режиме формирует **только вводную часть**, указывая `claim_ids`
для каждого утверждения. Python проверяет все ID, структуру и ограничения.
Навыки, проекты и достижения **не перефразируются повторно**: они уже
сформулированы Profile Builder и копируются в `ResumeItem` с точными
ссылками на соответствующие `CandidateClaim.id`. Так не возникает второго
шанса выдумать детали проектов. При пустом профиле не делается ни одного
вызова LLM, выход остаётся честно неполным.

Пример вызова из Python:

```python
from github2cv_ml.agent.runner import LocalOpenAIModel
from github2cv_ml.cv_writer import write_resume
from github2cv_ml.domain import CandidateProfile

profile = CandidateProfile.model_validate_json(profile_json)
model = LocalOpenAIModel(model="qwen-3.5", temperature=0.5)
resume = write_resume(profile, model=model)
# resume.model_dump_json() соответствует ResumeDocument v1.0.
```

Для демонстрации можно использовать уже готовый `candidate_profile.json`
или архив из предыдущего Colab — **повторный анализ GitHub не требуется**:

```bash
python -m github2cv_ml.cv_demo \
  --profile ./github2cv_live_profile_demo.zip \
  --model qwen-3.5 --temperature 0.5 \
  --output ./results/cv_demo
```

Команда обращается только к локальному OpenAI-совместимому LLM-серверу и
создаёт `resume.json`, `resume_preview.md` и `cv_writer_trace.json`.
Важная граница: сохранённые ID доказывают **происхождение текста из
профильных claims**, но не истинность этих claims. Семантическая проверка
на уровне исходного кода остаётся задачей #29; CV Writer не должен выдавать
непроверенное авторство, метрики, опыт работы или квалификацию.


### Несколько репозиториев в Colab

Ноутбук [`colab_repo_analyzer_toolcalling_smoke.ipynb`](notebooks/colab_repo_analyzer_toolcalling_smoke.ipynb)
теперь принимает несколько адресов в первом поле `TARGET_REPOS`:
```python
TARGET_REPOS = "Ferraronp/military-charters-rag, Ferraronp/github2CV_ML"
```
Допускаются `owner/repo` и прямые `https://github.com/owner/repo` ссылки,
разделённые запятыми, точками с запятой или переводом строки. Повторяющиеся
адреса в разных регистрах удаляются.

Каждый репозиторий **последовательно** анализируется через `Repo Analyzer`,
результаты складываются в отдельные `analyses/NN_owner_repo/analysis.json`.
Один ошибочный проект не сбрасывает другие; если evidence нет, этот
результат не передаётся Profile Builder. Все успешные анализы вместе с
`CACHED_ANALYSIS_FILES` передаются единому `Profile Builder`, который
формирует **общий CandidateProfile**. Затем один `CV Writer` генерирует
**общее резюме с отдельными абзацами проектов**.

Скачиваемый ZIP содержит полный список входов, ошибок и результатов в
`analysis_manifest.json`, отдельные анализы в `analyses/...`, общий
профиль в `profile/...` и итоговое резюме в `cv/...`.
Корневой `analysis.json` в ZIP сохранён для совместимости со старым
однорепозиторным демо и относится **только к первому свежему репозиторию**.

Это сценарий **мультипроектного последовательного запуска**, а не
реализация параллельной A2A-оркестрации из #11. В Colab не нужно писать
свой цикл для нескольких GitHub URL: он уже добавлен в ноутбук.

### Углублённый анализ крупных репозиториев

После smoke-прогона двух репозиториев (15 доступных исходных файлов у
`military-charters-rag`, 46 у `github2CV_ML`) стало видно, что четыре
чтения недостаточны: у второго проекта все три evidence пришли из одного
файла `tests/test_benchmark_report_csv.py`, хотя существовали модули
реальной ML-архитектуры. Это ограничение полноты анализа, а не доказательство
отсутствия остальной функциональности.

**Новые настройки в Colab** (применяются **к каждому** репозиторию):

```python
MAX_STEPS = 28
MAX_READS = 12
MIN_PLANNED_READS = 8
```

Repo Analyzer теперь формирует до **12 разных запланированных исходных
файлов** (раньше максимум 7). Приоритет — основные модули реализации;
тесты/бенчмарки и миграции идут после них. `runner.py` и
`profile_builder.py` распознаются как кандидаты на инфраструктуру
pipeline, но их назначение по-прежнему **не считается доказанным по имени**.

Если LLM преждевременно вызывает `finish_analysis`, настройка
`--min-planned-reads 8` заставляет контроллер сначала прочитать
недостающие разные модули из плана (если они доступны). Неудачные чтения
не выдают искусственных evidence. При исчерпании шагов или недоступности
файлов результат остаётся частичным, с явным предупреждением.

Через CLI можно выбирать бюджет самостоятельно:

```bash
python -m github2cv_ml.agent Ferraronp/github2CV_ML \
  --max-steps 28 --max-reads 12 --min-planned-reads 8 \
  --model qwen-3.5 --temperature 0.5
```

`analysis.json.coverage` теперь включает `minimum_planned_files` и
`source_files_with_evidence`, позволяя отдельно видеть количество
изученных модулей и количество файлов, реально давших подтверждения.
**Это не равно полному покрытию репозитория**: читается ограниченное
число фрагментов, и semantic grounding остаётся задачей #29.
Увеличение числа чтений увеличивает длительность GPU-прогона.

### Проверка ссылок в тексте «О себе»

CV Writer по-прежнему не получает сырые файлы GitHub. Он берёт
`CandidateProfile` и проверяет все `claim_ids`. Если текст,
предложенный моделью, явно упоминает существующий навык-тег из профиля,
но забывает его ID, Python **добавляет ровно этот уже существующий
`CandidateClaim.id`** (по совпадению полного названия без совпадений
подстрок). Никакие новые навыки не создаются.

Фактическое дополнение ссылок записывается в `cv_writer_trace.json`
(`recovered_summary_claim_ids`, `recovered_headline_claim_ids`).
Это убирает потерянные ссылки на известные теги в предыдущем
smoke-прогоне. **Не путать со смысловой проверкой**: корректный ID
не делает истинным утверждение о роли технологии или личном опыте.
Такие проверки необходимо реализовать/оценить отдельно в #29.
