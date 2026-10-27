# Repo Analyzer tool-calling MVP (issue #4, experimental slice)

The CLI accepts a **GitHub repository URL**, freezes its default branch to a full commit SHA, and
lets a local model select read-only tools to examine project code. Each `read_file` call causes
a **separate isolated extraction request**; only the resulting short summary and validated
file/line references are returned to the Explorer. Source text never enters the accumulated
Explorer context.

**This is an initial tool-calling prototype, not a completed semantic verifier or the full
Repo Analyzer.** In particular, the model's JSON/function-calling reliability on the actual
GTX 1660 Super still needs to be measured.

## Colab: реальная проверка Qwen2.5-Coder 7B Q4

[**Открыть smoke notebook в Google Colab**](https://colab.research.google.com/github/Ferraronp/github2CV_ML/blob/feat/deep-repo-exploration/notebooks/colab_repo_analyzer_toolcalling_smoke.ipynb)

[Исходный ноутбук в PR #28](../notebooks/colab_repo_analyzer_toolcalling_smoke.ipynb).

В Colab переключи runtime на GPU (T4), затем выполни ячейки сверху вниз. Сервер автоматически выбирает свободный loopback-порт, так что занятый `8080` не требует перезапуска среды; этот же адрес используется для probe и CLI.
Ноутбук клонирует ветку PR, устанавливает пакет, загружает ту же
Qwen2.5-Coder-7B-Instruct Q4_K_M GGUF, что использовалась в v8 benchmark,
разворачивает проверенный по SHA-256 CUDA prebuilt llama-server, выполняет
отдельный diagnostic tool call и запускает **реальный** Repo Analyzer CLI
на публичном `Ferraronp/military-charters-rag`.

Итог: `analysis.json`, `report.md`, история инструментов, счётчики
успешных `read_file` и evidence с прямыми ссылками на GitHub, ZIP для
ручного скачивания. `status=completed` **не** гарантирует семантическую
правильность claims; проверь хотя бы несколько ссылок вручную. Для
`status=incomplete` проверь `tool_trace` и `llama-server.log`.

**Не тестировалось на реальном GPU в CI:** автоматические проверки гарантируют
формат ноутбука, но не качество function calling Qwen или совместимость
выделенной Colab GPU/драйвера с CUDA 12.8. Не отправляй приватный код
в небезопасную среду; фильтр файлов не исключает секреты внутри .py/.js.

## Deep-exploration increment (issue #17)

The Explorer now receives a **deterministic candidate exploration plan** generated
from the safe repository file tree. The plan favors a small, role-diverse set of
files from implementation-relevant roles: pipeline, retrieval, generation,
service, data access, security, infrastructure and entrypoints. Supporting
core utilities and tests are lower priority. On the backend regression
fixture, this favors authentication services and user repositories over
router definitions and generic settings.
The planner excludes paths with missing, empty, or over-limit size metadata
because these cannot pass the browser's initial read checks; the actual content
may still fail validation after fetching. The number of candidates is capped by
`min(7, max_read_calls)` (six by default), avoiding a plan whose size already
exceeds the allowed read count. Covering the plan is still not guaranteed because
other tool calls use the step budget and the model may revisit files.
These roles are **hypotheses based on path names**, not proof of implemented
capabilities. The model still chooses its read-only tool calls; it is prompted
to visit different important files rather than rely on repeated keyword matches.

The read-only tools now support pagination:
- `list_files(path="", offset=0, limit=40)` returns `files`, `total`,
  `offset`, `next_offset` (null on final page), and `truncated`.
- `search_code(query, path="", offset=0)` searches literal text in at most
  12 candidate files per page (case-insensitive; no regex). The offset
  advances by *candidate files*, not by matches, and may omit additional
  matches within a scanned file after the 12-match cap. The result includes
  `next_offset`, `total_candidate_files`, and `search_budget_exhausted`.
  Follow `next_offset` for more files; it is not a whole-repo search.
- `read_file` remains limited to 120 lines and 12K characters.

To avoid using the entire GitHub fetch budget on searches alone,
`RepositoryBrowser` defaults to 40 distinct file fetch attempts total,
of which searches can initiate at most 24. Metadata-rejected large,
empty or unknown-size files do not consume search fetch quota, though they still
advance the pagination offset. Only actual GitHub fetch attempts count toward
the search quota. Files already cached can be reused, and remaining fetches
are available for explicit `read_file` calls.
These are bounds, not promises of exhaustive scanning.

`analysis.json` now includes `exploration_plan` and `coverage`, and
`report.md` lists visited/unvisited selected files. The ratio
`planned_files_read / planned_files` measures only how much of the
**heuristically selected plan** was inspected, **not** project-wide
coverage or semantic evidence quality. `status=completed` means the
Explorer called `finish_analysis`; it does not imply the selected plan
was fully covered. When the allowed number of `read_file` calls is used,
the Explorer stops without further LLM tool calls and returns
`status=read_budget_exhausted`, preserving extracted evidence and explicitly
recording the read-budget limit. This avoids repeated rejected read calls.
If the final available step consumes the last read, the status still names
the read-budget limit rather than the generic step budget.

### Bounded loop recovery and evidence quality

A real smoke on a separate FastAPI/PostgreSQL/Redis backend found **90
duplicate tool errors in 100 Explorer steps**, despite 7/7 heuristic targets
having been visited. A successful result from an identical call remains
cached and is not executed twice; a different requested range that resolves
to already analyzed source lines is now also rejected **without another
extractor call**.

The orchestration layer now triggers deterministic recovery after **one
duplicate/already-inspected read**, three consecutive tool errors, or three
steps without a new successful source read (including otherwise successful
navigation). A single duplicate search/list call remains recoverable by the
model; it does not immediately trigger fallback. The explorer instructions
prioritize unread planned files rather than restarting navigation every turn.

Recovery reads the next unread planned candidate using the same pinned,
read-only browser, isolated extractor and evidence validation as an ordinary
read. After visiting the selected candidates, it can deepen an already-read
planned module by reading its first *uninspected* source range, if budgets
allow. In either case the controller shrinks oversized excerpts to respect
the 12K character limit and skips unreadable candidates. The trace records
`origin=automatic_recovery` and `recovery_reason`.

If no useful planned range remains **and every candidate has been visited**,
the agent stops with `status=plan_covered` instead of falsely claiming that
the model stalled. This means only that the filename-heuristic plan was
covered, **not** that the whole repository was traversed, every line was
verified, or the model called `finish_analysis`. If relevant planned
candidates remain unreadable/unvisited, the status stays `stalled` and
partial evidence is preserved. The total tool-error cap remains 24.

The filename planner now assigns Alembic/migration files a lower-priority
`migration` role so a migration named `*_pipeline.py` no longer displaces
runtime service or repository implementations. Migration content remains
available to the explorer and may still be selected if budget remains.

These are **progress and stop guarantees**, not guarantees of full repository
coverage or correct semantic evidence. Real Colab runs on the previous HEAD
showed Qwen3.5-9B (reasoning disabled) extracting observations but sometimes
claiming optimistic locking where `with_for_update` implements pessimistic
locking, or conflating a semaphore with brute-force rate limiting. Those
errors are **not** resolved by this control-flow fix. Treat extracted claims
as unverified until a separate semantic grounding/verifier phase exists.

These guards **do not** assert that extracted claims are semantically correct.
The extractor prompt now distinguishes tests/deployment from HTTP handling,
observability and application cleanup, and discourages assuming that middleware
is registered simply because it was defined. Those are prompt constraints,
not a substitute for a cross-file verifier.


## Local model setup

Use a recent `llama.cpp` build with OpenAI-compatible tool calling enabled. For example:

```bash
llama-server \
  -m /path/to/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --jinja --alias qwen-coder \
  --ctx-size 8192 --n-gpu-layers 99 --port 8080
```

Some GGUF builds may require a compatible `--chat-template-file`. Check the
llama-server startup logs to confirm tool-calling support. The benchmark selected
Qwen2.5-Coder 7B Q4 as a **candidate**; it did not measure tool-call accuracy.

Install and run the CLI in a separate terminal:

```bash
python -m pip install -e ".[dev]"
python -m github2cv_ml.agent \
  https://github.com/Ferraronp/military-charters-rag \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen-coder \
  --max-steps 12 --max-reads 6 \
  --output ./results/military-rag
```

The output contains:
- `analysis.json`: project description, observations as **existing `RepoEvidence` Pydantic
  contracts**, pinned SHA, visited file summaries, tool trace and limitations.
- `report.md`: readable descriptions and GitHub permalinks for cited lines.

`GITHUB_TOKEN` (optional) is used only for GitHub API requests. The model API key is
`LLM_API_KEY` (usually not needed for localhost). By default `--base-url` **must be a
loopback address**; remote inference, which might transfer private code, requires explicit
`--allow-remote-model`. Start with a public repository.

## Agent design

1. Fetch metadata, dependencies, file tree and README using the existing collector,
   resolving the default branch to an immutable SHA before loading source.
2. Explorer chooses one of `list_files`, `search_code` (literal text, not regex),
   `read_file`, `get_project_info`, `finish_analysis`.
3. A `read_file` call loads at most **120 source lines**. A fresh model call extracts
   `file_summary` + 0–5 observations `{claim, category, source_path, line_start, line_end}`.
4. The Python validator only accepts observations referring to the exact inspected file
   and its actual range, then maps them to `RepoEvidence` with permalink + commit SHA.
5. Explorer receives the summary and counts instead of original source; its next
   request is freshly constructed. The final pass merges findings and drafts a
   provisional project summary.

**The claim itself is not semantically verified** by the source locator checker.
A valid line number is not proof that the claimed implementation exists. This design
reduces unchecked citations but does not eliminate hallucinations. The final
`verification_level` is explicitly `locator_only`.

### Safety and budgets

The tools do not expose shell commands or file writes. The browser uses an
**allowlist of known source-code extensions** (Python, JS/TS, Go, Rust, Java,
Kotlin, C/C++, C#, Ruby and PHP); unknown formats, configuration files
(`.npmrc`, `.pypirc`, `.netrc`, `pip.conf`), environment files, binary
artifacts, large files and generated directories are excluded. Source files
are not executed. This is **not a guarantee that source code itself contains
no embedded secrets**. Treat private repositories and remote model endpoints
as sensitive; do not assume the filter is sufficient for untrusted private data.
Per run: default 12 Explorer steps, 6 read calls, 120 lines/read,
80 KB/file, 12K characters/read, 40 total distinct fetches (24 initiated by search), 12 files/scanned search page and 12 search matches.
Search is **partial**, not a guarantee of finding every occurrence.
Implementation paths under `src/` and other conventional source directories
are scanned before similarly named evaluation utilities. Repeated **successful**
tool calls with identical arguments are rejected as recoverable errors rather than
re-executed; the Explorer is instructed to change the query/file/range.
The extractor is prompted to prefer substantive engineering facts over
boilerplate, but their relevance and semantic correctness still require review.

Repository source and README are treated as untrusted data, never as agent instructions.
The model can terminate early, exhaust the budget or return malformed tool calls; such
runs are explicitly marked as incomplete rather than silently counted as success.
If an LLM request fails or it returns malformed tool calls, `RepoAnalyzer.run()`
keeps any already-extracted evidence and records an `incomplete` status
with a warning; the CLI still writes `analysis.json` and `report.md`.
Missing/invalid tool arguments are recorded as recoverable tool errors.
Model prompts and tool metadata will be logged in the output, so avoid committing reports
containing private repository content.

## Verification

```bash
ruff check .
python -m pytest
```

Unit tests use a deterministic fake model (no GPU/network), verifying
tool dispatch, read isolation, SHA pinning, evidence range validation,
unsafe path rejection, malformed tool arguments, endpoint outages, incomplete
report persistence, budget checks and remote inference opt-in.

### No `tool_calls` returned by a real model

The first Colab smoke produced `status=model_stopped_without_tool`,
zero `file_summaries`, zero `evidence`, and a metadata-only final summary.
That means the collector and final summary ran, **not** that the model
successfully inspected source code.

The Explorer now requests `tool_choice="required"` from `llama-server`.
This is stronger than `auto`, but some server/template combinations may
still return plain text. When that happens, the agent records a bounded
`tool_trace[].response_preview` for diagnosis instead of silently
ending with no trace. The preview may contain model-generated sensitive
text, so handle output artifacts accordingly.

Use the notebook's tool-calling diagnostic to check the response and
the server log's chat template. If forced tool calling still fails, the
model/server/template needs investigation, not a claim that the MVP passed.

### Actual-model smoke test still needed

Run the above CLI against at least one public GitHub repo with
Qwen2.5-Coder-7B-Instruct Q4. Inspect `analysis.json` for:
- correct use of `tool_calls` instead of free-form prose;
- choosing meaningful source files with <12 steps;
- useful file summaries and facts without malformed JSON;
- GitHub links pointing to correct lines **and supporting the claims**;
- no loops/excessive duplicate reads and realistic latency/VRAM.

If Qwen2.5-Coder 7B struggles with tool calling, keep the
**isolated source extraction** step and replace Explorer with deterministic
file selection. LangGraph can wrap the current steps later, but is not
needed for the first end-to-end experiment.


## Experimental whole-file extraction (branch only)

Branch: `experiment/adaptive-whole-file-extraction`. This is an **opt-in A/B
experiment**, not a replacement for the bounded tools or a semantic verifier.

Run the same repo, pinned model, model settings, and budgets twice:

```bash
python -m github2cv_ml.agent OWNER/REPO --max-steps 16 --max-reads 7 \
  --prefer-whole-files --output ./results-whole
python -m github2cv_ml.agent OWNER/REPO --max-steps 16 --max-reads 7 \
  --output ./results-ranges
```

With `--prefer-whole-files`, a normal initial `read_file(1, 120)`
request can be expanded to all numbered lines in the pinned file **only if**
the complete numbered text fits `max_excerpt_chars=12000`. Otherwise
the original 120-line fragment is used. Explicit smaller or later ranges
are unchanged, as are the frozen file tree, 80 KB file guard, read-call quota,
deduplication, provenance line numbers, and `list_files`/`search_code`
pagination. The old mode remains the CLI default.

Compare `coverage.source_read_strategy`, `file_summaries[*].whole_file`,
source ranges, trace/error counts, number of observations, and **manual
correctness of claims**. A higher observation count or wider source range
alone is not evidence of better semantic accuracy. Larger files may still
need multiple reads. Character-budget fitting is not a guarantee of fitting
every model's token context.

The Colab smoke notebook on this branch defaults to Qwen3.5-9B with
`--reasoning off`. Flip `PREFER_WHOLE_FILES` in its setup cell and
rerun the analysis to compare both strategies; outputs use separate ZIP
names and directories. No PR is opened for this experiment.
