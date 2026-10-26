# Repo Analyzer tool-calling MVP (issue #4, experimental slice)

The CLI accepts a **GitHub repository URL**, freezes its default branch to a full commit SHA, and
lets a local model select read-only tools to examine project code. Each `read_file` call causes
a **separate isolated extraction request**; only the resulting short summary and validated
file/line references are returned to the Explorer. Source text never enters the accumulated
Explorer context.

**This is an initial tool-calling prototype, not a completed semantic verifier or the full
Repo Analyzer.** In particular, the model's JSON/function-calling reliability on the actual
GTX 1660 Super still needs to be measured.

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

The tools do not expose shell commands or file writes. Only text files in the frozen
file tree can be opened; environment/secrets, binary artifacts, large files, generated
directories and traversal paths are excluded. Source files are not executed.
Per run: default 12 Explorer steps, 6 read calls, 120 lines/read,
80 KB/file, 12K characters/read, 25 distinct fetched files, 12 files/scanned search and 12 search matches.
Search is **partial**, not a guarantee of finding every occurrence.

Repository source and README are treated as untrusted data, never as agent instructions.
The model can terminate early, exhaust the budget or return malformed tool calls; such
runs are explicitly marked as incomplete rather than silently counted as success.
Model prompts and tool metadata will be logged in the output, so avoid committing reports
containing private repository content.

## Verification

```bash
ruff check .
python -m pytest
```

Unit tests use a deterministic fake model (no GPU/network), verifying
tool dispatch, read isolation, SHA pinning, evidence range validation,
unsafe path rejection, budget checks and remote inference opt-in.

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
