"""Static checks for the real-model Colab smoke notebook in PR #27."""

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).parents[2]
    / "notebooks"
    / "colab_repo_analyzer_toolcalling_smoke.ipynb"
)


def test_colab_smoke_notebook_is_valid_and_unexecuted():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) >= 8
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)


def test_colab_smoke_exercises_real_llm_server_and_cli():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "feat/repo-analyzer-toolcalling-mvp" in source
    assert "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF" in source
    assert "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" in source
    assert "PREBUILT_SHA256" in source
    assert '"--jinja"' in source
    assert '"--host", "127.0.0.1"' in source
    assert "LocalOpenAIModel" in source
    assert '"-m", "github2cv_ml.agent"' in source
    assert 'analysis["status"]' in source
    assert 'analysis["tool_trace"]' in source
    assert 'analysis["evidence"]' in source
    assert '"report.md"' in source
