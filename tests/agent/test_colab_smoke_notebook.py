"""Static checks for real-model Repo Analyzer and Candidate Profile Colab demo."""

import ast
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
    assert "feat/candidate-profile-builder" in source
    assert "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF" in source
    assert "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" in source
    assert 'LLAMA_CPP_REF = "b8660"' in source
    assert '"-DGGML_CUDA=ON"' in source
    assert '"--target", "llama-server"' in source
    assert '"--jinja"' in source
    assert '"--host", "127.0.0.1"' in source
    assert "LocalOpenAIModel" in source
    assert '"-m", "github2cv_ml.agent"' in source
    assert 'analysis["status"]' in source
    assert 'analysis["tool_trace"]' in source
    assert 'analysis["evidence"]' in source
    assert '"report.md"' in source
    assert 'analysis.get("coverage", {})' in source
    assert 'analysis.get("exploration_plan", [])' in source


def test_colab_server_uses_dynamic_port_consistently():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codes = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    server = next(code for code in codes if "SERVER_LOG_PATH = WORKDIR" in code)
    probe = next(code for code in codes if "diagnostic_tool = [{" in code)
    command = next(code for code in codes if '"-m", "github2cv_ml.agent"' in code)

    # Port 8080 may already be occupied by an unrelated Colab process.
    assert 'port_probe.bind(("127.0.0.1", 0))' in server
    assert '"--port", str(SERVER_PORT)' in server
    assert 'SERVER_BASE_URL = f"http://127.0.0.1:{SERVER_PORT}/v1"' in server
    assert 'urlopen(SERVER_HEALTH_URL, timeout=3)' in server
    assert 'base_url=SERVER_BASE_URL' in probe
    assert '"--base-url", SERVER_BASE_URL' in command
    assert '"--port", "8080"' not in server
    assert 'base_url="http://127.0.0.1:8080/v1"' not in probe


def test_colab_classifies_read_budget_stop_as_real_exploration():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'analysis["status"] == "read_budget_exhausted"' in source
    assert "not an explicit finish_analysis" in source


def test_colab_reports_stalled_runs_and_automatic_recovery():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'analysis["status"] == "stalled"' in source
    assert 'action.get("origin") == "automatic_recovery"' in source


def test_colab_distinguishes_covered_plan_from_model_completion():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'analysis["status"] == "plan_covered"' in source
    assert "not whole-repository or semantic coverage" in source


def test_colab_runs_real_profile_builder_and_displays_its_model_proposals():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "CANDIDATE_LOGIN" in source
    assert "CACHED_ANALYSIS_FILES" in source
    assert '"-m", "github2cv_ml.profile_demo"' in source
    assert '"--analysis", str(RESULT_DIR / "analysis.json")' in source
    assert '"--base-url", SERVER_BASE_URL' in source
    assert 'candidate_profile.json' in source
    assert 'profile_preview.md' in source
    assert 'llm_profile_trace.json' in source
    assert 'call["llm_proposed_skills"]' in source
    assert 'call["llm_proposed_project_paragraph"]' in source


def test_colab_builds_pinned_upstream_llama_cpp_instead_of_ai_dock_release():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    build = next(cell for cell in cells if 'LLAMA_CPP_REF = "b8660"' in cell)
    launch = next(cell for cell in cells if 'SERVER_PROCESS = subprocess.Popen' in cell)
    assert 'https://github.com/ggml-org/llama.cpp.git' in build
    assert '"git", "clone", "--depth", "1", "--branch", LLAMA_CPP_REF' in build
    assert '"-DGGML_CUDA=ON"' in build
    assert 'f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"' in build
    assert '"--target", "llama-server"' in build
    assert 'shutil.which("nvcc")' in build
    assert '"ldd", str(library)' in build
    assert 'PREBUILT_URL' not in build
    assert 'ai-dock/llama.cpp-cuda' not in build
    assert '"--jinja"' in launch
    assert '"--reasoning", "off"' in launch
    assert '"--flash-attn", "on"' in launch
    assert '"--cache-type-k", "q8_0"' in launch
    assert '"--cache-type-v", "q8_0"' in launch
    assert '"--host", "127.0.0.1"' in launch
    assert 'SERVER_ENV' in launch


def test_colab_python_cells_parse_without_syntax_errors():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            ast.parse(code, filename=f"notebook_cell_{index}.py")
