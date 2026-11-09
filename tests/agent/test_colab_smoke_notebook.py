"""Static checks for the three-stage live Qwen3.5 Colab demo."""

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
    assert "feat/cv-writer" in source
    assert "unsloth/Qwen3.5-9B-GGUF" in source
    assert "Qwen3.5-9B-Q4_K_M.gguf" in source
    assert 'RELEASE_TAG = "b11896"' in source
    assert 'ggml-org/llama.cpp/releases/download/' in source
    assert 'llama-b11896-bin-ubuntu-cuda-12.8-x64.tar.gz' in source
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
    assert "MODEL_TEMPERATURE = 0.5" in source
    assert 'temperature=MODEL_TEMPERATURE' in source
    assert '"--temperature", str(MODEL_TEMPERATURE)' in source
    assert "CACHED_ANALYSIS_FILES" in source
    assert '"-m", "github2cv_ml.profile_demo"' in source
    assert '"--analysis", str(RESULT_DIR / "analysis.json")' in source
    assert '"--base-url", SERVER_BASE_URL' in source
    assert 'candidate_profile.json' in source
    assert 'profile_preview.md' in source
    assert 'llm_profile_trace.json' in source
    assert 'call["llm_proposed_skills"]' in source
    assert 'call["llm_proposed_project_paragraph"]' in source


def test_colab_uses_verified_official_cuda_release_without_building():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    install = next(cell for cell in cells if 'RELEASE_TAG = "b11896"' in cell)
    launch = next(cell for cell in cells if 'SERVER_PROCESS = subprocess.Popen' in cell)
    assert "https://github.com/ggml-org/llama.cpp/releases/download/" in install
    assert "BIN_SHA256" in install
    assert "CUDART_SHA256" in install
    assert "download_verified(" in install
    assert "unpack_verified(" in install
    assert '"--version"' in install
    assert '"--list-devices"' in install
    assert '"ldd", str(binary)' in install
    assert 'cuda_backend' in install
    assert 'platform.libc_ver()' in install
    assert 'tarfile.open(' in install
    assert 'filter="data"' in install
    assert 'PREBUILT_URL' not in install
    assert 'ai-dock/llama.cpp-cuda' not in install
    assert "nvcc" not in install
    assert 'cmake' not in install
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



def test_colab_runs_cv_writer_and_exports_full_resume():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'profile_available = True' in source
    assert 'cv_available = False' in source
    assert 'if not profile_available:' in source
    assert '"-m", "github2cv_ml.cv_demo"' in source
    assert '"--profile", str(PROFILE_DIR / "candidate_profile.json")' in source
    assert '"--temperature", str(MODEL_TEMPERATURE)' in source
    assert '"resume.json"' in source
    assert '"resume_preview.md"' in source
    assert '"cv_writer_trace.json"' in source
    assert 'bundle.write(result_file, arcname=f"cv/{filename}")' in source
    assert 'if cv_available:' in source
