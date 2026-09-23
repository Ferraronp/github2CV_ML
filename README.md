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

## Quality checks

```bash
ruff check .
python -m pytest
```

CI runs the same lint and test checks for pull requests and pushes to `main`.
