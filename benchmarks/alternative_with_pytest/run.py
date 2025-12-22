from pathlib import Path
import os
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

os_cwd = Path.cwd()
if os_cwd != REPO_ROOT:
    print(f"[benchmarks] Changing cwd from {os_cwd} → {REPO_ROOT}")
    os.chdir(REPO_ROOT)

raise SystemExit(pytest.main(["benchmarks", "-m", "bench", "-v"]))
