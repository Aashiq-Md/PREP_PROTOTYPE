"""
PREP — Master Runner
Starts the backend API and generates demo data.

Requires a 64-bit Python with numpy + pandas (e.g. `py -3.14 run.py demo`).

Usage:
    python run.py demo      # Generate demo data only
    python run.py backend   # Start FastAPI backend
    python run.py all       # Generate demo data + start backend
    python run.py test      # Run tests
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DEMO_DIR = ROOT / "demo"
BACKEND_DIR = ROOT / "backend"

# numpy/pandas publish no 32-bit wheels for modern Python. Fail fast with a
# readable message instead of an OpenBLAS allocation crash deep in the pipeline.
_PREFLIGHT = (
    "import struct;"
    "assert struct.calcsize('P') * 8 == 64, 'interpreter is 32-bit';"
    "import numpy, pandas"
)


def preflight() -> bool:
    """Verify the running interpreter is 64-bit and has numpy + pandas."""
    result = subprocess.run(
        [sys.executable, "-c", _PREFLIGHT], capture_output=True, text=True
    )
    if result.returncode == 0:
        return True

    print("[ERROR] This Python interpreter cannot run the PREP pipeline.")
    print(f"        interpreter: {sys.executable}")
    detail = [ln for ln in (result.stderr or "").strip().splitlines() if ln]
    if detail:
        print(f"        reason: {detail[-1]}")
    print("[ERROR] Use 64-bit Python 3.11+ with numpy and pandas installed, e.g.:")
    print("          py -3.14 run.py all")
    return False


def _env() -> dict:
    """Environment with OpenBLAS threading capped for low-memory machines."""
    env = dict(os.environ)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def generate_demo():
    if not preflight():
        sys.exit(1)
    print("[PREP] Generating synthetic demo data...")
    result = subprocess.run(
        [sys.executable, str(DEMO_DIR / "generate_demo_data.py")],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("[ERROR] Demo data generation failed.")
        sys.exit(1)
    print("[PREP] Demo data ready.")


def start_backend():
    print("[PREP] Starting FastAPI backend on http://localhost:8000")
    print("[PREP] API docs: http://localhost:8000/docs")
    # `python -m uvicorn` — the uvicorn console script is not always on PATH.
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn", "app.main:app",
            "--reload", "--host", "0.0.0.0", "--port", "8000",
        ],
        cwd=str(BACKEND_DIR),
        env=_env(),
    )


def run_tests():
    if not preflight():
        sys.exit(1)
    print("[PREP] Running tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=str(ROOT),
        env=_env(),
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd == "demo":
        generate_demo()
    elif cmd == "backend":
        start_backend()
    elif cmd == "test":
        run_tests()
    elif cmd == "all":
        generate_demo()
        start_backend()
    else:
        print(__doc__)
        sys.exit(1)
