"""Run all nine tutorials sequentially, stopping on any failed check."""
import os
from pathlib import Path
import subprocess
import sys

folder = Path(__file__).resolve().parent
if __name__ == "__main__":
    environment = os.environ.copy()
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        environment[variable] = "1"
    for script in sorted(folder.glob("[0-9][0-9]_*.py")):
        print(f"Running {script.name}", flush=True)
        subprocess.run([sys.executable, str(script), *sys.argv[1:]], check=True, env=environment)
