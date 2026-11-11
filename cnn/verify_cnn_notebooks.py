"""Execute both notebooks in smoke mode, keeping outputs outside the source files.

Run with the CPU environment containing requirements-notebooks.txt:
    .venv/Scripts/python.exe cnn/verify_cnn_notebooks.py
"""

import os
import hashlib
import re
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


CNN_DIR = Path(__file__).resolve().parent
ROOT = CNN_DIR.parent
OUTPUT = CNN_DIR / "artifacts" / "_verification"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for variable, directory in (("JUPYTER_RUNTIME_DIR", "runtime"), ("IPYTHONDIR", "ipython"),
                                ("MPLCONFIGDIR", "matplotlib")):
        location = OUTPUT / directory
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)
    for name in ("cnn_tuning.ipynb", "cnn_analysis.ipynb"):
        notebook = nbformat.read(CNN_DIR / name, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{name}:cell{index}", "exec")
                cell.source = re.sub(r"(?m)^SMOKE_MODE\s*=.*$", "SMOKE_MODE = True", cell.source)
                cell.source = re.sub(r"(?m)^RUN_TEST\s*=.*$", "RUN_TEST = False", cell.source)
                cell.source = re.sub(r"(?m)^DEVICE_REQUEST\s*=.*$", 'DEVICE_REQUEST = "cpu"', cell.source)
                verification_name = "verification_" + hashlib.sha256((CNN_DIR / "cnn_experiments.py").read_bytes()).hexdigest()[:12]
                cell.source = re.sub(r"(?m)^RUN_NAME\s*=.*$", f'RUN_NAME = "{verification_name}"', cell.source)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]

        def progress(cell, cell_index):
            if cell.cell_type == "code":
                print(f"{name}: executing cell {cell_index}", flush=True)

        working_directory = CNN_DIR if name == "cnn_tuning.ipynb" else ROOT
        client = NotebookClient(notebook, km=manager, timeout=180,
                                resources={"metadata": {"path": str(working_directory)}}, on_cell_start=progress)
        try:
            executed = client.execute()
        finally:
            # Explicitly supplied managers are not owned/cleaned up by nbclient.
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
        nbformat.write(executed, OUTPUT / name)
        images = sum("image/png" in output.get("data", {})
                     for cell in executed.cells for output in cell.get("outputs", []))
        print(f"PASS {name}: {len(executed.cells)} cells, {images} rendered figures", flush=True)
    print(f"Executed smoke notebooks: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
