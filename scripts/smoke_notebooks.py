from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute notebooks in memory as a smoke test.")
    parser.add_argument(
        "notebooks",
        nargs="*",
        default=["02_model_training.ipynb", "03_knockout_simulation.ipynb"],
        help="Notebook paths to execute. Defaults to notebooks 02 and 03.",
    )
    parser.add_argument(
        "--allow-output-writes",
        action="store_true",
        help="Do not skip cells that write CSVs or figures.",
    )
    return parser.parse_args()


def configure_windows_jupyter_runtime() -> None:
    os.environ.setdefault("WC2026_SAVE_FIGURES", "0")
    if os.name != "nt":
        return
    os.environ.setdefault(
        "JUPYTER_RUNTIME_DIR",
        str(Path(tempfile.gettempdir()) / "wc2026-jupyter-runtime"),
    )
    # Windows ACL setup can fail in restricted shells without pywin32. This only affects
    # temporary Jupyter connection files used for the smoke test.
    os.environ.setdefault("JUPYTER_ALLOW_INSECURE_WRITES", "1")
    Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)


def load_notebook(path: Path, allow_output_writes: bool):
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    if allow_output_writes:
        return notebook, []

    skipped: list[int] = []
    kept = []
    write_markers = ["to_csv("]
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type == "code" and any(marker in cell.source for marker in write_markers):
            skipped.append(index)
            continue
        kept.append(cell)
    notebook.cells = kept
    return notebook, skipped


def execute_notebook(path: Path, allow_output_writes: bool) -> None:
    from nbclient import NotebookClient

    notebook, skipped = load_notebook(path, allow_output_writes)
    if skipped:
        print(f"{path.name}: skipped write cells {skipped}")
    started = time.time()
    client = NotebookClient(
        notebook,
        timeout=240,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    print(f"{path.name}: OK in {time.time() - started:.1f}s")


def main() -> int:
    args = parse_args()
    configure_windows_jupyter_runtime()
    for notebook_name in args.notebooks:
        execute_notebook(ROOT / notebook_name, args.allow_output_writes)
    print("Notebook smoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
