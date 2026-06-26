"""Execute code cells from the HMM grid-search notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("hmm_selected_features_grid_search.ipynb")


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        print(f"\n--- cell {index} ---", flush=True)
        code = "".join(cell.get("source", []))
        exec(compile(code, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
