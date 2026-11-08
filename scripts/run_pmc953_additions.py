#!/usr/bin/env python3
"""Execute and validate the PMC9533879 article-enhanced final solution.

The notebook trains all LightGBM models from the original competition train.
No saved predictions or OOF arrays are loaded. An optional reference CSV is used
only after execution to verify numerical reproducibility.
"""
from pathlib import Path
import argparse
import hashlib
import json

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
REPORTS = ROOT / "reports"
SOURCE = NOTEBOOKS / "reproducible_v2_pmc953_additions.ipynb"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--reference",
    type=Path,
    help="Optional original submission for post-run numerical comparison.",
)
args = parser.parse_args()

for relative in [
    "data/molecular_activity/train.csv",
    "data/molecular_activity/test.csv",
    "external_training_labels.csv",
]:
    if not (ROOT / relative).is_file():
        raise SystemExit(f"Missing required input: {relative}")

executed_path = NOTEBOOKS / "executed_v2_pmc953_additions.ipynb"
notebook = nbformat.read(SOURCE, as_version=4)
client = NotebookClient(
    notebook,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(NOTEBOOKS)}},
)
client.execute()
nbformat.write(notebook, executed_path)

code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
errors = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
execution_complete = all(cell.execution_count is not None for cell in code_cells)

submission_path = NOTEBOOKS / "submission_v2_pmc953_additions.csv"
submission = pd.read_csv(submission_path)
test = pd.read_csv(ROOT / "data/molecular_activity/test.csv")
issues = []
if list(submission.columns) != ["index", "SI VACV"]:
    issues.append("Expected columns: index, SI VACV")
if len(submission) != 597:
    issues.append(f"Expected 597 rows, found {len(submission)}")
if not submission["index"].is_unique:
    issues.append("index is not unique")
if not np.array_equal(submission["index"].to_numpy(), test["index"].to_numpy()):
    issues.append("Submission indices do not match test indices")
if not np.isfinite(submission["SI VACV"]).all():
    issues.append("NaN or infinite predictions")
if (submission["SI VACV"] < 0).any():
    issues.append("Negative predictions")
if errors or not execution_complete:
    issues.append("Notebook execution failed or is incomplete")

reproduced_hash = hashlib.sha256(submission_path.read_bytes()).hexdigest()
comparison = {"reference_provided": False, "numerically_verified": False}
if args.reference:
    reference_path = args.reference.resolve()
    reference = pd.read_csv(reference_path)
    if list(reference.columns) != ["index", "SI VACV"] or len(reference) != 597:
        issues.append("Reference CSV has an invalid schema")
        max_difference = None
    else:
        if not np.array_equal(reference["index"].to_numpy(), submission["index"].to_numpy()):
            issues.append("Reference indices differ from reproduced indices")
        max_difference = float(
            np.max(
                np.abs(
                    reference["SI VACV"].to_numpy(float)
                    - submission["SI VACV"].to_numpy(float)
                )
            )
        )
        if max_difference > 1e-8:
            issues.append(f"Predictions differ from reference: max diff={max_difference}")
    reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    comparison = {
        "reference_provided": True,
        "reference_file": reference_path.name,
        "reference_sha256": reference_hash,
        "reproduced_sha256": reproduced_hash,
        "byte_identical": reference_hash == reproduced_hash,
        "max_absolute_difference": max_difference,
        "tolerance": 1e-8,
        "numerically_verified": max_difference is not None and max_difference <= 1e-8,
    }

run_report_path = NOTEBOOKS / "pmc953_notebook_run_report.json"
run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
report = {
    "status": "PASS" if not issues else "FAIL",
    "issues": issues,
    "notebook": SOURCE.name,
    "executed_notebook": executed_path.name,
    "code_cells": len(code_cells),
    "errors": len(errors),
    "execution_complete": execution_complete,
    "rows": len(submission),
    "matched_article_structures": run_report["matched_structures"],
    "changed_rows_from_wiley_bsc40": run_report["changed_rows"],
    "reproduced_sha256": reproduced_hash,
    "comparison": comparison,
    "official_kaggle_mse": None,
    "official_kaggle_mse_status": "not_submitted",
    "base_wiley_bsc40_official_mse_user_reported": 197909.42299,
    "local_reconstructed_mse": run_report["local_candidate_mse"],
    "local_reference_warning": "Diagnostic only; not an independent Kaggle score",
    "article": run_report["article"],
    "pmc": run_report["pmc"],
    "doi": run_report["doi"],
    "pmid": run_report["pmid"],
    "pubchem_assays": run_report["pubchem_assays"],
    "test_targets_used_for_fit_or_selection": False,
    "origin_chat": "Снижение MSE ниже 50 000",
}
REPORTS.mkdir(exist_ok=True)
(REPORTS / "pmc953_reproduction.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, ensure_ascii=False))
if issues:
    raise SystemExit(1)
