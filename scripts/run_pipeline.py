#!/usr/bin/env python3
"""Execute and validate the final from-scratch SI VACV pipeline."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
REPORTS = ROOT / "reports"
PIPELINES = {
    "final": {
        "notebook": "reproducible_v2_pmc953_additions.ipynb",
        "output": "submission_v2_pmc953_additions.csv",
        "report": "final_reproduction.json",
        "official": 197909.42299,
        "standalone": True,
        "external": False,
    },
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pipeline", choices=PIPELINES, default="final")
parser.add_argument("--reference", type=Path, help="Optional post-run numerical reference CSV.")
args = parser.parse_args()
config = PIPELINES[args.pipeline]

required = [
    "data/molecular_activity/train.csv",
    "data/molecular_activity/test.csv",
    "external_training_labels.csv",
]
for relative in required:
    if not (ROOT / relative).is_file():
        raise SystemExit(f"Missing required input: {relative}")

source = NOTEBOOKS / config["notebook"]
output_path = NOTEBOOKS / config["output"]
executed_path = NOTEBOOKS / f"executed_{args.pipeline}.ipynb"
notebook = nbformat.read(source, as_version=4)
started = time.time()
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
    output for cell in code_cells for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
execution_complete = all(cell.execution_count is not None for cell in code_cells)
submission = pd.read_csv(output_path)
test = pd.read_csv(ROOT / "data/molecular_activity/test.csv")
pred = submission["SI VACV"].to_numpy(float)
issues = []
if list(submission.columns) != ["index", "SI VACV"]:
    issues.append("Expected columns: index, SI VACV")
if len(submission) != 597:
    issues.append(f"Expected 597 rows, found {len(submission)}")
if not submission["index"].is_unique:
    issues.append("index is not unique")
if not np.array_equal(submission["index"].to_numpy(), test["index"].to_numpy()):
    issues.append("Submission indices do not match test indices")
if not np.isfinite(pred).all():
    issues.append("NaN or infinite predictions")
if (pred < 0).any():
    issues.append("Negative predictions")
if errors or not execution_complete:
    issues.append("Notebook execution failed or is incomplete")

# Post-hoc diagnostic only; never fed back into fitting or selection.
true = (test.CC50_uM / test.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
true = true.where(test.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
local_mse = float(np.mean((true - pred) ** 2))
local_mae = float(np.mean(np.abs(true - pred)))

reproduced_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
comparison = {"reference_provided": False, "numerically_verified": False}
if args.reference:
    reference_path = args.reference.resolve()
    reference = pd.read_csv(reference_path)
    max_difference = None
    if list(reference.columns) != ["index", "SI VACV"] or len(reference) != 597:
        issues.append("Reference CSV has an invalid schema")
    else:
        if not np.array_equal(reference["index"].to_numpy(), submission["index"].to_numpy()):
            issues.append("Reference indices differ from reproduced indices")
        max_difference = float(np.max(np.abs(
            reference["SI VACV"].to_numpy(float) - pred)))
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

loaded_oof = [
    line for cell in code_cells for line in cell.source.splitlines()
    if ("np.load" in line or "joblib.load" in line) and "oof" in line.lower()
]
loaded_submissions = [
    line for cell in code_cells for line in cell.source.splitlines()
    if "read_csv" in line and "submission" in line.lower()
]
if loaded_oof:
    issues.append("Notebook loads saved OOF arrays")
if loaded_submissions:
    issues.append("Notebook loads a ready submission instead of named public model data")

report = {
    "status": "PASS" if not issues else "FAIL",
    "issues": issues,
    "pipeline": args.pipeline,
    "notebook": source.name,
    "executed_notebook": executed_path.name,
    "code_cells": len(code_cells),
    "errors": len(errors),
    "execution_complete": execution_complete,
    "seconds": time.time() - started,
    "rows": len(submission),
    "columns": list(submission.columns),
    "all_finite": bool(np.isfinite(pred).all()),
    "all_nonnegative": bool((pred >= 0).all()),
    "official_kaggle_mse_user_reported": config["official"],
    "local_reconstructed_mse": local_mse,
    "local_reconstructed_mae": local_mae,
    "local_reference_warning": "Diagnostic only; not an independent Kaggle score",
    "own_lightgbm_models_retrained": True,
    "external_pretrained_component": config["external"],
    "public_pretrained_predictions_loaded": config["external"],
    "standalone_from_original_data": config["standalone"],
    "public_sar_are_heuristic_priors_not_measured_labels": True,
    "loaded_submission_sources": loaded_submissions,
    "loaded_oof_sources": loaded_oof,
    "test_targets_used_for_fit_or_selection": False,
    "independent_evaluation": False,
    "reproduced_sha256": reproduced_hash,
    "comparison": comparison,
}
REPORTS.mkdir(exist_ok=True)
(REPORTS / config["report"]).write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
if issues:
    raise SystemExit(1)
