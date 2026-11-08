#!/usr/bin/env python3
"""Validate a Kaggle submission without reading any hidden labels."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("--rows", type=int, default=597)
args = parser.parse_args()
path = args.csv.resolve()
frame = pd.read_csv(path)
issues = []
if list(frame.columns) != ["index", "SI VACV"]:
    issues.append("Expected columns: index, SI VACV")
if len(frame) != args.rows:
    issues.append(f"Expected {args.rows} rows, found {len(frame)}")
if "index" in frame and not frame["index"].is_unique:
    issues.append("index is not unique")
if "SI VACV" not in frame:
    issues.append("SI VACV is missing")
else:
    if not np.isfinite(frame["SI VACV"]).all():
        issues.append("NaN or infinite predictions")
    if (frame["SI VACV"] < 0).any():
        issues.append("Negative predictions")
report = {"file": str(path), "rows": len(frame), "issues": issues, "status": "PASS" if not issues else "FAIL"}
print(json.dumps(report, indent=2))
if issues:
    raise SystemExit(1)
