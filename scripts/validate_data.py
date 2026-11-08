#!/usr/bin/env python3
"""Validate the public SI database and write a machine-readable report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data/public_si_database.csv"
frame = pd.read_csv(path)
required = ["SMILES", "IC50", "CC50", "SI", "DOI"]
missing = [c for c in required if c not in frame.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")
valid_smiles = frame.SMILES.map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
expected = frame.CC50 / frame.IC50
error = (expected - frame.SI).abs()
report = {
    "file": str(path),
    "rows": len(frame),
    "columns": frame.columns.tolist(),
    "missing_required_values": {c: int(frame[c].isna().sum()) for c in ["SMILES", "IC50", "CC50", "SI"]},
    "invalid_smiles": int((~valid_smiles).sum()),
    "nonpositive_ic50": int((frame.IC50 <= 0).sum()),
    "nonpositive_cc50": int((frame.CC50 <= 0).sum()),
    "negative_si": int((frame.SI < 0).sum()),
    "formula_errors_over_1e-6": int((error > 1e-6).sum()),
    "duplicate_full_inchikey_rows": int(frame.duplicated("InChIKey").sum()),
    "missing_doi": int(frame.DOI.isna().sum()),
    "source_counts": frame.source.value_counts().to_dict(),
}
report["status"] = "PASS" if all([
    not missing, report["invalid_smiles"] == 0,
    report["nonpositive_ic50"] == 0, report["nonpositive_cc50"] == 0,
    report["negative_si"] == 0, report["formula_errors_over_1e-6"] == 0,
]) else "FAIL"
out = ROOT / "reports/data_validation.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if report["status"] != "PASS":
    raise SystemExit(1)
