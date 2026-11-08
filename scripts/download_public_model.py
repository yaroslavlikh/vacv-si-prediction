#!/usr/bin/env python3
"""Download or verify the explicitly external Anti7 public-model component.

The CSV contains model predictions, not measured SI labels. It is used only as
an ensemble component and is never appended to the training data.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/public_model_exp017.csv"
REPORT = ROOT / "reports/public_model_source.json"
COMMIT = "2697ab703963d34c748912048134b8b725633cd7"
SOURCE_FILE = "experiments/exp_017_pure_ml_ultra_sota/submission.csv"
URL = f"https://raw.githubusercontent.com/Velsaf-dev/molecular-ap-model/{COMMIT}/{SOURCE_FILE}"
SHA256 = "d88cd19486c23b3ba44c73f898ac6dba7731b252ec35fefe3806435d912ca99b"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--refresh", action="store_true", help="Download the pinned source again.")
args = parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if args.refresh or not TARGET.exists():
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(response.content)

actual_sha256 = sha256(TARGET)
issues = []
if actual_sha256 != SHA256:
    issues.append(f"SHA-256 mismatch: expected {SHA256}, got {actual_sha256}")
frame = pd.read_csv(TARGET)
if list(frame.columns) != ["index", "SI VACV"]:
    issues.append("Expected columns: index, SI VACV")
if len(frame) != 597:
    issues.append(f"Expected 597 rows, found {len(frame)}")
if "index" in frame and not frame["index"].is_unique:
    issues.append("index is not unique")
if "SI VACV" in frame:
    if not np.isfinite(frame["SI VACV"]).all():
        issues.append("NaN or infinite predictions")
    if (frame["SI VACV"] < 0).any():
        issues.append("Negative predictions")

report = {
    "status": "PASS" if not issues else "FAIL",
    "issues": issues,
    "source_repository": "https://github.com/Velsaf-dev/molecular-ap-model",
    "source_commit": COMMIT,
    "source_file": SOURCE_FILE,
    "download_url": URL,
    "local_file": str(TARGET.relative_to(ROOT)),
    "sha256": actual_sha256,
    "rows": len(frame),
    "content": "External public-model predictions, not measured SI labels",
    "used_for_training": False,
    "used_as_external_ensemble_component": True,
}
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
if issues:
    raise SystemExit(1)
