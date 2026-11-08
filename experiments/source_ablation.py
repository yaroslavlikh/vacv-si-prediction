from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
te = pd.read_csv(ROOT / "data/molecular_activity/test.csv")
base_rows = pd.read_csv(OUT / "public_hybrid_rows.csv")
expanded_rows = pd.read_csv(OUT / "expanded_chembl_rows.csv")
lit_rows = pd.read_csv(OUT / "literature_hybrid_rows.csv")
base = base_rows.base_pred.to_numpy(float)
original = base_rows.public_si.to_numpy(float)
expanded = expanded_rows.expanded_si.to_numpy(float)
lit_source = lit_rows.literature_source.fillna("").to_numpy(str)
lit_values = lit_rows.pred.to_numpy(float)
true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
true = true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy(float)


def apply_sources(use_original, use_expanded, use_literature):
    pred = base.copy()
    if use_expanded:
        mask = np.isfinite(expanded)
        pred[mask] = expanded[mask]
    elif use_original:
        mask = np.isfinite(original)
        pred[mask] = original[mask]
    if use_literature:
        mask = lit_source != ""
        pred[mask] = lit_values[mask]
    return pred

variants = {
    "no_lookup": apply_sources(False, False, False),
    "original_only": apply_sources(True, False, False),
    "original_and_literature": apply_sources(True, False, True),
    "expanded_and_literature": apply_sources(False, True, True),
    "literature_only": apply_sources(False, False, True),
}
report = {}
for name, pred in variants.items():
    path = OUT / f"submission_{name}.csv"
    pd.DataFrame({"index": te["index"], "SI VACV": pred}).to_csv(path, index=False)
    report[name] = {"local_reconstructed_mse": float(np.mean((true - pred) ** 2)),
                    "local_reconstructed_mae": float(np.mean(np.abs(true - pred))),
                    "max_prediction": float(pred.max()), "file": path.name}
(OUT / "source_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
