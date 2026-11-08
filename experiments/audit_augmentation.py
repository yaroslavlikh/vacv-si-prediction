from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "data/molecular_activity"


def mol(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return None
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    except Exception:
        pass
    return m


def full(s):
    m = mol(s)
    return Chem.MolToInchiKey(m) if m is not None else ""


def k14(s):
    m = mol(s)
    return Chem.MolToInchiKey(m)[:14] if m is not None else ""


aug = pd.read_csv(ROOT / "exports/vacv_augmentation_dataset_v2.csv")
aug["valid_smiles"] = aug.smiles.map(lambda s: mol(s) is not None)
aug["full_key"] = aug.smiles.map(full)
aug["computed_ik14"] = aug.smiles.map(k14)
labeled = aug[aug.si.notna()].copy()
labeled["ratio"] = labeled.cc50_uM / labeled.ec50_uM
labeled["formula_abs_error"] = (labeled.ratio - labeled.si).abs()
labeled["log_si"] = np.log1p(labeled.si.clip(lower=0))
tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0
tr["target"] = y
tr["full_key"] = tr.SMILES.map(full)
te["full_key"] = te.SMILES.map(full)
train_by_key = tr.groupby("full_key").target.median()
test_keys = set(te.full_key)

# Verified manual labels are kept separate from the bulk augmentation sources.
current = pd.read_csv(OUT / "external_training_labels.csv")
bulk_sources = {"PubChem_NCGC_screen", "PubChem_retrieval", "PubChem_lit", "ChEMBL exact EC50/CC50"}
verified = current[~current.source.isin(bulk_sources)].copy()
verified_by_key = verified.groupby("key").SI.median()

summary = {
    "rows_total": len(aug), "rows_labeled": len(labeled),
    "invalid_smiles_labeled": int((~labeled.valid_smiles).sum()),
    "ik14_mismatch_labeled": int((labeled.ik14 != labeled.computed_ik14).sum()),
    "negative_si": int((labeled.si < 0).sum()), "zero_si": int((labeled.si == 0).sum()),
    "formula_max_abs_error": float(labeled.formula_abs_error.max()),
    "formula_rows_over_1e-6": int((labeled.formula_abs_error > 1e-6).sum()),
    "unique_full_keys": int(labeled.full_key.nunique()),
    "duplicate_rows": int(labeled.duplicated("full_key").sum()),
    "duplicate_keys_with_over_2x_spread": int((labeled.groupby("full_key").si.agg(lambda x: x.max() / max(x.min(), 1e-12)) > 2).sum()),
    "test_structures": int(labeled[labeled.full_key.isin(test_keys)].full_key.nunique()),
    "train_rows": int(labeled.full_key.isin(set(train_by_key.index)).sum()),
    "verified_manual_structures": len(verified_by_key),
}

source_rows = []
for source, g in labeled.groupby("source"):
    overlap = g[g.full_key.isin(train_by_key.index)].copy()
    overlap["train_si"] = overlap.full_key.map(train_by_key)
    ver = g[g.full_key.isin(verified_by_key.index)].copy()
    ver["verified_si"] = ver.full_key.map(verified_by_key)
    row = {
        "source": source, "rows": len(g), "unique_full_keys": g.full_key.nunique(),
        "doi_present": int(g.doi.notna().sum()),
        "test_unique": g[g.full_key.isin(test_keys)].full_key.nunique(),
        "train_overlap": len(overlap), "verified_overlap": len(ver),
        "si_median": float(g.si.median()), "si_p95": float(g.si.quantile(.95)),
        "si_max": float(g.si.max()), "over_10000": int((g.si > 10000).sum()),
    }
    if len(overlap):
        row.update({
            "train_log_mae": float(np.mean(np.abs(np.log1p(overlap.train_si.clip(lower=0)) - overlap.log_si))),
            "train_median_log_ratio": float(np.median(overlap.log_si - np.log1p(overlap.train_si.clip(lower=0)))),
            "train_over_2x_mismatch": int((np.abs(overlap.log_si - np.log1p(overlap.train_si.clip(lower=0))) > np.log(2)).sum()),
        })
    if len(ver):
        row.update({
            "verified_log_mae": float(np.mean(np.abs(np.log1p(ver.verified_si.clip(lower=0)) - ver.log_si))),
            "verified_over_2x_mismatch": int((np.abs(ver.log_si - np.log1p(ver.verified_si.clip(lower=0))) > np.log(2)).sum()),
        })
    source_rows.append(row)
sources = pd.DataFrame(source_rows).sort_values("source")
sources.to_csv(OUT / "augmentation_quality_by_source.csv", index=False)

train_conflicts = labeled[labeled.full_key.isin(train_by_key.index)].copy()
train_conflicts["train_si"] = train_conflicts.full_key.map(train_by_key)
train_conflicts["log_error"] = np.abs(np.log1p(train_conflicts.train_si.clip(lower=0)) - train_conflicts.log_si)
train_conflicts.sort_values("log_error", ascending=False).to_csv(OUT / "augmentation_train_conflicts.csv", index=False)
verified_conflicts = labeled[labeled.full_key.isin(verified_by_key.index)].copy()
verified_conflicts["verified_si"] = verified_conflicts.full_key.map(verified_by_key)
verified_conflicts["log_error"] = np.abs(np.log1p(verified_conflicts.verified_si.clip(lower=0)) - verified_conflicts.log_si)
verified_conflicts.sort_values("log_error", ascending=False).to_csv(OUT / "augmentation_verified_conflicts.csv", index=False)
labeled.nlargest(30, "si").to_csv(OUT / "augmentation_largest_si.csv", index=False)

report = {"summary": summary, "by_source": sources.to_dict(orient="records"),
          "note": "Quality audit only; no augmented rows were accepted into training by this script."}
(OUT / "augmentation_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(sources.to_string(index=False))
