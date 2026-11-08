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
ALLOWED = {"ChEMBL", "PubChem_lit", "PubChem_retrieval", "paper_PMC9865139"}


def full_key(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return ""
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    except Exception:
        pass
    return Chem.MolToInchiKey(m)


aug = pd.read_csv(ROOT / "exports/vacv_augmentation_dataset_v2.csv")
initial = aug[aug.si.notna()].copy()
initial["SI"] = initial.cc50_uM / initial.ec50_uM
initial["key"] = initial.smiles.map(full_key)
initial["valid"] = initial.key != ""
excluded = {
    "unlabeled": int(aug.si.isna().sum()),
    "invalid_smiles": int((~initial.valid).sum()),
    "ncgc_unverified": int((initial.source == "PubChem_NCGC_screen").sum()),
    "other_source": int((~initial.source.isin(ALLOWED) & (initial.source != "PubChem_NCGC_screen")).sum()),
    "chembl_over_10000": int(((initial.source == "ChEMBL") & (initial.SI > 10000)).sum()),
}
filtered = initial[initial.valid & initial.source.isin(ALLOWED) & ~((initial.source == "ChEMBL") & (initial.SI > 10000))].copy()
filtered["source"] = filtered.source + "; ratio=CC50/EC50"
filtered = filtered.groupby("key", as_index=False).agg(
    SMILES=("smiles", "first"), SI=("SI", "median"),
    source=("source", lambda x: ";".join(sorted(set(x)))),
)
te = pd.read_csv(DATA / "test.csv")
test_keys = set(te.SMILES.map(full_key))
filtered.to_csv(OUT / "augmentation_filtered.csv", index=False)
report = {
    "input_labeled_rows": len(initial),
    "accepted_unique_structures": len(filtered),
    "accepted_test_structures": int(filtered[filtered.key.isin(test_keys)].key.nunique()),
    "accepted_by_source": filtered.source.str.split(";", regex=False).str[0].value_counts().to_dict(),
    "excluded": excluded,
    "rules": [
        "Exclude PubChem_NCGC_screen: no DOI and no independent train/verified label validation",
        "Use CC50/EC50 instead of rounded SI column",
        "Exclude ChEMBL SI>10000 as high-leverage assay outliers",
        "Aggregate full InChIKey by median; verified manual and original train labels take priority later",
    ],
    "training_started_by_this_script": False,
}
(OUT / "augmentation_filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
