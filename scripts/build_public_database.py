#!/usr/bin/env python3
"""Build the public SI database: SMILES, IC50, CC50, SI and DOI.

Only public measurements are used. Competition labels and test concentrations are
not read. Rounded SI columns are recomputed as CC50/IC50 where possible.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
ALLOWED = {"ChEMBL", "PubChem_lit", "PubChem_retrieval", "paper_PMC9865139"}


def full_key(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    except Exception:
        pass
    return Chem.MolToInchiKey(mol)


def record(smiles, ic50, cc50, doi, source, bound=""):
    return {"SMILES": smiles, "IC50": float(ic50), "CC50": float(cc50),
            "SI": float(cc50) / float(ic50), "DOI": doi or "",
            "source": source, "cc50_bound": bound}


pairs = pd.read_csv(OUT / "molecular_activity/external/ec50_cc50_pairs_vacv.csv")
pairs = pairs[(pairs.ec50_nm > 0) & (pairs.cc50_nm > 0)]
rows = [record(r.smiles, r.ec50_nm / 1000, r.cc50_nm / 1000, r.doi,
               "ChEMBL exact pair", r.cc50_bound) for r in pairs.itertuples(index=False)]

aug = pd.read_csv(ROOT / "exports/vacv_augmentation_dataset_v2.csv")
aug = aug[aug.si.notna() & aug.source.isin(ALLOWED)].copy()
aug["ratio"] = aug.cc50_uM / aug.ec50_uM
aug = aug[~((aug.source == "ChEMBL") & (aug.ratio > 10000))]
rows.extend(record(r.smiles, r.ec50_uM, r.cc50_uM, r.doi, r.source)
            for r in aug.itertuples(index=False))

# Wiley Table 1, DOI 10.1002/cmdc.202100771.
wiley = {
    "7a-as-520": ("CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)C1=CC=CC=C1)C2", 196.4, 15.2),
    "7d-as-522": ("CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)C1=CC=C(F)C=C1)C2", 58.1, 14.9),
    "7e-as-515": ("CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)C1=CC=C([N+](=O)[O-])C=C1)C2", 626.1, .9),
    "7h-as-517": ("CC1=CC=C(C(=O)N[C@@H]2C[C@H]3CC[C@]2(C)C3(C)C)C=C1", 47.9, 2.6),
    "7k-as-528": ("CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)C1=CN=CC=N1)C2", 170.9, 74.1),
    "7l-as-529": ("CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)C1CCCCC1)C2", 542.9, 4.3),
    "8b-as-556": ("CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O)C1=CC=C(Cl)C=C1)C2", 3210.2, .24),
    "8c-as-536": ("CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O)C1=CC=C(Br)C=C1)C2", 1582.9, .36),
    "2c": ("CC1(C)[C@@H]2CC[C@@]1(C)/C(=N/NC(=O)C1=CC=C(Br)C=C1)C2", 2904.9, 11.1),
    "3c-ks-347": ("CC1(C)/C(=N\\NC(=O)C2=CC=C(Br)C=C2)[C@]2(C)CC[C@H]1C2", 946.8, 2.2),
}
for name, (smiles, cc50, ic50) in wiley.items():
    rows.append(record(smiles, ic50, cc50, "10.1002/cmdc.202100771", f"Wiley {name}"))

rows.append(record("NC1=NC(=O)N([C@@H]2S[C@H](CO)[C@@H](O)[C@@H]2O)C=C1F",
                   .1, 4.6, "", "PMC3185742 4-thio-5-fluoro-cytidine"))
rows.append(record("C[C@H](O)C1=C[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
                   .1, 400.0, "10.1021/jm00080a018", "PubChem AID156533/AID156532", ">"))

frame = pd.DataFrame(rows)
frame["InChIKey"] = frame.SMILES.map(full_key)
frame = frame[frame.InChIKey != ""].drop_duplicates(subset=["SMILES", "IC50", "CC50", "DOI", "source", "cc50_bound"])
frame = frame[["SMILES", "IC50", "CC50", "SI", "DOI", "source", "cc50_bound", "InChIKey"]]
frame.to_csv(OUT / "public_si_database.csv", index=False)
print(f"Saved {len(frame)} public SI records to {OUT / 'public_si_database.csv'}")
print(frame.source.value_counts().to_string())
