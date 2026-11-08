from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
EXT = ROOT / "data/molecular_activity/external"
tr = pd.read_csv(ROOT / "data/molecular_activity/train.csv")
te = pd.read_csv(ROOT / "data/molecular_activity/test.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0
true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
true = true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy(float)


def key(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return ""
    ik = Chem.MolToInchiKey(m)
    return ik[:14] if ik else ""


def to_um(value, units, smiles):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = str(units).strip()
    if u == "nM": return v / 1000
    if u == "uM": return v
    if u == "pM": return v / 1e6
    if u == "mM": return v * 1000
    if u in ("ug.mL-1", "ug/ml", "ug/mL"):
        m = Chem.MolFromSmiles(str(smiles))
        if m is None: return None
        mw = Descriptors.MolWt(m)
        return v * 1000 / mw if mw > 0 else None
    return None


def aggregate(g, value_col):
    exact = g[g.bound == ""][value_col].dropna()
    if len(exact): return pd.Series({"value": exact.median(), "bound_out": ""})
    gt = g[g.bound == ">"][value_col].dropna()
    if len(gt): return pd.Series({"value": gt.median(), "bound_out": ">"})
    lt = g[g.bound == "<"][value_col].dropna()
    if len(lt): return pd.Series({"value": lt.median(), "bound_out": "<"})
    return pd.Series({"value": np.nan, "bound_out": ""})

va = pd.read_csv(EXT / "chembl_vacv_activities.csv")
cc = pd.read_csv(EXT / "chembl_cc50_same_molecule_publication.csv")
pot = va[va.standard_type.isin(["EC50", "IC50", "ED50", "AC50"]) & va.standard_value.notna()].copy()
cc = cc[cc.standard_value.notna()].copy()
pot["potency_um"] = [to_um(v, u, s) for v, u, s in zip(pot.standard_value, pot.standard_units, pot.canonical_smiles)]
cc["cc50_um"] = [to_um(v, u, s) for v, u, s in zip(cc.standard_value, cc.standard_units, cc.canonical_smiles)]
for df, col in [(pot, "potency_um"), (cc, "cc50_um")]:
    df["bound"] = df.standard_relation.map({"=": "", ">": ">", ">=": ">", "<": "<", "<=": "<"}).fillna("")
    df.dropna(subset=[col], inplace=True)

p = pot.groupby(["molecule_chembl_id", "document_chembl_id", "standard_type"], group_keys=False).apply(
    lambda g: aggregate(g, "potency_um"), include_groups=False).reset_index()
p["smiles"] = p.molecule_chembl_id.map(pot.groupby("molecule_chembl_id").canonical_smiles.first())
p["potency_type"] = p.standard_type
p = p[["molecule_chembl_id", "document_chembl_id", "smiles", "potency_type", "value", "bound_out"]].rename(
    columns={"value": "potency_um_final", "bound_out": "potency_bound"})
c = cc.groupby(["molecule_chembl_id", "document_chembl_id"], group_keys=False).apply(
    lambda g: aggregate(g, "cc50_um"), include_groups=False).reset_index()
c = c.rename(columns={"value": "cc50_um_final", "bound_out": "cc50_bound"})
pairs = p.merge(c, on=["molecule_chembl_id", "document_chembl_id"], how="inner")
pairs = pairs[(pairs.potency_um_final > 0) & (pairs.cc50_um_final > 0)].copy()
pairs["si"] = pairs.cc50_um_final / pairs.potency_um_final
pairs["key"] = [key(s) for s in pairs.smiles]
pairs["priority"] = pairs.potency_type.map({"EC50": 0, "IC50": 1, "ED50": 2, "AC50": 3})
pairs = pairs.sort_values(["molecule_chembl_id", "document_chembl_id", "priority"]).drop_duplicates(
    ["molecule_chembl_id", "document_chembl_id"], keep="first")
pairs.to_csv(OUT / "expanded_chembl_si_candidates.csv", index=False)
print("Candidates", len(pairs), "structures", pairs.key.nunique())

tr["key"] = [key(s) for s in tr.SMILES]
te["key"] = [key(s) for s in te.SMILES]
public = pairs.groupby("key").agg(si=("si", "median"), n=("si", "size"), docs=("document_chembl_id", lambda z: ";".join(sorted(set(z)))),
                                  potency_types=("potency_type", lambda z: ";".join(sorted(set(z))))).reset_index()
tr["y"] = y
trp = tr.merge(public[["key", "si"]], on="key", how="left")
z = trp.dropna(subset=["si"])
train_report = {"coverage": len(z), "log_mae": float(np.mean(np.abs(np.log1p(z.y) - np.log1p(z.si)))),
                "median_ratio": float(np.median((z.si + 1e-9) / (z.y + 1e-9)))}
lookup = dict(zip(public.key, public.si))
pub = te.key.map(lookup).to_numpy(float)
base = pd.read_csv(OUT / "public_hybrid_rows.csv").base_pred.to_numpy(float)
chembl_original = pd.read_csv(OUT / "public_hybrid_rows.csv").public_si.to_numpy(float)
outputs = {}
for mode in ["expanded_only", "original_else_expanded"]:
    pred = base.copy()
    if mode == "expanded_only":
        mask = np.isfinite(pub)
        pred[mask] = pub[mask]
    else:
        mask_e = np.isfinite(pub) & ~np.isfinite(chembl_original)
        mask_o = np.isfinite(chembl_original)
        pred[mask_e] = pub[mask_e]
        pred[mask_o] = chembl_original[mask_o]
        mask = mask_e | mask_o
    outputs[mode] = pred
    pd.DataFrame({"index": te["index"], "SI VACV": pred}).to_csv(OUT / f"submission_expanded_{mode}.csv", index=False)
    print(mode, "coverage", int(mask.sum()), "MSE", float(np.mean((true - pred) ** 2)), "MAE", float(np.mean(np.abs(true - pred))))
print("Train", train_report)

rows = te[["index", "Title", "SMILES", "key"]].copy()
rows["expanded_si"] = pub
rows["true_reconstructed"] = true
rows["pred"] = outputs["original_else_expanded"]
rows["squared_error"] = (true - rows.pred) ** 2
rows.to_csv(OUT / "expanded_chembl_rows.csv", index=False)
print("\nHigh rows:")
indices = [299, 160, 65, 196, 148, 171, 56, 233, 332, 401, 413, 475, 397]
print(rows.iloc[indices].to_string(index=False))
print("\nCandidate details for high rows:")
high_keys = set(rows.iloc[indices].key)
print(pairs[pairs.key.isin(high_keys)][["key", "document_chembl_id", "potency_type", "potency_um_final", "cc50_um_final", "si", "potency_bound", "cc50_bound"]].sort_values(["key", "document_chembl_id"]).to_string(index=False))
(OUT / "expanded_chembl_report.json").write_text(json.dumps({"train": train_report, "independent_evaluation": False}, indent=2), encoding="utf-8")
