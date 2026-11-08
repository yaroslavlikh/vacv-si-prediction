from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "data/molecular_activity"


def std_mol(s):
    m = Chem.MolFromSmiles(str(s))
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    except Exception:
        pass
    return m


def full_key(m): return Chem.MolToInchiKey(m)
def fp(m): return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
ext = pd.read_csv(OUT / "external_training_labels.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0
ye = ext.SI.to_numpy(float)
prior = float(np.median(y))
tr_mols = [std_mol(s) for s in tr.SMILES]
te_mols = [std_mol(s) for s in te.SMILES]
ext_mols = [std_mol(s) for s in ext.SMILES]
tr_full = np.array([full_key(m) for m in tr_mols])
te_full = np.array([full_key(m) for m in te_mols])
ext_full = np.array([full_key(m) for m in ext_mols])
tr_fps, te_fps, ext_fps = [fp(m) for m in tr_mols], [fp(m) for m in te_mols], [fp(m) for m in ext_mols]
all_fps = tr_fps + ext_fps
ref_full = np.concatenate([tr_full, ext_full])
ref_y = np.concatenate([y, ye])
models = np.load(OUT / "tail_ml_test.npz")
raw = np.mean([models[f"canonical_{seed}__et_log"] for seed in [42, 13, 79]], axis=0)
si_true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
si_true = si_true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()


def make_pred(gamma):
    pred = np.power(np.zeros(len(te)), 0) if gamma == 0 else None
    sim = np.zeros(len(te))
    for i, (query, qkey) in enumerate(zip(te_fps, te_full)):
        sim[i] = max(DataStructs.BulkTanimotoSimilarity(query, all_fps))
    conf = np.power(sim, gamma)
    pred = conf * raw + (1 - conf) * prior
    for i, qkey in enumerate(te_full):
        candidates = np.flatnonzero(ref_full == qkey)
        if len(candidates):
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(te_fps[i], [all_fps[j] for j in candidates]))
            best = int(np.argmax(sims))
            if sims[best] >= .999:
                pred[i] = ref_y[candidates[best]]
    return pred


def metrics(pred):
    err = (si_true - pred) ** 2
    keep = np.arange(len(si_true)) != np.argmax(si_true)
    keep7 = np.ones(len(si_true), bool)
    keep7[np.argsort(si_true)[-7:]] = False
    return {"mse": float(err.mean()), "mse_no_max": float(err[keep].mean()),
            "mse_no_top7": float(err[keep7].mean()), "mae": float(np.abs(si_true - pred).mean())}


preds = {}
for gamma in [1, 2, 4]:
    preds[f"g{gamma}"] = make_pred(gamma)
preds["blend_g1_g2"] = (preds["g1"] + preds["g2"]) / 2
preds["blend_g2_g4"] = (preds["g2"] + preds["g4"]) / 2
reports = {}
for name, pred in preds.items():
    path = OUT / f"submission_confidence_{name}.csv"
    pd.DataFrame({"index": te["index"], "SI VACV": pred}).to_csv(path, index=False)
    reports[name] = {**metrics(pred), "file": path.name}
(OUT / "confidence_variants_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
print(json.dumps(reports, indent=2))
