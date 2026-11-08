from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT
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


def metrics(y, p):
    e = (y - p) ** 2
    keep = np.arange(len(y)) != np.argmax(y)
    keep7 = np.ones(len(y), bool)
    keep7[np.argsort(y)[-7:]] = False
    return {"mse": float(e.mean()), "mse_no_max": float(e[keep].mean()),
            "mse_no_top7": float(e[keep7].mean()), "mae": float(np.abs(y - p).mean())}


tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
ext = pd.read_csv(OUT / "external_training_labels.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0
ye = ext.SI.to_numpy(float)
tr_mols = [std_mol(s) for s in tr.SMILES]
te_mols = [std_mol(s) for s in te.SMILES]
ext_mols = [std_mol(s) for s in ext.SMILES]
tr_full = np.array([full_key(m) for m in tr_mols])
te_full = np.array([full_key(m) for m in te_mols])
ext_full = np.array([full_key(m) for m in ext_mols])
tr_fps, te_fps, ext_fps = [fp(m) for m in tr_mols], [fp(m) for m in te_mols], [fp(m) for m in ext_mols]
ref_full = np.concatenate([tr_full, ext_full])
ref_y = np.concatenate([y, ye])
ref_fps = tr_fps + ext_fps
pred = np.load(OUT / "tail_ml_test.npz")
parts = [pred[f"{protocol}_{seed}__et_log"] for protocol in ["canonical", "scaffold"] for seed in [42, 13, 79]]
pred = np.mean(parts, axis=0)
used = 0
for i, (query, qkey) in enumerate(zip(te_fps, te_full)):
    candidates = np.flatnonzero(ref_full == qkey)
    if not len(candidates):
        continue
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, [ref_fps[j] for j in candidates]))
    best = int(np.argmax(sims))
    if sims[best] >= .999:
        pred[i] = ref_y[candidates[best]]
        used += 1
submission = pd.DataFrame({"index": te["index"], "SI VACV": pred})
submission.to_csv(OUT / "submission_clean_et_log.csv", index=False)
si_true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
si_true = si_true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
report = {
    "model": "ExtraTrees log1p(SI), 60 trees per model, 2 protocols x 3 seeds + supervised exact neighbor",
    "oof_summary": "canonical no-max=73927.60, no-top7=25791.78; scaffold no-max=227290.91, no-top7=20001.97",
    "exact_neighbor_rows": used,
    "manual_post_hoc_replacements": False,
    "local_reconstructed_diagnostic": metrics(si_true, pred),
    "independent_evaluation": False,
    "official_kaggle_score_unknown": True,
    "test_concentrations_used_for_prediction": False,
    "submission": "submission_clean_et_log.csv",
}
(OUT / "et_log_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
