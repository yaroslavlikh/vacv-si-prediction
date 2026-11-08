from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT
DATA = ROOT / "data/molecular_activity"
SEEDS = [42, 13, 79]


def std_mol(s):
    m = Chem.MolFromSmiles(str(s))
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    except Exception:
        pass
    return m


def key14(m): return Chem.MolToInchiKey(m)[:14]
def full_key(m): return Chem.MolToInchiKey(m)
def fp(m): return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def scaffold(m):
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        if sc.GetNumAtoms() == 0: return "ACYCLIC"
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc), isomericSmiles=False)
    except Exception:
        return "ACYCLIC"


def features(mols):
    rows = []
    for m in mols:
        arr = np.zeros(2048, np.uint8)
        DataStructs.ConvertToNumpyArray(fp(m), arr)
        maccs = np.zeros(167, np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), maccs)
        desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m),
                Descriptors.RingCount(m), Descriptors.FractionCSP3(m), Descriptors.HeavyAtomCount(m)]
        rows.append(np.concatenate([arr, maccs, desc]))
    return np.asarray(rows, dtype=np.float32)


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
# Adefovir is deliberately left as an ML prediction task, not a lower-bound lookup.
ext = ext[~ext.source.str.contains("named-drug lower bound", na=False)].reset_index(drop=True)
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0
ye = ext.SI.to_numpy(float)
tr_mols = [std_mol(s) for s in tr.SMILES]
te_mols = [std_mol(s) for s in te.SMILES]
ext_mols = [std_mol(s) for s in ext.SMILES]
tr_keys = np.array([key14(m) for m in tr_mols])
tr_full = np.array([full_key(m) for m in tr_mols])
te_full = np.array([full_key(m) for m in te_mols])
tr_scafs = np.array([scaffold(m) for m in tr_mols])
ext_keys14 = np.array([key14(m) for m in ext_mols])
ext_full = np.array([full_key(m) for m in ext_mols])
ext_scafs = np.array([scaffold(m) for m in ext_mols])
X, Xt, Xe = features(tr_mols), features(te_mols), features(ext_mols)
tr_fps, te_fps, ext_fps = [fp(m) for m in tr_mols], [fp(m) for m in te_mols], [fp(m) for m in ext_mols]


def fit_et(Xfit, yfit, wfit, seed, transform):
    model = ExtraTreesRegressor(n_estimators=60, max_features=.4, min_samples_leaf=1,
                                n_jobs=4, random_state=seed)
    target = np.log1p(yfit) if transform == "log" else yfit
    model.fit(Xfit, target, sample_weight=wfit / wfit.mean())
    return model


def predict_et(model, Xq, transform):
    pred = model.predict(Xq)
    return np.clip(np.expm1(pred) if transform == "log" else pred, 0, None)


def knn_raw(query_fps, query_full, ref_fps, ref_full, ref_y, k=5, gamma=4):
    out = np.zeros(len(query_fps))
    for i, (query, qkey) in enumerate(zip(query_fps, query_full)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, ref_fps))
        exact = np.flatnonzero((ref_full == qkey) & (sims >= .999))
        if len(exact):
            out[i] = ref_y[exact[0]]
            continue
        idx = np.argpartition(sims, -k)[-k:]
        w = np.maximum(sims[idx], 1e-6) ** gamma
        out[i] = np.sum(w * ref_y[idx]) / w.sum()
    return np.clip(out, 0, None)


reports = []
test_predictions = {}
oof_predictions = {}
for protocol, groups in [("canonical", tr_keys), ("scaffold", tr_scafs)]:
    for seed in SEEDS:
        folds = np.empty(len(y), dtype=int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups)):
            folds[va] = f
        transform = "log"
        oof = np.zeros(len(y))
        for f in range(5):
            ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            em = ~np.isin(ext_keys14, np.unique(tr_keys[vi]))
            if protocol == "scaffold":
                em &= ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
            Xfit = np.vstack([X[ti], Xe[em]])
            yfit = np.concatenate([y[ti], ye[em]])
            wfit = np.concatenate([1 + y[ti] / 100, np.full(em.sum(), .1)])
            model = fit_et(Xfit, yfit, wfit, seed, transform)
            oof[vi] = predict_et(model, X[vi], transform)
        model = fit_et(np.vstack([X, Xe]), np.concatenate([y, ye]),
                       np.concatenate([1 + y / 100, np.full(len(ye), .1)]), seed, transform)
        pt = predict_et(model, Xt, transform)
        name = "et_log"
        key = f"{protocol}_{seed}__{name}"
        oof_predictions[key] = oof
        test_predictions[key] = pt
        reports.append({"protocol": protocol, "seed": seed, "model": name, **metrics(y, oof)})
        print(protocol, seed, "done", flush=True)

np.savez_compressed(OUT / "tail_ml_oof.npz", **oof_predictions)
np.savez_compressed(OUT / "tail_ml_test.npz", **test_predictions)
reports = pd.DataFrame(reports)
reports.to_csv(OUT / "tail_ml_scores.csv", index=False)
summary = reports.groupby(["protocol", "model"]).agg(
    mse=("mse", "mean"), mse_no_max=("mse_no_max", "mean"),
    mse_no_top7=("mse_no_top7", "mean"), mae=("mae", "mean")).reset_index()
summary.to_csv(OUT / "tail_ml_summary.csv", index=False)
blockers = te[te.Title.isin(["Adefovir dipivoxil", "sid103210702"])][["index", "Title"]].copy()
for key, pred in test_predictions.items():
    blockers[key] = pred[blockers.index.to_numpy()]
blockers.to_csv(OUT / "tail_ml_blocker_predictions.csv", index=False)
print(summary.to_string(index=False))
print(blockers.to_string(index=False))
