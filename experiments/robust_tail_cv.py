from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "data/molecular_activity"
N_FOLDS = 5
SEEDS = [42, 13, 79]
tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0


def std_mol(s):
    m = Chem.MolFromSmiles(str(s))
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    except Exception:
        pass
    return m


def key(m):
    ik = Chem.MolToInchiKey(m)
    return ik[:14] if ik else ""


def scaffold(m):
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        if s.GetNumAtoms() == 0:
            return "ACYCLIC"
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(s), isomericSmiles=False)
    except Exception:
        return "ACYCLIC"


def features(mols):
    rows = []
    for m in mols:
        fp = np.zeros(2048, np.uint8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), fp)
        maccs = np.zeros(167, np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), maccs)
        desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m),
                Descriptors.RingCount(m), Descriptors.FractionCSP3(m), Descriptors.HeavyAtomCount(m)]
        rows.append(np.concatenate([fp, maccs, desc]))
    return np.asarray(rows, dtype=np.float32)


def transform(v, power):
    return np.log1p(v) if power == 0 else np.power(v + 1, power) - 1


def inverse(v, power):
    out = np.expm1(v) if power == 0 else np.power(np.maximum(v + 1, 0), 1 / power) - 1
    return np.clip(out, 0, 1e7)


def metrics(pred):
    err = (y - pred) ** 2
    keep = np.arange(len(y)) != np.argmax(y)
    keep7 = np.ones(len(y), bool)
    keep7[np.argsort(y)[-7:]] = False
    return {"mse": float(err.mean()), "mse_no_max": float(err[keep].mean()),
            "mse_no_top7": float(err[keep7].mean()), "mae": float(np.abs(y - pred).mean()),
            "max_target_pred": float(pred[np.argmax(y)])}


tr_mols = [std_mol(s) for s in tr.SMILES]
te_mols = [std_mol(s) for s in te.SMILES]
X = features(tr_mols)
Xt = features(te_mols)
tr_keys = np.array([key(m) for m in tr_mols])
te_keys = set(key(m) for m in te_mols)
tr_scafs = np.array([scaffold(m) for m in tr_mols])

pairs = pd.read_csv(DATA / "external/ec50_cc50_pairs_vacv.csv")
pairs = pairs[(pairs.ec50_bound.fillna("") == "") & (pairs.cc50_nm > 0) & (pairs.ec50_nm > 0)].copy()
pairs["si"] = pairs.cc50_nm / pairs.ec50_nm
ext_mols = [std_mol(s) for s in pairs.smiles]
pairs["key"] = [key(m) for m in ext_mols]
pairs["smi_std"] = [Chem.MolToSmiles(m) for m in ext_mols]
agg = pairs.groupby("key", as_index=False).agg(si=("si", "median"), smi_std=("smi_std", "first"))
agg = agg[(agg.key != "") & ~agg.key.isin(set(tr_keys)) & ~agg.key.isin(te_keys)].reset_index(drop=True)
ext_mols = [Chem.MolFromSmiles(s) for s in agg.smi_std]
Xe = features(ext_mols)
ye = agg.si.to_numpy(float)
ext_scafs = np.array([scaffold(m) for m in ext_mols])


def fit_base(ti, vi, seed, power, ext_weight, tail_alpha, protocol):
    Xs, ys = [X[ti]], [y[ti]]
    ws = [np.power(1 + y[ti] / 100, tail_alpha)]
    if ext_weight > 0:
        emask = np.ones(len(ye), dtype=bool)
        if protocol == "scaffold":
            emask = ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
        Xs.append(Xe[emask]); ys.append(ye[emask]); ws.append(np.full(emask.sum(), ext_weight))
    Xf, yf, wf = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    wf = wf / wf.mean()
    model = LGBMRegressor(n_estimators=450, learning_rate=.03, num_leaves=15,
                          subsample=.85, colsample_bytree=.7, reg_lambda=5,
                          verbose=-1, n_jobs=4, random_state=seed)
    model.fit(Xf, transform(yf, power), sample_weight=wf)
    return model


def run_regular(folds, seed, protocol, power, ext_weight, tail_alpha):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        model = fit_base(ti, vi, seed, power, ext_weight, tail_alpha, protocol)
        oof[vi] = inverse(model.predict(X[vi]), power)
        test_folds.append(inverse(model.predict(Xt), power))
    return oof, np.mean(test_folds, axis=0)


def run_mixture(folds, seed, protocol, threshold, class_weight, gamma):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        base = fit_base(ti, vi, seed, 0, 0, .5, protocol)
        base_v = inverse(base.predict(X[vi]), 0)
        base_t = inverse(base.predict(Xt), 0)
        tail = y[ti] >= threshold
        clf = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1, max_features=.8,
                                   class_weight=class_weight, n_jobs=-1, random_state=seed)
        clf.fit(X[ti], tail.astype(int))
        p_v = clf.predict_proba(X[vi])[:, 1] ** gamma
        p_t = clf.predict_proba(Xt)[:, 1] ** gamma
        tail_mean = float(y[ti][tail].mean())
        oof[vi] = (1 - p_v) * base_v + p_v * tail_mean
        test_folds.append((1 - p_t) * base_t + p_t * tail_mean)
    return oof, np.mean(test_folds, axis=0)


regular_configs = [
    ("log", 0, 0, 0),
    ("tailw0.5", 0, 0, .5),
    ("tailw0.75", 0, 0, .75),
    ("tailw1.0", 0, 0, 1.0),
    ("p0.05_e0.1", .05, .1, 0),
    ("p0.1_e0.05", .1, .05, 0),
    ("p0.1_e0.1", .1, .1, 0),
    ("p0.1_e0.1_w0.25", .1, .1, .25),
    ("p0.1_e0.1_w0.5", .1, .1, .5),
    ("p0.25", .25, 0, 0),
]
mixture_configs = [
    ("mix100_bal_g1", 100, "balanced", 1),
    ("mix100_none_g1", 100, None, 1),
    ("mix500_bal_g1", 500, "balanced", 1),
    ("mix500_bal_g2", 500, "balanced", 2),
    ("mix1000_bal_g1", 1000, "balanced", 1),
    ("mix1000_bal_g2", 1000, "balanced", 2),
    ("mix1000_none_g1", 1000, None, 1),
]

results = []
oof_store = {}
test_store = {}
for protocol, groups in [("canonical", tr_keys), ("scaffold", tr_scafs)]:
    for seed in SEEDS:
        folds = np.empty(len(y), dtype=int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups)):
            folds[va] = f
        print(f"\n{protocol} seed={seed} folds={np.bincount(folds).tolist()}", flush=True)
        for name, power, ew, alpha in regular_configs:
            start = time.time()
            oof, pred = run_regular(folds, seed, protocol, power, ew, alpha)
            row = {"protocol": protocol, "seed": seed, "model": name, **metrics(oof), "seconds": time.time() - start}
            results.append(row)
            oof_store[f"{protocol}_{seed}__{name}"] = oof
            test_store[f"{protocol}_{seed}__{name}"] = pred
            print(f"{name:20s} full={row['mse']:11,.0f} no_max={row['mse_no_max']:11,.0f} no7={row['mse_no_top7']:10,.0f}", flush=True)
        for name, threshold, cw, gamma in mixture_configs:
            start = time.time()
            oof, pred = run_mixture(folds, seed, protocol, threshold, cw, gamma)
            row = {"protocol": protocol, "seed": seed, "model": name, **metrics(oof), "seconds": time.time() - start}
            results.append(row)
            oof_store[f"{protocol}_{seed}__{name}"] = oof
            test_store[f"{protocol}_{seed}__{name}"] = pred
            print(f"{name:20s} full={row['mse']:11,.0f} no_max={row['mse_no_max']:11,.0f} no7={row['mse_no_top7']:10,.0f}", flush=True)
        pd.DataFrame(results).to_csv(OUT / "robust_tail_scores_by_repeat.csv", index=False)
        np.savez_compressed(OUT / "robust_tail_oof.npz", **oof_store)
        np.savez_compressed(OUT / "robust_tail_test_predictions.npz", **test_store)

scores = pd.DataFrame(results)
summary = scores.groupby(["protocol", "model"]).agg(
    mse_mean=("mse", "mean"), mse_std=("mse", "std"),
    mse_no_max_mean=("mse_no_max", "mean"), mse_no_max_std=("mse_no_max", "std"),
    mse_no_top7_mean=("mse_no_top7", "mean"), mae_mean=("mae", "mean"),
    max_target_pred_mean=("max_target_pred", "mean"),
).reset_index()
summary.to_csv(OUT / "robust_tail_scores_summary.csv", index=False)
(OUT / "robust_tail_report.json").write_text(json.dumps({
    "seeds": SEEDS, "folds": N_FOLDS, "test_targets_used": False,
    "external_novel_rows": len(ye), "results": results
}, indent=2), encoding="utf-8")
print("\nSUMMARY\n", summary.sort_values(["protocol", "mse_no_max_mean"]).to_string(index=False), flush=True)
print("DONE", flush=True)
