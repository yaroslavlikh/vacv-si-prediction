from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5
tr = pd.read_csv(ROOT / "data/molecular_activity/train.csv")
te = pd.read_csv(ROOT / "data/molecular_activity/test.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0
is_zero = y == 0
cc50 = tr["CC50_uM"].to_numpy(float)
ic50 = tr["IC50_uM_VACV"].to_numpy(float)


def mol(s):
    return Chem.MolFromSmiles(str(s))


def std_key(m):
    return Chem.MolToInchiKey(m)[:14]


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
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), fp)
        maccs = np.zeros(167, np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), maccs)
        desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m),
                Descriptors.RingCount(m), Descriptors.FractionCSP3(m), Descriptors.HeavyAtomCount(m)]
        rows.append(np.concatenate([fp, maccs, desc]))
    return np.asarray(rows, dtype=np.float32)


def scaffold_folds(keys):
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    order = sorted(groups.values(), key=lambda idx: (-len(idx), min(idx)))
    sizes = [0] * N_FOLDS
    assign = np.empty(len(keys), dtype=int)
    for idx in order:
        f = int(np.argmin(sizes))
        assign[idx] = f
        sizes[f] += len(idx)
    return assign


def metrics(pred):
    err2 = (y - pred) ** 2
    order = np.argsort(y)[::-1]
    keep1 = np.ones(len(y), bool)
    keep1[order[0]] = False
    keep7 = np.ones(len(y), bool)
    keep7[order[:7]] = False
    return {
        "mse": float(err2.mean()),
        "mse_no_max": float(err2[keep1].mean()),
        "mse_no_top7": float(err2[keep7].mean()),
        "mae": float(np.abs(y - pred).mean()),
        "max_target_pred": float(pred[order[0]]),
        "max_target_squared_error": float(err2[order[0]]),
        "top7_squared_error_share": float(err2[order[:7]].sum() / err2.sum()),
    }


tr_mols = [mol(s) for s in tr.SMILES]
te_mols = [mol(s) for s in te.SMILES]
X = features(tr_mols)
Xt = features(te_mols)
tr_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in tr_mols]
te_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in te_mols]
sim_train = np.asarray([[DataStructs.TanimotoSimilarity(a, b) for b in tr_fps] for a in tr_fps], dtype=np.float32)
sim_test = np.asarray([[DataStructs.TanimotoSimilarity(a, b) for b in tr_fps] for a in te_fps], dtype=np.float32)
keys = np.array([std_key(m) for m in tr_mols])
scafs = np.array([scaffold(m) for m in tr_mols])
canonical_assign = np.empty(len(y), dtype=int)
for f, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y, keys)):
    canonical_assign[va] = f
protocols = {"canonical": canonical_assign, "scaffold": scaffold_folds(scafs)}


def make_reg(name):
    log_target = name.endswith("_log")
    base = name[:-4] if log_target else name
    if base == "et1":
        model = ExtraTreesRegressor(n_estimators=350, min_samples_leaf=1, max_features=1.0, n_jobs=-1, random_state=SEED)
    elif base == "et2":
        model = ExtraTreesRegressor(n_estimators=350, min_samples_leaf=2, max_features=.8, n_jobs=-1, random_state=SEED)
    elif base == "rf1":
        model = RandomForestRegressor(n_estimators=350, min_samples_leaf=1, max_features=.8, n_jobs=-1, random_state=SEED)
    elif base == "rf2":
        model = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, max_features=.8, n_jobs=-1, random_state=SEED)
    elif base == "hgb":
        model = HistGradientBoostingRegressor(max_iter=350, learning_rate=.04, max_leaf_nodes=15, l2_regularization=3, random_state=SEED)
    elif base == "lgb4":
        model = LGBMRegressor(n_estimators=450, learning_rate=.03, num_leaves=4, max_depth=-1,
                              subsample=.85, colsample_bytree=.7, reg_lambda=5, verbose=-1, n_jobs=4, random_state=SEED)
    elif base == "lgb15":
        model = LGBMRegressor(n_estimators=450, learning_rate=.03, num_leaves=15,
                              subsample=.85, colsample_bytree=.7, reg_lambda=5, verbose=-1, n_jobs=4, random_state=SEED)
    else:
        raise ValueError(name)
    return model, log_target


def knn_predict(train_idx, query_idx, k, power, log_target, query="train"):
    sim = sim_train[np.ix_(query_idx, train_idx)] if query == "train" else sim_test[:, train_idx]
    vals = y[train_idx]
    if query == "train":
        # query rows are validation and are disjoint from train_idx.
        pass
    kk = min(k, len(train_idx))
    top = np.argpartition(sim, -kk, axis=1)[:, -kk:]
    pred = []
    for row, inds in zip(sim, top):
        w = np.maximum(row[inds], 1e-6) ** power
        v = vals[inds]
        if log_target:
            p = np.expm1(np.sum(w * np.log1p(v)) / w.sum())
        else:
            p = np.sum(w * v) / w.sum()
        pred.append(max(0.0, p))
    return np.asarray(pred)


def run_knn(folds, k, power, log_target):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti = np.flatnonzero(folds != f)
        vi = np.flatnonzero(folds == f)
        oof[vi] = knn_predict(ti, vi, k, power, log_target)
        test_folds.append(knn_predict(ti, np.arange(len(te)), k, power, log_target, query="test"))
    return oof, np.mean(test_folds, axis=0)


def run_reg(folds, name):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti = np.flatnonzero(folds != f)
        vi = np.flatnonzero(folds == f)
        model, log_target = make_reg(name)
        model.fit(X[ti], np.log1p(y[ti]) if log_target else y[ti])
        pv = model.predict(X[vi])
        pt = model.predict(Xt)
        if log_target:
            pv, pt = np.expm1(pv), np.expm1(pt)
        oof[vi] = np.clip(pv, 0, None)
        test_folds.append(np.clip(pt, 0, None))
    return oof, np.mean(test_folds, axis=0)


def run_two_head(folds, reg_name):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti = np.flatnonzero(folds != f)
        vi = np.flatnonzero(folds == f)
        active = ti[~is_zero[ti]]
        clf = ExtraTreesClassifier(n_estimators=350, class_weight="balanced", n_jobs=-1, random_state=SEED)
        clf.fit(X[ti], is_zero[ti].astype(int))
        reg, log_target = make_reg(reg_name)
        reg.fit(X[active], np.log1p(y[active]) if log_target else y[active])
        pv = reg.predict(X[vi])
        pt = reg.predict(Xt)
        if log_target:
            pv, pt = np.expm1(pv), np.expm1(pt)
        oof[vi] = (1 - clf.predict_proba(X[vi])[:, 1]) * np.clip(pv, 0, None)
        test_folds.append((1 - clf.predict_proba(Xt)[:, 1]) * np.clip(pt, 0, None))
    return oof, np.mean(test_folds, axis=0)


def run_endpoints(folds):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti = np.flatnonzero(folds != f)
        vi = np.flatnonzero(folds == f)
        active = ti[~is_zero[ti]]
        cc_idx = ti[np.isfinite(cc50[ti]) & (cc50[ti] > 0)]
        clf = ExtraTreesClassifier(n_estimators=350, class_weight="balanced", n_jobs=-1, random_state=SEED)
        clf.fit(X[ti], is_zero[ti].astype(int))
        cc_model = ExtraTreesRegressor(n_estimators=350, max_features=.8, n_jobs=-1, random_state=SEED)
        ic_model = ExtraTreesRegressor(n_estimators=350, max_features=.8, n_jobs=-1, random_state=SEED)
        cc_model.fit(X[cc_idx], np.log(cc50[cc_idx]))
        ic_model.fit(X[active], np.log(ic50[active]))
        for query, dest in ((vi, "oof"), (np.arange(len(te)), "test")):
            Xq = X[query] if dest == "oof" else Xt
            ratio = np.exp(cc_model.predict(Xq) - ic_model.predict(Xq))
            pred = (1 - clf.predict_proba(Xq)[:, 1]) * np.clip(ratio, 0, None)
            if dest == "oof":
                oof[query] = pred
            else:
                test_folds.append(pred)
    return oof, np.mean(test_folds, axis=0)


configs = []
for k in [1, 3, 5, 10]:
    for power in ([0] if k == 1 else [2, 6]):
        for log_target in [False, True]:
            configs.append(("knn", f"knn{k}_p{power}" + ("_log" if log_target else ""), (k, power, log_target)))
for name in ["et1", "et1_log", "et2", "et2_log", "rf1", "rf1_log", "rf2", "rf2_log", "hgb", "hgb_log", "lgb4", "lgb4_log", "lgb15", "lgb15_log"]:
    configs.append(("reg", name, name))
for name in ["et1", "et1_log", "lgb4", "lgb4_log"]:
    configs.append(("two_head", "two_" + name, name))
configs.append(("endpoints", "endpoints_log", None))

all_results = []
test_predictions = {}
oof_store = {}
for protocol, folds in protocols.items():
    print(f"\nPROTOCOL {protocol}; folds={np.bincount(folds).tolist()}", flush=True)
    for kind, name, params in configs:
        start = time.time()
        if kind == "knn":
            oof, pred = run_knn(folds, *params)
        elif kind == "reg":
            oof, pred = run_reg(folds, params)
        elif kind == "two_head":
            oof, pred = run_two_head(folds, params)
        else:
            oof, pred = run_endpoints(folds)
        result = {"protocol": protocol, "model": name, **metrics(oof), "seconds": time.time() - start}
        all_results.append(result)
        key = f"{protocol}__{name}"
        test_predictions[key] = pred
        oof_store[key] = oof
        print(f"{name:20s} MSE={result['mse']:12,.0f} no_max={result['mse_no_max']:12,.0f} no_top7={result['mse_no_top7']:10,.0f} MAE={result['mae']:8.1f} ({result['seconds']:.1f}s)", flush=True)
        pd.DataFrame(all_results).sort_values(["protocol", "mse_no_max"]).to_csv(OUT / "baseline_cv_scores.csv", index=False)
        np.savez_compressed(OUT / "baseline_oof.npz", **oof_store)
        np.savez_compressed(OUT / "baseline_test_predictions.npz", **test_predictions)

summary = {
    "labels": "original train with missing IC50 mapped to SI=0; no test targets read",
    "n_train": len(tr), "n_test": len(te), "n_folds": N_FOLDS, "seed": SEED,
    "protocols": {k: np.bincount(v).tolist() for k, v in protocols.items()},
    "results": all_results,
}
(OUT / "baseline_cv_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nDONE", flush=True)
