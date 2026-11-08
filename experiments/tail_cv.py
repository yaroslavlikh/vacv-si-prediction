from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "data/molecular_activity"
SEED = 42
N_FOLDS = 5
tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0


def mol(s):
    return Chem.MolFromSmiles(str(s))


def standardize(m):
    try:
        m = rdMolStandardize.FragmentParent(m)
        return Chem.MolFromSmiles(Chem.MolToSmiles(m))
    except Exception:
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
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    sizes = [0] * N_FOLDS
    assign = np.empty(len(keys), dtype=int)
    for idx in sorted(groups.values(), key=lambda z: (-len(z), min(z))):
        f = int(np.argmin(sizes))
        assign[idx] = f
        sizes[f] += len(idx)
    return assign


def transform(vals, power):
    return np.log1p(vals) if power == 0 else np.power(vals + 1, power) - 1


def inverse(vals, power):
    if power == 0:
        out = np.expm1(vals)
    else:
        out = np.power(np.maximum(vals + 1, 0), 1 / power) - 1
    return np.clip(out, 0, 1e7)


def metrics(pred):
    err = (y - pred) ** 2
    keep = np.arange(len(y)) != np.argmax(y)
    top7 = np.argsort(y)[-7:]
    return {"mse": float(err.mean()), "mse_no_max": float(err[keep].mean()),
            "mae": float(np.abs(y - pred).mean()), "max_target_pred": float(pred[np.argmax(y)]),
            "top7_squared_error_share": float(err[top7].sum() / err.sum())}


tr_mols = [standardize(mol(s)) for s in tr.SMILES]
te_mols = [standardize(mol(s)) for s in te.SMILES]
tr_keys = np.array([key(m) for m in tr_mols])
te_keys = set(key(m) for m in te_mols)
tr_scafs = np.array([scaffold(m) for m in tr_mols])
X = features(tr_mols)
Xt = features(te_mols)
tr_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in tr_mols]
te_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in te_mols]
sim_train = np.asarray([[DataStructs.TanimotoSimilarity(a, b) for b in tr_fps] for a in tr_fps], dtype=np.float32)
sim_test = np.asarray([[DataStructs.TanimotoSimilarity(a, b) for b in tr_fps] for a in te_fps], dtype=np.float32)

# Only public structures distinct from every competition train/test structure.
pairs = pd.read_csv(DATA / "external/ec50_cc50_pairs_vacv.csv")
pairs = pairs[(pairs.ec50_bound.fillna("") == "") & (pairs.cc50_nm > 0) & (pairs.ec50_nm > 0)].copy()
pairs["si"] = pairs.cc50_nm / pairs.ec50_nm
ext_mols = [standardize(mol(s)) for s in pairs.smiles]
pairs["key"] = [key(m) if m is not None else "" for m in ext_mols]
pairs["smi_std"] = [Chem.MolToSmiles(m) if m is not None else "" for m in ext_mols]
agg = pairs.groupby("key", as_index=False).agg(si=("si", "median"), smi_std=("smi_std", "first"))
agg = agg[(agg.key != "") & ~agg.key.isin(set(tr_keys)) & ~agg.key.isin(te_keys)].reset_index(drop=True)
ext_mols = [mol(s) for s in agg.smi_std]
Xe = features(ext_mols)
ye = agg.si.to_numpy(float)
ext_scafs = np.array([scaffold(m) for m in ext_mols])
print("External novel:", len(ye), "high>1000:", int((ye > 1000).sum()), flush=True)

canonical = np.empty(len(y), dtype=int)
for f, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y, tr_keys)):
    canonical[va] = f
protocols = {"canonical": canonical, "scaffold": scaffold_folds(tr_scafs)}


def run_lgb(folds, power, ext_weight, weight_alpha=0.0):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti = np.flatnonzero(folds != f)
        vi = np.flatnonzero(folds == f)
        Xs, ys = [X[ti]], [y[ti]]
        ws = [np.power(1 + y[ti] / 100, weight_alpha)]
        if ext_weight > 0:
            emask = np.ones(len(ye), dtype=bool)
            if folds is protocols["scaffold"]:
                emask = ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
            Xs.append(Xe[emask])
            ys.append(ye[emask])
            ws.append(np.full(emask.sum(), ext_weight))
        Xf = np.vstack(Xs)
        yf = np.concatenate(ys)
        wf = np.concatenate(ws)
        wf = wf / wf.mean()
        model = LGBMRegressor(n_estimators=450, learning_rate=.03, num_leaves=15,
                              subsample=.85, colsample_bytree=.7, reg_lambda=5,
                              verbose=-1, n_jobs=4, random_state=SEED)
        model.fit(Xf, transform(yf, power), sample_weight=wf)
        oof[vi] = inverse(model.predict(X[vi]), power)
        test_folds.append(inverse(model.predict(Xt), power))
    return oof, np.mean(test_folds, axis=0)


def run_rf(folds, power, ext_weight):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti = np.flatnonzero(folds != f)
        vi = np.flatnonzero(folds == f)
        Xs, ys, ws = [X[ti]], [y[ti]], [np.ones(len(ti))]
        if ext_weight > 0:
            emask = np.ones(len(ye), dtype=bool)
            if folds is protocols["scaffold"]:
                emask = ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
            Xs.append(Xe[emask]); ys.append(ye[emask]); ws.append(np.full(emask.sum(), ext_weight))
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_features=.8,
                                      n_jobs=-1, random_state=SEED)
        model.fit(np.vstack(Xs), transform(np.concatenate(ys), power), sample_weight=np.concatenate(ws))
        oof[vi] = inverse(model.predict(X[vi]), power)
        test_folds.append(inverse(model.predict(Xt), power))
    return oof, np.mean(test_folds, axis=0)


def run_knn(folds, k, sim_power, target_power):
    oof = np.zeros(len(y))
    test_folds = []
    for f in range(N_FOLDS):
        ti = np.flatnonzero(folds != f)
        vi = np.flatnonzero(folds == f)
        for query, dest in ((vi, "oof"), (np.arange(len(te)), "test")):
            sim = sim_train[np.ix_(query, ti)] if dest == "oof" else sim_test[:, ti]
            top = np.argpartition(sim, -k, axis=1)[:, -k:]
            pred = []
            for row, inds in zip(sim, top):
                w = np.maximum(row[inds], 1e-6) ** sim_power
                z = transform(y[ti][inds], target_power)
                pred.append(inverse(np.sum(w * z) / w.sum(), target_power))
            if dest == "oof":
                oof[query] = pred
            else:
                test_folds.append(np.asarray(pred))
    return oof, np.mean(test_folds, axis=0)


configs = []
for power in [.1, .25, .4, .5, .65, .8]:
    for ew in [0, .1, .3]:
        configs.append(("lgb", f"lgb_p{power}_e{ew}", (power, ew, 0.0)))
for alpha in [.25, .5]:
    configs.append(("lgb", f"lgb_log_tailw{alpha}", (0, 0, alpha)))
for power in [.25, .5]:
    for ew in [0, .1]:
        configs.append(("rf", f"rf_p{power}_e{ew}", (power, ew)))
for power in [.1, .25, .5]:
    configs.append(("knn", f"knn3_p2_t{power}", (3, 2, power)))
    configs.append(("knn", f"knn3_p6_t{power}", (3, 6, power)))

results = []
oof_store = {}
test_store = {}
for protocol, folds in protocols.items():
    print("\nPROTOCOL", protocol, flush=True)
    for kind, name, params in configs:
        start = time.time()
        if kind == "lgb":
            oof, pred = run_lgb(folds, *params)
        elif kind == "rf":
            oof, pred = run_rf(folds, *params)
        else:
            oof, pred = run_knn(folds, *params)
        row = {"protocol": protocol, "model": name, **metrics(oof), "seconds": time.time() - start}
        results.append(row)
        key_name = protocol + "__" + name
        oof_store[key_name] = oof
        test_store[key_name] = pred
        print(f"{name:22s} MSE={row['mse']:12,.0f} no_max={row['mse_no_max']:12,.0f} MAE={row['mae']:8.1f} maxpred={row['max_target_pred']:9.1f} ({row['seconds']:.1f}s)", flush=True)
        pd.DataFrame(results).sort_values(["protocol", "mse_no_max"]).to_csv(OUT / "tail_cv_scores.csv", index=False)
        np.savez_compressed(OUT / "tail_oof.npz", **oof_store)
        np.savez_compressed(OUT / "tail_test_predictions.npz", **test_store)

report = {"labels": "corrected missing IC50 -> SI=0", "test_targets_used": False,
          "external": {"source": "ec50_cc50_pairs_vacv.csv exact EC50", "novel_rows": len(ye),
                       "all_train_and_test_structures_excluded": True}, "results": results}
(OUT / "tail_cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("DONE", flush=True)
