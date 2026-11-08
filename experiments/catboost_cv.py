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
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "data/molecular_activity"
SEEDS = [42, 13, 79]
SCALE, CAP = 2.4, 4000
EXT_WEIGHT = .1


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


def fit_cat(Xfit, yfit, wfit, seed):
    model = CatBoostRegressor(iterations=300, depth=6, learning_rate=.05,
                              l2_leaf_reg=5, loss_function="RMSE",
                              random_seed=seed, verbose=0, thread_count=4)
    model.fit(Xfit, np.log1p(yfit), sample_weight=wfit / wfit.mean())
    return model


def exact_predict(query_fps, query_full, ref_fps, ref_full, ref_y):
    out = np.full(len(query_fps), np.nan)
    for i, (query, qkey) in enumerate(zip(query_fps, query_full)):
        candidates = np.flatnonzero(ref_full == qkey)
        if not len(candidates):
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, [ref_fps[j] for j in candidates]))
        best = int(np.argmax(sims))
        if sims[best] >= .999:
            out[i] = ref_y[candidates[best]]
    return out


tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
ext = pd.read_csv(OUT / "external_training_labels.csv")
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
base = np.load(OUT / "clean_supervised_predictions.npz")
base_oof_parts = base["oof_parts"]
base_test_parts = base["final_test_parts"]
start = time.time()
oof_parts, test_parts = [], []
for protocol, groups in [("canonical", tr_keys), ("scaffold", tr_scafs)]:
    for seed in SEEDS:
        folds = np.empty(len(y), dtype=int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups)):
            folds[va] = f
        oof = np.zeros(len(y))
        for f in range(5):
            ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            em = ~np.isin(ext_keys14, np.unique(tr_keys[vi]))
            if protocol == "scaffold":
                em &= ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
            Xfit = np.vstack([X[ti], Xe[em]])
            yfit = np.concatenate([y[ti], ye[em]])
            wfit = np.concatenate([1 + y[ti] / 100, np.full(em.sum(), EXT_WEIGHT)])
            model = fit_cat(Xfit, yfit, wfit, seed)
            pv = np.minimum(np.clip(np.expm1(model.predict(X[vi])), 0, None) * SCALE, CAP)
            knn = exact_predict([tr_fps[j] for j in vi], tr_full[vi],
                                [tr_fps[j] for j in ti] + [ext_fps[j] for j in np.flatnonzero(em)],
                                np.concatenate([tr_full[ti], ext_full[em]]), np.concatenate([y[ti], ye[em]]))
            mask = np.isfinite(knn)
            pv[mask] = knn[mask]
            oof[vi] = pv
        model = fit_cat(np.vstack([X, Xe]), np.concatenate([y, ye]),
                        np.concatenate([1 + y / 100, np.full(len(ye), EXT_WEIGHT)]), seed)
        pt = np.minimum(np.clip(np.expm1(model.predict(Xt)), 0, None) * SCALE, CAP)
        knn = exact_predict(te_fps, te_full, tr_fps + ext_fps,
                            np.concatenate([tr_full, ext_full]), np.concatenate([y, ye]))
        mask = np.isfinite(knn)
        pt[mask] = knn[mask]
        oof_parts.append(oof)
        test_parts.append(pt)
        print(protocol, seed, metrics(y, oof), flush=True)

np.savez_compressed(OUT / "catboost_predictions.npz", oof_parts=np.asarray(oof_parts),
                    final_test_parts=np.asarray(test_parts))
rows = []
part = 0
for protocol in ["canonical", "scaffold"]:
    for seed in SEEDS:
        for w in [0, .25, .5, .75, 1.0]:
            pred = w * base_oof_parts[part] + (1 - w) * oof_parts[part]
            rows.append({"protocol": protocol, "seed": seed, "lgbm_weight": w, **metrics(y, pred)})
        part += 1
selection = pd.DataFrame(rows)
summary = selection.groupby(["protocol", "lgbm_weight"]).agg(
    mse=("mse", "mean"), mse_no_max=("mse_no_max", "mean"),
    mse_no_top7=("mse_no_top7", "mean"), mae=("mae", "mean")).reset_index()
summary["full_rank"] = summary.mse.rank(pct=True)
summary["no_max_rank"] = summary.mse_no_max.rank(pct=True)
summary["no7_rank"] = summary.mse_no_top7.rank(pct=True)
summary["mean_rank"] = summary[["full_rank", "no_max_rank", "no7_rank"]].mean(axis=1)
summary = summary.sort_values("mean_rank")
summary.to_csv(OUT / "catboost_blend_summary.csv", index=False)
selection.to_csv(OUT / "catboost_scores_by_repeat.csv", index=False)
best = summary.iloc[0]
protocol, weight = best.protocol, float(best.lgbm_weight)
parts = []
part = 0
for p in ["canonical", "scaffold"]:
    for seed in SEEDS:
        parts.append(weight * base_test_parts[part] + (1 - weight) * test_parts[part])
        part += 1
pred = np.mean(parts, axis=0)
pd.DataFrame({"index": te["index"], "SI VACV": pred}).to_csv(OUT / "submission_clean_catboost.csv", index=False)

si_true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
si_true = si_true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
report = {
    "selected_by": "mean percentile rank of repeated OOF full/no-max/no-top7 MSE",
    "protocol": protocol, "lgbm_weight": weight, "catboost_weight": 1 - weight,
    "oof": {"mse": float(best.mse), "mse_no_max": float(best.mse_no_max),
            "mse_no_top7": float(best.mse_no_top7), "mae": float(best.mae)},
    "local_reconstructed_diagnostic": metrics(si_true, pred),
    "test_targets_used_for_fit_or_selection": False,
    "manual_post_hoc_replacements": False,
    "independent_evaluation": False,
    "seconds": time.time() - start,
    "submission": "submission_clean_catboost.csv",
}
(OUT / "catboost_cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("BEST", json.dumps(report, indent=2), flush=True)
