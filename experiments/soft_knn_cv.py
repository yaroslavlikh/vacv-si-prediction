from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
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
tr_keys = np.array([key14(m) for m in tr_mols])
tr_full = np.array([full_key(m) for m in tr_mols])
te_full = np.array([full_key(m) for m in te_mols])
tr_scafs = np.array([scaffold(m) for m in tr_mols])
ext_keys14 = np.array([key14(m) for m in ext_mols])
ext_full = np.array([full_key(m) for m in ext_mols])
ext_scafs = np.array([scaffold(m) for m in ext_mols])
tr_fps = [fp(m) for m in tr_mols]
te_fps = [fp(m) for m in te_mols]
ext_fps = [fp(m) for m in ext_mols]
base = np.load(OUT / "clean_supervised_predictions.npz")
base_oof_parts = base["oof_parts"]
base_test_parts = base["final_test_parts"]
configs = [(1, 1, "raw"), (3, 4, "log"), (5, 4, "log"), (5, 4, "raw"), (5, 8, "log"), (10, 8, "log")]


def soft_predict(query_fps, query_full, ref_fps, ref_full, ref_y, k, gamma, transform):
    out = np.zeros(len(query_fps))
    for i, (query, qkey) in enumerate(zip(query_fps, query_full)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, ref_fps))
        exact = np.flatnonzero((ref_full == qkey) & (sims >= .999))
        if len(exact):
            out[i] = ref_y[exact[0]]
            continue
        kk = min(k, len(ref_y))
        idx = np.argpartition(sims, -kk)[-kk:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        w = np.maximum(sims[idx], 1e-6) ** gamma
        if transform == "raw":
            out[i] = np.sum(w * ref_y[idx]) / w.sum()
        else:
            out[i] = np.expm1(np.sum(w * np.log1p(ref_y[idx])) / w.sum())
    return np.clip(out, 0, None)


oof_store = {}
test_store = {}
repeat_rows = []
for protocol, groups in [("canonical", tr_keys), ("scaffold", tr_scafs)]:
    for seed in SEEDS:
        folds = np.empty(len(y), dtype=int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(tr, y, groups)):
            folds[va] = f
        for k, gamma, transform in configs:
            name = f"k{k}_g{gamma}_{transform}"
            oof = np.zeros(len(y))
            for f in range(5):
                ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                em = ~np.isin(ext_keys14, np.unique(tr_keys[vi]))
                if protocol == "scaffold":
                    em &= ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
                ref_fps = [tr_fps[j] for j in ti] + [ext_fps[j] for j in np.flatnonzero(em)]
                ref_full = np.concatenate([tr_full[ti], ext_full[em]])
                ref_y = np.concatenate([y[ti], ye[em]])
                oof[vi] = soft_predict([tr_fps[j] for j in vi], tr_full[vi], ref_fps, ref_full, ref_y, k, gamma, transform)
            key = f"{protocol}_{seed}__{name}"
            oof_store[key] = oof
            ref_fps = tr_fps + ext_fps
            ref_full = np.concatenate([tr_full, ext_full])
            ref_y = np.concatenate([y, ye])
            test_store[key] = soft_predict(te_fps, te_full, ref_fps, ref_full, ref_y, k, gamma, transform)
            row = {"protocol": protocol, "seed": seed, "model": name, **metrics(y, oof)}
            repeat_rows.append(row)
            print(protocol, seed, name, f"no_max={row['mse_no_max']:.1f}", flush=True)

np.savez_compressed(OUT / "soft_knn_oof.npz", **oof_store)
np.savez_compressed(OUT / "soft_knn_test.npz", **test_store)
repeats = pd.DataFrame(repeat_rows)
repeats.to_csv(OUT / "soft_knn_scores_by_repeat.csv", index=False)
selection = []
part = 0
for protocol in ["canonical", "scaffold"]:
    for seed in SEEDS:
        base_oof = base_oof_parts[part]
        part += 1
        for k, gamma, transform in configs:
            name = f"k{k}_g{gamma}_{transform}"
            soft = oof_store[f"{protocol}_{seed}__{name}"]
            for base_weight in [0, .25, .5, .75, 1.0]:
                pred = base_weight * base_oof + (1 - base_weight) * soft
                selection.append({"protocol": protocol, "seed": seed, "model": name,
                                  "base_weight": base_weight, **metrics(y, pred)})
selection = pd.DataFrame(selection)
selection.to_csv(OUT / "soft_knn_blend_selection.csv", index=False)
summary = selection.groupby(["protocol", "model", "base_weight"]).agg(
    mse=("mse", "mean"), mse_no_max=("mse_no_max", "mean"), mse_no_top7=("mse_no_top7", "mean"),
    mae=("mae", "mean")).reset_index()
summary["full_rank"] = summary.mse.rank(pct=True)
summary["no_max_rank"] = summary.mse_no_max.rank(pct=True)
summary["no7_rank"] = summary.mse_no_top7.rank(pct=True)
summary["mean_rank"] = summary[["full_rank", "no_max_rank", "no7_rank"]].mean(axis=1)
summary = summary.sort_values("mean_rank")
summary.to_csv(OUT / "soft_knn_blend_summary.csv", index=False)
best = summary.iloc[0]
protocol, model_name, base_weight = best.protocol, best.model, float(best.base_weight)
parts = []
part = 0
for p in ["canonical", "scaffold"]:
    for seed in SEEDS:
        soft = test_store[f"{p}_{seed}__{model_name}"]
        parts.append(base_weight * base_test_parts[part] + (1 - base_weight) * soft)
        part += 1
pred = np.mean(parts, axis=0)
submission = pd.DataFrame({"index": te["index"], "SI VACV": pred})
submission.to_csv(OUT / "submission_clean_soft_knn.csv", index=False)

si_true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
si_true = si_true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
report = {
    "selected_by": "mean percentile rank of repeated OOF full/no-max/no-top7 MSE",
    "protocol": protocol, "model": model_name, "base_weight": base_weight,
    "oof": {"mse": float(best.mse), "mse_no_max": float(best.mse_no_max),
            "mse_no_top7": float(best.mse_no_top7), "mae": float(best.mae)},
    "local_reconstructed_diagnostic": metrics(si_true, pred),
    "test_targets_used_for_fit_or_selection": False,
    "manual_post_hoc_replacements": False,
    "independent_evaluation": False,
    "submission": "submission_clean_soft_knn.csv",
}
(OUT / "soft_knn_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("BEST", json.dumps(report, indent=2), flush=True)
