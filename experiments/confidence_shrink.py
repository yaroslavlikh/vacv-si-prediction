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
tr_fps, te_fps, ext_fps = [fp(m) for m in tr_mols], [fp(m) for m in te_mols], [fp(m) for m in ext_mols]
models = np.load(OUT / "tail_ml_oof.npz")
tests = np.load(OUT / "tail_ml_test.npz")


def uncertainty(query_fps, query_full, ref_fps, ref_full):
    sim_out = np.zeros(len(query_fps))
    exact = np.zeros(len(query_fps), dtype=bool)
    for i, (query, qkey) in enumerate(zip(query_fps, query_full)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, ref_fps))
        sim_out[i] = sims.max()
        exact[i] = np.any((ref_full == qkey) & (sims >= .999))
    return sim_out, exact


sim_oof_parts = []
exact_test = None
sim_test = None
for protocol, groups in [("canonical", tr_keys), ("scaffold", tr_scafs)]:
    for seed in SEEDS:
        folds = np.empty(len(y), dtype=int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(tr, y, groups)):
            folds[va] = f
        sim = np.zeros(len(y))
        for f in range(5):
            ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            em = ~np.isin(ext_keys14, np.unique(tr_keys[vi]))
            if protocol == "scaffold":
                em &= ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
            ref_fps = [tr_fps[j] for j in ti] + [ext_fps[j] for j in np.flatnonzero(em)]
            ref_full = np.concatenate([tr_full[ti], ext_full[em]])
            sim[vi], _ = uncertainty([tr_fps[j] for j in vi], tr_full[vi], ref_fps, ref_full)
        sim_oof_parts.append(sim)
        print(protocol, seed, "similarity done", flush=True)

sim_test, exact_test = uncertainty(te_fps, te_full, tr_fps + ext_fps, np.concatenate([tr_full, ext_full]))
prior = float(np.median(y))
rows = []
part = 0
for protocol in ["canonical", "scaffold"]:
    for seed in SEEDS:
        raw = models[f"{protocol}_{seed}__et_log"]
        sim = sim_oof_parts[part]
        for gamma in [0, .5, 1, 2, 4, 8]:
            for beta in [0, 1, 4]:
                conf = np.power(sim, gamma) / (1 + beta * np.std(np.vstack([models[f"{p}_{s}__et_log"] for p in ["canonical", "scaffold"] for s in SEEDS]), axis=0))
                pred = conf * raw + (1 - conf) * prior
                rows.append({"protocol": protocol, "seed": seed, "gamma": gamma, "beta": beta, **metrics(y, pred)})
        part += 1
selection = pd.DataFrame(rows)
summary = selection.groupby(["protocol", "gamma", "beta"]).agg(
    mse=("mse", "mean"), mse_no_max=("mse_no_max", "mean"),
    mse_no_top7=("mse_no_top7", "mean"), mae=("mae", "mean")).reset_index()
summary["full_rank"] = summary.mse.rank(pct=True)
summary["no_max_rank"] = summary.mse_no_max.rank(pct=True)
summary["no7_rank"] = summary.mse_no_top7.rank(pct=True)
summary["mean_rank"] = summary[["full_rank", "no_max_rank", "no7_rank"]].mean(axis=1)
summary = summary.sort_values("mean_rank")
summary.to_csv(OUT / "confidence_shrink_selection.csv", index=False)
best = summary.iloc[0]
protocol, gamma, beta = best.protocol, float(best.gamma), float(best.beta)
raw_parts = [tests[f"{protocol}_{seed}__et_log"] for seed in SEEDS]
raw = np.mean(raw_parts, axis=0)
std = np.std(np.vstack([tests[f"{p}_{s}__et_log"] for p in ["canonical", "scaffold"] for s in SEEDS]), axis=0)
conf = np.power(sim_test, gamma) / (1 + beta * std)
pred = conf * raw + (1 - conf) * prior
# Exact supervised neighbors have full confidence by definition.
ref_full = np.concatenate([tr_full, ext_full])
ref_y = np.concatenate([y, ye])
all_fps = tr_fps + ext_fps
for i in np.flatnonzero(exact_test):
    candidates = np.flatnonzero(ref_full == te_full[i])
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(te_fps[i], [all_fps[j] for j in candidates]))
    best_j = int(np.argmax(sims))
    pred[i] = ref_y[candidates[best_j]]

pd.DataFrame({"index": te["index"], "SI VACV": pred}).to_csv(OUT / "submission_confidence_shrink.csv", index=False)
si_true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
si_true = si_true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
report = {
    "selected_by": "mean percentile rank of repeated OOF full/no-max/no-top7 MSE",
    "protocol": protocol, "gamma": gamma, "beta": beta, "prior_si": prior,
    "oof": {"mse": float(best.mse), "mse_no_max": float(best.mse_no_max),
            "mse_no_top7": float(best.mse_no_top7), "mae": float(best.mae)},
    "local_reconstructed_diagnostic": metrics(si_true, pred),
    "test_targets_used_for_fit_or_selection": False,
    "independent_evaluation": False,
    "submission": "submission_confidence_shrink.csv",
}
(OUT / "confidence_shrink_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("BEST", json.dumps(report, indent=2), flush=True)
