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
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "data/molecular_activity"
SEEDS = [42, 13, 79]
N_FOLDS = 5
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


def key14(m):
    return Chem.MolToInchiKey(m)[:14]


def scaffold(m):
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        if sc.GetNumAtoms() == 0:
            return "ACYCLIC"
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc), isomericSmiles=False)
    except Exception:
        return "ACYCLIC"


def fp_features(mols):
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


tr_mols = [std_mol(s) for s in tr.SMILES]
te_mols = [std_mol(s) for s in te.SMILES]
fp = fp_features(tr_mols)
fpt = fp_features(te_mols)
emb = np.load(OUT / "chemberta_embeddings.npz")
features = {
    "cls": (emb["cls_train"], emb["cls_test"]),
    "mean": (emb["mean_train"], emb["mean_test"]),
    "cls_mean": (np.hstack([emb["cls_train"], emb["mean_train"]]),
                 np.hstack([emb["cls_test"], emb["mean_test"]])),
    "fp_mean": (np.hstack([fp, emb["mean_train"]]), np.hstack([fpt, emb["mean_test"]])),
}
groups_by_protocol = {
    "canonical": np.array([key14(m) for m in tr_mols]),
    "scaffold": np.array([scaffold(m) for m in tr_mols]),
}
max_i = int(np.argmax(y))
keep = np.arange(len(y)) != max_i
keep7 = np.ones(len(y), bool)
keep7[np.argsort(y)[-7:]] = False


def metrics(p):
    e = (y - p) ** 2
    return {"mse": float(e.mean()), "mse_no_max": float(e[keep].mean()),
            "mse_no_top7": float(e[keep7].mean()), "mae": float(np.abs(y - p).mean()),
            "max_target_pred": float(p[max_i])}


def fit_predict(X, Xt, ti, vi, seed):
    weights = 1 + y[ti] / 100
    weights = weights / weights.mean()
    model = LGBMRegressor(n_estimators=450, learning_rate=.03, num_leaves=15,
                          subsample=.85, colsample_bytree=.7, reg_lambda=5,
                          verbose=-1, n_jobs=4, random_state=seed)
    model.fit(X[ti], np.log1p(y[ti]), sample_weight=weights)
    return (np.clip(np.expm1(model.predict(X[vi])), 0, None),
            np.clip(np.expm1(model.predict(Xt)), 0, None))


results = []
oof_store = {}
test_store = {}
for protocol, groups in groups_by_protocol.items():
    for seed in SEEDS:
        folds = np.empty(len(y), dtype=int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(fp, y, groups)):
            folds[va] = f
        for name, (X, Xt) in features.items():
            start = time.time()
            oof = np.zeros(len(y))
            fold_test = []
            for f in range(N_FOLDS):
                ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                oof[vi], pred = fit_predict(X, Xt, ti, vi, seed)
                fold_test.append(pred)
            test_pred = np.mean(fold_test, axis=0)
            row = {"protocol": protocol, "seed": seed, "features": name,
                   **metrics(oof), "seconds": time.time() - start}
            results.append(row)
            oof_store[f"{protocol}_{seed}__{name}"] = oof
            test_store[f"{protocol}_{seed}__{name}"] = test_pred
            print(f"{protocol:9s} {seed} {name:9s} full={row['mse']:11,.0f} "
                  f"no_max={row['mse_no_max']:11,.0f} no7={row['mse_no_top7']:10,.0f}", flush=True)
            pd.DataFrame(results).to_csv(OUT / "chemberta_cv_scores_by_repeat.csv", index=False)
            np.savez_compressed(OUT / "chemberta_oof.npz", **oof_store)
            np.savez_compressed(OUT / "chemberta_test_predictions.npz", **test_store)

scores = pd.DataFrame(results)
summary = scores.groupby(["protocol", "features"]).agg(
    mse_mean=("mse", "mean"), mse_std=("mse", "std"),
    mse_no_max_mean=("mse_no_max", "mean"), mse_no_max_std=("mse_no_max", "std"),
    mse_no_top7_mean=("mse_no_top7", "mean"), mae_mean=("mae", "mean"),
    max_target_pred_mean=("max_target_pred", "mean"),
).reset_index()
summary.to_csv(OUT / "chemberta_cv_summary.csv", index=False)

# Compare calibration and blending with the previously frozen fingerprint model using OOF only.
base_oofs = np.load(OUT / "robust_tail_oof.npz")
base_tests = np.load(OUT / "robust_tail_test_predictions.npz")
selection_rows = []
for protocol in groups_by_protocol:
    base_keys = [f"{protocol}_{s}__tailw1.0" for s in SEEDS]
    base_oof = np.mean([base_oofs[k] for k in base_keys], axis=0)
    base_test = np.mean([base_tests[k] for k in base_keys], axis=0)
    for name in ["base"] + list(features):
        if name == "base":
            raw_oof, raw_test = base_oof, base_test
        else:
            keys = [f"{protocol}_{s}__{name}" for s in SEEDS]
            raw_oof = np.mean([oof_store[k] for k in keys], axis=0)
            raw_test = np.mean([test_store[k] for k in keys], axis=0)
        for blend in ([1.0] if name == "base" else [0.0, .25, .5, .75, 1.0]):
            mixed_oof = blend * base_oof + (1 - blend) * raw_oof
            mixed_test = blend * base_test + (1 - blend) * raw_test
            for scale in np.arange(.5, 3.01, .1):
                p = np.clip(mixed_oof * scale, 0, None)
                selection_rows.append({"protocol": protocol, "features": name,
                                       "base_weight": blend, "scale": scale, **metrics(p)})
                # Store only uncalibrated components; final selection is made in a separate explicit step.
selection = pd.DataFrame(selection_rows)
selection.to_csv(OUT / "chemberta_blend_selection.csv", index=False)
report = {
    "model": "DeepChem/ChemBERTa-77M-MTR frozen embeddings + LightGBM",
    "target": "log1p(SI)", "sample_weight": "1+SI/100", "seeds": SEEDS, "folds": N_FOLDS,
    "test_targets_used_for_fit_or_selection": False,
    "external_pretraining": "77M PubChem structures; no task labels",
    "summary": summary.to_dict(orient="records"),
}
(OUT / "chemberta_cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nSUMMARY\n", summary.sort_values(["protocol", "mse_no_max_mean"]).to_string(index=False), flush=True)
print("\nBEST BLENDS NO-MAX\n", selection.nsmallest(20, "mse_no_max").to_string(index=False), flush=True)
print("DONE", flush=True)
