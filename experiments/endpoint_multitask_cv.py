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


def std_mol(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return None
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    except Exception:
        pass
    return m


def key14(m):
    ik = Chem.MolToInchiKey(m)
    return ik[:14] if ik else ""


def scaffold(m):
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        if sc.GetNumAtoms() == 0:
            return "ACYCLIC"
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc), isomericSmiles=False)
    except Exception:
        return "ACYCLIC"


def one_feature(m):
    fp = np.zeros(2048, np.uint8)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), fp)
    maccs = np.zeros(167, np.uint8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), maccs)
    desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
            Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m),
            Descriptors.RingCount(m), Descriptors.FractionCSP3(m), Descriptors.HeavyAtomCount(m)]
    return np.concatenate([fp, maccs, desc]).astype(np.float32)


def to_um(value, units, mol):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(v) or v <= 0:
        return np.nan
    u = str(units).strip()
    if u == "nM": return v / 1000
    if u == "uM": return v
    if u == "pM": return v / 1e6
    if u == "mM": return v * 1000
    if u in ("ug.mL-1", "ug/ml", "ug/mL"):
        mw = Descriptors.MolWt(mol) if mol is not None else 0
        return v * 1000 / mw if mw > 0 else np.nan
    return np.nan


def prepare_external(rows):
    grouped = {}
    for row in rows:
        if not np.isfinite(row["potency"]) and not np.isfinite(row["cc50"]):
            continue
        k = row["key"]
        if not k:
            continue
        grouped.setdefault(k, {"mol": row["mol"], "potency": [], "cc50": []})
        if np.isfinite(row["potency"]): grouped[k]["potency"].append(row["potency"])
        if np.isfinite(row["cc50"]): grouped[k]["cc50"].append(row["cc50"])
    records = []
    for k, v in grouped.items():
        records.append({"key": k, "mol": v["mol"], "scaffold": scaffold(v["mol"]),
                        "potency": np.median(v["potency"]) if v["potency"] else np.nan,
                        "cc50": np.median(v["cc50"]) if v["cc50"] else np.nan})
    X = np.vstack([one_feature(r["mol"]) for r in records]) if records else np.empty((0, 2224), np.float32)
    return records, X


tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0
tr_mols = [std_mol(s) for s in tr.SMILES]
te_mols = [std_mol(s) for s in te.SMILES]
X = np.vstack([one_feature(m) for m in tr_mols])
Xt = np.vstack([one_feature(m) for m in te_mols])
tr_keys = np.array([key14(m) for m in tr_mols])
tr_scafs = np.array([scaffold(m) for m in tr_mols])
y_pot = tr["IC50_uM_VACV"].fillna(10000).to_numpy(float)
y_cc = tr["CC50_uM"].to_numpy(float)

chembl_rows = []
va = pd.read_csv(DATA / "external/chembl_vacv_activities.csv")
va = va[va.standard_type.isin(["EC50", "IC50", "ED50", "AC50"]) &
       (va.standard_relation.fillna("") == "=") & va.standard_value.notna()]
for r in va.itertuples(index=False):
    m = std_mol(r.canonical_smiles)
    value = to_um(r.standard_value, r.standard_units, m)
    if m is not None and np.isfinite(value):
        chembl_rows.append({"mol": m, "key": key14(m), "potency": value, "cc50": np.nan})
cc = pd.read_csv(DATA / "external/chembl_cc50_same_molecule_publication.csv")
cc = cc[(cc.standard_relation.fillna("") == "=") & cc.standard_value.notna()]
for r in cc.itertuples(index=False):
    m = std_mol(r.canonical_smiles)
    value = to_um(r.standard_value, r.standard_units, m)
    if m is not None and np.isfinite(value):
        chembl_rows.append({"mol": m, "key": key14(m), "potency": np.nan, "cc50": value})
chembl_records, X_chembl = prepare_external(chembl_rows)

pubchem_rows = []
pc = pd.read_csv(DATA / "external/pubchem_enrichment/enriched_db.csv")
for r in pc.itertuples(index=False):
    m = std_mol(r.smiles)
    if m is None:
        continue
    pubchem_rows.append({"mol": m, "key": key14(m),
                         "potency": float(r.vacv_potency_um) if pd.notna(r.vacv_potency_um) else np.nan,
                         "cc50": float(r.cc50_um) if pd.notna(r.cc50_um) else np.nan})
pubchem_records, X_pubchem = prepare_external(pubchem_rows)

sources = {
    "none": ([], np.empty((0, X.shape[1]), np.float32)),
    "chembl": (chembl_records, X_chembl),
    "pubchem": (pubchem_records, X_pubchem),
    "both": (chembl_records + pubchem_records, np.vstack([X_chembl, X_pubchem])),
}
print("External unique structures:", {k: len(v[0]) for k, v in sources.items()}, flush=True)

max_i = int(np.argmax(y))
keep = np.arange(len(y)) != max_i
keep7 = np.ones(len(y), bool)
keep7[np.argsort(y)[-7:]] = False


def metrics(p):
    e = (y - p) ** 2
    return {"mse": float(e.mean()), "mse_no_max": float(e[keep].mean()),
            "mse_no_top7": float(e[keep7].mean()), "mae": float(np.abs(y - p).mean()),
            "max_target_pred": float(p[max_i])}


def fit_endpoint(Xfit, target, weights, Xpred, seed):
    weights = weights / weights.mean()
    model = LGBMRegressor(n_estimators=300, learning_rate=.035, num_leaves=15,
                          subsample=.85, colsample_bytree=.7, reg_lambda=5,
                          verbose=-1, n_jobs=4, random_state=seed)
    model.fit(Xfit, np.log(np.clip(target, 1e-8, None)), sample_weight=weights)
    return model.predict(Xpred)


configs = [("none", 0.0), ("chembl", .1), ("chembl", .5), ("pubchem", .1), ("both", .1), ("both", .5)]
results = []
oof_store = {}
test_store = {}
for protocol, groups in [("canonical", tr_keys), ("scaffold", tr_scafs)]:
    for seed in SEEDS:
        folds = np.empty(len(y), dtype=int)
        for f, (_, va_idx) in enumerate(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups)):
            folds[va_idx] = f
        for source_name, ext_weight in configs:
            start = time.time()
            records, Xe = sources[source_name]
            if records:
                ext_keys = np.array([r["key"] for r in records])
                ext_scafs = np.array([r["scaffold"] for r in records])
                ext_pot = np.array([r["potency"] for r in records])
                ext_cc = np.array([r["cc50"] for r in records])
            oof = np.zeros(len(y))
            fold_test = []
            for f in range(N_FOLDS):
                ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                ext_allowed = np.ones(len(records), dtype=bool)
                if records:
                    ext_allowed &= ~np.isin(ext_keys, np.unique(tr_keys[vi]))
                    if protocol == "scaffold":
                        ext_allowed &= ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
                # Potency: missing train IC50 is the organizer-corrected inactive value 10000 uM.
                Xpot = [X[ti]]; ypot = [y_pot[ti]]; wpot = [1 + y[ti] / 100]
                if records:
                    em = ext_allowed & np.isfinite(ext_pot)
                    Xpot.append(Xe[em]); ypot.append(ext_pot[em]); wpot.append(np.full(em.sum(), ext_weight))
                Xpot = np.vstack(Xpot); ypot = np.concatenate(ypot); wpot = np.concatenate(wpot)
                pot_pred = fit_endpoint(Xpot, ypot, wpot, np.vstack([X[vi], Xt]), seed)
                log_pot_v, log_pot_t = pot_pred[:len(vi)], pot_pred[len(vi):]
                # CC50: two unknown train values are excluded, never imputed from test.
                ti_cc = ti[np.isfinite(y_cc[ti])]
                Xcc = [X[ti_cc]]; ycc = [y_cc[ti_cc]]; wcc = [1 + y[ti_cc] / 100]
                if records:
                    em = ext_allowed & np.isfinite(ext_cc)
                    Xcc.append(Xe[em]); ycc.append(ext_cc[em]); wcc.append(np.full(em.sum(), ext_weight))
                Xcc = np.vstack(Xcc); ycc = np.concatenate(ycc); wcc = np.concatenate(wcc)
                cc_pred = fit_endpoint(Xcc, ycc, wcc, np.vstack([X[vi], Xt]), seed)
                log_cc_v, log_cc_t = cc_pred[:len(vi)], cc_pred[len(vi):]
                oof[vi] = np.exp(np.clip(log_cc_v - log_pot_v, -20, 20))
                fold_test.append(np.exp(np.clip(log_cc_t - log_pot_t, -20, 20)))
            name = f"{source_name}_w{ext_weight:g}"
            pred = np.mean(fold_test, axis=0)
            row = {"protocol": protocol, "seed": seed, "model": name,
                   **metrics(np.minimum(oof, 4000)), "seconds": time.time() - start}
            results.append(row)
            oof_store[f"{protocol}_{seed}__{name}"] = oof
            test_store[f"{protocol}_{seed}__{name}"] = pred
            print(f"{protocol:9s} {seed} {name:13s} full={row['mse']:11,.0f} "
                  f"no_max={row['mse_no_max']:11,.0f} no7={row['mse_no_top7']:10,.0f}", flush=True)
            pd.DataFrame(results).to_csv(OUT / "endpoint_cv_scores_by_repeat.csv", index=False)
            np.savez_compressed(OUT / "endpoint_oof.npz", **oof_store)
            np.savez_compressed(OUT / "endpoint_test_predictions.npz", **test_store)

scores = pd.DataFrame(results)
summary = scores.groupby(["protocol", "model"]).agg(
    mse_mean=("mse", "mean"), mse_std=("mse", "std"),
    mse_no_max_mean=("mse_no_max", "mean"), mse_no_max_std=("mse_no_max", "std"),
    mse_no_top7_mean=("mse_no_top7", "mean"), mae_mean=("mae", "mean"),
    max_target_pred_mean=("max_target_pred", "mean"),
).reset_index()
summary.to_csv(OUT / "endpoint_cv_summary.csv", index=False)

# OOF-only selection of simple calibration and blends with the direct-SI baseline.
base_oofs = np.load(OUT / "robust_tail_oof.npz")
base_tests = np.load(OUT / "robust_tail_test_predictions.npz")
selection_rows = []
for protocol in ["canonical", "scaffold"]:
    base_keys = [f"{protocol}_{s}__tailw1.0" for s in SEEDS]
    base_oof = np.mean([base_oofs[k] for k in base_keys], axis=0)
    for name in [f"{s}_w{w:g}" for s, w in configs]:
        keys = [f"{protocol}_{seed}__{name}" for seed in SEEDS]
        ratio_oof = np.mean([oof_store[k] for k in keys], axis=0)
        for base_weight in [0, .25, .5, .75, 1.0]:
            mixed = base_weight * base_oof + (1 - base_weight) * ratio_oof
            for scale in np.arange(.1, 3.01, .1):
                for cap in [1000, 4000, 10000, 36067]:
                    p = np.minimum(mixed * scale, cap)
                    selection_rows.append({"protocol": protocol, "model": name,
                                           "base_weight": base_weight, "scale": scale, "cap": cap,
                                           **metrics(p)})
selection = pd.DataFrame(selection_rows)
selection.to_csv(OUT / "endpoint_blend_selection.csv", index=False)
report = {
    "model": "separate LightGBM log(IC50) and log(CC50), SI=exp(logCC50-logIC50)",
    "seeds": SEEDS, "folds": N_FOLDS, "external_structures": {k: len(v[0]) for k, v in sources.items()},
    "missing_train_IC50": "10000 uM per corrected organizer convention",
    "validation_structure_exclusion": "InChIKey14; scaffold protocol also excludes validation scaffolds from external data",
    "test_targets_used_for_fit_or_selection": False,
    "summary": summary.to_dict(orient="records"),
}
(OUT / "endpoint_cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\nSUMMARY\n", summary.sort_values(["protocol", "mse_no_max_mean"]).to_string(index=False), flush=True)
print("\nBEST NO-MAX\n", selection.nsmallest(20, "mse_no_max").to_string(index=False), flush=True)
print("DONE", flush=True)
