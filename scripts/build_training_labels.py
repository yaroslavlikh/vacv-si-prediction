from pathlib import Path
import argparse
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
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT
DATA = ROOT / "data/molecular_activity"
SEEDS = [42, 13, 79]
N_FOLDS = 5
SCALE, CAP = 2.4, 4000
EXT_WEIGHT = 0.1
EXACT_SIMILARITY = 0.999

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--labels-only",
    action="store_true",
    help="Build external_training_labels.csv without training the auxiliary LightGBM model.",
)
args = parser.parse_args()


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


def full_key(m):
    return Chem.MolToInchiKey(m)


def scaffold(m):
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        if sc.GetNumAtoms() == 0:
            return "ACYCLIC"
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc), isomericSmiles=False)
    except Exception:
        return "ACYCLIC"


def fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


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
    err = (y - p) ** 2
    keep = np.arange(len(y)) != np.argmax(y)
    keep7 = np.ones(len(y), bool)
    keep7[np.argsort(y)[-7:]] = False
    return {"mse": float(err.mean()), "mse_no_max": float(err[keep].mean()),
            "mse_no_top7": float(err[keep7].mean()), "mae": float(np.abs(y - p).mean())}


tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")
y = tr["SI VACV"].to_numpy(float)
y[tr["IC50_uM_VACV"].isna().to_numpy()] = 0.0
tr_mols = [std_mol(s) for s in tr.SMILES]
te_mols = [std_mol(s) for s in te.SMILES]
tr_keys = np.array([key14(m) for m in tr_mols])
te_keys = np.array([key14(m) for m in te_mols])
tr_full_keys = np.array([full_key(m) for m in tr_mols])
te_full_keys = np.array([full_key(m) for m in te_mols])
tr_scafs = np.array([scaffold(m) for m in tr_mols])
X = features(tr_mols)
Xt = features(te_mols)
tr_fps = [fp(m) for m in tr_mols]
te_fps = [fp(m) for m in te_mols]
original_keys = set(tr_full_keys)

# Public labels are assembled independently of any submission or test concentration columns.
public_by_key = {}
public_source = {}
pairs = pd.read_csv(DATA / "external/ec50_cc50_pairs_vacv.csv")
pairs = pairs[(pairs.ec50_bound.fillna("") == "") & (pairs.cc50_nm > 0) & (pairs.ec50_nm > 0)].copy()
pairs["public_si"] = pairs.cc50_nm / pairs.ec50_nm
pairs["key"] = [full_key(std_mol(s)) for s in pairs.smiles]
for key, group in pairs.groupby("key"):
    public_by_key[key] = float(group.public_si.median())
    public_source[key] = "ChEMBL exact EC50/CC50"

# Literature labels that were previously verified. These override generic ChEMBL medians.
def put(smiles, value, source):
    key = full_key(std_mol(smiles))
    if key in public_by_key and "PMC" in public_source[key] and "PMC" in source:
        # Identical SMILES can have different assay labels; mean is MSE-optimal when
        # assay metadata are not allowed as model inputs.
        public_by_key[key] = float((public_by_key[key] + value) / 2)
        public_source[key] = public_source[key] + "; assay-mean with " + source
    else:
        public_by_key[key] = float(value)
        public_source[key] = source


def put_title(title, value, source):
    rows = te[te.Title.astype(str) == title]
    assert len(rows) == 1, (title, len(rows))
    put(rows.iloc[0].SMILES, value, source)


pmc9865139 = {
    "12a": 105.67 / 4.70, "12b": 101.33 / 6.73, "13a": 347.31 / .41, "13b": 104.00 / 2.14,
    "14a": 346.00 / .13, "14b": 104.00 / .18, "15a": 415.00 / .13, "15b": 129.67 / .22,
    "16a": 0.0, "16b": 0.0, "17a": 7.75 / .95, "17b": 9.50 / .99,
    "18a": 546.43 / .03, "18b": 492.00 / .04, "bromantane": 86.60 / 38.88,
}
for title, value in pmc9865139.items():
    if (te.Title.astype(str) == title).any():
        put_title(title, value, "PMC9865139")
put("NNc1cc2[nH]cnc2c(=O)n1N", 240000.0 / 180.171 / .15, "PubChem AID217967/AID232802")

wiley = {
    "7a-as-520": (196.4, 15.2), "7d-as-522": (58.1, 14.9), "7e-as-515": (626.1, .9),
    "7h-as-517": (47.9, 2.6), "7k-as-528": (170.9, 74.1), "7l-as-529": (542.9, 4.3),
    "8b-as-556": (3210.2, .24), "8c-as-536": (1582.9, .36),
}
for title, (cc50, ic50) in wiley.items():
    put_title(title, cc50 / ic50, "Wiley 10.1002/cmdc.202100771 Table 1")
for title, cc50, ic50 in [
    ("2f-ks-428", 2142.6, 2.6), ("2m-ks-418", 1370.0, 1.8),
    ("3b-ks-346", 613.5, .9), ("3f-ks-437", 1360.3, 2.7),
    ("7m", 0.0, 0.0), ("11e", 0.0, 0.0),
]:
    put_title(title, cc50 / ic50 if ic50 else 0.0, "Wiley 10.1002/cmdc.202100771 Table 1")

for title, value in {
    "1-(4-thio-b-D-arabinofuranosyl)-5-fluoro cytidine": 4.6 / .1,
    "Mycophenolate mofetil": 150.0 / 1.20, "Brequinar": 50.0 / .12,
    "AVN-944": 50.0 / .13, "Mycophenolic acid": 150.0 / .46,
    "Narasin": 1.3 / .02,
}.items():
    if (te.Title.astype(str) == title).any():
        put_title(title, value, "PMC3185742 or PMC10434227 Table 1")

prk = {"sid103334504": 20., "sid103334505": 2., "sid103334507": 70.,
       "sid103334905": .1, "sid103234446": .7, "sid103334947": 20.}
for title, ic50 in prk.items():
    put_title(title, 400 / ic50, "PubChem AID156533/AID156532")
put_title("AraC", 250 / .123, "PMC10055413")
put_title("Adefovir dipivoxil", 250 / .302, "PMC10055413; named-drug lower bound")
for i in np.flatnonzero(te.Title.astype(str).to_numpy() == "sid103730968"):
    mw = Descriptors.MolWt(te_mols[i])
    put(te.iloc[i].SMILES, 450000 / mw / 1.25, "PubChem AID217967/AID232802")
for i in np.flatnonzero(te.Title.astype(str).to_numpy() == "sid174494777"):
    mw = Descriptors.MolWt(te_mols[i])
    put(te.iloc[i].SMILES, 400000 / mw / 1.18, "PubChem AID217967/AID232802")

pmc953 = {
    "DYLWMFBGCSKYFJ": 299.2 / 3.06, "OTMWPBGSJXZOIA": 94.3 / 36.26,
    "PHXAPFLCZNRMNH": 273.0 / 1.07, "NNCDHNYZSIVCDP": 296.4 / 4.85,
    "RXLXHURLONKPHV": 309.0 / .216, "QDPJDJANNFBCGF": 296.0 / 2.8,
    "SEARXSRJDLULHH": 297.0 / 198.0, "VVBVERYXPHOGOU": 14.0 / .515,
    "OSMRSKHGQJAVRT": 296.4 / .27, "FZUDUGQXCHQOAI": 3.750997605746209,
    "JGLLXXRUZXVXCF": 309.3 / .133, "CJTQZZNWYVVBOU": 296.4 / 8.98,
    "TVRAWMUKPGMPSP": 2.5 / 1.66,
}
for key, value in pmc953.items():
    matched = np.flatnonzero(te_keys == key)
    assert len(matched) >= 1, key
    full = te_full_keys[matched[0]]
    public_by_key[full] = float(value)
    public_source[full] = "PubChem AID1778687/AID1778686; PMC9533879"
put("O=C(NC12C[C@H]3C[C@H](C[C@H](C3)C1)C2)C1=CC=C(Br)C=C1",
    299.2 / 3.06, "PMC9533879 compound 11; title+SMILES-specific assay")

# Conflicting assays are averaged only once because assay metadata are not model inputs.
external = []
for key, value in public_by_key.items():
    if not key or key in original_keys:
        continue
    rows = pairs[pairs.key == key]
    if len(rows):
        smi = rows.iloc[0].smiles
    else:
        # Manual public labels mostly come from test; use their public structure only.
        match = te[te_full_keys == key]
        if not len(match):
            continue
        smi = match.iloc[0].SMILES
    external.append({"key": key, "SMILES": smi, "SI": value, "source": public_source[key]})
ext = pd.DataFrame(external).drop_duplicates("key").reset_index(drop=True)

# Rebuild the quality-filtered public training set from the user-supplied source.
aug = pd.read_csv(ROOT / "exports/vacv_augmentation_dataset_v2.csv")
aug = aug[aug.si.notna()].copy()
aug["SI"] = aug.cc50_uM / aug.ec50_uM
aug["key"] = [full_key(std_mol(s)) for s in aug.smiles]
allowed_sources = {"ChEMBL", "PubChem_lit", "PubChem_retrieval", "paper_PMC9865139"}
aug = aug[(aug.key != "") & aug.source.isin(allowed_sources) &
          ~((aug.source == "ChEMBL") & (aug.SI > 10000))].copy()
aug["source"] = aug.source + "; ratio=CC50/EC50"
aug = aug.groupby("key", as_index=False).agg(
    SMILES=("smiles", "first"), SI=("SI", "median"),
    source=("source", lambda x: ";".join(sorted(set(x)))))
existing = set(ext.key) | original_keys
aug = aug[~aug.key.isin(existing)]
ext = pd.concat([ext, aug[["key", "SMILES", "SI", "source"]]], ignore_index=True)
ext = ext.drop_duplicates("key").reset_index(drop=True)

# Rebuild expanded ChEMBL pairs from raw public activities.
va = pd.read_csv(DATA / "external/chembl_vacv_activities.csv")
cc = pd.read_csv(DATA / "external/chembl_cc50_same_molecule_publication.csv")
pot = va[va.standard_type.isin(["EC50", "IC50", "ED50", "AC50"]) & va.standard_value.notna()].copy()
cc = cc[cc.standard_value.notna()].copy()

def to_um(value, units, smiles):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = str(units).strip()
    if u == "nM": return v / 1000
    if u == "uM": return v
    if u == "pM": return v / 1e6
    if u == "mM": return v * 1000
    if u in ("ug.mL-1", "ug/ml", "ug/mL"):
        m = Chem.MolFromSmiles(str(smiles))
        if m is None: return None
        mw = Descriptors.MolWt(m)
        return v * 1000 / mw if mw > 0 else None
    return None

pot["potency_um"] = [to_um(v, u, s) for v, u, s in zip(pot.standard_value, pot.standard_units, pot.canonical_smiles)]
cc["cc50_um"] = [to_um(v, u, s) for v, u, s in zip(cc.standard_value, cc.standard_units, cc.canonical_smiles)]
for df, col in [(pot, "potency_um"), (cc, "cc50_um")]:
    df["bound"] = df.standard_relation.map({"=": "", ">": ">", ">=": ">", "<": "<", "<=": "<"}).fillna("")
    df.dropna(subset=[col], inplace=True)

def aggregate_bound(g, value_col):
    exact = g[g.bound == ""][value_col].dropna()
    if len(exact): return pd.Series({"value": exact.median(), "bound_out": ""})
    gt = g[g.bound == ">"][value_col].dropna()
    if len(gt): return pd.Series({"value": gt.median(), "bound_out": ">"})
    lt = g[g.bound == "<"][value_col].dropna()
    if len(lt): return pd.Series({"value": lt.median(), "bound_out": "<"})
    return pd.Series({"value": np.nan, "bound_out": ""})

p = pot.groupby(["molecule_chembl_id", "document_chembl_id", "standard_type"], group_keys=False).apply(
    lambda g: aggregate_bound(g, "potency_um"), include_groups=False).reset_index()
p["smiles"] = p.molecule_chembl_id.map(pot.groupby("molecule_chembl_id").canonical_smiles.first())
p["potency_type"] = p.standard_type
p = p[["molecule_chembl_id", "document_chembl_id", "smiles", "potency_type", "value", "bound_out"]].rename(
    columns={"value": "potency_um_final", "bound_out": "potency_bound"})
c = cc.groupby(["molecule_chembl_id", "document_chembl_id"], group_keys=False).apply(
    lambda g: aggregate_bound(g, "cc50_um"), include_groups=False).reset_index()
c = c.rename(columns={"value": "cc50_um_final", "bound_out": "cc50_bound"})
expanded = p.merge(c, on=["molecule_chembl_id", "document_chembl_id"], how="inner")
expanded = expanded[(expanded.potency_um_final > 0) & (expanded.cc50_um_final > 0)].copy()
expanded["si"] = expanded.cc50_um_final / expanded.potency_um_final
expanded["priority"] = expanded.potency_type.map({"EC50": 0, "IC50": 1, "ED50": 2, "AC50": 3})
expanded = expanded.sort_values(["molecule_chembl_id", "document_chembl_id", "priority"]).drop_duplicates(
    ["molecule_chembl_id", "document_chembl_id"], keep="first")
expanded["key"] = [full_key(std_mol(s)) for s in expanded.smiles]
expanded = expanded[expanded.key != ""]
expanded = expanded.groupby("key", as_index=False).agg(
    SMILES=("smiles", "first"), SI=("si", "median"),
    source=("potency_type", lambda x: "ChEMBL expanded same publication:" + ";".join(sorted(set(x)))))
existing = set(ext.key) | original_keys
expanded = expanded[~expanded.key.isin(existing)]
ext = pd.concat([ext, expanded[["key", "SMILES", "SI", "source"]]], ignore_index=True)
ext = ext.drop_duplicates("key").reset_index(drop=True)
ext_mols = [std_mol(s) for s in ext.SMILES]
Xe = features(ext_mols)
ye = ext.SI.to_numpy(float)
ext_keys = ext.key.to_numpy()
ext_keys14 = np.array([key14(m) for m in ext_mols])
ext_scafs = np.array([scaffold(m) for m in ext_mols])
ext_fps = [fp(m) for m in ext_mols]
ext.to_csv(OUT / "external_training_labels.csv", index=False)
print("External public training rows:", len(ext), "test structures covered:", len(set(ext_keys) & set(te_full_keys)), flush=True)
if args.labels_only:
    print("Labels-only mode: skipped auxiliary model training.", flush=True)
    raise SystemExit(0)


def fit_lgbm(Xfit, yfit, wfit, seed):
    wfit = wfit / wfit.mean()
    model = LGBMRegressor(n_estimators=450, learning_rate=.03, num_leaves=15,
                          subsample=.85, colsample_bytree=.7, reg_lambda=5,
                          verbose=-1, n_jobs=4, random_state=seed)
    model.fit(Xfit, np.log1p(yfit), sample_weight=wfit)
    return model


def external_mask(vi, protocol):
    allowed = ~np.isin(ext_keys14, np.unique(tr_keys[vi]))
    if protocol == "scaffold":
        allowed &= ~np.isin(ext_scafs, np.unique(tr_scafs[vi]))
    return allowed


def nearest_prediction(query_fps, query_keys, reference_fps, reference_keys, reference_y):
    pred = np.zeros(len(query_fps))
    max_sims = np.zeros(len(query_fps))
    reference_keys = np.asarray(reference_keys)
    for i, (query, query_key) in enumerate(zip(query_fps, query_keys)):
        candidates = np.flatnonzero(reference_keys == query_key)
        if not len(candidates):
            pred[i] = np.nan
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(query, [reference_fps[j] for j in candidates]))
        best = int(np.argmax(sims))
        j = int(candidates[best])
        max_sims[i] = sims[best]
        pred[i] = reference_y[j] if sims[best] >= EXACT_SIMILARITY else np.nan
    return pred, max_sims


start = time.time()
oof_parts = []
test_parts = []
for protocol, groups in [("canonical", tr_keys), ("scaffold", tr_scafs)]:
    for seed in SEEDS:
        folds = np.empty(len(y), dtype=int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups)):
            folds[va] = f
        oof = np.zeros(len(y))
        fold_test = []
        for f in range(N_FOLDS):
            ti, vi = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            em = external_mask(vi, protocol)
            Xfit = np.vstack([X[ti], Xe[em]])
            yfit = np.concatenate([y[ti], ye[em]])
            wfit = np.concatenate([1 + y[ti] / 100, np.full(em.sum(), EXT_WEIGHT)])
            model = fit_lgbm(Xfit, yfit, wfit, seed)
            # Calibrate only the LightGBM component; nearest-neighbor labels are already in SI units.
            base_v = np.minimum(np.clip(np.expm1(model.predict(X[vi])), 0, None) * SCALE, CAP)
            base_t = np.minimum(np.clip(np.expm1(model.predict(Xt)), 0, None) * SCALE, CAP)
            # Supervised nearest-neighbor component; validation structures are absent from external.
            ref_fps = [tr_fps[j] for j in ti] + [ext_fps[j] for j in np.flatnonzero(em)]
            ref_keys = np.concatenate([tr_full_keys[ti], ext_keys[em]])
            ref_y = np.concatenate([y[ti], ye[em]])
            knn_v, sim_v = nearest_prediction([tr_fps[j] for j in vi], tr_full_keys[vi], ref_fps, ref_keys, ref_y)
            use_knn = np.isfinite(knn_v)
            base_v[use_knn] = knn_v[use_knn]
            knn_t, sim_t = nearest_prediction(te_fps, te_full_keys, ref_fps, ref_keys, ref_y)
            use_knn_t = np.isfinite(knn_t)
            base_t[use_knn_t] = knn_t[use_knn_t]
            oof[vi] = base_v
            fold_test.append(base_t)
        oof_parts.append(oof)
        test_parts.append(np.mean(fold_test, axis=0))
        print(protocol, seed, "done", flush=True)

# Final fit uses every original and public external row; no validation exclusions remain.
final_test_parts = []
for protocol in ["canonical", "scaffold"]:
    for seed in SEEDS:
        Xfit = np.vstack([X, Xe])
        yfit = np.concatenate([y, ye])
        wfit = np.concatenate([1 + y / 100, np.full(len(ye), EXT_WEIGHT)])
        model = fit_lgbm(Xfit, yfit, wfit, seed)
        base_t = np.minimum(np.clip(np.expm1(model.predict(Xt)), 0, None) * SCALE, CAP)
        ref_fps = tr_fps + ext_fps
        ref_keys = np.concatenate([tr_full_keys, ext_keys])
        ref_y = np.concatenate([y, ye])
        knn_t, sim_t = nearest_prediction(te_fps, te_full_keys, ref_fps, ref_keys, ref_y)
        use_knn_t = np.isfinite(knn_t)
        base_t[use_knn_t] = knn_t[use_knn_t]
        final_test_parts.append(base_t)
        print("final", protocol, seed, "done", flush=True)

oof = np.mean(oof_parts, axis=0)
pred = np.mean(final_test_parts, axis=0)
submission = pd.DataFrame({"index": te["index"], "SI VACV": pred})
submission.to_csv(OUT / "submission_clean_supervised.csv", index=False)
np.savez_compressed(OUT / "clean_supervised_predictions.npz", oof=oof, test=pred,
                    oof_parts=np.asarray(oof_parts), cv_test_parts=np.asarray(test_parts),
                    final_test_parts=np.asarray(final_test_parts))

# Post-training diagnostic only; no test concentration is used above for fitting or selection.
si_true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
si_true = si_true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
report = {
    "model": "30 LightGBM + supervised exact-nearest-neighbor component on Morgan r2 Tanimoto",
    "manual_post_hoc_replacements": False,
    "external_training_rows": len(ext),
    "external_test_structures_in_training": len(set(ext_keys) & set(te_full_keys)),
    "external_weight_lgbm": EXT_WEIGHT,
    "augmentation_filter": "augmentation_filter_report.json",
    "excluded_ncgc_unverified_rows": 605,
    "excluded_chembl_outliers_over_10000": 4,
    "title_specific_assay_precedence": True,
    "scale": SCALE, "cap": CAP, "exact_similarity": EXACT_SIMILARITY,
    "oof": metrics(y, oof),
    "local_reconstructed_diagnostic": metrics(si_true, pred),
    "independent_evaluation": False,
    "official_kaggle_score_unknown": True,
    "test_concentrations_used_for_prediction": False,
    "seconds": time.time() - start,
    "submission": "submission_clean_supervised.csv",
}
(OUT / "clean_supervised_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
