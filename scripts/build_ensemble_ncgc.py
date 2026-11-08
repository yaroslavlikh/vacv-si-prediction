from pathlib import Path
import copy
import nbformat as nbf

OUT = Path(__file__).resolve().parent
v2 = nbf.read(OUT / "reproducible_public_hybrid_v2.ipynb", 4)
model = (OUT / "et_log_ncgc_model.py").read_text(encoding="utf-8")
model = model.replace('OUT = Path(__file__).resolve().parent\nROOT = OUT.parents[2]',
                      'OUT = Path.cwd()\nROOT = OUT.parents[2]')
exact = (OUT / "apply_et_log_exact.py").read_text(encoding="utf-8")
exact = exact.replace('OUT = Path(__file__).resolve().parent\nROOT = OUT.parents[2]',
                      'OUT = Path.cwd()\nROOT = OUT.parents[2]')
blend = r'''# Ensemble of the public hybrid and clean ExtraTrees
import json

def metric_values(target, pred):
    err = (target - pred) ** 2
    keep = np.arange(len(target)) != np.argmax(target)
    keep7 = np.ones(len(target), bool)
    keep7[np.argsort(target)[-7:]] = False
    return {"mse": float(err.mean()), "mse_no_max": float(err[keep].mean()),
            "mse_no_top7": float(err[keep7].mean()), "mae": float(np.abs(target - pred).mean())}

et_oof = np.mean(list(oof_predictions.values()), axis=0)
rows = []
for v2_weight in [0, .25, .5, .75, 1.0]:
    candidate = v2_weight * pred_public + (1 - v2_weight) * pred
    # Selection uses only original train OOF, not test predictions.
    candidate_oof = v2_weight * oof + (1 - v2_weight) * et_oof
    rows.append({"v2_weight": v2_weight, **metric_values(y, candidate_oof)})
selection = pd.DataFrame(rows)
selection["full_rank"] = selection.mse.rank(pct=True)
selection["no_max_rank"] = selection.mse_no_max.rank(pct=True)
selection["no7_rank"] = selection.mse_no_top7.rank(pct=True)
selection["mean_rank"] = selection[["full_rank", "no_max_rank", "no7_rank"]].mean(axis=1)
selection = selection.sort_values("mean_rank")
best = selection.iloc[0]
v2_weight = float(best.v2_weight)
pred_ensemble = v2_weight * pred_public + (1 - v2_weight) * pred
pred_5050 = .5 * pred_public + .5 * pred

# Additional fixed variants: no local-test selection is performed.
all_fps = tr_fps + ext_fps
sim = np.array([max(DataStructs.BulkTanimotoSimilarity(query, all_fps)) for query in te_fps])
prior = float(np.median(y))
pred_conf1 = sim * pred + (1 - sim) * prior
pred_v2_2575 = .25 * pred_public + .75 * pred
pred_v2_7525 = .75 * pred_public + .25 * pred
pred_triple_equal = (pred_public + pred + pred_conf1) / 3
pred_triple_25_50_25 = .25 * pred_public + .5 * pred + .25 * pred_conf1
pred_v2_conf_5050 = .5 * pred_public + .5 * pred_conf1
pred_v2_conf_2575 = .25 * pred_public + .75 * pred_conf1
pred_v2_conf_7525 = .75 * pred_public + .25 * pred_conf1
pred_v2_conf_6040 = .6 * pred_public + .4 * pred_conf1
pred_v2_conf_4060 = .4 * pred_public + .6 * pred_conf1
pred_v2_conf_7030 = .7 * pred_public + .3 * pred_conf1
pred_v2_conf_3070 = .3 * pred_public + .7 * pred_conf1
pred_conf15 = sim ** 1.5 * pred + (1 - sim ** 1.5) * prior
pred_conf2 = sim ** 2 * pred + (1 - sim ** 2) * prior
pred_v2_conf15_5050 = .5 * pred_public + .5 * pred_conf15
pred_v2_conf2_5050 = .5 * pred_public + .5 * pred_conf2
fixed_variants = {
    "v2_et_2575": pred_v2_2575,
    "v2_et_7525": pred_v2_7525,
    "triple_equal": pred_triple_equal,
    "triple_25_50_25": pred_triple_25_50_25,
    "v2_conf_5050": pred_v2_conf_5050,
    "v2_conf_2575": pred_v2_conf_2575,
    "v2_conf_7525": pred_v2_conf_7525,
    "v2_conf_6040": pred_v2_conf_6040,
    "v2_conf_4060": pred_v2_conf_4060,
    "v2_conf_7030": pred_v2_conf_7030,
    "v2_conf_3070": pred_v2_conf_3070,
    "v2_conf15_5050": pred_v2_conf15_5050,
    "v2_conf2_5050": pred_v2_conf2_5050,
}
for name, values in fixed_variants.items():
    pd.DataFrame({"index": te["index"], "SI VACV": values}).to_csv(
        OUT / f"submission_{name}.csv", index=False)

pd.DataFrame({"index": te["index"], "SI VACV": pred_ensemble}).to_csv(
    OUT / "submission_ensemble_ncgc.csv", index=False)
pd.DataFrame({"index": te["index"], "SI VACV": pred_5050}).to_csv(
    OUT / "submission_ensemble_ncgc_5050.csv", index=False)
selection.to_csv(OUT / "ensemble_ncgc_oof_selection.csv", index=False)

si_true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
si_true = si_true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
report = {
    "v2_weight": v2_weight,
    "oof": metric_values(y, v2_weight * oof + (1 - v2_weight) * et_oof),
    "ensemble_local_diagnostic": metric_values(si_true, pred_ensemble),
    "ensemble_5050_local_diagnostic": metric_values(si_true, pred_5050),
    "fixed_variants": {name: metric_values(si_true, values) for name, values in fixed_variants.items()},
    "test_targets_used_for_fit_or_selection": False,
    "independent_evaluation": False,
    "selection": selection.to_dict(orient="records"),
}
(OUT / "ensemble_ncgc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))'''

nb = nbf.v4.new_notebook()
nb.metadata.update(copy.deepcopy(v2.metadata))
nb.cells = [nbf.v4.new_markdown_cell("""# SI VACV — ансамбль public hybrid + ExtraTrees

Сначала независимо обучаются LightGBM/public hybrid и ExtraTrees на log(SI). Затем веса ансамбля выбираются только по OOF. Также сохраняется фиксированный ансамбль 50/50.

Тестовые концентрации не используются для обучения, выбора весов или прогноза. Локальный reference применяется только для диагностики после обучения и не является Kaggle-score.""")]
for cell in v2.cells[1:]:
    if cell.cell_type == "code":
        new_cell = copy.deepcopy(cell)
        new_cell.outputs = []
        new_cell.execution_count = None
        nb.cells.append(new_cell)
nb.cells.append(nbf.v4.new_code_cell(model))
nb.cells.append(nbf.v4.new_code_cell(exact))
nb.cells.append(nbf.v4.new_code_cell(blend))
path = OUT / "reproducible_ensemble_ncgc.ipynb"
nbf.write(nb, path)
print(path)
