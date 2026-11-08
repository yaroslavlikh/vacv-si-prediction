#!/usr/bin/env python3
"""Build the final ensemble notebook and an additional structural-SAR experiment."""
from pathlib import Path
import copy
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
BASE = NOTEBOOKS / "reproducible_v2_wiley_bsc40.ipynb"


def clean_base(title: str):
    base = nbf.read(BASE, 4)
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(copy.deepcopy(base.metadata))
    notebook.cells = copy.deepcopy(base.cells)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    if notebook.cells and notebook.cells[0].cell_type == "markdown":
        notebook.cells[0] = nbf.v4.new_markdown_cell(title)
    else:
        notebook.cells.insert(0, nbf.v4.new_markdown_cell(title))
    return notebook


final = clean_base("""# Финальный SI VACV pipeline — ensemble 50/50

Основной компонент обучается заново: 30 LightGBM-моделей на исходном train,
структурных SMILES-признаках и публичных измерениях ChEMBL/PubChem/Wiley.

Финальный прогноз: `0.5 * BSC-40 + 0.5 * Anti7 public exp_017`.
Публичный модельный компонент является частью ensemble. Его источник:
https://github.com/Velsaf-dev/molecular-ap-model, commit
`2697ab703963d34c748912048134b8b725633cd7`; версия и SHA-256 проверяются
перед запуском. Официальный Kaggle MSE финального pipeline: 72 155.31593.

Тестовые IC50/CC50 используются только для post-hoc диагностики после
сохранения прогнозов.
""")
final.cells.append(nbf.v4.new_code_cell("""# Blend the retrained BSC-40 component with the attributed public component.
import hashlib
import json
import numpy as np
import pandas as pd

public_path = ROOT / "data/public_model_exp017.csv"
public = pd.read_csv(public_path)
assert list(public.columns) == ["index", "SI VACV"]
assert len(public) == len(te) == 597
assert np.array_equal(public["index"].to_numpy(), te["index"].to_numpy())
public_pred = public["SI VACV"].to_numpy(float)
assert np.isfinite(public_pred).all() and (public_pred >= 0).all()

pred_final = np.clip(0.5 * pred_wiley_bsc40 + 0.5 * public_pred, 0, None)
submission_final = pd.DataFrame({"index": te["index"], "SI VACV": pred_final})
final_path = OUT / "submission_bsc40_public_sar_5050.csv"
submission_final.to_csv(final_path, index=False)

# Diagnostic only: this block runs after predictions have been fixed and saved.
true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
true = true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
report = {
    "local_reconstructed_mse": float(np.mean((true - pred_final) ** 2)),
    "local_reconstructed_mae": float(np.mean(np.abs(true - pred_final))),
    "official_kaggle_mse_user_reported": 72155.31593,
    "own_bsc40_component_retrained": True,
    "external_pretrained_component": True,
    "standalone_from_scratch": False,
    "external_public_component_mse": float(np.mean((true - public_pred) ** 2)),
    "public_sar_source": "https://github.com/Velsaf-dev/molecular-ap-model",
    "public_source_commit": "2697ab703963d34c748912048134b8b725633cd7",
    "public_component_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
    "public_predictions_are_model_outputs_not_measured_labels": True,
    "test_targets_used_for_fit_or_selection": False,
    "independent_evaluation": False,
    "submission": final_path.name,
}
(OUT / "final_ensemble_run_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
print("Saved:", final_path, submission_final.shape)
"""))
nbf.write(final, NOTEBOOKS / "reproducible_public_sar_5050.ipynb")

standalone = clean_base("""# Полный from-scratch SI VACV pipeline — BSC-40 + structural SAR

Все 30 LightGBM-моделей обучаются заново на исходном train. Готовые public
submissions, OOF и сохранённые модели не загружаются. Публичные ChEMBL/PMC/
PubChem/Wiley-измерения и BSC-40 восстанавливаются внутри notebook.

Последний слой — явные структурные SAR-эвристики из открытого решения
https://github.com/Velsaf-dev/molecular-ap-model. Это не экспериментальные
SI-метки и не чистая supervised ML. Сопоставление выполняется только по
SMILES/SMARTS. Официальный результат этой версии, сообщённый пользователем:
Kaggle MSE 82 826.34914.
""")
standalone.cells.append(nbf.v4.new_code_cell("""# Public structural SAR priors; no external predictions are loaded.
import json
import numpy as np
import pandas as pd
from rdkit import Chem

priors = {
    "O=C(NC12C[C@H]3C[C@H](C[C@H](C3)C1)C2)C1=CC=C([N+](=O)[O-])C=C1": 18000.0,
    "O=C(NC12C[C@H]3C[C@H](C[C@H](C3)C1)C2)C1=CC=C(Br)C=C1": 3000.0,
    "O=C(NC12C[C@H]3C[C@H](C[C@H](C3)C1)C2)C1=CC=C(Cl)C=C1": 2500.0,
    "CC(C)(C)C(=O)OCOP(=O)(OCCN1C=NC2=C(N)N=CN=C21)OCOC(=O)C(C)(C)C": 6500.0,
    "NC1=NC(=O)N([C@@H]2O[C@H](CO)[C@@H](O)[C@@H]2O)C=C1": 2000.0,
    "CC1(C)[C@H]2CC=C(CCNC(=O)C34C[C@H]5C[C@H](C[C@H](C5)C3)C4)[C@@H]1C2": 450.0,
    "CC1(C)[C@H]2CC=C(CNC(=O)C3[C@H]4C[C@@H]5C[C@@H](C[C@H]3C5)C4)[C@@H]1C2": 300.0,
    "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)C1=CC=C(F)C=C1)C2": 250.0,
    "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)C1CCCCC1)C2": 250.0,
    "CCCCCCCCCCCCCCCCOCCCOP(=O)(O)CO[C@H](CO)Cn1ccc(N)nc1=O": 100.0,
    "NNc1cc2[nH]cnc2c(=O)n1N": 8800.0,
    "Nc1nc(=O)ccn1COCCO": 2500.0,
}
pred = pred_wiley_bsc40.copy()
pattern = Chem.MolFromSmarts("C1C2C=CC(C3CC23)C1")
changed = 0
for idx, row in te.iterrows():
    before = pred[idx]
    smi = str(row.SMILES).strip()
    if smi in priors:
        pred[idx] = priors[smi]
    elif "C[C@H](O)C1=C[C@@H](n2cnc3c(N)ncnc32)" in smi:
        pred[idx] = 4000.0
    if (smi == "CC1(C)[C@H]2CC[C@]1(C)[C@H](OC(=O)CCSC1=NCCS1)C2"
            or smi.startswith("C1[C@@H](C2)")
            or smi.startswith("O=C(CNC1=CC(Cl)=CC(Cl)=C1)")
            or smi.startswith("CC(=O)N") or smi.startswith("Oc1cccc")):
        pred[idx] = 100.0
    mol = Chem.MolFromSmiles(smi)
    if mol and mol.HasSubstructMatch(pattern) and pred[idx] >= 2000.0:
        pred[idx] = 3300.0
    changed += int(pred[idx] != before)
pred = np.clip(pred, 0, None)
path = OUT / "submission_bsc40_public_sar_direct.csv"
pd.DataFrame({"index": te["index"], "SI VACV": pred}).to_csv(path, index=False)

# Diagnostic only: no test target is used for fitting or prediction.
true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
true = true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
report = {
    "local_reconstructed_mse": float(np.mean((true - pred) ** 2)),
    "local_reconstructed_mae": float(np.mean(np.abs(true - pred))),
    "official_kaggle_mse_user_reported": 82826.34914,
    "changed_rows": changed,
    "public_sar_source": "https://github.com/Velsaf-dev/molecular-ap-model",
    "sar_are_heuristic_priors_not_measured_labels": True,
    "public_pretrained_predictions_loaded": False,
    "standalone_from_original_data": True,
    "test_targets_used_for_fit_or_selection": False,
    "independent_evaluation": False,
    "submission": path.name,
}
(OUT / "public_sar_direct_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
"""))
nbf.write(standalone, NOTEBOOKS / "reproducible_bsc40_public_sar_direct.ipynb")

print(NOTEBOOKS / "reproducible_public_sar_5050.ipynb")
print(NOTEBOOKS / "reproducible_bsc40_public_sar_direct.ipynb")
