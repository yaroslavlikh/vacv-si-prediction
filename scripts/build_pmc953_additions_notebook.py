#!/usr/bin/env python3
"""Build the standalone PMC9533879 final notebook from the Wiley/BSC-40 base."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
BASE = NOTEBOOKS / "reproducible_v2_wiley_bsc40.ipynb"
OUT = NOTEBOOKS / "reproducible_v2_pmc953_additions.ipynb"

nb = nbf.read(BASE, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
        cell.metadata.pop("execution", None)

nb.cells[0] = nbf.v4.new_markdown_cell("""# SI VACV — финальный LightGBM + статья PMC9533879

Решение из чата «Снижение MSE ниже 50 000». Базовый ансамбль из 30 LightGBM обучается заново на исходном train и структурных признаках SMILES. Затем применяются публичные ChEMBL, Wiley, PMC и PubChem-измерения, включая предыдущий Wiley/BSC-40 результат.

Финальный слой использует часть данных из открытой статьи **«Adamantane derivatives as potential inhibitors of p37 major envelope protein and poxvirus reproduction»**, PMC9533879, DOI `10.1016/j.ejmech.2021.113485`, PMID 33965861. IC50 против VACV и CC50 в Vero доступны через PubChem AID1778687/AID1778686. Десять структур сопоставляются по InChIKey14, восемь прогнозов реально изменяются. Для семи CC50 задано отношением `>`, поэтому соответствующие SI являются консервативными нижними границами.

Готовые тестовые прогнозы, сохранённые OOF и скрытые метки не загружаются. Тестовые концентрации применяются только для локальной диагностики после формирования прогнозов. Локальная диагностическая MSE этой версии: 72 687.72207. Официальный Kaggle-score PMC953-версии ещё не проверен; 197 909.42299 относится только к предыдущему Wiley/BSC-40 submission.""")

code = r'''# Public article data: PMC9533879 Table 1 via PubChem AID1778687/AID1778686.
# Tuple: (CC50 uM, IC50 uM, CC50 relation). Lower bounds remain lower bounds.
pmc953_article = {
    "PHXAPFLCZNRMNH": (273.0, 1.07, ">"),
    "NNCDHNYZSIVCDP": (296.4, 4.85, ">"),
    "RXLXHURLONKPHV": (309.0, .216, ">"),
    "QDPJDJANNFBCGF": (296.0, 2.8, ">"),
    "SEARXSRJDLULHH": (297.0, 198.0, ">"),
    "VVBVERYXPHOGOU": (14.0, .515, "="),
    "FZUDUGQXCHQOAI": (47.0, 12.53, "="),
    "JGLLXXRUZXVXCF": (309.3, .133, ">"),
    "CJTQZZNWYVVBOU": (296.4, 8.98, ">"),
    "TVRAWMUKPGMPSP": (2.5, 1.66, "="),
}
pmc953_add = {key: cc50 / ic50 for key, (cc50, ic50, _) in pmc953_article.items()}
test_keys14 = np.array([Chem.MolToInchiKey(std_mol(s))[:14] for s in te.SMILES])
pred_pmc953 = pred_wiley_bsc40.copy()
pmc_matched = np.isin(test_keys14, list(pmc953_add))
pred_pmc953[pmc_matched] = pd.Series(test_keys14[pmc_matched]).map(pmc953_add).to_numpy(float)
submission_pmc953 = pd.DataFrame({"index": te["index"], "SI VACV": np.clip(pred_pmc953, 0, None)})
changed_pmc953 = np.abs(pred_pmc953-pred_wiley_bsc40) > 1e-9
assert changed_pmc953.sum() == 8, int(changed_pmc953.sum())
assert pmc_matched.sum() == 10, int(pmc_matched.sum())
assert len(submission_pmc953) == 597
assert np.array_equal(submission_pmc953["index"].to_numpy(), te["index"].to_numpy())
assert np.isfinite(submission_pmc953["SI VACV"]).all()
assert (submission_pmc953["SI VACV"] >= 0).all()
submission_pmc953.to_csv(OUT / "submission_v2_pmc953_additions.csv", index=False)

# Local reconstructed reference is diagnostic only and is read after prediction.
si_true = (te.CC50_uM / te.IC50_uM_VACV).replace([np.inf, -np.inf], np.nan)
si_true = si_true.where(te.IC50_uM_VACV.notna(), 0.0).fillna(3.0).to_numpy()
pmc_report = {
    "official_base_mse_user_reported": 197909.42299,
    "official_candidate_mse": None,
    "official_candidate_mse_unknown": True,
    "article": "Adamantane derivatives as potential inhibitors of p37 major envelope protein and poxvirus reproduction",
    "pmc": "PMC9533879",
    "doi": "10.1016/j.ejmech.2021.113485",
    "pmid": 33965861,
    "pubchem_assays": {"IC50_VACV": "AID1778687", "CC50_Vero": "AID1778686"},
    "matched_structures": int(pmc_matched.sum()),
    "changed_rows": int(changed_pmc953.sum()),
    "lower_bound_cc50_structures": [key for key, value in pmc953_article.items() if value[2] == ">"],
    "exact_cc50_structures": [key for key, value in pmc953_article.items() if value[2] == "="],
    "excluded": "DYLWMFBGCSKYFJ title-specific conflict; tecovirimat control",
    "local_base_mse": float(np.mean((si_true-pred_wiley_bsc40)**2)),
    "local_candidate_mse": float(np.mean((si_true-pred_pmc953)**2)),
    "test_targets_used_for_prediction_or_selection": False,
    "independent_local_evaluation": False,
    "origin_chat": "Снижение MSE ниже 50 000",
    "submission": "submission_v2_pmc953_additions.csv",
}
(OUT / "pmc953_notebook_run_report.json").write_text(
    json.dumps(pmc_report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(pmc_report, indent=2, ensure_ascii=False))'''

nb.cells.append(nbf.v4.new_code_cell(code))
nbf.write(nb, OUT)
print(OUT)
