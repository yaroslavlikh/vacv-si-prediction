# Эксперименты из чата «Достижение MSE ниже 200 000»

В этом файле перечислены только эксперименты из указанного чата. Другие чаты и исторические модели не включены.

## Базовые модели и валидация

- `baseline_cv.py` — Morgan r2 2048 + MACCS + 9 RDKit-дескрипторов, несколько target transforms. Лучший canonical HGB+log1p: full 4 234 101, no-max 191 808.
- `tail_cv.py` и `robust_tail_cv.py` — tail weighting и внешние ChEMBL-строки. Лучший прямой LightGBM tailw1.0 после выбора: OOF no-max 136 910.66.
- `endpoint_multitask_cv.py` — раздельные log(IC50) и log(CC50), SI как отношение. OOF-only blend: full 4 086 612, no-max 69 454, no-top7 34 861. На локальном test чистый endpoint оказался нестабилен.
- `scaffold_target_encoding.py` — хуже базового ансамбля и не используется в финале.

## Предобученные представления

- `extract_chemberta.py`, `chemberta_cv.py` — frozen DeepChem/ChemBERTa-77M-MTR, 384-мерные embeddings. Canonical no-max 238 847, scaffold no-max 232 611. Хуже fingerprint baseline, в финал не включено.

## Дополнительные модели

- `catboost_cv.py` — OOF-отбор выбрал 100% LightGBM, 0% CatBoost.
- `tail_ml_candidates.py` — ExtraTrees на log(SI) показал canonical no-max 73 927.60 и стал основой финала. Raw ExtraTrees и raw kNN нестабильны.
- `et_cap_models.py` — capped raw MSE с tail weights. Лучший OOF no-max 223 017.73, хуже ExtraTrees log(SI).
- `soft_knn_cv.py` — мягкий kNN улучшил одну OOF-метрику, но ухудшил хвост и локальную диагностику, поэтому отклонён.

## Публичные данные и аналоги

- `expanded_chembl_pairs.py` — пары potency + CC50 по молекуле и публикации.
- `filter_augmentation.py`, `audit_augmentation.py` — 985 labeled строк проверены по формуле, train и первоисточникам. PubChem_NCGC_screen (605 строк) исключён из-за отсутствия DOI и независимой проверки; четыре ChEMBL-выброса SI>10000 удалены. Итог: 375 отфильтрованных уникальных структур.
- `search_pubchem_analogs.py` — similarity-поиск до 200 аналогов Adefovir и sid103210702. Максимальный SI аналогов Adefovir = 66.67, sid103210702 = 314.38; хвост не закрыт.
- `source_ablation.py` — сравнение original/expanded/literature lookup.

## Confidence-aware варианты

- `confidence_shrink.py` — прогноз уменьшается пропорционально Tanimoto² при низком сходстве. OOF no-top7 = 17 780.53, локальная MSE = 80 889.37.
- `confidence_variants.py` — пять вариантов Tanimoto¹/²/⁴ и две смеси. Локальные MSE находятся в диапазоне 80 775–81 163. Официальные score этих вариантов пока не проверены.

## Финальный ансамбль

`reproducible_ensemble_ncgc.ipynb` объединяет public hybrid и confidence-aware ExtraTrees в пропорции 50/50. Локальная MSE `v2_conf_5050` = **75 990.20**, OOF no-max = 98 591.12, no-top7 = 16 642.30. Файл `ensemble_ncgc_report.json` содержит также фиксированные варианты весов.

## Официальные результаты

- Гибрид v2: около 203 000 MSE на Kaggle.
- Чистый ExtraTrees: 211 415.38617 MSE на Kaggle.
- Финальный ансамбль `v2_conf_5050` ещё не отправлялся на Kaggle.
