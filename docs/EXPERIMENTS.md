# Эксперименты и происхождение финального решения

## Эксперименты из чата «Достижение MSE ниже 200 000»

В разделах ниже перечислены только эксперименты из указанного чата. Финальное решение, перенесённое позже из чата «Снижение MSE ниже 50 000», описано отдельно и не смешивается с этим списком.

### Базовые модели и валидация

- `baseline_cv.py` — Morgan r2 2048 + MACCS + 9 RDKit-дескрипторов, несколько target transforms. Лучший canonical HGB+log1p: full 4 234 101, no-max 191 808.
- `tail_cv.py` и `robust_tail_cv.py` — tail weighting и внешние ChEMBL-строки. Лучший прямой LightGBM tailw1.0 после выбора: OOF no-max 136 910.66.
- `endpoint_multitask_cv.py` — раздельные log(IC50) и log(CC50), SI как отношение. OOF-only blend: full 4 086 612, no-max 69 454, no-top7 34 861. На локальном test чистый endpoint оказался нестабилен.
- `scaffold_target_encoding.py` — хуже базового ансамбля и не используется в финале.

### Предобученные представления

- `extract_chemberta.py`, `chemberta_cv.py` — frozen DeepChem/ChemBERTa-77M-MTR, 384-мерные embeddings. Canonical no-max 238 847, scaffold no-max 232 611. Хуже fingerprint baseline, в финал не включено.

### Дополнительные модели

- `catboost_cv.py` — OOF-отбор выбрал 100% LightGBM, 0% CatBoost.
- `tail_ml_candidates.py` — ExtraTrees на log(SI) показал canonical no-max 73 927.60. Raw ExtraTrees и raw kNN нестабильны.
- `et_cap_models.py` — capped raw MSE с tail weights. Лучший OOF no-max 223 017.73, хуже ExtraTrees log(SI).
- `soft_knn_cv.py` — мягкий kNN улучшил одну OOF-метрику, но ухудшил хвост и локальную диагностику, поэтому отклонён.

### Публичные данные и аналоги

- `expanded_chembl_pairs.py` — пары potency + CC50 по молекуле и публикации.
- `filter_augmentation.py`, `audit_augmentation.py` — проверка внешней разметки по формуле, train и первоисточникам. PubChem_NCGC_screen (605 строк) исключён из-за отсутствия DOI и независимой проверки; четыре ChEMBL-выброса SI>10000 удалены.
- `search_pubchem_analogs.py` — similarity-поиск до 200 аналогов Adefovir и sid103210702. Максимальный SI аналогов Adefovir = 66.67, sid103210702 = 314.38; хвост не закрыт.
- `source_ablation.py` — сравнение original/expanded/literature lookup.

### Confidence-aware варианты

- `confidence_shrink.py` — прогноз уменьшается пропорционально Tanimoto² при низком сходстве. OOF no-top7 = 17 780.53, локальная MSE = 80 889.37.
- `confidence_variants.py` — пять вариантов Tanimoto¹/²/⁴ и две смеси. Локальные MSE находятся в диапазоне 80 775–81 163.
- `reproducible_ensemble_ncgc.ipynb` — public hybrid и confidence-aware ExtraTrees 50/50. Локальная MSE `v2_conf_5050` = **75 990.19713**, OOF no-max = 98 591.12, no-top7 = 16 642.30. Это прежний кандидат, а не текущий финал.

### Официальные результаты экспериментов

- Public hybrid v2: около 203 000 MSE на Kaggle.
- Чистый ExtraTrees: 211 415.38617 MSE на Kaggle.
- Confidence-ансамбль `v2_conf_5050` на Kaggle не проверялся.

## Финал из чата «Снижение MSE ниже 50 000»

По отдельному запросу пользователя текущей рабочей версией выбран `submission_v2_pmc953_additions.csv`. Готовый CSV в Git не включается: он заново создаётся notebook `reproducible_v2_pmc953_additions.ipynb`.

Финальный pipeline:

1. 30 LightGBM-моделей обучаются с нуля на исходном train и структурных SMILES-признаках.
2. Применяются публичные ChEMBL/PMC/PubChem/Wiley-измерения.
3. Добавляются 12 post-v2 public-source соответствий; база имеет сообщённый пользователем Kaggle MSE 201 013.27709 и локальную диагностическую MSE 74 041.22172.
4. Wiley `2c`, `3c-ks-347` и ChEMBL BSC-40 для тековиримата изменяют четыре строки. Для тековиримата CC50>100 µM сохраняется как нижняя граница. Этот промежуточный baseline имеет сообщённый пользователем официальный **Kaggle MSE 197 909.42299** и локальную диагностическую MSE **72 892.61237**.
5. Добавляются некоторые измерения из Table 1 статьи **«Adamantane derivatives as potential inhibitors of p37 major envelope protein and poxvirus reproduction»**, PMC9533879, DOI `10.1016/j.ejmech.2021.113485`, PMID 33965861. IC50 против VACV и CC50 в Vero проверены через PubChem AID1778687/AID1778686.
6. Статьёвный слой сопоставляет 10 структур по InChIKey14 и реально меняет 8 строк. Для 7 CC50 действует отношение `>`, поэтому SI сохраняется как консервативная нижняя граница. Конфликтный DYLWMFBGCSKYFJ и контрольный тековиримат исключены.
7. Локальная диагностическая MSE PMC953-версии — **72 687.72207**. Официальный score этой версии пока неизвестен; результат 197 909.42299 к ней не переносится.

Локальная диагностика не является независимой оценкой и не используется для обучения или выбора финальных значений. Числовая проверка независимого запуска записывается в `reports/pmc953_reproduction.json`; официальные результаты сохранены в `reports/official_scores.json`. Проверка предыдущего Wiley/BSC-40 baseline остаётся в `reports/wiley_bsc40_reproduction.json`.
