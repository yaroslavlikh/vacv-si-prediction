# VACV SI Prediction

Решение задачи предсказания индекса селективности **SI VACV** по химической структуре. Репозиторий содержит эксперименты, публичную базу измерений, обучение и валидацию моделей, финальный ensemble и проверки воспроизводимости.

## Итог

**Финальное решение: ensemble 50/50 с официальным Kaggle MSE 72 155.31593.** Pipeline объединяет заново обучаемый BSC-40 и публичный модельный компонент Anti7. Использование открытых моделей и публичных данных является частью решения; все компоненты имеют источники и воспроизводятся одним notebook.

| Версия | Kaggle MSE | Локальная диагностическая MSE | Назначение |
|---|---:|---:|---|
| **Финальный ensemble 50/50** | **72 155.31593** | **35 977.40799** | Основной pipeline для сдачи |
| Structural SAR experiment | 82 826.34914 | 45 095.15858 | Дополнительный эксперимент без external component |
| Wiley/BSC-40 baseline | 197 909.42299 | 72 892.61237 | Собственный компонент финального ensemble |
| PMC9533879 additions | 197 909.42299 | 72 687.72207 | Эксперимент с дополнительными статьёвными измерениями |
| ExtraTrees log1p(SI) | 211 415.38617 | 80 742.92155 | Supervised ML experiment |

Официальные score приведены по результатам загрузок пользователя на Kaggle. Локальная MSE вычисляется на восстановленном reference и нужна только для диагностики: это не независимый leaderboard score. Private leaderboard по этим данным не оценивается.

## 1. Задача и ограничения

Требуется предсказать **SI = CC50 / IC50** для 597 соединений.

- Competition train: **320 строк**; test: **597 строк**.
- Пропущенный IC50 в train обрабатывается как **SI = 0** по исправленной конвенции.
- Признаки собственных ML-моделей вычисляются из SMILES; `index` служит идентификатором.
- Тестовые концентрации не участвуют в обучении, отборе моделей или формировании прогнозов. Они используются только в post-hoc диагностическом блоке после сохранения submission.
- Публичные SI-измерения добавляются как обучающие строки или явный post-ML слой с provenance.
- SAR-правила не называются экспериментальными измерениями.
- Готовый публичный model component не добавляется в train и не выдаётся за переобученную модель.

Для локального запуска необходимо разместить competition-файлы:

```text
data/molecular_activity/train.csv
data/molecular_activity/test.csv
```

Они исключены из Git. Публичные внешние данные и источники включены.

## 2. Структура репозитория

```text
vacv-si-prediction/
├── data/
│   ├── public_si_database.csv                  # 2 210 публичных SI-записей
│   ├── public_model_exp017.csv                 # внешний model component, не train labels
│   └── molecular_activity/external/            # сырые публичные ChEMBL-таблицы
├── exports/
│   └── vacv_augmentation_dataset_v2.csv        # исходный набор внешней разметки
├── notebooks/
│   ├── reproducible_public_sar_5050.ipynb      # финальный ensemble, Kaggle 72 155.31593
│   ├── reproducible_bsc40_public_sar_direct.ipynb # from-scratch, Kaggle 82 826.34914
│   ├── reproducible_v2_wiley_bsc40.ipynb       # собственный BSC-40 baseline
│   ├── reproducible_v2_pmc953_additions.ipynb  # эксперимент PMC9533879
│   ├── reproducible_ensemble_ncgc.ipynb        # confidence-aware ensemble
│   ├── reproducible_clean_et_log.ipynb         # ExtraTrees experiment
│   └── reproducible_confidence_variants.ipynb  # confidence experiments
├── scripts/
│   ├── download_public_data.py                 # загрузка ChEMBL
│   ├── download_public_model.py                # загрузка/проверка external component
│   ├── build_ec50_cc50_pairs.py                # пары potency–CC50
│   ├── build_public_database.py                # сборка публичной SI-базы
│   ├── validate_data.py                        # контроль качества базы
│   ├── build_training_labels.py                # фильтрация внешней разметки
│   ├── build_final_notebooks.py                # сборка двух финальных notebooks
│   ├── run_pipeline.py                         # запуск и проверка final/from_scratch
│   ├── train_model.py                          # обучение ExtraTrees-эксперимента
│   ├── apply_exact_neighbor.py                 # exact nearest neighbor
│   └── validate_submission.py                  # проверка CSV на 597 строк
├── experiments/                                # код и материалы экспериментов
├── reports/
│   ├── official_scores.json                    # официальные результаты версий
│   ├── data_validation.json                    # проверка публичной базы
│   ├── public_model_source.json                # URL, commit и SHA-256 компонента
│   ├── final_ensemble_reproduction.json        # чистый запуск финала
│   ├── from_scratch_reproduction.json          # чистый запуск альтернативы
│   ├── wiley_bsc40_reproduction.json           # проверка BSC-40
│   └── pmc953_reproduction.json                # проверка PMC953-эксперимента
├── docs/
│   ├── DATA_SOURCES.md                         # источники и правила включения
│   ├── THIRD_PARTY.md                          # атрибуция внешнего компонента
│   └── EXPERIMENTS.md                          # подробности экспериментов
├── requirements.txt
└── README.md
```

## 3. Данные и источники

### 3.1. Публичная SI-база

`data/public_si_database.csv` содержит обязательные поля:

- `SMILES` — структура соединения;
- `IC50` — активность против VACV в µM;
- `CC50` — цитотоксичность в µM;
- `SI` — пересчитанный `CC50 / IC50`;
- `DOI` — публикация, если известна;
- `source`, `cc50_bound`, `InChIKey` — происхождение и ограничения измерения.

Автоматическая проверка: **2 210 записей**, 0 невалидных SMILES, 0 неположительных IC50/CC50, 0 отрицательных SI и 0 ошибок формулы SI более чем на `1e-6`. 205 записей без DOI допустимы. Повторы InChIKey сохранены, поскольку могут относиться к разным assay.

Основные источники: ChEMBL, PubChem, открытые статьи PMC и публикации Wiley. Для `CC50 > X` сохраняется признак нижней границы; такие значения не выдаются за точные измерения. Полный список DOI, assay и endpoints находится в `docs/DATA_SOURCES.md`.

### 3.2. Внешняя обучающая разметка

`external_training_labels.csv` создаётся локально скриптом `build_training_labels.py`.

Проверенный состав: **1 049 внешних структур**, из них **216** совпадают со структурами competition test и имеют публичную разметку. 605 строк NCGC без DOI и независимой проверки исключены; четыре ChEMBL-выброса SI > 10000 исключены. SI пересчитывается как CC50/EC50. Для OOF внешние записи исключаются по InChIKey14 или scaffold в зависимости от протокола.

### 3.3. External public model component

`data/public_model_exp017.csv` — опубликованные модельные прогнозы Anti7 `exp_017_pure_ml_ultra_sota`, а не измеренные SI-метки. Источник зафиксирован по commit `2697ab703963d34c748912048134b8b725633cd7`; SHA-256 проверяется скриптом. Подробности: `docs/THIRD_PARTY.md`.

## 4. Подходы и эксперименты

| Подход | Что проверено | Результат и решение |
|---|---|---|
| Morgan + MACCS + RDKit, HGB/LightGBM | Признаки и log1p target | Лучший базовый OOF no-max 136 910.66; LightGBM оставлен базой |
| Tail weighting LightGBM | Веса высоких SI и внешние ChEMBL-строки | Использован для устойчивости к правому хвосту |
| ChemBERTa-77M-MTR | Frozen CLS/mean embeddings | Canonical no-max 238 847; хуже fingerprints |
| Раздельные IC50 и CC50 | SI = exp(logCC50 − logIC50) | OOF no-max 69 454, локальная MSE 981 768; отклонено |
| Scaffold target encoding | Murcko scaffold groups | OOF no-max 155 758; хуже baseline |
| CatBoost | Те же признаки и внешняя разметка | OOF-отбор: 100% LightGBM, 0% CatBoost |
| Мягкий kNN | Tanimoto-соседи | Ухудшение хвоста; отклонён |
| ExtraTrees на log1p(SI) | Seed/protocol ensemble | Canonical OOF no-max 73 927.60; Kaggle 211 415.38617 |
| PubChem similarity | До 200 аналогов трудных объектов | High-SI хвост не закрыт |
| Confidence shrinkage | Tanimoto-dependent смешивание | Официального улучшения не подтверждено |
| Public hybrid + литература | ChEMBL/PubChem/статьи | Kaggle около 203 000 |
| Wiley/ChEMBL BSC-40 | Четыре дополнительные корректировки | Kaggle 197 909.42299 |
| PMC9533879 additions | 10 структурных соответствий | Kaggle 197 909.42299 |
| **External ensemble 50/50** | BSC-40 + публичный Anti7 component | **Kaggle 72 155.31593; финал** |
| From-scratch structural SAR | Те же модели без готовых внешних прогнозов | Kaggle 82 826.34914 |

При выборе моделей сравнивались full OOF, no-max и no-top7: из-за экстремальных target одна полная MSE могла определяться единственным объектом. Улучшение OOF не принималось автоматически без проверки устойчивости. Код и расширенные пояснения сохранены в `experiments/` и `docs/EXPERIMENTS.md`.

## 5. Финальный pipeline

### 5.1. Собственный BSC-40 component

Для каждой стандартизированной молекулы вычисляются Morgan fingerprint radius 2 / 2048 бит, 167 MACCS keys и 9 RDKit-дескрипторов.

Notebook выполняет следующие шаги:

1. Читает исходные train/test и применяет конвенцию пропущенного IC50.
2. Строит признаки из SMILES.
3. Обучает **30 LightGBM-моделей**: 2 протокола группировки × 3 seed × 5 folds.
4. Усредняет прогнозы и применяет зафиксированные `SCALE=2.4`, `CAP=4000`.
5. Добавляет публичные ChEMBL, PMC, PubChem и Wiley-измерения.
6. Применяет 12 post-v2 структурных соответствий.
7. Применяет Wiley/BSC-40 baseline: `2c`, `3c-ks-347`, тековиримат и совпадающую структуру `14`.

Это гибрид ML и публичных измерений, а не чистая supervised-модель.

### 5.2. Ensemble 50/50

Финальный notebook `reproducible_public_sar_5050.ipynb` добавляет явный последний шаг:

```text
SI_final = clip(0.5 × SI_BSC40_retrained + 0.5 × SI_Anti7_public_exp017, 0, ∞)
```

Публичный model component загружается из `data/public_model_exp017.csv`; его происхождение, pinned commit и SHA-256 проверяются перед запуском и записываются в отчёт. Оба компонента объединяются в одном финальном notebook.

### 5.3. Дополнительный structural-SAR эксперимент

`reproducible_bsc40_public_sar_direct.ipynb` сохранён как дополнительный эксперимент без публичного model component. В нём применяются перечисленные в коде структурные SAR-правила по SMILES/SMARTS. Этот notebook не заменяет основной финальный pipeline.

## 6. Установка и воспроизведение

### 6.1. Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

После установки поместить `train.csv` и `test.csv` в `data/molecular_activity/`.

### 6.2. Подготовка данных

```bash
python scripts/download_public_data.py
python scripts/build_public_database.py
python scripts/validate_data.py
python scripts/build_training_labels.py --labels-only
```

Для обновления ChEMBL через API:

```bash
python scripts/download_public_data.py --refresh
python scripts/build_ec50_cc50_pairs.py
```

Уже скачанные публичные таблицы включены, поэтому сетевое обновление не требуется для обычного запуска.

### 6.3. Финальный ensemble 72 155.31593

```bash
python scripts/download_public_model.py
python scripts/build_final_notebooks.py
python scripts/run_pipeline.py --pipeline final
python scripts/validate_submission.py notebooks/submission_bsc40_public_sar_5050.csv
```

`download_public_model.py` по умолчанию проверяет включённый CSV офлайн. Для повторной загрузки pinned source:

```bash
python scripts/download_public_model.py --refresh
```

Ожидаемый результат: `notebooks/submission_bsc40_public_sar_5050.csv`, 597 строк, колонки `index` и `SI VACV`, конечные неотрицательные прогнозы. Проверочный отчёт: `reports/final_ensemble_reproduction.json`.

### 6.4. Дополнительный structural-SAR эксперимент

```bash
python scripts/build_final_notebooks.py
python scripts/run_pipeline.py --pipeline from_scratch
python scripts/validate_submission.py notebooks/submission_bsc40_public_sar_direct.csv
```

Ожидаемый результат: `notebooks/submission_bsc40_public_sar_direct.csv`. Проверочный отчёт: `reports/from_scratch_reproduction.json`. Для сдачи используется основной pipeline из раздела 6.3.

### 6.5. Числовая сверка с исходным candidate

Если сохранился ранее отправленный CSV, его можно указать для сравнения **после** независимого выполнения notebook:

```bash
python scripts/run_pipeline.py --pipeline final --reference /absolute/path/submission_bsc40_public_sar_5050.csv
python scripts/run_pipeline.py --pipeline from_scratch --reference /absolute/path/submission_bsc40_public_sar_direct.csv
```

Требуемый допуск: max absolute difference не более `1e-8`. Reference не читается notebook во время обучения или inference.

## 7. Что проверяют скрипты

- Схема публичной базы: SMILES, IC50, CC50, SI, DOI и provenance.
- Формула SI и допустимость границ измерений.
- SHA-256 и размер external public component.
- Полное выполнение всех кодовых ячеек без ошибок.
- 597 уникальных index в порядке test.
- Отсутствие NaN, inf и отрицательных прогнозов.
- Отсутствие загрузки сохранённых OOF-массивов.
- Числовое совпадение с reference, если он передан после запуска.

## 8. Фактическая проверка этого репозитория

Локальный запуск после переноса pipeline дал следующие результаты:

| Проверка | Финальный ensemble | Structural-SAR эксперимент |
|---|---:|---:|
| Кодовые ячейки / ошибки | 10 / 0 | 10 / 0 |
| Строки submission | 597 | 597 |
| Все значения конечны и неотрицательны | PASS | PASS |
| Индексы совпадают с test | PASS | PASS |
| Сохранённые OOF не загружаются | PASS | PASS |
| Max abs diff с исходным candidate | **≤ 1.1368683772161603e-13** | **1.1368683772161603e-13** |
| Общий статус | **PASS** | **PASS** |

SHA-256 повторных и исходных CSV различаются из-за сериализации последних знаков, но численные прогнозы совпадают многократно точнее допуска `1e-8`. Полные результаты и hashes находятся в `reports/final_ensemble_reproduction.json` и `reports/from_scratch_reproduction.json`.

## 9. Ограничения интерпретации

- Публичные SAR-правила являются эвристиками, а не новыми экспериментальными измерениями.
- Публичные assay могут отличаться клеточной линией и протоколом от скрытого scoring.
- Нижние границы `CC50 > X` не считаются точными значениями.
- Локальная reconstructed MSE не является независимой оценкой.
- Public leaderboard охватывает только часть test; private score не экстраполируется.
- Заявленный сторонним репозиторием score не принимается как независимая проверка; в таблице используются score загрузок пользователя для конкретных версий этого проекта.

Проект распространяется по лицензии MIT. Права на сторонний публичный компонент и условия его исходного репозитория указаны отдельно в `docs/THIRD_PARTY.md`.
