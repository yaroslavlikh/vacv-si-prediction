# VACV SI Prediction

Воспроизводимый pipeline предсказания индекса селективности **SI VACV** по химической структуре. Репозиторий содержит эксперименты, публичную базу измерений, обучение и валидацию моделей, финальный notebook и проверки воспроизводимости.

## Итог

**Финальное решение: BSC-40 + structural SAR, официальный Kaggle MSE — 82 826.34914.**

- 30 LightGBM-моделей обучаются заново на исходном train.
- Используются структурные SMILES-признаки и публичные ChEMBL/PubChem/Wiley-измерения.
- Structural SAR-правила применяются только по SMILES/SMARTS.
- Готовые внешние модели, чужие model predictions и сохранённые OOF не загружаются.
- Финальный notebook воспроизводит 597 прогнозов без ошибок.

Официальный score приведён по результату загрузки пользователя на Kaggle. Локальная reconstructed MSE равна **45 095.15858**, но используется только для диагностики и не является независимым leaderboard score. Private leaderboard по этим данным не оценивается.

## 1. Задача и ограничения

Требуется предсказать **SI = CC50 / IC50** для 597 соединений.

- Competition train: **320 строк**; test: **597 строк**.
- Пропущенный IC50 в train обрабатывается как **SI = 0** по исправленной конвенции.
- Признаки ML-моделей вычисляются из SMILES; `index` служит идентификатором.
- Тестовые концентрации не участвуют в обучении, отборе моделей или формировании прогнозов. Они используются только в post-hoc диагностическом блоке после сохранения submission.
- Публичные SI-измерения добавляются как обучающие строки или явный post-ML слой с provenance.
- SAR-правила не называются экспериментальными измерениями.
- Готовый внешний model component в финальном pipeline не используется.

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
│   └── molecular_activity/external/            # сырые публичные ChEMBL-таблицы
├── exports/
│   └── vacv_augmentation_dataset_v2.csv        # исходный набор внешней разметки
├── notebooks/
│   ├── reproducible_bsc40_public_sar_direct.ipynb # финальный pipeline, Kaggle 82 826.34914
│   ├── reproducible_v2_wiley_bsc40.ipynb       # BSC-40 baseline
│   ├── reproducible_v2_pmc953_additions.ipynb  # эксперимент PMC9533879
│   ├── reproducible_ensemble_ncgc.ipynb        # confidence-aware ensemble
│   ├── reproducible_clean_et_log.ipynb         # ExtraTrees experiment
│   └── reproducible_confidence_variants.ipynb  # confidence experiments
├── scripts/
│   ├── download_public_data.py                 # загрузка ChEMBL
│   ├── build_ec50_cc50_pairs.py                # пары potency–CC50
│   ├── build_public_database.py                # сборка публичной SI-базы
│   ├── validate_data.py                        # контроль качества базы
│   ├── build_training_labels.py                # фильтрация внешней разметки
│   ├── run_pipeline.py                         # запуск и проверка финала
│   ├── train_model.py                          # обучение ExtraTrees-эксперимента
│   ├── apply_exact_neighbor.py                 # exact nearest neighbor
│   └── validate_submission.py                  # проверка CSV на 597 строк
├── experiments/                                # код и материалы экспериментов
├── reports/
│   ├── official_scores.json                    # официальные результаты версий
│   ├── data_validation.json                    # проверка публичной базы
│   ├── final_reproduction.json                 # чистый запуск финала
│   ├── wiley_bsc40_reproduction.json           # проверка BSC-40
│   └── pmc953_reproduction.json                # проверка PMC953-эксперимента
├── docs/
│   ├── DATA_SOURCES.md                         # источники и правила включения
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

### 3.3. Structural SAR

Финальный слой содержит явно перечисленные структурные эвристики из открытого решения `Velsaf-dev/molecular-ap-model`. Они сопоставляются по SMILES/SMARTS и меняют 35 прогнозов относительно BSC-40. Это алгоритмические приоритеты, а не новые assay-измерения. Готовые модели и прогнозы из этого источника не используются.

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
| **BSC-40 + structural SAR** | 35 структурных правил без внешней модели | **Kaggle 82 826.34914; финал** |

При выборе моделей сравнивались full OOF, no-max и no-top7: из-за экстремальных target одна полная MSE могла определяться единственным объектом. Улучшение OOF не принималось автоматически без проверки устойчивости. Код и расширенные пояснения сохранены в `experiments/` и `docs/EXPERIMENTS.md`.

## 5. Финальный pipeline

### 5.1. Структурные признаки и обучение

Для каждой стандартизированной молекулы вычисляются Morgan fingerprint radius 2 / 2048 бит, 167 MACCS keys и 9 RDKit-дескрипторов.

Notebook выполняет следующие шаги:

1. Читает исходные train/test и применяет конвенцию пропущенного IC50.
2. Строит признаки из SMILES.
3. Обучает **30 LightGBM-моделей**: 2 протокола группировки × 3 seed × 5 folds.
4. Усредняет прогнозы и применяет зафиксированные `SCALE=2.4`, `CAP=4000`.
5. Добавляет публичные ChEMBL, PMC, PubChem и Wiley-измерения.
6. Применяет 12 post-v2 структурных соответствий.
7. Применяет Wiley/BSC-40 baseline: `2c`, `3c-ks-347`, тековиримат и совпадающую структуру `14`.
8. Применяет structural SAR-правила только по SMILES/SMARTS.
9. Сохраняет 597 прогнозов и только после этого вычисляет локальную диагностику.

Это гибрид собственных ML-моделей, публичных измерений и атрибутированных SAR-эвристик. Готовые submissions, сохранённые OOF и внешние модели не загружаются.

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

### 6.3. Финальный запуск

```bash
python scripts/run_pipeline.py --pipeline final
python scripts/validate_submission.py notebooks/submission_bsc40_public_sar_direct.csv
```

Ожидаемый результат: `notebooks/submission_bsc40_public_sar_direct.csv`, 597 строк, колонки `index` и `SI VACV`, конечные неотрицательные прогнозы. Проверочный отчёт: `reports/final_reproduction.json`.

### 6.4. Числовая сверка с исходным candidate

Если сохранился ранее отправленный CSV, его можно указать для сравнения **после** независимого выполнения notebook:

```bash
python scripts/run_pipeline.py --pipeline final \
  --reference /absolute/path/submission_bsc40_public_sar_direct.csv
```

Требуемый допуск: max absolute difference не более `1e-8`. Reference не читается notebook во время обучения или inference.

## 7. Что проверяют скрипты

- Схема публичной базы: SMILES, IC50, CC50, SI, DOI и provenance.
- Формула SI и допустимость границ измерений.
- Полное выполнение всех кодовых ячеек без ошибок.
- 597 уникальных index в порядке test.
- Отсутствие NaN, inf и отрицательных прогнозов.
- Отсутствие загрузки готовых submissions и сохранённых OOF-массивов.
- Числовое совпадение с reference, если он передан после запуска.

## 8. Фактическая проверка этого репозитория

| Проверка | Результат |
|---|---:|
| Кодовые ячейки / ошибки | 10 / 0 |
| Строки submission | 597 |
| Все значения конечны и неотрицательны | PASS |
| Индексы совпадают с test | PASS |
| Готовые submissions и OOF не загружаются | PASS |
| Max abs diff с исходным candidate | **≤ 2.2737367544323206e-13** |
| Общий статус | **PASS** |

SHA-256 повторного и исходного CSV могут различаться из-за сериализации последних знаков, но численные прогнозы совпадают многократно точнее допуска `1e-8`. Полный результат и hashes находятся в `reports/final_reproduction.json`.

## 9. Ограничения интерпретации

- Публичные SAR-правила являются эвристиками, а не новыми экспериментальными измерениями.
- Публичные assay могут отличаться клеточной линией и протоколом от скрытого scoring.
- Нижние границы `CC50 > X` не считаются точными значениями.
- Локальная reconstructed MSE не является независимой оценкой.
- Public leaderboard охватывает только часть test; private score не экстраполируется.

Проект распространяется по лицензии MIT.
