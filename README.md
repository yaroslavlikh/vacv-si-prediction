# VACV SI Prediction

Воспроизводимый pipeline предсказания индекса селективности **SI VACV** по SMILES. Репозиторий содержит обучение и валидацию моделей, публичную базу измерений, источники данных, эксперименты и проверку финального запуска.

## Итог

**Финальное решение: Wiley/BSC-40 + PMC9533879, официальный Kaggle MSE — 197 909.42299.**

- 30 LightGBM-моделей обучаются заново на исходном train.
- Признаки вычисляются из SMILES: Morgan fingerprints, MACCS и RDKit-дескрипторы.
- Используются публичные измерения ChEMBL, PubChem, PMC и Wiley с указанием источников.
- Готовые внешние модели, чужие model predictions, сохранённые OOF и неподтверждённые SAR-числа не используются.
- Финальный notebook воспроизводит 597 прогнозов без ошибок.

Официальный score приведён по загрузке пользователя. Локальная reconstructed MSE равна **72 687.72207**, но нужна только для диагностики и не является независимым leaderboard score. Private leaderboard не оценивается.

## 1. Задача

Требуется предсказать **SI = CC50 / IC50** для 597 соединений.

- Competition train: **320 строк**; test: **597 строк**.
- Пропущенный IC50 в train обрабатывается как **SI = 0** по исправленной конвенции.
- `index` используется только как идентификатор.
- Тестовые концентрации не участвуют в обучении, отборе или прогнозировании. Они вычисляются только в post-hoc диагностике после сохранения submission.
- Для сопоставления структур используются публичные измерения с provenance; нижние границы вида `CC50 > X` не выдаются за точные значения.

Для локального запуска необходимо разместить:

```text
data/molecular_activity/train.csv
data/molecular_activity/test.csv
```

Эти файлы исключены из Git.

## 2. Структура репозитория

```text
vacv-si-prediction/
├── data/
│   ├── public_si_database.csv                  # 2 210 публичных SI-записей
│   └── molecular_activity/external/            # сырые публичные ChEMBL-таблицы
├── exports/
│   └── vacv_augmentation_dataset_v2.csv        # исходный набор внешней разметки
├── notebooks/
│   ├── reproducible_v2_pmc953_additions.ipynb       # финальный pipeline
│   ├── reproducible_v2_pmc953_additions.ipynb  # дополнительный литературный эксперимент
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
│   ├── run_wiley_bsc40.py                      # независимый runner BSC-40
│   ├── train_model.py                          # обучение ExtraTrees-эксперимента
│   └── validate_submission.py                  # проверка CSV на 597 строк
├── experiments/                                # код и материалы экспериментов
├── reports/
│   ├── official_scores.json                    # официальные результаты версий
│   ├── data_validation.json                    # проверка публичной базы
│   ├── final_reproduction.json                 # чистый запуск финала
│   ├── wiley_bsc40_reproduction.json           # независимая проверка BSC-40
│   └── pmc953_reproduction.json                # проверка PMC953-эксперимента
├── docs/
│   ├── DATA_SOURCES.md                         # источники и правила включения
│   ├── PUBLIC_SOLUTION_DATA_SOURCES.md         # инвентарь источников публичного решения
│   └── EXPERIMENTS.md                          # подробности экспериментов
├── requirements.txt
└── README.md
```

## 3. Данные и источники

### 3.1. Публичная SI-база

`data/public_si_database.csv` содержит:

- `SMILES` — структуру соединения;
- `IC50` — активность против VACV в µM;
- `CC50` — цитотоксичность в µM;
- `SI` — пересчитанный `CC50 / IC50`;
- `DOI` — публикацию, если известна;
- `source`, `cc50_bound`, `InChIKey` — происхождение и ограничения измерения.

Проверка: **2 210 записей**, 0 невалидных SMILES, 0 неположительных IC50/CC50, 0 отрицательных SI и 0 ошибок формулы более чем на `1e-6`. 205 записей без DOI допустимы. Повторы InChIKey сохранены, поскольку могут относиться к разным assay.

Основные источники: ChEMBL, PubChem, открытые статьи PMC и Wiley. Полный список DOI, assay и endpoints находится в `docs/DATA_SOURCES.md`.

### 3.2. Внешняя обучающая разметка

`external_training_labels.csv` создаётся скриптом `build_training_labels.py`.

Проверенный состав: **1 049 внешних структур**, из них **216** совпадают со структурами test и имеют публичную разметку. 605 строк NCGC без DOI и независимой проверки исключены; четыре ChEMBL-выброса SI > 10000 исключены. SI пересчитывается как CC50/EC50. Для OOF внешние записи исключаются по InChIKey14 или scaffold в зависимости от протокола.

## 4. Подходы и эксперименты

| Подход | Результат и решение |
|---|---|
| Morgan + MACCS + RDKit, HGB/LightGBM | Лучший базовый OOF no-max 136 910.66; LightGBM оставлен базой |
| Tail weighting LightGBM | Использован для устойчивости к правому хвосту |
| ChemBERTa-77M-MTR | Canonical no-max 238 847; хуже fingerprints |
| Раздельные IC50 и CC50 | OOF no-max 69 454, локальная MSE 981 768; отклонено |
| Scaffold target encoding | OOF no-max 155 758; хуже baseline |
| CatBoost | OOF-отбор: 100% LightGBM, 0% CatBoost |
| Мягкий kNN | Ухудшение хвоста; отклонён |
| ExtraTrees на log1p(SI) | Canonical OOF no-max 73 927.60; Kaggle 211 415.38617 |
| Confidence shrinkage | Официального улучшения не подтверждено |
| Public hybrid + литература | Kaggle около 203 000 |
| Post-v2 public labels | Kaggle 201 013.27709 |
| **Wiley/ChEMBL BSC-40** | **Kaggle 197 909.42299; финал** |
| PMC9533879 additions | Тот же Kaggle 197 909.42299; дополнительный эксперимент |

При выборе моделей сравнивались full OOF, no-max и no-top7, поскольку полная MSE могла определяться одним экстремальным объектом. Улучшение OOF не принималось автоматически без проверки устойчивости. Подробности сохранены в `experiments/` и `docs/EXPERIMENTS.md`.

## 5. Финальный pipeline

Для каждой стандартизированной молекулы вычисляются Morgan fingerprint radius 2 / 2048 бит, 167 MACCS keys и 9 RDKit-дескрипторов.

`reproducible_v2_pmc953_additions.ipynb` выполняет:

1. Читает train/test и применяет конвенцию пропущенного IC50.
2. Строит признаки из SMILES.
3. Обучает **30 LightGBM-моделей**: 2 протокола группировки × 3 seed × 5 folds.
4. Усредняет прогнозы и применяет зафиксированные `SCALE=2.4`, `CAP=4000`.
5. Добавляет публичные ChEMBL, PMC, PubChem и Wiley-измерения.
6. Применяет 12 post-v2 структурных соответствий.
7. Применяет Wiley/BSC-40: `2c`, `3c-ks-347`, тековиримат и совпадающую структуру `14`.
8. Добавляет 10 структурных соответствий из Table 1 статьи PMC9533879, реально изменяя 8 строк; DOI `10.1016/j.ejmech.2021.113485`, PubChem AID1778687/AID1778686.
9. Сохраняет 597 прогнозов и только после этого вычисляет локальную диагностику.

Для тековиримата используется ChEMBL1142975: EC50 = 0.04 µM, CC50 > 100 µM, поэтому SI = 2500 является консервативной нижней границей. Для семи записей PMC9533879 CC50 также задано отношением `>` и сохраняется как нижняя граница. Это гибрид ML и публичных измерений, а не чистая supervised-модель.

## 6. Установка и воспроизведение

### 6.1. Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2. Подготовка данных

```bash
python scripts/download_public_data.py
python scripts/build_public_database.py
python scripts/validate_data.py
python scripts/build_training_labels.py --labels-only
```

Уже скачанные публичные таблицы включены. Для обновления ChEMBL:

```bash
python scripts/download_public_data.py --refresh
python scripts/build_ec50_cc50_pairs.py
```

### 6.3. Финальный запуск

```bash
python scripts/run_pipeline.py --pipeline final
python scripts/validate_submission.py notebooks/submission_v2_pmc953_additions.csv
```

Результат: `notebooks/submission_v2_pmc953_additions.csv`, 597 строк, колонки `index` и `SI VACV`, конечные неотрицательные прогнозы. Отчёт: `reports/final_reproduction.json`.

### 6.4. Числовая сверка

```bash
python scripts/run_pipeline.py --pipeline final \
  --reference /absolute/path/submission_v2_pmc953_additions.csv
```

Reference используется только после независимого выполнения notebook. Требуемый допуск: max absolute difference не более `1e-8`.

## 7. Проверки

- Схема базы: SMILES, IC50, CC50, SI, DOI и provenance.
- Формула SI и границы измерений.
- Полное выполнение ячеек без ошибок.
- 597 уникальных index в порядке test.
- Отсутствие NaN, inf и отрицательных значений.
- Отсутствие загрузки готовых submissions и OOF.
- Числовое совпадение с reference, если он передан.

Фактический запуск финала: **10 кодовых ячеек, 0 ошибок, 597 строк**, числовое расхождение с исходным candidate не более **2.28e-13**, статус **PASS**.

## 8. Ограничения интерпретации

- Публичные assay могут отличаться клеточной линией и протоколом от скрытого scoring.
- Нижние границы `CC50 > X` не считаются точными значениями.
- Локальная reconstructed MSE не является независимой оценкой.
- Public leaderboard охватывает только часть test; private score не экстраполируется.

Проект распространяется по лицензии MIT.
