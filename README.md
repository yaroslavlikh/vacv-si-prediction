# VACV SI Prediction

Воспроизводимая **чистая supervised ML-модель** для предсказания индекса селективности **SI VACV** по SMILES. Репозиторий содержит обучение, валидацию, публичную базу измерений, источники данных и эксперименты.

## Итог

**Финальное решение: ExtraTrees на `log1p(SI)` + supervised exact nearest neighbor. Официальный Kaggle MSE — 211 415.38617.**

- Модель обучается заново на исходном train и проверенных внешних обучающих метках.
- Входные признаки вычисляются из SMILES.
- Литературный post-ML lookup, ручные замены прогнозов и SAR-числа не используются.
- Готовые внешние модели и чужие model predictions не загружаются.
- Результат: 597 прогнозов, полный запуск проверен.

Официальный score приведён по загрузке пользователя. Локальная reconstructed MSE равна **80 742.92155** и используется только для диагностики, а не как независимый Kaggle-score. Private leaderboard не оценивается.

Прежний гибрид LightGBM + литературные измерения с Kaggle MSE 197 909.42299 сохранён в экспериментах, но не используется в текущем финале.

## 1. Задача

Требуется предсказать **SI = CC50 / IC50** для 597 соединений.

- Competition train: **320 строк**; test: **597 строк**.
- Пропущенный IC50 в train обрабатывается как **SI = 0** по исправленной конвенции.
- `index` служит только идентификатором.
- Тестовые концентрации не участвуют в обучении, отборе или прогнозировании. Они используются только в post-hoc диагностике после сохранения submission.

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
│   ├── source_inventory/                       # ссылки и метаданные источников
│   └── molecular_activity/external/            # сырые публичные ChEMBL-таблицы
├── exports/
│   └── vacv_augmentation_dataset_v2.csv        # исходный набор внешней разметки
├── notebooks/
│   ├── reproducible_clean_et_log.ipynb         # финальный чистый ML pipeline
│   ├── reproducible_v2_pmc953_additions.ipynb  # прежний гибрид с литературой
│   ├── reproducible_v2_wiley_bsc40.ipynb       # Wiley/BSC-40 experiment
│   ├── reproducible_ensemble_ncgc.ipynb        # confidence-aware ensemble
│   └── reproducible_confidence_variants.ipynb  # confidence experiments
├── scripts/
│   ├── download_public_data.py                 # загрузка ChEMBL
│   ├── build_ec50_cc50_pairs.py                # пары potency–CC50
│   ├── build_public_database.py                # сборка публичной SI-базы
│   ├── validate_data.py                        # контроль качества базы
│   ├── build_training_labels.py                # фильтрация внешней разметки
│   ├── run_pipeline.py                         # запуск и проверка финала
│   ├── train_model.py                          # обучение ExtraTrees-эксперимента
│   ├── inventory_public_solution_sources.py    # инвентарь внешних источников
│   └── validate_submission.py                  # проверка CSV на 597 строк
├── experiments/                                # код и материалы экспериментов
├── reports/
│   ├── official_scores.json                    # официальные результаты версий
│   ├── data_validation.json                    # проверка публичной базы
│   ├── final_reproduction.json                 # чистый запуск финала
│   └── public_source_inventory.json            # сводка инвентаря источников
├── docs/
│   ├── DATA_SOURCES.md                         # источники и правила включения
│   ├── PUBLIC_SOLUTION_DATA_SOURCES.md         # обзор источников публичного решения
│   └── EXPERIMENTS.md                          # подробности экспериментов
├── requirements.txt
└── README.md
```

## 3. Данные и источники

### 3.1. Публичная SI-база

`data/public_si_database.csv` содержит поля:

- `SMILES` — структуру соединения;
- `IC50` — активность против VACV в µM;
- `CC50` — цитотоксичность в µM;
- `SI` — пересчитанный `CC50 / IC50`;
- `DOI` — публикацию, если известна;
- `source`, `cc50_bound`, `InChIKey` — происхождение и ограничения измерения.

Проверка: **2 210 записей**, 0 невалидных SMILES, 0 неположительных IC50/CC50, 0 отрицательных SI и 0 ошибок формулы более чем на `1e-6`. 205 записей без DOI допустимы.

Основные источники: ChEMBL, PubChem, открытые статьи PMC и Wiley. Полный список находится в `docs/DATA_SOURCES.md`. Дополнительные 263 литературных идентификатора и 474 PubChem AID собраны в `data/source_inventory/` и описаны в `docs/PUBLIC_SOLUTION_DATA_SOURCES.md`.

### 3.2. Внешняя обучающая разметка

`external_training_labels.csv` создаётся скриптом `build_training_labels.py`.

Проверенный состав: **1 049 внешних структур**, из них **216** совпадают со структурами test и имеют публичную разметку. 605 строк NCGC без DOI и независимой проверки исключены; четыре ChEMBL-выброса SI > 10000 исключены. Для OOF внешние записи исключаются по InChIKey14 или scaffold в зависимости от протокола.

Внешние метки используются **только для обучения и supervised nearest neighbor**, а не как готовый post-ML lookup.

## 4. Финальная модель

### 4.1. ExtraTrees на log1p(SI)

Модель использует структурные признаки молекул и ансамбль ExtraTrees:

- **60 деревьев на модель**;
- **2 протокола группировки**;
- **3 seed**;
- **5 folds**;
- целевая переменная — `log1p(SI)`;
- прогнозы усредняются и переводятся обратно через `expm1`.

Разные протоколы и seed нужны для устойчивости ансамбля. Качество оценивается по OOF, включая no-max и no-top7, чтобы не выбирать модель только по одному экстремальному объекту.

### 4.2. Supervised exact nearest neighbor

Для точных структурных совпадений применяется supervised nearest neighbor по обучающим структурам. Он изменяет **252 строки** и не является ручной литературной заменой: значение берётся из обучающей разметки по правилу модели.

### 4.3. Результаты валидации

- Canonical OOF no-max: **73 927.60**.
- Canonical OOF no-top7: **25 791.78**.
- Scaffold OOF no-max: **227 290.91**.
- Scaffold OOF no-top7: **20 001.97**.
- Локальная диагностическая MSE: **80 742.92155**.
- Официальный Kaggle MSE: **211 415.38617**.

## 5. Другие эксперименты

| Подход | Результат и решение |
|---|---|
| Morgan + MACCS + RDKit, HGB/LightGBM | Лучший базовый OOF no-max 136 910.66 |
| ChemBERTa-77M-MTR | Canonical no-max 238 847; хуже fingerprints |
| Раздельные IC50 и CC50 | OOF no-max 69 454, локальная MSE 981 768; отклонено |
| Scaffold target encoding | OOF no-max 155 758; хуже baseline |
| CatBoost | OOF-отбор выбрал 100% LightGBM, 0% CatBoost |
| Мягкий kNN | Ухудшение хвоста; отклонён |
| Confidence shrinkage | Официального улучшения не подтверждено |
| Public hybrid + литература | Kaggle около 203 000 |
| Wiley/BSC-40 + PMC9533879 | Kaggle 197 909.42299; прежний гибрид |
| **ExtraTrees log1p(SI) + exact neighbor** | **Kaggle 211 415.38617; текущий чистый ML финал** |

Подробности находятся в `experiments/` и `docs/EXPERIMENTS.md`.

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
python scripts/validate_submission.py notebooks/submission_clean_et_log.csv
```

Результат: `notebooks/submission_clean_et_log.csv`, 597 строк, колонки `index` и `SI VACV`, конечные неотрицательные прогнозы. Отчёт: `reports/final_reproduction.json`.

Notebook сначала обучает модели и сохраняет собственные промежуточные OOF/test-массивы, затем формирует submission. Эти массивы создаются внутри того же запуска, а не загружаются как готовые артефакты.

### 6.4. Числовая сверка

```bash
python scripts/run_pipeline.py --pipeline final \
  --reference /absolute/path/submission_clean_et_log.csv
```

Reference используется только после независимого выполнения. Требуемый допуск: max absolute difference не более `1e-8`.

## 7. Проверки и ограничения

Проверяются схема данных, формула SI, выполнение всех ячеек, 597 уникальных index, конечные неотрицательные прогнозы и отсутствие загрузки готовых submissions/OOF.

Ограничения:

- Публичные assay могут отличаться клеточной линией и протоколом от скрытого scoring.
- Нижние границы `CC50 > X` не считаются точными значениями.
- Локальная reconstructed MSE не является независимой оценкой.
- Public leaderboard охватывает только часть test; private score не экстраполируется.

Проект распространяется по лицензии MIT.
