# VACV SI Prediction

Воспроизводимый ML-пайплайн для предсказания индекса селективности **SI VACV** по SMILES. Репозиторий содержит финальную модель, публичную базу измерений, скрипты загрузки и проверки данных, а также эксперименты из чата **«Достижение MSE ниже 200 000»**.

## 1. Задача и ограничения

Цель — предсказать **SI = CC50 / IC50** для 597 соединений из тестовой выборки.

Исходные данные:

- **320 строк** competition train;
- **597 строк** competition test;
- пропущенный IC50 в train обрабатывается как **SI = 0** по исправленной конвенции организаторов;
- входы модели — только структурная информация из SMILES, а `index` используется как идентификатор;
- тестовые концентрации **не используются** для обучения, выбора модели или прогнозирования;
- скрытые метки, OOF-массивы и готовые тестовые прогнозы в репозиторий не включены.

Конкретный `train.csv` и `test.csv` не публикуются. Для воспроизведения соревнования их нужно разместить локально в `data/molecular_activity/`.

## 2. Что сделано

1. Собраны публичные ChEMBL и PubChem assay-данные и открытые таблицы научных публикаций.
2. Проверены fingerprint-бейзлайны, LightGBM, CatBoost, ExtraTrees, kNN, ChemBERTa и раздельные IC50/CC50-модели.
3. Отфильтрована внешняя разметка по формуле SI, InChIKey, пересечениям с train и независимыми первоисточниками.
4. Собрана единая публичная база **SMILES–IC50–CC50–SI–DOI** на **2 198 записей**.
5. Выбран финальный ансамбль **public hybrid + confidence-aware ExtraTrees** (`v2_conf_5050`) с supervised exact nearest neighbor.
6. Полный pipeline и все три notebook запущены и проверены с нуля.

## 3. Эксперименты и решения

| Подход | Что проверено | Результат и решение |
|---|---|---|
| Morgan + MACCS + RDKit descriptors, HGB/LightGBM | Базовые признаки и log1p target | Лучший canonical baseline: no-max 136 910.66 |
| Tail weighting LightGBM | Веса для высоких SI и внешние ChEMBL-строки | Лучший OOF no-max 136 910.66; использован как baseline |
| ChemBERTa-77M-MTR | Frozen CLS/mean embeddings и их смеси с fingerprints | Canonical no-max 238 847, scaffold no-max 232 611 — хуже baseline |
| Раздельные IC50 и CC50 | SI = exp(logCC50 − logIC50) | OOF no-max 69 454, но локальная диагностика 981 768 — нестабильно |
| Scaffold target encoding | Группировка по Murcko scaffold | OOF no-max 155 758 — хуже baseline |
| CatBoost | Те же признаки и внешняя разметка | OOF-отбор выбрал 100% LightGBM, 0% CatBoost |
| Мягкий kNN | Соседи по Tanimoto и смешивание с LightGBM | Улучшение одной метрики, но ухудшение хвоста и локальной MSE — отклонён |
| Capped raw-target MSE | Cap 5000/10000 и tail weight 0/0.5 | Лучший OOF no-max 223 018 — хуже ExtraTrees log(SI) |
| **ExtraTrees на log1p(SI)** | Прямая supervised-модель и ансамбль seed/protocol | **Canonical OOF no-max 73 927.60, no-top7 25 791.78 — выбрана** |
| PubChem similarity | До 200 аналогов Adefovir и sid103210702 | Максимальный SI аналогов 66.67 и 314.38 — хвост не закрыт |
| Confidence shrinkage | Затухание прогноза по Tanimoto¹/²/⁴ и смеси | OOF no-top7 до 17 780.53; локальная MSE около 80.8–81.2 тысяч |

Подробности находятся в `experiments/` и `docs/EXPERIMENTS.md`.

## 4. Данные и их качество

### Публичная SI-база

`data/public_si_database.csv` содержит:

- `SMILES` — структура соединения;
- `IC50` — активность против VACV в µM;
- `CC50` — цитотоксичность в µM;
- `SI` — рассчитанный `CC50 / IC50`;
- `DOI` — публикация, если она известна;
- `source`, `cc50_bound`, `InChIKey` — происхождение и ограничения измерения.

Проверка `scripts/validate_data.py` подтвердила:

- 2 198 записей;
- 0 невалидных SMILES;
- 0 неположительных IC50/CC50;
- 0 отрицательных SI;
- 0 ошибок формулы SI более чем на `1e-6`;
- 205 записей без DOI — допустимо;
- повторы InChIKey сохранены, потому что они относятся к разным assay и не усредняются в самой базе.

### Внешняя обучающая разметка

В `external_training_labels.csv` (создаётся при запуске) используются **1 049 внешних структур**, из них **216** присутствуют в competition test.

Фильтрация:

- 605 строк NCGC без DOI и независимой проверки исключены;
- 4 ChEMBL-выброса SI>10000 исключены;
- округлённый SI пересчитан как CC50/EC50;
- для одинаковых SMILES с разными assay используются средние значения, поскольку assay metadata не являются входами модели;
- исходные competition-метки не перезаписываются;
- внешние записи исключаются из OOF-фолдов по InChIKey14, а для scaffold-протокола — по scaffold.

Источники и ссылки приведены в `docs/DATA_SOURCES.md`.

## 5. Финальный пайплайн

### Признаки

Для каждой стандартизированной молекулы:

- Morgan fingerprint, radius 2, 2048 бит;
- 167 MACCS keys;
- 9 RDKit-дескрипторов: MolWt, MolLogP, TPSA, HDonors, HAcceptors, rotatable bonds, ring count, FractionCSP3, heavy atom count.

### Обучение и ансамбль

Финальная версия `v2_conf_5050` состоит из двух независимо обучаемых компонентов:

1. **Public hybrid**: 30 LightGBM на `log1p(SI)` с Morgan, MACCS и RDKit-признаками, затем публичные измерения из ChEMBL, PubChem и открытых статей.
2. **Confidence-aware ExtraTrees**: 60 деревьев, `max_features=0.4`, `min_samples_leaf=1`, с затуханием прогноза пропорционально Tanimoto-similarity при низкой уверенности.

Они объединяются **50/50**. Параметры ансамбля и confidence не подбираются по скрытому test.

Общие настройки ExtraTrees:

- Цель: `log1p(SI)`.
- Протоколы: canonical InChIKey14 и scaffold.
- Seeds: 42, 13, 79.
- Валидация: 5-fold GroupKFold для каждого seed.
- Веса train: `1 + SI/100`, внешних структур: `0.1`.
- Финальный fit выполняется на всех train и внешних данных.

### Supervised exact nearest neighbor

Это часть алгоритма, а не ручная lookup-замена:

- вычисляется Morgan/Tanimoto similarity;
- совпадение принимается только при `similarity >= 0.999` **и совпадении полного InChIKey**;
- полный InChIKey нужен, чтобы не склеивать диастереомеры;
- для OOF внешние совпадения с валидационными структурами удаляются из reference.

### Результаты

| Версия | Официальный Kaggle MSE | Локальная MSE | Комментарий |
|---|---:|---:|---|
| **v2_conf_5050: public hybrid + confidence ExtraTrees** | Пока не проверено | **75 990.20** | **Финальная версия** |
| Гибрид v2: LightGBM + public literature lookup | ≈203 000 | 81 312.20 | Не чистая supervised-модель |
| ExtraTrees log1p(SI) + exact neighbor | 211 415.38617 | 80 742.92 | Предыдущая чистая ML-модель |
| Confidence Tanimoto² | Не проверено | 80 889.37 | Более осторожные прогнозы на незнакомых структурах |

OOF финального ансамбля:

- full MSE: 4 144 285.85;
- no-max: **98 591.12**;
- no-top7: **16 642.30**.

Для сравнения, у отдельного ExtraTrees canonical no-max было 73 927.60, no-top7 — 25 791.78.

Локальная диагностика не является независимой оценкой. Она восстановлена из test-концентраций только после обучения и **не используется для выбора модели**. Public leaderboard охватывает лишь часть test, поэтому частный score не экстраполируется.

## 6. Структура репозитория

- `notebooks/reproducible_ensemble_ncgc.ipynb` — **финальный ансамбль `v2_conf_5050`**.
- `notebooks/reproducible_clean_et_log.ipynb` — отдельный ExtraTrees без ансамбля.
- `notebooks/reproducible_confidence_variants.ipynb` — пять confidence-aware вариантов.
- `scripts/download_public_data.py` — проверка и загрузка ChEMBL-данных.
- `scripts/build_ec50_cc50_pairs.py` — сборка пар EC50/CC50 из сырых таблиц.
- `scripts/build_public_database.py` — сборка публичной SI-базы.
- `scripts/validate_data.py` — проверка SMILES, SI и качества базы.
- `scripts/build_training_labels.py` — фильтрация и сборка внешней разметки.
- `scripts/train_model.py` — обучение ExtraTrees и OOF-валидация.
- `scripts/apply_exact_neighbor.py` — supervised nearest-neighbor компонент.
- `scripts/validate_submission.py` — проверка структуры submission.
- `experiments/` — скрипты экспериментов из указанного чата.
- `reports/` — небольшие отчёты и метрики, без OOF-массивов и скрытых прогнозов.
- `docs/` — источники и описание экспериментов.
- `data/public_si_database.csv` — публичная SI-база.
- `requirements.txt` — зависимости Python.

## 7. Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 8. Воспроизведение результатов

### Размещение competition-данных

```text
data/molecular_activity/train.csv
data/molecular_activity/test.csv
```

Эти файлы нужны только для воспроизведения соревнования, но не для сборки публичной SI-базы.

### Полный pipeline

Из корня репозитория:

```bash
python scripts/download_public_data.py
python scripts/build_public_database.py
python scripts/validate_data.py
python scripts/build_training_labels.py
python scripts/train_model.py
python scripts/apply_exact_neighbor.py
cd notebooks
jupyter nbconvert --to notebook --execute reproducible_ensemble_ncgc.ipynb --output executed_ensemble_ncgc.ipynb
cd ..
python scripts/validate_submission.py notebooks/submission_v2_conf_5050.csv
```

Ожидаемый результат: файл `notebooks/submission_v2_conf_5050.csv` на 597 строк, с колонками `index` и `SI VACV`, без NaN и отрицательных значений. Проверка `validate_submission.py` должна вернуть `PASS`.

### Запуск финального notebook

Сначала выполнить `build_training_labels.py`, затем:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute reproducible_ensemble_ncgc.ipynb --output executed_ensemble_ncgc.ipynb
```

### Обновление ChEMBL

```bash
python scripts/download_public_data.py --refresh
python scripts/build_ec50_cc50_pairs.py
```

ChEMBL API может временно ограничивать запросы; уже скачанные публичные таблицы включены в репозиторий.

## 9. Ограничения интерпретации

- Низкие границы вида `CC50 > X` не выдаются за точные измерения.
- Публичные assay могут отличаться клеточной линией и протоколом от скрытого scoring.
- Adefovir и sid103210702 остаются трудными хвостовыми объектами: релевантных high-SI аналогов мало, и простое усложнение модели их не решает.
- Локальная MSE существенно отличается от официальной, поэтому выводы делаются только с явным указанием источника оценки.

Проект распространяется по лицензии MIT.
