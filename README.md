# VACV SI Prediction

Воспроизводимый ML-пайплайн для предсказания индекса селективности **SI VACV** по SMILES.

## Финальная модель

**ExtraTrees на `log1p(SI)` + supervised exact nearest neighbor по Morgan/Tanimoto с проверкой полного InChIKey.**

- 2 протокола разбиения × 3 seed, 60 деревьев ExtraTrees на модель.
- 320 исходных объектов train + публичная внешняя разметка ChEMBL, PubChem и открытых статей.
- Валидация external-данных исключает валидационные структуры по InChIKey14/scaffold.
- Никаких ручных post-hoc замен в test и никаких тестовых концентраций в обучении или прогнозе.
- Официальный Kaggle-score финальной версии: **211 415.38617 MSE**.
- Локальная диагностика: **80 742.92 MSE**. Она не является независимой оценкой.

Также приведён гибридный исторический baseline: **около 203 000 MSE на Kaggle** и **81 312.20** локально. Он не является финальной чистой ML-моделью.

## Структура

- `notebooks/reproducible_clean_et_log.ipynb` — финальный notebook.
- `notebooks/reproducible_confidence_variants.ipynb` — пять confidence-aware вариантов ExtraTrees.
- `scripts/download_public_data.py` — проверка/загрузка публичных ChEMBL-данных.
- `scripts/build_ec50_cc50_pairs.py` — сборка пар EC50/CC50 из сырых ChEMBL-таблиц.
- `scripts/build_public_database.py` — единая публичная база SMILES–IC50–CC50–SI–DOI.
- `scripts/validate_data.py` — проверка базы и формулы SI.
- `scripts/build_training_labels.py` — фильтрация и сборка внешней разметки.
- `scripts/train_model.py` — обучение ExtraTrees и OOF-валидация.
- `scripts/apply_exact_neighbor.py` — supervised nearest-neighbor компонент.
- `scripts/validate_submission.py` — проверка структуры submission.
- `experiments/` — скрипты экспериментов из чата «Достижение MSE ниже 200 000».
- `reports/` — небольшие отчёты и метрики, без OOF-массивов и скрытых прогнозов.
- `data/public_si_database.csv` — публичная база измерений, после запуска `build_public_database.py`.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

Публичные ChEMBL-таблицы и augmentation-набор уже включены. Исходные competition `train.csv` и `test.csv` **не публикуются** и должны быть размещены локально:

```text
data/molecular_activity/train.csv
data/molecular_activity/test.csv
```

Они нужны только для воспроизведения соревнования, а не для сборки публичной SI-базы. Скрытые метки, OOF и готовые тестовые прогнозы в репозиторий не включены.

## Запуск пайплайна

Из корня репозитория:

```bash
python scripts/download_public_data.py
python scripts/build_public_database.py
python scripts/validate_data.py
python scripts/build_training_labels.py
python scripts/train_model.py
python scripts/apply_exact_neighbor.py
python scripts/validate_submission.py submission_clean_et_log.csv
```

При необходимости обновления ChEMBL:

```bash
python scripts/download_public_data.py --refresh
python scripts/build_ec50_cc50_pairs.py
```

## Запуск notebook

```bash
cd notebooks
jupyter nbconvert --to notebook --execute reproducible_clean_et_log.ipynb --output executed_clean_et_log.ipynb
```

Перед запуском notebook должен быть создан `external_training_labels.csv` в корне репозитория шагом `build_training_labels.py`.

## Метрики и ограничения

- OOF используется только для выбора моделей и коэффициентов.
- Локальный reference восстановлен из предоставленных test-концентраций только для диагностики и не совпадает по надёжности с Kaggle.
- Официальный public leaderboard охватывает лишь часть test, поэтому частный score не экстраполируется.
- Низкие границы вида `CC50 > X` не выдаются за точные измерения.
- Для одинаковых SMILES с разными assay используются средние значения, потому что assay metadata не входят в модель.

Источники и ограничения внешней разметки описаны в `docs/DATA_SOURCES.md`, история экспериментов — в `docs/EXPERIMENTS.md`.
