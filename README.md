# VACV SI Prediction

Воспроизводимый пайплайн предсказания индекса селективности **SI VACV** по SMILES. Текущая финальная рабочая версия — **LightGBM + публичные измерения + additions из статьи PMC9533879**. Часть значений IC50 и CC50 действительно взята из Table 1 научной статьи; источник, DOI и границы измерений указаны явно.

Предыдущая Wiley/BSC-40 версия имеет сообщённый пользователем официальный **Kaggle MSE 197 909.42299**. Для новой PMC953-версии официальный score пока не проверен и не приписывается.

Репозиторий также содержит эксперименты из чата **«Достижение MSE ниже 200 000»**. Финальный pipeline перенесён из чата **«Снижение MSE ниже 50 000»** и описан отдельно.

## 1. Задача и ограничения

Цель — предсказать **SI = CC50 / IC50** для 597 соединений из тестовой выборки.

- Competition train: **320 строк**.
- Competition test: **597 строк**.
- Пропущенный IC50 в train обрабатывается как **SI = 0** по исправленной конвенции.
- ML-признаки получаются только из SMILES; `index` используется как идентификатор.
- Тестовые концентрации не используются для обучения, выбора модели или формирования прогнозов.
- Локальная MSE восстановленного reference применяется только для диагностики и не является независимым Kaggle-score.
- Скрытые метки, OOF-массивы, исходные competition-таблицы и готовые тестовые прогнозы в репозиторий не включены.

Для воспроизведения соревнования `train.csv` и `test.csv` нужно локально разместить в `data/molecular_activity/`.

## 2. Что сделано

1. Собраны публичные ChEMBL/PubChem assay-данные и открытые таблицы научных публикаций.
2. Проверены fingerprint-бейзлайны, LightGBM, CatBoost, ExtraTrees, kNN, ChemBERTa и раздельные IC50/CC50-модели.
3. Отфильтрована внешняя разметка по формуле SI, InChIKey, пересечениям с train и независимым первоисточникам.
4. Сформирована публичная база **SMILES–IC50–CC50–SI–DOI** на **2 210 записей**.
5. Проверен прежний ансамбль public hybrid + confidence-aware ExtraTrees.
6. Воспроизведён Wiley/BSC-40 baseline с официальным MSE 197 909.42299.
7. Добавлены измерения из открытой статьи **PMC9533879**: 10 сопоставленных структур, 8 реальных изменений прогноза.
8. Финальный notebook создаёт `submission_v2_pmc953_additions.csv` с нуля, не читая готовый submission.

## 3. Эксперименты из чата «Достижение MSE ниже 200 000»

| Подход | Что проверено | Результат и решение |
|---|---|---|
| Morgan + MACCS + RDKit descriptors, HGB/LightGBM | Базовые признаки и log1p target | Лучший canonical baseline: OOF no-max 136 910.66 |
| Tail weighting LightGBM | Веса для высоких SI и внешние ChEMBL-строки | Использован как baseline |
| ChemBERTa-77M-MTR | Frozen CLS/mean embeddings и смеси с fingerprints | Canonical no-max 238 847, scaffold no-max 232 611 — хуже baseline |
| Раздельные IC50 и CC50 | SI = exp(logCC50 − logIC50) | OOF no-max 69 454, локальная диагностика 981 768 — нестабильно |
| Scaffold target encoding | Группировка по Murcko scaffold | OOF no-max 155 758 — хуже baseline |
| CatBoost | Те же признаки и внешняя разметка | OOF-отбор выбрал 100% LightGBM, 0% CatBoost |
| Мягкий kNN | Соседи по Tanimoto и смешивание с LightGBM | Ухудшение хвоста и локальной MSE — отклонён |
| Capped raw-target MSE | Cap 5000/10000 и tail weight 0/0.5 | Лучший OOF no-max 223 018 — хуже ExtraTrees log(SI) |
| ExtraTrees на log1p(SI) | Supervised-модель и ансамбль seed/protocol | Canonical OOF no-max 73 927.60, no-top7 25 791.78 |
| PubChem similarity | До 200 аналогов Adefovir и sid103210702 | Максимальный SI аналогов 66.67 и 314.38 — хвост не закрыт |
| Confidence shrinkage | Затухание прогноза по Tanimoto и смеси | Лучший прежний кандидат: `v2_conf_5050`, локальная MSE 75 990.20 |

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

Проверка `scripts/validate_data.py`:

- **2 210 записей**;
- 0 невалидных SMILES;
- 0 неположительных IC50/CC50;
- 0 отрицательных SI;
- 0 ошибок формулы SI более чем на `1e-6`;
- 205 записей без DOI — допустимо;
- повторы InChIKey сохранены, поскольку могут относиться к разным assay и источникам и не усредняются в самой базе.

В том числе добавлены **10 записей из PMC9533879 Table 1** с DOI `10.1016/j.ejmech.2021.113485`. Для семи записей `cc50_bound=">"`, для трёх CC50 измерено точно.

### Внешняя обучающая разметка

`external_training_labels.csv` создаётся локально скриптом `build_training_labels.py`. В проверенной версии используются **1 049 внешних структур**, из них **216** присутствуют в competition test как публично размеченные структуры.

Фильтрация:

- 605 строк NCGC без DOI и независимой проверки исключены;
- 4 ChEMBL-выброса SI>10000 исключены;
- округлённый SI пересчитан как CC50/EC50;
- одинаковые структуры с разными assay агрегируются по заранее заданному правилу;
- исходные competition-метки не перезаписываются;
- для OOF внешние записи исключаются по InChIKey14 или scaffold в зависимости от протокола.

Финальный LightGBM обучается на исходном train. Внешняя таблица нужна notebook для независимо собранных public-source соответствий, а не для загрузки готовых прогнозов.

Источники и ссылки приведены в `docs/DATA_SOURCES.md`.

## 5. Финальный pipeline: PMC9533879 additions

### Структурные признаки

Для каждой стандартизированной молекулы:

- Morgan fingerprint, radius 2, 2048 бит;
- 167 MACCS keys;
- 9 RDKit-дескрипторов: MolWt, MolLogP, TPSA, HDonors, HAcceptors, rotatable bonds, ring count, FractionCSP3, heavy atom count.

### Обучение и публичные измерения

`notebooks/reproducible_v2_pmc953_additions.ipynb` выполняет полный путь:

1. Читает исходные train/test и исправляет конвенцию пропущенного IC50.
2. Строит признаки из SMILES.
3. Обучает **30 LightGBM-моделей**: 2 протокола группировки × 3 seed × 5 folds.
4. Усредняет прогнозы и применяет зафиксированные `SCALE=2.4`, `CAP=4000`.
5. Добавляет публичные ChEMBL, PMC, PubChem и Wiley-измерения.
6. Применяет **12 post-v2 структурных соответствий**.
7. Применяет Wiley/BSC-40 baseline: `2c`, `3c-ks-347`, тековиримат и совпадающая структура `14`. Для тековиримата CC50>100 µM сохраняется как нижняя граница.
8. **Добавляет часть данных из научной статьи** «Adamantane derivatives as potential inhibitors of p37 major envelope protein and poxvirus reproduction»:
   - открытая версия: **PMC9533879**;
   - DOI: `10.1016/j.ejmech.2021.113485`;
   - PMID: `33965861`;
   - IC50 против VACV: PubChem **AID1778687**;
   - CC50 в Vero: PubChem **AID1778686**;
   - используются 10 структурных InChIKey14-соответствий, реально изменяются 8 строк;
   - для 7 соединений CC50 имеет отношение `>`, поэтому SI используется как консервативная нижняя граница;
   - конфликтный `DYLWMFBGCSKYFJ` и контрольный тековиримат из этого слоя исключены.
9. Сохраняет 597 прогнозов и только после этого считает локальную диагностику.

Это **гибрид ML и публичных измерений**, а не чистая supervised-модель. Публикационные названия используются для сопоставления соединений с первоисточниками; LightGBM получает только структурные признаки. Готовые submissions, сохранённые OOF и скрытые метки не загружаются.

### Результаты

| Версия | Официальный Kaggle MSE | Локальная диагностическая MSE | Статус |
|---|---:|---:|---|
| **LightGBM + Wiley/BSC-40 + статья PMC9533879** | **Не проверено** | **72 687.72207** | **Текущая финальная рабочая версия** |
| LightGBM + post-v2 + Wiley/ChEMBL BSC-40 | 197 909.42299 | 72 892.61237 | Предыдущий финал; score сообщён пользователем |
| Public hybrid v2 + 12 post-v2 labels | 201 013.27709 | 74 041.22172 | База Wiley/BSC-40 |
| Public hybrid + confidence ExtraTrees 50/50 | Не проверено | 75 990.19713 | Прежний кандидат |
| Public hybrid v2 | ≈203 000 | 81 312.20240 | Более ранний baseline |
| ExtraTrees log1p(SI) + exact neighbor | 211 415.38617 | 80 742.92155 | Чистая supervised-модель |

OOF базового LightGBM до публичных постобработок:

- full MSE: 4 169 967.37;
- MSE без глобального максимума target: **136 910.66**.

OOF и локальная диагностика измеряют разные выборки и не заменяют официальный score. Улучшение локальной MSE на 204.89 относительно Wiley/BSC-40 не гарантирует улучшение Kaggle. Public leaderboard охватывает только часть test; private score не экстраполируется.

## 6. Структура репозитория

- `notebooks/reproducible_v2_pmc953_additions.ipynb` — **текущий финальный pipeline со статьёй PMC9533879**.
- `notebooks/reproducible_v2_wiley_bsc40.ipynb` — предыдущий Wiley/BSC-40 финал и база новой версии.
- `notebooks/reproducible_ensemble_ncgc.ipynb` — прежний confidence-ансамбль.
- `notebooks/reproducible_clean_et_log.ipynb` — отдельный ExtraTrees.
- `notebooks/reproducible_confidence_variants.ipynb` — confidence-aware эксперименты.
- `scripts/build_pmc953_additions_notebook.py` — сборка standalone PMC953-notebook из Wiley/BSC-40 базы.
- `scripts/run_pmc953_additions.py` — запуск текущего финала, проверка и числовая сверка.
- `scripts/run_wiley_bsc40.py` — запуск предыдущего Wiley/BSC-40 baseline.
- `scripts/download_public_data.py` — проверка и загрузка ChEMBL-данных.
- `scripts/build_ec50_cc50_pairs.py` — сборка пар EC50/CC50 из сырых таблиц.
- `scripts/build_public_database.py` — сборка публичной SI-базы, включая статью.
- `scripts/validate_data.py` — проверка SMILES, SI и качества базы.
- `scripts/build_training_labels.py` — фильтрация внешней разметки; `--labels-only` отключает обучение вспомогательной модели.
- `scripts/train_model.py`, `scripts/apply_exact_neighbor.py` — обучение и проверка прежнего supervised ExtraTrees.
- `scripts/validate_submission.py` — проверка структуры submission.
- `experiments/` — эксперименты из чата «Достижение MSE ниже 200 000».
- `reports/` — метрики и результаты проверок без OOF-массивов и скрытых прогнозов.
- `docs/` — источники и описание экспериментов.
- `data/public_si_database.csv` — публичная SI-база.

## 7. Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 8. Воспроизведение результата

### Competition-данные

```text
data/molecular_activity/train.csv
data/molecular_activity/test.csv
```

Эти файлы нужны только для локального воспроизведения соревнования и не публикуются.

### Полный запуск

Из корня репозитория:

```bash
python scripts/download_public_data.py
python scripts/build_public_database.py
python scripts/validate_data.py
python scripts/build_training_labels.py --labels-only
python scripts/build_pmc953_additions_notebook.py
python scripts/run_pmc953_additions.py
python scripts/validate_submission.py notebooks/submission_v2_pmc953_additions.csv
```

Результат: `notebooks/submission_v2_pmc953_additions.csv` на **597 строк**, с колонками `index` и `SI VACV`, без NaN и отрицательных значений. Оба валидатора должны вернуть `PASS`.

Скрипт также сохраняет:

- `notebooks/executed_v2_pmc953_additions.ipynb` — выполненный notebook, исключён из Git;
- `reports/pmc953_reproduction.json` — строки, ошибки выполнения, источник статьи, SHA-256 и результат сверки.

### Проверка числовой идентичности исходному решению

Если сохранился исходный CSV из чата «Снижение MSE ниже 50 000»:

```bash
python scripts/run_pmc953_additions.py --reference /absolute/path/submission_v2_pmc953_additions.csv
```

Сравнение выполняется только после независимого обучения. Требуется максимальное абсолютное расхождение не более `1e-8`; в отчёт записываются hashes и фактическое расхождение.

### Обновление ChEMBL

```bash
python scripts/download_public_data.py --refresh
python scripts/build_ec50_cc50_pairs.py
```

ChEMBL API может временно ограничивать запросы; уже скачанные публичные таблицы включены в репозиторий.

## 9. Ограничения интерпретации

- Нижние границы вида `CC50 > X` не выдаются за точные измерения.
- Публичные assay могут отличаться клеточной линией и протоколом от скрытого scoring.
- Adefovir и sid103210702 остаются трудными хвостовыми объектами: релевантных high-SI аналогов мало.
- Локальная MSE существенно отличается от официальной, поэтому источник каждой оценки указан явно.
- Воспроизводимость PMC953-версии проверяется числовым совпадением с исходным `submission_v2_pmc953_additions.csv`; её официальный Kaggle-score до отправки остаётся неизвестным.
- MSE 197 909.42299 относится только к предыдущему `submission_v2_wiley_bsc40.csv` и не переносится на PMC953-версию.

Проект распространяется по лицензии MIT.
