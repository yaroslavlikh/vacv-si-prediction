# Обзор источников публичного решения Anti7

## Назначение документа

Здесь собраны источники данных из открытого репозитория `Velsaf-dev/molecular-ap-model`, commit `2697ab703963d34c748912048134b8b725633cd7`.

**Это справочный инвентарь для дальнейшей проверки, а не часть финального обучающего набора.** Финальная модель проекта остаётся `reproducible_v2_pmc953_additions.ipynb` с пользовательским Kaggle MSE **197 909.42299**. Чужие модели, готовые model predictions и `external_database.csv` публичного решения в финале не используются.

Инвентарь создан скриптом:

```bash
python scripts/inventory_public_solution_sources.py
```

Скрипт скачивает публичные таблицы в память и сохраняет только агрегированные ссылки и метаданные. Он не копирует SMILES, IC50, CC50, SI, test index или predictions в репозиторий.

## 1. Инвентарь источников

| Файл | Содержимое | Количество |
|---|---|---:|
| `data/source_inventory/literature_sources.csv` | DOI/другие идентификаторы, URL, число строк и статус проверки | 263 |
| `data/source_inventory/pubchem_assays.csv` | Уникальные PubChem AID, ссылки, названия assay и типы измерений | 474 |

Сводка: `reports/public_source_inventory.json`.

## 2. PubChem

В публичном решении используются следующие способы доступа.

### Поиск assay через NCBI E-utilities

```text
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pcassay&term=("Vaccinia virus"[Organism] OR "Cowpox virus"[Organism] OR "Monkeypox virus"[Organism]) AND ("IC50" OR "EC50" OR "Selectivity")&retmode=json&retmax=120
```

### Загрузка результатов и метаданных

```text
https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/CSV
https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/description/JSON
https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{CIDS}/property/CanonicalSMILES/JSON
```

В инвентаре присутствуют, среди прочих:

- AID2135800 — VACV IC50 в Vero, ChEMBL5625829, DOI `10.1016/j.ejmech.2023.115601`;
- AID2135802 — therapeutic index для той же серии;
- AID1778687 — VACV IC50 для PMC9533879;
- AID1778686 — Vero CC50 для PMC9533879;
- AID156533/AID156532 — PRK potency и cell morphology для SID-серии;
- AID217967/AID232802 — virus rating и toxic level.

Полный список из 474 AID находится в `pubchem_assays.csv`. Ссылка вида `PubChem_AID_NIAID_sid…` из итоговой базы публичного решения не считается полноценной assay-ссылкой: для воспроизведения нужен конкретный AID.

## 3. ChEMBL

В `chembl_miner.py` выполняется поиск assay по ключевым словам:

```text
vaccinia, cowpox, monkeypox, mpox, variola, ectromelia, poxvirus
```

Дополнительно ищутся target organisms:

```text
Vaccinia virus, Cowpox virus, Monkeypox virus, Variola virus, Poxviridae
```

Запрашиваемые типы измерений:

```text
IC50, EC50, CC50, TC50, TD50, Selectivity index,
Ratio CC50/EC50, Ratio CC50/IC50, Ratio, SI, MIC, MIC50
```

Используются официальные ChEMBL API resources `assay`, `activity`, `document` и `target`. Документы проверяются через endpoints вида:

```text
https://www.ebi.ac.uk/chembl/api/data/document/{DOCUMENT_CHEMBL_ID}.json
```

Пример: CHEMBL1142975, DOI `10.1021/jm061484y`, PMID 17335190 — источник BSC-40/тековиримата в нашем финале.

## 4. Литература

Инвентарь содержит все DOI и другие идентификаторы из публичной таблицы, включая:

- `10.3390/v15010029` — адамантановые амиды; открытая статья PMC9865139, подтверждены значения 18a = 18214 и 18b = 12300;
- `10.1002/cmdc.202100771` — camphor/fenchone derivatives; источник Wiley-серии в нашем решении;
- `10.1002/ardp.202100038` — дополнительная серия клеточных измерений;
- `10.1128/jvi.01894-06` — серия аналогов тековиримата;
- `10.1016/j.ejmech.2021.113485` — PMC9533879, используется в нашем финале;
- `10.1016/j.ejmech.2023.115601` — disulfide-incorporated lipid prodrugs of cidofovir;
- `10.1128/spectrum.04745-22` — monkeypox inhibitors;
- `10.1021/jm00080a018` — PRK neplanocin series;
- `10.1021/jm061484y` — orthopoxvirus egress inhibitors, тековиримат/BSC-40.

Полный список из 263 идентификаторов находится в `literature_sources.csv`.

## 5. Ограничения проверки

Инвентарь **не означает подтверждение каждой строки** публичной `external_database.csv`.

Аудит выявил:

- итоговая таблица содержит все 597 test-структур в том же порядке по полному InChIKey;
- их ChEMBL/PubChem-майнеры удаляют test-структуры и не создают эту итоговую таблицу;
- `fetch_open_data.py` лишь читает уже готовый `external_poxvirus_dataset.csv` и не показывает происхождение измерений;
- DOI `10.3390/v15010029` приписан 275 строкам и 264 уникальным структурам, хотя статья содержит небольшую серию соединений, поэтому массовая атрибуция требует отдельной проверки;
- часть PubChem-идентификаторов не содержит AID, а Drugbank-ссылка является общей, а не record-level citation.

Поэтому статус `not_independently_verified` в инвентаре нельзя читать как «источник не существует». Он означает только то, что конкретная связка **структура → IC50/CC50/SI → источник** пока не подтверждена нашим проектом.

## 6. Связь с финальным решением и Habr

Финальный pipeline использует собственную проверенную базу `data/public_si_database.csv` и PMC9533879. Источники из этого документа можно добавлять в обучение только после независимой проверки конкретных assay и значений.

Статья с Habr в финале не используется. Она относилась только к экспериментам с GCN и RDKit-дескрипторами.
