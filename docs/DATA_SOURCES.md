# Источники данных

## ChEMBL

Сырые API endpoints:

- `https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL613493`
- `https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL613168`
- `https://www.ebi.ac.uk/chembl/api/data/activity.json?standard_type=CC50&document_chembl_id=<DOCUMENT_ID>`

Файлы:

- `data/molecular_activity/external/chembl_vacv_activities.csv`
- `data/molecular_activity/external/chembl_cc50_same_molecule_publication.csv`
- `data/molecular_activity/external/ec50_cc50_pairs_vacv.csv`

Пары строятся по молекуле и публикации. Одна публикация не гарантирует одинаковую клеточную линию или идентичный протокол assay, поэтому источники проверяются на train и независимых пересечениях.

### BSC-40 / тековиримат

BSC-40 component использует запись **CHEMBL221054** из документа **CHEMBL1142975**:

- DOI: `10.1021/jm061484y`;
- PMID: `17335190`;
- EC50 = 40 nM = 0.04 µM;
- CC50 > 100000 nM = 100 µM;
- консервативная нижняя граница SI = 2500.

Проверка документа: `https://www.ebi.ac.uk/chembl/api/data/document/CHEMBL1142975.json`.

CC50 является нижней границей, а не точным значением. Одинаковые стандартизованные структуры получают одно правило независимо от test index.

## PubChem

- `fastsimilarity_2d` для поиска аналогов.
- `assaysummary/JSON` для разбора potency, cytotoxicity, AID, PMID.
- AID156533/AID156532 — PRK potency и cell morphology для SID-серии.
- AID1778687/AID1778686 — адамантановая серия PMC9533879.
- AID217967/AID232802 — virus rating и toxic level.

### Статья PMC9533879

Дополнительный эксперимент использует Table 1 статьи **«Adamantane derivatives as potential inhibitors of p37 major envelope protein and poxvirus reproduction. Design, synthesis and antiviral activity»**:

- открытая статья: `https://pmc.ncbi.nlm.nih.gov/articles/PMC9533879/`;
- DOI: `10.1016/j.ejmech.2021.113485`;
- PMID: `33965861`;
- журнал: European Journal of Medicinal Chemistry, 2021, volume 221, 113485;
- IC50 против Vaccinia virus Copenhagen в Vero: PubChem **AID1778687**, ChEMBL assay **CHEMBL4838274**;
- CC50 в Vero: PubChem **AID1778686**, ChEMBL assay **CHEMBL4838273**.

Проверяемые assay endpoints:

- `https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1778687/JSON`;
- `https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1778686/JSON`.

В эксперименте используются 10 структурных соответствий. Для `PHXAPFLCZNRMNH`, `NNCDHNYZSIVCDP`, `RXLXHURLONKPHV`, `QDPJDJANNFBCGF`, `SEARXSRJDLULHH`, `JGLLXXRUZXVXCF`, `CJTQZZNWYVVBOU` CC50 задано отношением `>`; вычисленный SI является нижней границей. Для `VVBVERYXPHOGOU`, `FZUDUGQXCHQOAI`, `TVRAWMUKPGMPSP` используется точное отношение `=`.

Все 10 пар IC50/CC50 внесены в `data/public_si_database.csv` с DOI, assay source и `cc50_bound`. SI вычисляется как CC50/IC50, а не копируется из округлённого значения. PMC953-слой сохранён как эксперимент; финальный ensemble 72k использует Wiley/BSC-40 component без этого дополнительного слоя.

## Открытые публикации

- Wiley, «Design, Synthesis, and Biological Evaluation of (+)-Camphor- and (−)-Fenchone-Based Derivatives as Potent Orthopoxvirus Inhibitors», DOI `10.1002/cmdc.202100771`. В baseline используются в том числе Table 1: `2c` (CC50 = 2904.9 µM, IC50 = 11.1 µM) и `3c-ks-347` (CC50 = 946.8 µM, IC50 = 2.2 µM).
- «Adamantane derivatives as potential inhibitors of p37 major envelope protein and poxvirus reproduction», DOI `10.1016/j.ejmech.2021.113485`, PMC9533879.
- «Antiviral activities of two nucleos(t)ide analogs against vaccinia and mpox viruses in primary human fibroblasts», PMC10055413.
- «Antiviral Activity of 4′-thioIDU and Thymidine Analogs», PMC3185742.
- «Identification of In Vitro Inhibitors of Monkeypox Replication», DOI `10.1128/spectrum.04745-22`, PMC10434227.
- Viruses 2023 adamantane series, DOI `10.3390/v15010029`, PMC9865139.
- PRK neplanocin series, DOI `10.1021/jm00080a018`.

## Правила включения

- SI пересчитывается как CC50/IC50, а не копируется из округлённого столбца.
- Для нижних границ сохраняется `cc50_bound`; такие значения не описываются как точные измерения.
- Неподтверждённые DOI/assay источники не используются как точные значения.
- NCGC без DOI и независимой проверки исключён из supervised SI-разметки.
- Скрытые competition labels и test-концентрации в публичную базу не включаются.
- Готовые test predictions не используются как источник обучающих данных или как компонент финального pipeline.
