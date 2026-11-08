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

## PubChem

- `fastsimilarity_2d` для поиска аналогов.
- `assaysummary/JSON` для разбора potency, cytotoxicity, AID, PMID.
- AID156533/AID156532 — PRK potency и cell morphology для SID-серии.
- AID1778687/AID1778686 — адамантановая серия PMC9533879.
- AID217967/AID232802 — virus rating и toxic level.

## Открытые публикации

- Wiley, «Design, Synthesis, and Biological Evaluation of (+)-Camphor- and (−)-Fenchone-Based Derivatives as Potent Orthopoxvirus Inhibitors», DOI `10.1002/cmdc.202100771`.
- «Adamantane derivatives as potential inhibitors of p37 major envelope protein and poxvirus reproduction», DOI `10.1016/j.ejmech.2021.113485`, PMC9533879.
- «Antiviral activities of two nucleos(t)ide analogs against vaccinia and mpox viruses in primary human fibroblasts», PMC10055413.
- «Antiviral Activity of 4′-thioIDU and Thymidine Analogs», PMC3185742.
- «Identification of In Vitro Inhibitors of Monkeypox Replication», DOI `10.1128/spectrum.04745-22`, PMC10434227.
- Viruses 2023 adamantane series, DOI `10.3390/v15010029`, PMC9865139.
- PRK neplanocin series, DOI `10.1021/jm00080a018`.

## Правила включения

- SI пересчитывается как CC50/IC50, а не копируется из округлённого столбца.
- Для нижних границ сохраняется `cc50_bound`.
- Не подтверждённые DOI/assay источники не используются как точные значения.
- NCGC без DOI и независимой проверки исключён из supervised SI-разметки.
- Скрытые competition labels и test-концентрации в публичную базу не включаются.
