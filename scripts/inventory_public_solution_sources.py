#!/usr/bin/env python3
"""Build source-only inventory from the public Anti7 repository.

No SMILES, IC50, CC50, SI, test index or model prediction is copied into the
repository. The inventory is for source review, not for model training.
"""
from pathlib import Path
import io
import json

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/source_inventory"
REPORT = ROOT / "reports/public_source_inventory.json"
COMMIT = "2697ab703963d34c748912048134b8b725633cd7"
BASE = f"https://raw.githubusercontent.com/Velsaf-dev/molecular-ap-model/{COMMIT}/"
SOURCES = {
    "literature": BASE + "data/external_database.csv",
    "pubchem": BASE + "data/mined/pubchem_sids_poxvirus_assays.csv",
}


def fetch_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def status(source: str) -> str:
    if source == "10.3390/v15010029":
        return "partially_verified_18a_18b_but_mass_attribution_requires_review"
    if source.startswith("PubChem_AID_NIAID_"):
        return "sid_present_but_exact_aid_missing"
    if "drugbank.com" in source:
        return "generic_drugbank_url_not_record_level_citation"
    return "not_independently_verified"


OUT.mkdir(parents=True, exist_ok=True)
literature = fetch_csv(SOURCES["literature"])
rows = []
for source, group in literature.groupby("DOI", dropna=False):
    source = str(source)
    if source.startswith("10."):
        url = "https://doi.org/" + source
    elif source.startswith("http"):
        url = source
    else:
        url = ""
    rows.append({
        "source": source,
        "url": url,
        "rows_in_public_file": len(group),
        "unique_smiles_in_public_file": group["SMILES"].nunique(),
        "verification_status": status(source),
        "used_in_final_model": False,
    })
literature_inventory = pd.DataFrame(rows).sort_values(
    ["verification_status", "source"], ascending=[True, True])
literature_inventory.to_csv(OUT / "literature_sources.csv", index=False)

pubchem = fetch_csv(SOURCES["pubchem"])
pubchem_rows = []
for aid, group in pubchem.groupby("aid", dropna=True):
    aid = int(aid)
    assay_names = group["assay_name"].dropna().astype(str).unique()
    activity_names = group["activity_name"].dropna().astype(str).unique()
    pubchem_rows.append({
        "aid": aid,
        "url": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
        "assay_name": assay_names[0] if len(assay_names) else "",
        "activity_names": " | ".join(sorted(activity_names)),
        "rows_in_public_mined_file": len(group),
        "used_in_final_model": False,
    })
pubchem_inventory = pd.DataFrame(pubchem_rows).sort_values("aid")
pubchem_inventory.to_csv(OUT / "pubchem_assays.csv", index=False)

report = {
    "source_repository": "https://github.com/Velsaf-dev/molecular-ap-model",
    "source_commit": COMMIT,
    "literature_source_identifiers": len(literature_inventory),
    "pubchem_unique_aids": len(pubchem_inventory),
    "literature_inventory": "data/source_inventory/literature_sources.csv",
    "pubchem_inventory": "data/source_inventory/pubchem_assays.csv",
    "contains_smiles_or_measurements_or_test_predictions": False,
    "used_in_final_model": False,
    "warning": "Inventory of leads, not validation of the public external_database",
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
