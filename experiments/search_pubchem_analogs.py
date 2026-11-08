from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import json
import requests
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "data/molecular_activity"
CACHE = OUT / "analog_cache"
CACHE.mkdir(exist_ok=True)
te = pd.read_csv(DATA / "test.csv")
tr = pd.read_csv(DATA / "train.csv")
queries = {title: te[te.Title == title].iloc[0].SMILES for title in ["Adefovir dipivoxil", "sid103210702"]}
# PubChem rejects the nonstandard test isomer; use the canonical drug for analog discovery.
queries["Adefovir dipivoxil"] = tr[tr.Title == "Adefovir dipivoxil"].iloc[0].SMILES
HEADERS = {"User-Agent": "it-hub-vacv-analog-research/1.0 academic"}


def get(url, path):
    if path.exists():
        return json.loads(path.read_text())
    r = requests.get(url, headers=HEADERS, timeout=40)
    if r.status_code in (400, 404, 429, 500, 502, 503, 504):
        path.write_text("{}")
        return {}
    r.raise_for_status()
    path.write_text(r.text)
    return r.json()


def parse_assays(cid, data):
    if not data or "Table" not in data:
        return []
    cols = data["Table"]["Columns"]["Column"]
    rows = []
    for row in data["Table"].get("Row", []):
        c = dict(zip(cols, row["Cell"]))
        assay = str(c.get("Assay Name") or "")
        name = str(c.get("Activity Name") or "").upper()
        try:
            value = float(c.get("Activity Value [uM]") or "")
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        text = assay.lower()
        if "vaccinia" in text and name in {"EC50", "IC50", "AC50", "ED50", "ID50"}:
            kind = "potency"
        elif name in {"CC50", "TC50", "GI50", "LC50", "IC50", "ID50"} and any(x in text for x in ["cytotox", "viability", "toxicity", "cell growth", "cell proliferation"]):
            kind = "toxicity"
        else:
            continue
        rows.append({"cid": cid, "kind": kind, "value_um": value, "activity_name": name,
                     "assay_name": assay, "aid": str(c.get("AID") or ""), "pmid": str(c.get("PubMed ID") or "")})
    return rows


records = []
for title, smiles in queries.items():
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{quote(smiles, safe='')}/cids/JSON?Threshold=50&MaxRecords=200"
    data = get(url, CACHE / f"similarity50_{title.replace(' ', '_')}.json")
    cids = data.get("IdentifierList", {}).get("CID", [])
    print(title, "analog CIDs", len(cids), flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        responses = list(ex.map(lambda cid: (cid, get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON", CACHE / f"{cid}.json")), cids))
    for cid, assay_data in responses:
        for row in parse_assays(cid, assay_data):
            row["query"] = title
            records.append(row)

raw = pd.DataFrame(records)
raw.to_csv(OUT / "pubchem_analog_activity.csv", index=False)
if raw.empty:
    print("No assay records returned")
    raise SystemExit(0)
candidates = []
for (query, cid, pmid), group in raw.groupby(["query", "cid", "pmid"], dropna=False):
    potency = group[group.kind == "potency"]
    toxicity = group[group.kind == "toxicity"]
    if not len(potency) or not len(toxicity) or not pmid:
        continue
    pv, tv = float(potency.value_um.median()), float(toxicity.value_um.median())
    candidates.append({"query": query, "cid": cid, "pmid": pmid,
                       "potency_um": pv, "cc50_um": tv, "si": tv / pv,
                       "aids": ";".join(sorted(set(group.aid))),
                       "assays": " | ".join(sorted(set(group.assay_name)))})
candidates = pd.DataFrame(candidates)
if len(candidates):
    candidates.to_csv(OUT / "pubchem_analog_si_candidates.csv", index=False)
    for title in queries:
        print("\nTOP", title)
        print(candidates[candidates["query"] == title].nlargest(15, "si").to_string(index=False))
else:
    print("No same-PMID numeric SI candidates")
