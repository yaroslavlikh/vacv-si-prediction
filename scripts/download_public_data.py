#!/usr/bin/env python3
"""Download public ChEMBL activity tables used by the pipeline.

The files are already included in this repository. Run with --refresh only when
fresh versions are required. Public competition train/test files are NOT downloaded.
"""
from pathlib import Path
import argparse
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGETS = ["CHEMBL613493", "CHEMBL613168"]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "vacv-si-prediction/1.0 academic"})


def get_page(url, params, retries=4):
    for attempt in range(retries):
        try:
            response = SESSION.get(url, params=params, timeout=60)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return None


def download_activities(params):
    rows = []
    params = dict(params)
    params["limit"] = 1000
    params["offset"] = 0
    while True:
        payload = get_page(BASE, params)
        rows.extend(payload.get("activities", []))
        next_url = payload.get("page_meta", {}).get("next")
        if not next_url:
            break
        params["offset"] += 1000
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Download instead of only checking paths")
    args = parser.parse_args()
    out = ROOT / "data/molecular_activity/external"
    out.mkdir(parents=True, exist_ok=True)
    if not args.refresh:
        for name in ["chembl_vacv_activities.csv", "chembl_cc50_same_molecule_publication.csv", "ec50_cc50_pairs_vacv.csv"]:
            path = out / name
            print(f"{path}: {'OK' if path.exists() else 'MISSING'}")
        return
    import pandas as pd
    activities = []
    for target in TARGETS:
        activities.extend(download_activities({"target_chembl_id": target}))
    pd.DataFrame(activities).to_csv(out / "chembl_vacv_activities.csv", index=False)
    documents = sorted({row.get("document_chembl_id") for row in activities if row.get("document_chembl_id")})
    cc50_rows = []
    for i, document in enumerate(documents, 1):
        cc50_rows.extend(download_activities({"standard_type": "CC50", "document_chembl_id": document}))
        if i % 25 == 0:
            print(f"CC50 documents: {i}/{len(documents)}", flush=True)
    pd.DataFrame(cc50_rows).to_csv(out / "chembl_cc50_same_molecule_publication.csv", index=False)
    print(f"Saved {len(activities)} VACV activities and {len(cc50_rows)} CC50 activities")


if __name__ == "__main__":
    main()
