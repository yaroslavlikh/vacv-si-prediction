#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ec50_cc50_pairs.py — сборка парной базы SMILES + EC50 + CC50 + DOI
из УЖЕ скачанных сырых таблиц ChEMBL (data/molecular_activity/external/).

Источники (не перекачиваются):
  - chembl_vacv_activities.csv            — активности против VACV (EC50 и др.)
  - chembl_cc50_same_molecule_publication.csv — CC50 тех же молекул в тех же публикациях

Логика:
  - Пары на зерне (молекула × публикация): EC50 против VACV + CC50 той же молекулы
    из той же публикации ChEMBL. Это прокси «одного исследования», но НЕ гарантия
    одной клеточной линии/условий (см. предупреждение в external/README.md).
  - Значения в ug/ml конвертируются в нМ через молекулярную массу из ChEMBL API
    (помечаются флагом, см. отчёт).
  - Цензурированные значения («>», «<») сохраняются с флагом bound, а не выбрасываются.

Выход: external/ec50_cc50_pairs_vacv.csv
  smiles, ec50_nm, ec50_bound, cc50_nm, cc50_bound, doi
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
EXTERNAL = Path(__file__).resolve().parent.parent / "data" / "molecular_activity" / "external"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "it-hub-vacv-research/1.0 (academic use)"})


def get_json(url, params=None, retries=5):
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch_mol_weights(mol_ids):
    """molecule_chembl_id -> full_molweight (float)."""
    out = {}
    ids = list(dict.fromkeys(mol_ids))
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        data = get_json(f"{BASE}/molecule.json",
                        params={"molecule_chembl_id__in": ",".join(chunk),
                                "limit": 1000})
        for m in data.get("molecules", []):
            props = m.get("molecule_properties") or {}
            try:
                out[m["molecule_chembl_id"]] = float(props.get("full_mwt") or props["mw_freebase"])
            except (TypeError, ValueError, KeyError):
                pass
        time.sleep(0.15)
    return out


def fetch_dois(doc_ids):
    out = {}
    ids = list(dict.fromkeys(d for d in doc_ids if pd.notna(d)))
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        data = get_json(f"{BASE}/document.json",
                        params={"document_chembl_id__in": ",".join(chunk),
                                "limit": 1000, "only": "document_chembl_id,doi"})
        for d in data.get("documents", []):
            out[d["document_chembl_id"]] = d.get("doi") or ""
        time.sleep(0.15)
    return out


def to_nm(value, units, mw):
    """Значение -> нМ. ug/ml конвертируется через MW, помечается None при нехватке данных."""
    if pd.isna(value) or not isinstance(units, str):
        return None, False
    v = float(value)
    u = units.strip()
    if u == "nM":
        return v, False
    if u == "uM":
        return v * 1000.0, False
    if u == "pM":
        return v / 1000.0, False
    if u == "mM":
        return v * 1_000_000.0, False
    if u in ("ug.mL-1", "ug/ml", "ug/mL") and mw:
        return v * 1_000_000.0 / mw, True  # µg/mL → нМ: v*1e6/MW
    return None, False


def aggregate(group: pd.DataFrame, value_col: str, bound_col: str):
    """Медиана точных значений; если нет — медиана цензурированных с флагом."""
    exact = group[group[bound_col] == ""][value_col].dropna()
    if len(exact):
        return float(exact.median()), ""
    gt = group[group[bound_col] == ">"][value_col].dropna()
    if len(gt):
        return float(gt.median()), ">"
    lt = group[group[bound_col] == "<"][value_col].dropna()
    if len(lt):
        return float(lt.median()), "<"
    return None, ""


def main():
    print("[1/5] Читаю сырые таблицы...", flush=True)
    va = pd.read_csv(EXTERNAL / "chembl_vacv_activities.csv")
    cc = pd.read_csv(EXTERNAL / "chembl_cc50_same_molecule_publication.csv")

    ec = va[(va["standard_type"] == "EC50") & va["standard_value"].notna()].copy()
    cc = cc[cc["standard_value"].notna()].copy()
    print(f"  EC50-записей против VACV: {len(ec)}; CC50-записей: {len(cc)}")

    print("[2/5] Тяну молекулярные массы из ChEMBL API...", flush=True)
    mol_ids = pd.concat([ec["molecule_chembl_id"], cc["molecule_chembl_id"]]).unique()
    mw_map = fetch_mol_weights(mol_ids)
    print(f"  молекул с MW: {len(mw_map)} из {len(mol_ids)}")

    print("[3/5] Конвертирую единицы в нМ...", flush=True)
    for df, pref in ((ec, "ec50"), (cc, "cc50")):
        conv = [to_nm(v, u, mw_map.get(m))
                for v, u, m in zip(df["standard_value"], df["standard_units"], df["molecule_chembl_id"])]
        df[f"{pref}_nm"] = [c[0] for c in conv]
        df[f"{pref}_converted"] = [c[1] for c in conv]
        df[f"{pref}_bound"] = df["standard_relation"].map({"=": "", ">": ">", ">=": ">", "<": "<", "<=": "<"}).fillna("")
        df.dropna(subset=[f"{pref}_nm"], inplace=True)
    n_conv_ec = int(ec["ec50_converted"].sum())
    n_conv_cc = int(cc["cc50_converted"].sum())
    print(f"  конвертировано из ug/ml: EC50 {n_conv_ec}, CC50 {n_conv_cc}")

    print("[4/5] Спариваю записи по (молекула × публикация)...", flush=True)
    ec_j = ec[["molecule_chembl_id", "document_chembl_id", "canonical_smiles",
               "assay_chembl_id", "ec50_nm", "ec50_bound"]].rename(
        columns={"assay_chembl_id": "assay_ec50"})
    cc_j = cc[["molecule_chembl_id", "document_chembl_id",
               "assay_chembl_id", "cc50_nm", "cc50_bound"]].rename(
        columns={"assay_chembl_id": "assay_cc50"})
    pairs = ec_j.merge(cc_j, on=["molecule_chembl_id", "document_chembl_id"], how="inner")
    pairs = pairs[pairs["canonical_smiles"].notna() & (pairs["canonical_smiles"] != "")]
    print(f"  пар записей: {len(pairs)}; уникальных молекул: {pairs['molecule_chembl_id'].nunique()}")

    print("[5/5] Подтягиваю DOI...", flush=True)
    doi_map = fetch_dois(pairs["document_chembl_id"].unique().tolist())
    pairs["doi"] = pairs["document_chembl_id"].map(doi_map).fillna("")

    final = pairs[["canonical_smiles", "ec50_nm", "ec50_bound", "cc50_nm", "cc50_bound", "doi"]].rename(
        columns={"canonical_smiles": "smiles"})
    out_path = EXTERNAL / "ec50_cc50_pairs_vacv.csv"
    final.to_csv(out_path, index=False)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": ["chembl_vacv_activities.csv", "chembl_cc50_same_molecule_publication.csv"],
        "grain": "запись EC50 × запись CC50 той же молекулы из той же публикации",
        "n_pairs": len(final),
        "n_unique_molecules": int(pairs["molecule_chembl_id"].nunique()),
        "n_exact_ec50_and_exact_cc50": int(((final.ec50_bound == "") & (final.cc50_bound == "")).sum()),
        "n_converted_from_ug_ml": {"ec50": n_conv_ec, "cc50": n_conv_cc},
        "warnings": [
            "Пара из одной публикации НЕ гарантирует одну клеточную линию/условия — "
            "проверяйте assay в сырых файлах перед использованием как SI-метки.",
            "Одна молекула встречается в нескольких строках (несколько ассэев/публикаций) — "
            "для обучения агрегируйте или группируйте, чтобы не было утечки между фолдами.",
            "ec50_bound/cc50_bound: '' — точное значение, '>' или '<' — цензурированное.",
            "Значения из ug/ml конвертированы через full_molweight из ChEMBL.",
        ],
    }
    (EXTERNAL / "ec50_cc50_pairs_vacv_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Готово: {out_path}")
    print(f"Отчёт: {EXTERNAL / 'ec50_cc50_pairs_vacv_report.json'}")


if __name__ == "__main__":
    main()
