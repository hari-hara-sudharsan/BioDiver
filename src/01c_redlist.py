"""Rebuild Red List status from the IUCN checklist published on GBIF.

GBIF's per-species iucnRedListCategory endpoint missed real assessments
(Aconitum heterophyllum, Bacopa monnieri), so it is not used for final status.
Join order: name (including IUCN synonyms), then GBIF backbone synonyms.

A synonym passes on its status only if the species epithet matches, which is the
signature of a genus transfer (Saussurea costus -> Dolomiaea costus). A different
epithet can be a separately assessed taxon merged into a widespread species:
Withania chevalieri is CR, and that must not make W. somnifera CR. Such cases are
kept in iucn_review for a human and do not change the status.

"NE" here means no record was found in this checklist. It is not proof that no
assessment exists anywhere.
"""
import csv
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from config import CACHE, GBIF, PROCESSED, RAW, USER_AGENT

DATASET = "19491596-35ae-4a91-9a98-85cf505f1bd3"
ARCHIVE = "https://hosted-datasets.gbif.org/datasets/iucn/iucn-latest.zip"
CODES = {"EXTINCT": "EX", "EXTINCT_IN_THE_WILD": "EW", "CRITICALLY_ENDANGERED": "CR",
         "ENDANGERED": "EN", "VULNERABLE": "VU", "NEAR_THREATENED": "NT",
         "LEAST_CONCERN": "LC", "DATA_DEFICIENT": "DD", "NOT_EVALUATED": "NE"}
EXPECTED = {"Withania somnifera": "NE", "Aconitum heterophyllum": "EN", "Bacopa monnieri": "LC", "Dolomiaea costus": "CR",
            "Nardostachys jatamansi": "CR", "Rauvolfia serpentina": "NT"}
S = requests.Session()
S.headers["User-Agent"] = USER_AGENT


def get(url, **params):
    for attempt in range(6):
        try:
            r = S.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"failed: {url} {params}")


def checklist():
    """Parse global plant assessments from the IUCN Darwin Core archive."""
    path = CACHE / "iucn_plants.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    archive = RAW / "iucn.zip"
    if not archive.exists():
        with requests.get(ARCHIVE, timeout=300, headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            archive.write_bytes(r.content)
    with zipfile.ZipFile(archive) as z:
        dist = pd.read_csv(z.open("distribution.txt"), sep="	", header=None, dtype=str,
                           quoting=csv.QUOTE_NONE, keep_default_na=False)
        taxa = pd.read_csv(z.open("taxon.txt"), sep="	", header=None, dtype=str,
                           quoting=csv.QUOTE_NONE, keep_default_na=False)
    dist = dist[dist[2] == "Global"]
    status = dict(zip(dist[0], dist[5].str.upper().str.replace(" ", "_")))
    taxa = taxa[(taxa[2] == "PLANTAE") & (taxa[10] == "species")]
    rows = []
    for t in taxa.itertuples(index=False):
        tid, accepted = t[0], t[13] or t[0]
        code = status.get(tid) or status.get(accepted)
        if not code:
            continue
        year = re.search(r"Threatened Species (\d{4})", t[14])
        rows.append({"name": f"{t[7]} {t[8]}", "nubKey": None,
                     "code": CODES.get(code, code), "year": int(year.group(1)) if year else None,
                     "synonym": t[12] != "accepted"})
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def same_epithet(a, b):
    """True when two binomials share an epithet stem, allowing gender endings."""
    a, b = a.split(), b.split()
    return len(a) > 1 and len(b) > 1 and a[1][:5] == b[1][:5]


def synonyms(key):
    d = get(f"{GBIF}/species/{key}/synonyms", limit=100)
    return key, [r.get("canonicalName") for r in d.get("results", []) if r.get("canonicalName")]


def main():
    rows = checklist()
    by_name = {r["name"]: r["code"] for r in rows}
    years = {r["name"]: r["year"] for r in rows}
    print(f"plant assessments loaded: {len(rows)}")

    table = pd.read_csv(PROCESSED / "species_candidates.csv")
    if "iucn_endpoint" not in table:
        table["iucn_endpoint"] = table["iucn"]

    def direct(row):
        for n in (row["species"], row["name"]):
            if n in by_name and same_epithet(n, row["species"]):
                return by_name[n], "name", years[n]
        return None, None, None

    found = table.apply(direct, axis=1, result_type="expand")
    table["iucn"], table["iucn_source"], table["iucn_year"] = found[0], found[1], found[2]

    missing = table[table["iucn"].isna()]["speciesKey"].tolist()
    print(f"direct matches {len(table) - len(missing)}, checking synonyms for {len(missing)}", flush=True)
    syn_path = CACHE / "synonyms.json"
    syn = json.loads(syn_path.read_text()) if syn_path.exists() else {}
    todo = [k for k in missing if str(k) not in syn]
    with ThreadPoolExecutor(2) as ex:
        for key, names in ex.map(synonyms, todo):
            syn[str(key)] = names
    syn_path.write_text(json.dumps(syn))

    conflicts = []
    table["iucn_review"] = None
    for i, row in table[table["iucn"].isna()].iterrows():
        names = syn.get(str(row["speciesKey"]), []) + [row["name"]]
        all_hits = sorted({n for n in names if n in by_name})
        hits = [n for n in all_hits if same_epithet(n, row["species"])]
        others = [f"{n}={by_name[n]}" for n in all_hits if n not in hits]
        if others:
            table.loc[i, "iucn_review"] = "; ".join(others)
        codes = {by_name[n] for n in hits}
        if codes:
            order = ["EX", "EW", "CR", "EN", "VU", "NT", "LC", "DD"]
            table.loc[i, "iucn"] = min(codes, key=order.index)
            table.loc[i, "iucn_year"] = max((years[n] or 0) for n in hits) or None
            table.loc[i, "iucn_source"] = "synonym"
            if len(codes) > 1:
                conflicts.append((row["species"], sorted(codes)))
    # Keep assessments the endpoint knows about but the name joins could not reach.
    fallback = table["iucn"].isna() & table["iucn_endpoint"].notna() & (table["iucn_endpoint"] != "NE")
    table.loc[fallback, "iucn"] = table.loc[fallback, "iucn_endpoint"]
    table.loc[fallback, "iucn_source"] = "endpoint"
    table["iucn"] = table["iucn"].fillna("NE")
    table["iucn_source"] = table["iucn_source"].fillna("none")
    table.to_csv(PROCESSED / "species_candidates.csv", index=False)

    print("\nsource:\n" + table["iucn_source"].value_counts().to_string())
    print("\nstatus now vs endpoint:\n" + pd.crosstab(table["iucn_endpoint"], table["iucn"]).to_string())
    print(f"\nno checklist record: {(table.iucn == 'NE').sum()} of {len(table)} "
          f"= {(table.iucn == 'NE').mean():.1%}")
    print("synonym conflicts (most severe kept):", conflicts[:20])
    pr = table[table["priority"] == True]
    print("\npriority species with no record:", sorted(pr[pr.iucn == "NE"]["species"]))
    print("\nregression test:")
    ok = True
    for name, want in EXPECTED.items():
        got = table.loc[table["species"] == name, "iucn"].squeeze()
        ok &= got == want
        print(f"  {name}: expected {want}, got {got}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
