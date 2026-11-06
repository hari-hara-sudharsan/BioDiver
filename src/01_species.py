"""Build the candidate species table.

Wikidata medicinal taxa -> GBIF backbone match -> regional record count ->
Red List category. Every lookup is cached so reruns are cheap.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from config import (CACHE, GBIF, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN,
                    MIN_OCCURRENCES, PROCESSED, USER_AGENT)

S = requests.Session()
S.headers["User-Agent"] = USER_AGENT
SPARQL = """SELECT DISTINCT ?t ?name WHERE { ?t wdt:P225 ?name .
 { ?t wdt:P366 wd:Q188840 } UNION { ?t wdt:P31 wd:Q188840 }
 UNION { ?t wdt:P279 wd:Q188840 } }"""


def cached(name, fn):
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = fn()
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def wikidata_taxa():
    r = S.get("https://query.wikidata.org/sparql", timeout=120,
              params={"query": SPARQL},
              headers={"Accept": "application/sparql-results+json"})
    r.raise_for_status()
    return [{"qid": b["t"]["value"].rsplit("/", 1)[1], "name": b["name"]["value"]}
            for b in r.json()["results"]["bindings"]]


def regional_counts():
    """Species-level plant record counts for the extent, in one facet call."""
    r = S.get(f"{GBIF}/occurrence/search", timeout=120, params={
        "kingdomKey": 6, "hasCoordinate": "true", "hasGeospatialIssue": "false",
        "decimalLatitude": f"{LAT_MIN},{LAT_MAX}",
        "decimalLongitude": f"{LON_MIN},{LON_MAX}",
        "facet": "speciesKey", "facetLimit": 100000, "limit": 0})
    r.raise_for_status()
    return {f["name"]: f["count"] for f in r.json()["facets"][0]["counts"]}


def match(name):
    for attempt in range(3):
        try:
            r = S.get(f"{GBIF}/species/match", timeout=30,
                      params={"name": name, "kingdom": "Plantae", "strict": "true"})
            r.raise_for_status()
            d = r.json()
            return {"name": name, "speciesKey": d.get("speciesKey"),
                    "species": d.get("species"), "family": d.get("family"),
                    "matchType": d.get("matchType"), "rank": d.get("rank")}
        except requests.RequestException:
            continue
    return {"name": name, "speciesKey": None}


def iucn(key):
    for attempt in range(3):
        try:
            r = S.get(f"{GBIF}/species/{key}/iucnRedListCategory", timeout=30)
            if r.status_code == 404 or not r.text.strip():
                return {"speciesKey": key, "iucn": "NOT_EVALUATED"}
            r.raise_for_status()
            return {"speciesKey": key, "iucn": r.json().get("code") or "NE"}
        except requests.RequestException:
            continue
    return {"speciesKey": key, "iucn": None}


def main():
    taxa = cached("wikidata_taxa.json", wikidata_taxa)
    counts = cached("regional_counts.json", regional_counts)
    print(f"wikidata taxa {len(taxa)}, regional plant species {len(counts)}")

    names = sorted({t["name"] for t in taxa})
    with ThreadPoolExecutor(8) as ex:
        matches = cached("matches.json", lambda: list(ex.map(match, names)))
    m = pd.DataFrame(matches).dropna(subset=["speciesKey"])
    m["speciesKey"] = m["speciesKey"].astype(int)
    m = m[m["matchType"].isin(["EXACT", "FUZZY"])].drop_duplicates("speciesKey")
    m["regional_records"] = m["speciesKey"].astype(str).map(counts).fillna(0).astype(int)
    print(f"matched species {len(m)}, with any regional record {(m.regional_records > 0).sum()}")

    keep = m[m["regional_records"] >= MIN_OCCURRENCES].copy()
    with ThreadPoolExecutor(8) as ex:
        status = pd.DataFrame(ex.map(iucn, keep["speciesKey"].tolist()))
    keep = keep.merge(status, on="speciesKey").sort_values("regional_records", ascending=False)
    keep.to_csv(PROCESSED / "species_candidates.csv", index=False)
    print(f"candidates with >= {MIN_OCCURRENCES} records: {len(keep)}")
    print(keep["iucn"].value_counts(dropna=False).to_string())
    print(keep.head(15)[["species", "family", "regional_records", "iucn"]].to_string())


if __name__ == "__main__":
    main()
