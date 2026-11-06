"""Merge the curated priority names into the candidate table.

Wikidata's medicinal tag misses major species, including Withania somnifera.
Names resolve to accepted GBIF species, so a synonym such as Saussurea costus
may appear under its accepted name.
"""
import importlib

import pandas as pd

from config import CACHE, MIN_OCCURRENCES, PROCESSED, ROOT

species = importlib.import_module("01_species")
import json


def main():
    lines = (ROOT / "data/curated/priority_species.txt").read_text(encoding="utf-8").splitlines()
    names = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    counts = json.loads((CACHE / "regional_counts.json").read_text(encoding="utf-8"))

    rows = pd.DataFrame([species.match(n) for n in names])
    unresolved = rows[rows["speciesKey"].isna()]["name"].tolist()
    rows = rows.dropna(subset=["speciesKey"]).copy()
    rows["speciesKey"] = rows["speciesKey"].astype(int)
    rows["regional_records"] = rows["speciesKey"].astype(str).map(counts).fillna(0).astype(int)
    rows = rows.merge(pd.DataFrame(species.iucn(k) for k in rows["speciesKey"]), on="speciesKey")
    rows["priority"] = True

    path = PROCESSED / "species_candidates.csv"
    base = pd.read_csv(path)
    base = base[~base["speciesKey"].isin(rows["speciesKey"])]
    if "priority" in base:
        base = base[base["priority"] != True]
    base["priority"] = False
    both = pd.concat([rows, base], ignore_index=True)
    both.to_csv(path, index=False)

    print("unresolved:", unresolved)
    print(rows.sort_values("regional_records")[["name", "species", "regional_records", "iucn"]].to_string())
    print("priority below record screen:",
          (rows["regional_records"] < MIN_OCCURRENCES).sum(), "of", len(rows))
    print("total candidates:", len(both))


if __name__ == "__main__":
    main()
