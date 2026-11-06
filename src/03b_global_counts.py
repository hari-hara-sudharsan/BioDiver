"""Fetch global plant record counts in one facet call.

Red List status is global, but occurrence records here are regional. Without a
global signal the triage model confuses "rare in South Asia" with "rare".
"""
import json

import pandas as pd
import requests

from config import CACHE, GBIF, PROCESSED, USER_AGENT


def main():
    path = CACHE / "global_counts.json"
    if not path.exists():
        r = requests.get(f"{GBIF}/occurrence/search", timeout=280,
                         headers={"User-Agent": USER_AGENT},
                         params={"kingdomKey": 6, "hasCoordinate": "true", "limit": 0,
                                 "facet": "speciesKey", "facetLimit": 600000})
        r.raise_for_status()
        counts = {f["name"]: f["count"] for f in r.json()["facets"][0]["counts"]}
        path.write_text(json.dumps(counts), encoding="utf-8")
    counts = json.loads(path.read_text(encoding="utf-8"))
    print("global species counted:", len(counts))

    table = pd.read_csv(PROCESSED / "species_candidates.csv")
    table["global_records"] = table["speciesKey"].astype(str).map(counts)
    missing = table["global_records"].isna().sum()
    table["region_share"] = (table["regional_records"] / table["global_records"]).clip(upper=1)
    table.to_csv(PROCESSED / "species_candidates.csv", index=False)
    print("candidates missing a global count:", missing)
    print(table[["global_records", "region_share"]].describe().round(3).to_string())
    check = table[table["species"].isin(["Clematis parviloba", "Hemidesmus indicus", "Withania somnifera"])]
    print(check[["species", "regional_records", "global_records", "region_share"]].round(3).to_string())


if __name__ == "__main__":
    main()
