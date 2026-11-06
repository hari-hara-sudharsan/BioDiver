"""Fetch compound-to-plant links from Wikidata (LOTUS-curated, CC0).

A missing compound means "not reported", not "absent". Reporting depends heavily
on research effort, so compound counts are stored for bias checks downstream.
"""
import json
import time

import pandas as pd
import requests

from config import CACHE, PROCESSED, USER_AGENT

BATCH = 100
ENDPOINT = "https://query.wikidata.org/sparql"
S = requests.Session()
S.headers.update({"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
DIR = CACHE / "chem"
DIR.mkdir(exist_ok=True)


def sparql(query):
    for attempt in range(6):
        try:
            r = S.post(ENDPOINT, data={"query": query}, timeout=120)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 30)))
                continue
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt * 3)
    raise RuntimeError("Wikidata query failed")


def qids_for(names):
    """Look up every taxon QID for each name; names can have duplicate entries."""
    out = []
    for i in range(0, len(names), BATCH):
        values = " ".join(json.dumps(n) for n in names[i:i + BATCH])
        for b in sparql(f"SELECT ?t ?n WHERE {{ VALUES ?n {{ {values} }} ?t wdt:P225 ?n . }}"):
            qid = b["t"]["value"].rsplit("/", 1)[1]
            if qid.startswith("Q"):
                out.append({"name": b["n"]["value"], "qid": qid})
    return pd.DataFrame(out).drop_duplicates()


def compounds(qids, index):
    path = DIR / f"batch_{index}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    values = " ".join(f"wd:{q}" for q in qids)
    rows = [{"qid": b["t"]["value"].rsplit("/", 1)[1],
             "compound": b["c"]["value"].rsplit("/", 1)[1],
             "smiles": b.get("s", {}).get("value"),
             "inchikey": b.get("k", {}).get("value")}
            for b in sparql(f"""SELECT ?t ?c ?s ?k WHERE {{ VALUES ?t {{ {values} }}
                ?c wdt:P703 ?t . OPTIONAL {{ ?c wdt:P233 ?s }}
                OPTIONAL {{ ?c wdt:P235 ?k }} }}""")]
    path.write_text(json.dumps(rows), encoding="utf-8")
    time.sleep(1)
    return rows


def main():
    table = pd.read_csv(PROCESSED / "species_candidates.csv")
    long = pd.concat([table[["speciesKey", "name"]],
                      table[["speciesKey", "species"]].rename(columns={"species": "name"})])
    long = long.dropna().drop_duplicates()
    long = long.merge(qids_for(sorted(long["name"].unique())), on="name")
    print(f"species with a QID: {long.speciesKey.nunique()} of {len(table)}", flush=True)

    qids = sorted(long["qid"].unique())
    rows = []
    for n, i in enumerate(range(0, len(qids), BATCH)):
        rows += compounds(qids[i:i + BATCH], n)
        print(f"batch {n + 1}/{-(-len(qids) // BATCH)} links={len(rows)}", flush=True)

    links = pd.DataFrame(rows).merge(long[["speciesKey", "qid"]].drop_duplicates(), on="qid")
    links = links.drop(columns="qid").drop_duplicates(["speciesKey", "compound"])
    links.to_parquet(PROCESSED / "compounds.parquet")

    per = links.groupby("speciesKey")["compound"].nunique()
    print(f"\nlinks {len(links)}, distinct compounds {links.compound.nunique()}, "
          f"species with compounds {len(per)} of {len(table)}")
    print(f"with SMILES {links.smiles.notna().mean():.1%}")
    print(per.describe().round(1).to_string())
    pr = table[table["priority"] == True].set_index("speciesKey")
    print("\npriority species with no compounds:",
          sorted(pr.loc[~pr.index.isin(per.index), "species"]))
    print(pr.join(per.rename("n")).sort_values("n", na_position="first")[["species", "n"]].head(15).to_string())


if __name__ == "__main__":
    main()
