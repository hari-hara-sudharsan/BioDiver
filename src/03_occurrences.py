"""Fetch wild GBIF records for each candidate species.

Licences: CC0, CC BY and CC BY-NC. The open-only filter removed most records for
threatened Himalayan species, so NC records are kept for this non-commercial
research and each record's licence is stored for a sensitivity check.

Species above MAX_RECORDS are sampled per 10-degree tile, because taking the
first pages of results returns clustered records (teak: 2,997 records, 188 cells).

Filters: 1970 onward (the WorldClim baseline era), no fossil
or living-collection records, no records flagged as managed or introduced.
Each species is written to its own cache file, so the run can resume.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from config import (CACHE, GBIF, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PROCESSED,
                    USER_AGENT)

MAX_RECORDS = 3000
PAGE = 300
TILE_DEG = 10
MAX_UNCERTAINTY_M = 10000
EXCLUDED_MEANS = {"MANAGED", "INTRODUCED", "INVASIVE", "NATURALISED"}
OCC = CACHE / "occ"
OCC.mkdir(exist_ok=True)
MIN_INTERVAL_S = 2.0          # shared across threads: GBIF returned 429 at 0.8 s
S = requests.Session()
S.headers["User-Agent"] = USER_AGENT
_lock = threading.Lock()
_next_slot = [0.0]


def throttle(delay=0.0):
    """Reserve the next request slot; a positive delay pushes back every thread."""
    with _lock:
        now = time.monotonic()
        slot = max(now + delay, _next_slot[0])
        _next_slot[0] = slot + MIN_INTERVAL_S
    time.sleep(max(0.0, slot - now))


def page(key, offset, lat=f"{LAT_MIN},{LAT_MAX}", lon=f"{LON_MIN},{LON_MAX}"):
    params = [("taxonKey", key), ("hasCoordinate", "true"),
              ("hasGeospatialIssue", "false"), ("occurrenceStatus", "PRESENT"),
              ("decimalLatitude", lat), ("decimalLongitude", lon),
              ("year", "1970,2026"), ("limit", PAGE), ("offset", offset)]
    params += [("basisOfRecord", b) for b in
               ("HUMAN_OBSERVATION", "PRESERVED_SPECIMEN", "OBSERVATION", "MATERIAL_SAMPLE")]
    params += [("license", l) for l in ("CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0")]
    error = None
    for attempt in range(8):
        try:
            throttle()
            r = S.get(f"{GBIF}/occurrence/search", params=params, timeout=90)
            if r.status_code == 429:
                throttle(float(r.headers.get("Retry-After", 60)))
                error = "429 Too Many Requests"
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            error = e
            time.sleep(2 ** attempt * 3)
    raise RuntimeError(f"GBIF failed for {key} at offset {offset}: {error}")


def keep(key, results, rows):
    for o in results:
        unc = o.get("coordinateUncertaintyInMeters")
        if unc is not None and unc > MAX_UNCERTAINTY_M:
            continue
        if (o.get("establishmentMeans") or "").upper() in EXCLUDED_MEANS:
            continue
        rows.append({"speciesKey": key, "lon": o["decimalLongitude"],
                     "lat": o["decimalLatitude"], "year": o.get("year"),
                     "basis": o.get("basisOfRecord"),
                     "datasetKey": o.get("datasetKey"),
                     "license": o.get("license")})


def fetch(key):
    out = OCC / f"{key}.parquet"
    if out.exists():
        return 0
    rows = []
    first = page(key, 0)
    keep(key, first["results"], rows)
    if first["count"] <= MAX_RECORDS:
        offset = PAGE
        while not first["endOfRecords"] and offset < first["count"]:
            d = page(key, offset)
            keep(key, d["results"], rows)
            offset += PAGE
            if d.get("endOfRecords"):
                break
    else:
        rows = []
        for la in range(LAT_MIN, LAT_MAX, TILE_DEG):
            for lo in range(LON_MIN, LON_MAX, TILE_DEG):
                d = page(key, 0, f"{la},{min(la + TILE_DEG, LAT_MAX)}",
                         f"{lo},{min(lo + TILE_DEG, LON_MAX)}")
                keep(key, d["results"], rows)
    pd.DataFrame(rows, columns=["speciesKey", "lon", "lat", "year", "basis",
                                "datasetKey", "license"]).to_parquet(out)
    return len(rows)


def safe_fetch(key):
    """One bad species must not stop the run; failures are logged and retried on rerun."""
    try:
        return key, fetch(key), None
    except RuntimeError as e:
        return key, 0, str(e)


def main():
    table = pd.read_csv(PROCESSED / "species_candidates.csv")
    table = table.sort_values(["priority", "regional_records"], ascending=[False, True])
    keys = table["speciesKey"].tolist()
    done = 0
    failed = []
    with ThreadPoolExecutor(2) as ex:
        for key, _, error in ex.map(safe_fetch, keys):
            done += 1
            if error:
                failed.append({"speciesKey": key, "error": error})
                print(error, flush=True)
            if done % 100 == 0:
                print(f"{done}/{len(keys)} failed={len(failed)}", flush=True)
    pd.DataFrame(failed, columns=["speciesKey", "error"]).to_csv(
        PROCESSED / "occurrence_failures.csv", index=False)
    print(f"occurrences complete, failed={len(failed)}", flush=True)


if __name__ == "__main__":
    main()
