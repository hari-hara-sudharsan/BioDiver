"""Export public dashboard data to docs/data/app.json.

Safeguards are applied here, so the page cannot leak what it never receives:
  * Nothing finer than 0.5-degree blocks (about 55 km) is exported.
  * Threatened species (CR, EN, VU, EW) have no map, no location list and no
    share in any public layer. Counting them in aggregates was tried and dropped:
    subtracting the guide list from a block's richness revealed where they were.
  * Only standard-grade models feed the map layers and the grower's guide.
  * No occurrence coordinates are exported.

Usage: python src/09_export.py [--results sdm_results.csv] [--hindcast hindcast.csv]
"""
import argparse
import json
from datetime import date

import numpy as np
import pandas as pd
import rasterio

from config import LAT_MAX, LON_MIN, OUTPUTS, PROCESSED, ROOT

BLOCK = 6                      # 6 cells of 5 arc-min = 0.5 degrees
BLOCK_DEG = 0.5
PRESENT_SHARE = 0.25           # share of a block's land cells to count as present
AGREE = 4
SENSITIVE = {"CR", "EN", "VU", "EW"}
MIN_RICHNESS = 1


def clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 3)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def records(frame):
    return [{k: clean(v) for k, v in row.items()} for row in frame.to_dict("records")]


def read(name):
    path = OUTPUTS / name
    return pd.read_csv(path) if path.exists() else None


def block_grid():
    with rasterio.open(PROCESSED / "climate_current.tif") as src:
        land = (src.read(1) > -9000)
    H, W = land.shape
    bw = -(-W // BLOCK)
    rows, cols = np.indices((H, W))
    ids = (rows // BLOCK) * bw + cols // BLOCK
    land_per_block = np.bincount(ids[land], minlength=ids.max() + 1)
    return ids.ravel(), land_per_block, bw


def presence(cells, ids, land_per_block):
    """Blocks where the given cells cover enough of the block's land."""
    if len(cells) == 0:
        return np.zeros(len(land_per_block), bool)
    counts = np.bincount(ids[cells], minlength=len(land_per_block))
    return (land_per_block > 0) & (counts >= PRESENT_SHARE * np.maximum(land_per_block, 1))


def block_key(lat, lon):
    """Same text JavaScript gives for these numbers: 35 rather than 35.0."""
    return f"{round(lat, 2):g},{round(lon, 2):g}"


def layers(res, species_guide):
    ids, land_per_block, bw = block_grid()
    now = np.zeros(len(land_per_block), "int32")
    keep = {s: np.zeros(len(land_per_block), "int32") for s in ("ssp245", "ssp585")}
    guide = {}
    for row in res.itertuples():
        path = PROCESSED / "species_maps" / f"{row.speciesKey}.npz"
        if not path.exists():
            continue
        m = np.load(path)
        cells = m["cells"]
        here = presence(cells, ids, land_per_block)
        now += here
        stay = {}
        for ssp in keep:
            if f"agree_{ssp}" not in m:
                continue
            kept = presence(cells[m[f"agree_{ssp}"] >= AGREE], ids, land_per_block)
            keep[ssp] += kept & here
            stay[ssp] = kept
        index = species_guide.get(row.speciesKey)
        if index is None or "ssp585" not in stay:
            continue
        for b in np.flatnonzero(here):
            guide.setdefault(int(b), []).append(
                [index, int(stay.get("ssp245", here)[b]), int(stay["ssp585"][b])])

    blocks = []
    for b in np.flatnonzero(now >= MIN_RICHNESS):
        r, c = divmod(int(b), bw)
        blocks.append([round(LAT_MAX - r * BLOCK_DEG, 2), round(LON_MIN + c * BLOCK_DEG, 2),
                       int(now[b]), int(keep["ssp245"][b]), int(keep["ssp585"][b])])
    return blocks, {block_key(LAT_MAX - (b // bw) * BLOCK_DEG, LON_MIN + (b % bw) * BLOCK_DEG): v
                    for b, v in guide.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="sdm_results.csv")
    ap.add_argument("--hindcast", default="hindcast.csv")
    args = ap.parse_args()

    res = pd.read_csv(OUTPUTS / args.results)
    meta = pd.read_csv(PROCESSED / "species_candidates.csv")
    extra = [c for c in meta.columns if c not in res.columns or c == "speciesKey"]
    res = res.merge(meta[extra], on="speciesKey", how="left")
    res["sensitive"] = res["iucn"].isin(SENSITIVE)
    ok = res[res["status"] == "ok"]
    standard = ok[ok["confidence"] == "standard"]

    links = pd.read_parquet(PROCESSED / "compounds.parquet")
    compounds = links.groupby("speciesKey")["compound"].nunique()
    triage = read("triage_predictions.csv")

    shown = res[(res["priority"] == True) | (res["confidence"] == "standard")].copy()
    shown["compounds"] = shown["speciesKey"].map(compounds)
    shown["grade"] = shown["confidence"].fillna("data_gap")
    if triage is not None:
        shown = shown.merge(triage[["species", "triage"]], on="species", how="left")
    cols = ["species", "family", "iucn", "iucn_year", "priority", "sensitive", "grade", "cells",
            "auc_ens", "loss_ssp245_median", "loss_ssp585_median", "loss_ssp585_min",
            "loss_ssp585_max", "novel_ssp585_median", "compounds", "triage", "speciesKey"]
    shown = shown[[c for c in cols if c in shown]].sort_values(["priority", "species"], ascending=[False, True])

    guide_rows = shown[(shown["priority"] == True) & (shown["grade"] == "standard") & ~shown["sensitive"]]
    guide_names = guide_rows["species"].tolist()
    guide_index = {k: i for i, k in enumerate(guide_rows["speciesKey"])}
    blocks, guide = layers(standard[~standard["sensitive"]], guide_index)

    hind = read(args.hindcast)
    if hind is not None:
        hind = hind[hind["status"] == "ok"]
    chem = read("chemistry_at_risk.csv")
    summary = {
        "generated": date.today().isoformat(),
        "preview": args.results != "sdm_results.csv",
        "candidates": len(meta),
        "no_checklist_record": int((meta["iucn"] == "NE").sum()),
        "threatened": int(meta["iucn"].isin({"CR", "EN", "VU"}).sum()),
        "priority": int((meta["priority"] == True).sum()),
        "priority_no_record": int(((meta["priority"] == True) & (meta["iucn"] == "NE")).sum()),
        "modelled": len(ok),
        "grades": ok["confidence"].value_counts().to_dict(),
        "priority_gaps": sorted(res[(res["priority"] == True) & (res["status"] != "ok")]["species"].tolist()),
        "auc": {k: clean(ok[f"auc_{k}"].median()) for k in ("bioclim", "glm", "rf", "ens", "rf_random_cv")},
        "hindcast": None if hind is None or hind.empty else {
            "species": len(hind), "auc": clean(hind["auc_hindcast"].median()),
            "distance": clean(hind["auc_distance_baseline"].median()),
            "above_chance": int((hind["auc_hindcast"] > 0.5).sum()),
            "beats_distance": int((hind["auc_hindcast"] > hind["auc_distance_baseline"]).sum())},
    }

    out = {"summary": summary, "blocks": blocks, "guide": guide, "guide_species": guide_names,
           "species": records(shown.drop(columns="speciesKey")),
           "formula": records(read("formula_stress.csv")) if read("formula_stress.csv") is not None else [],
           "ingredients": records(read("formula_ingredients.csv")[[
               "formula", "species", "traditional_name", "status_x", "source", "iucn", "grade",
               "loss_ssp585_median", "loss_ssp585_min", "loss_ssp585_max"]])
           if read("formula_ingredients.csv") is not None else [],
           "triage_validation": records(read("triage_validation.csv")) if read("triage_validation.csv") is not None else [],
           "triage_conformal": records(read("triage_conformal.csv")) if read("triage_conformal.csv") is not None else [],
           "triage_released": triage is not None,
           "chemistry": records(chem)[0] if chem is not None else None}

    target = ROOT / "docs/data"
    target.mkdir(parents=True, exist_ok=True)
    (target / "app.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    size = (target / "app.json").stat().st_size / 1e6
    leak = set(res[res["sensitive"]]["species"]) & set(guide_names)
    print(f"app.json {size:.2f} MB, blocks {len(blocks)}, guide species {len(guide_names)}, "
          f"guide locations {len(guide)}, sensitive species in guide: {len(leak)}")
    assert not leak, leak


if __name__ == "__main__":
    main()
