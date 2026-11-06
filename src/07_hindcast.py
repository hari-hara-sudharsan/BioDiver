"""Temporal hindcast: can a model trained on the past find later records?

Each species is trained on records up to CUTOFF only. The test presences are
cells first recorded after CUTOFF, so nothing in the test set was seen in
training. Background comes from the same collector-effort surface.

This tests transfer to new places under the same baseline climate. It does not
validate 2050 projections, because the climate layer is fixed at 1970-2000.

Usage: python src/07_hindcast.py [--priority] [--jobs N]
"""
import argparse
import os
from pathlib import Path

os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import CACHE, OUTPUTS, PROCESSED

CUTOFF = 2010
MIN_EARLY_CELLS = 15
MIN_LATE_CELLS = 10


def hindcast(key):
    from scipy.ndimage import distance_transform_edt
    from sklearn.metrics import roc_auc_score

    import sdm

    current, land, _, _, effort = sdm.grids()
    H, W = land.shape
    df = pd.read_parquet(CACHE / "occ" / f"{key}.parquet").dropna(subset=["year"])
    out = {"speciesKey": key, "status": "ok"}
    if df.empty:
        return {**out, "status": "no_dated_records"}

    row, col = sdm.to_cells(df)
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    df, row, col = df[ok], row[ok], col[ok]
    ok = land[row, col]
    cell = (row * W + col)[ok]
    year = df["year"].to_numpy()[ok]

    early = np.unique(cell[year <= CUTOFF])
    late = np.setdiff1d(np.unique(cell[year > CUTOFF]), early)
    out.update(early_cells=len(early), late_new_cells=len(late))
    if len(early) < MIN_EARLY_CELLS or len(late) < MIN_LATE_CELLS:
        return {**out, "status": "too_few_cells"}

    seen = np.zeros((H, W), bool)
    seen.flat[early] = True
    access = (distance_transform_edt(~seen) <= sdm.ACCESS_CELLS) & land
    late = late[access.ravel()[late]]
    out["late_new_cells_accessible"] = len(late)
    if len(late) < MIN_LATE_CELLS:
        return {**out, "status": "too_few_cells"}

    rng = np.random.default_rng(sdm.SEED + key % 100000)
    held = np.zeros(H * W, bool)
    held[late] = True
    pool = np.flatnonzero(access.ravel() & ~seen.ravel() & ~held & (effort.ravel() > 0))
    if len(pool) < 400:
        return {**out, "status": "too_little_background"}
    weights = effort.ravel()[pool].astype("float64")
    n_bg = int(min(max(1000, 10 * len(early)), 5000, len(pool) // 2))
    picked = rng.choice(pool, 2 * n_bg, replace=False, p=weights / weights.sum())
    train_bg, test_bg = picked[:n_bg], picked[n_bg:]

    grid = current.reshape(len(sdm.BIO), -1)
    X = grid[:, np.concatenate([early, train_bg])].T
    y = np.concatenate([np.ones(len(early), int), np.zeros(n_bg, int)])
    forest, linear = sdm.rf().fit(X, y), sdm.glm().fit(X, y)

    Xt = grid[:, np.concatenate([late, test_bg])].T
    yt = np.concatenate([np.ones(len(late), int), np.zeros(len(test_bg), int)])
    score = (forest.predict_proba(Xt)[:, 1] + linear.predict_proba(Xt)[:, 1]) / 2
    out["auc_hindcast"] = roc_auc_score(yt, score)

    # Distance baseline: does the model beat "near a known record is suitable"?
    near = -distance_transform_edt(~seen).ravel()[np.concatenate([late, test_bg])]
    out["auc_distance_baseline"] = roc_auc_score(yt, near)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    species = pd.read_csv(PROCESSED / "species_candidates.csv")
    species = species[[(CACHE / "occ" / f"{k}.parquet").exists() for k in species["speciesKey"]]]
    if args.priority:
        species = species[species["priority"] == True]

    rows = Parallel(n_jobs=args.jobs)(delayed(hindcast)(k) for k in species["speciesKey"])
    out = pd.DataFrame(rows).merge(species[["speciesKey", "species", "iucn"]], on="speciesKey")
    name = "hindcast_sample.csv" if args.priority else "hindcast.csv"
    out.to_csv(OUTPUTS / name, index=False)

    print(out["status"].value_counts().to_string())
    good = out[out["status"] == "ok"]
    if len(good):
        print(good[["auc_hindcast", "auc_distance_baseline"]].describe().round(3).to_string())
        print(f"better than chance (AUC > 0.5): {(good.auc_hindcast > 0.5).sum()} of {len(good)}")
        print(f"beats distance baseline: {(good.auc_hindcast > good.auc_distance_baseline).sum()} of {len(good)}")
        pd.set_option("display.width", 160)
        print(good.sort_values("auc_hindcast")[["species", "early_cells", "late_new_cells_accessible",
                                                "auc_hindcast", "auc_distance_baseline"]].round(3).to_string())


if __name__ == "__main__":
    main()
