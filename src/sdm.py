"""Species distribution models with spatial validation and climate projection.

For each species:
  1. Thin records to one per 5 arc-min cell.
  2. Restrict to an accessible area (500 km around records).
  3. Draw target-group background, which mirrors collector effort.
  4. Compare BIOCLIM, quadratic GLM, Random Forest and an RF+GLM mean under
     spatial block CV. In a priority-species preview RF alone did not beat the GLM,
     so the final model is the fixed RF+GLM mean, chosen for all species up front
     rather than per species, which would inflate the reported scores.
  5. Project to 5 GCMs x 2 SSPs and grade each species:
       standard  skilful, enough data, mostly within sampled climate
       low       weak skill, few cells or heavy climate extrapolation
       no_skill  spatial AUC below 0.6; range change must not be interpreted

Elevation is deliberately not a predictor: it stays fixed in future scenarios
and would mask climate-driven change.

Run through 04_sdm.py: python src/04_sdm.py [--limit N] [--jobs N]
"""
import argparse
import json
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import CACHE, LAT_MAX, LON_MIN, OUTPUTS, PROCESSED

warnings.filterwarnings("ignore")

BIO = [1, 2, 4, 6, 12, 15, 17]     # low-collinearity subset of the 19 variables
RES = 1 / 12
GCMS = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MIROC6", "IPSL-CM6A-LR", "EC-Earth3-Veg"]
SSPS = ["ssp245", "ssp585"]
PERIOD = "2041-2060"
ACCESS_CELLS = 54                  # about 500 km at 9.3 km per cell
BLOCK_DEG = 5
MIN_CELLS = 10                     # 10-19 cells are modelled but flagged low confidence
STANDARD_CELLS = 20
STANDARD_AUC = 0.7
NO_SKILL_AUC = 0.6
MAX_NOVEL = 0.5                    # above this share of range, projection is extrapolation
SEED = 42
MAPS = PROCESSED / "species_maps"
MAPS.mkdir(exist_ok=True)
RESULTS = CACHE / "sdm"
RESULTS.mkdir(exist_ok=True)


def read_stack(path):
    with rasterio.open(path) as src:
        return src.read([b for b in BIO]).astype("float32")


@lru_cache(maxsize=1)
def grids():
    """Climate stacks, land mask, elevation and collector-effort surface."""
    current = read_stack(PROCESSED / "climate_current.tif")
    land = (current > -9000).all(axis=0)
    with rasterio.open(PROCESSED / "elevation.tif") as src:
        elev = src.read(1)
    future = {}
    for ssp in SSPS:
        for gcm in GCMS:
            path = PROCESSED / f"climate_{gcm}_{ssp}_{PERIOD}.tif"
            if path.exists():
                future[(gcm, ssp)] = read_stack(path)
    effort = np.load(PROCESSED / "effort.npy")
    return current, land, elev, future, effort


def to_cells(df):
    row = ((LAT_MAX - df["lat"].to_numpy()) / RES).astype(int)
    col = ((df["lon"].to_numpy() - LON_MIN) / RES).astype(int)
    return row, col


def build_effort():
    """Count distinct species recorded per cell across every candidate."""
    current = read_stack(PROCESSED / "climate_current.tif")
    effort = np.zeros(current.shape[1:], dtype="int32")
    for path in (CACHE / "occ").glob("*.parquet"):
        df = pd.read_parquet(path, columns=["lon", "lat"])
        if df.empty:
            continue
        row, col = to_cells(df)
        ok = (row >= 0) & (row < effort.shape[0]) & (col >= 0) & (col < effort.shape[1])
        cells = np.unique(row[ok] * effort.shape[1] + col[ok])
        effort.flat[cells] += 1
    np.save(PROCESSED / "effort.npy", effort)
    return effort


def bioclim_score(train, test):
    """Classic envelope: lowest two-tailed percentile across variables."""
    ranks = np.stack([np.searchsorted(np.sort(train[:, j]), test[:, j]) / len(train)
                      for j in range(train.shape[1])], axis=1)
    return (2 * np.minimum(ranks, 1 - ranks)).min(axis=1)


def rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features=2,
                                  class_weight="balanced_subsample", n_jobs=1,
                                  random_state=SEED)


def glm():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                         LogisticRegression(max_iter=500, class_weight="balanced", C=0.1))


def best_tss(y, score):
    """Threshold maximising TSS, returned with the TSS it achieves."""
    order = np.argsort(-score)
    ys = y[order]
    tp = np.cumsum(ys) / max(ys.sum(), 1)
    fp = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    i = int(np.argmax(tp - fp))
    return float(score[order][i]), float((tp - fp)[i])


def cross_validate(X, y, splits):
    """Out-of-fold scores for each model family."""
    oof = {m: np.full(len(y), np.nan) for m in ("rf", "glm", "bioclim", "ens")}
    for tr, te in splits:
        if y[tr].sum() < 5 or y[te].sum() == 0:
            continue
        oof["rf"][te] = rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof["glm"][te] = glm().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof["bioclim"][te] = bioclim_score(X[tr][y[tr] == 1], X[te])
        oof["ens"][te] = (oof["rf"][te] + oof["glm"][te]) / 2
    return oof


def auc(y, score):
    ok = ~np.isnan(score)
    if ok.sum() == 0 or len(set(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], score[ok])


def model_species(key):
    current, land, elev, future, effort = grids()
    H, W = land.shape
    df = pd.read_parquet(CACHE / "occ" / f"{key}.parquet")
    result = {"speciesKey": key, "records": len(df), "status": "ok"}
    if df.empty:
        return {**result, "cells": 0, "status": "too_few_cells"}

    row, col = to_cells(df)
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    row, col = row[ok], col[ok]
    ok = land[row, col]
    cells = np.unique(row[ok] * W + col[ok])
    result["cells"] = len(cells)
    if len(cells) < MIN_CELLS:
        return {**result, "status": "too_few_cells"}

    pres = np.zeros((H, W), bool)
    pres.flat[cells] = True
    access = (distance_transform_edt(~pres) <= ACCESS_CELLS) & land

    rng = np.random.default_rng(SEED + key % 100000)
    pool = np.flatnonzero((access & ~pres & (effort > 0)).ravel())
    n_bg = int(min(max(1000, 10 * len(cells)), 5000, len(pool)))
    if n_bg < 200:
        return {**result, "status": "too_little_background"}
    weights = effort.ravel()[pool].astype("float64")
    bg = rng.choice(pool, n_bg, replace=False, p=weights / weights.sum())

    idx = np.concatenate([cells, bg])
    y = np.concatenate([np.ones(len(cells), int), np.zeros(n_bg, int)])
    X = current.reshape(len(BIO), -1)[:, idx].T
    r, c = idx // W, idx % W
    lat, lon = LAT_MAX - (r + .5) * RES, LON_MIN + (c + .5) * RES
    blocks = (np.floor(lon / BLOCK_DEG) * 1000 + np.floor(lat / BLOCK_DEG)).astype(int)

    n_blocks = len(np.unique(blocks[y == 1]))
    result["presence_blocks"] = n_blocks
    if n_blocks < 3:
        return {**result, "status": "too_few_blocks"}

    folds = min(5, n_blocks)
    spatial = list(StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(X, y, blocks))
    oof = cross_validate(X, y, spatial)
    for name, score in oof.items():
        result[f"auc_{name}"] = auc(y, score)

    random_splits = StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)
    random_oof = np.full(len(y), np.nan)
    for tr, te in random_splits:
        random_oof[te] = rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    result["auc_rf_random_cv"] = auc(y, random_oof)

    valid = ~np.isnan(oof["ens"])
    if valid.sum() == 0 or len(set(y[valid])) < 2:
        return {**result, "status": "cv_failed"}
    threshold, tss = best_tss(y[valid], oof["ens"][valid])
    result.update(threshold=threshold, tss=tss)

    forest, linear = rf().fit(X, y), glm().fit(X, y)

    def predict(data):
        return (forest.predict_proba(data)[:, 1] + linear.predict_proba(data)[:, 1]) / 2

    a_idx = np.flatnonzero(access.ravel())
    now = predict(current.reshape(len(BIO), -1)[:, a_idx].T) >= threshold
    result["suitable_cells"] = int(now.sum())
    result["suitable_frac"] = float(now.mean())
    if now.sum() == 0:
        return {**result, "status": "empty_projection"}
    result["elev_mean_m"] = float(np.nanmean(elev.ravel()[a_idx][now]))

    lo, hi = X[y == 1].min(axis=0), X[y == 1].max(axis=0)
    maps = {"cells": a_idx[now].astype("int32")}
    for ssp in SSPS:
        kept = np.zeros(len(a_idx), "uint8")
        losses, gains, novel = [], [], []
        for gcm in GCMS:
            if (gcm, ssp) not in future:
                continue
            Xf = future[(gcm, ssp)].reshape(len(BIO), -1)[:, a_idx].T
            # Cells with no future data are left out of the change statistics.
            # Filling them with current climate would imply "no change".
            has_data = (Xf > -9000).all(axis=1)
            base = now & has_data
            if base.sum() == 0:
                continue
            fut = np.zeros(len(a_idx), bool)
            fut[has_data] = predict(Xf[has_data]) >= threshold
            kept += fut.astype("uint8")
            losses.append(1 - (base & fut).sum() / base.sum())
            gains.append((~now & fut).sum() / base.sum())
            novel.append(((Xf[base] < lo) | (Xf[base] > hi)).any(axis=1).mean())
        if not losses:
            continue
        result.update({f"loss_{ssp}_median": float(np.median(losses)),
                       f"loss_{ssp}_min": float(np.min(losses)),
                       f"loss_{ssp}_max": float(np.max(losses)),
                       f"gain_{ssp}_median": float(np.median(gains)),
                       f"novel_{ssp}_median": float(np.median(novel))})
        agree = 0.8 * len(losses)
        refugia = now & (kept >= agree)
        result[f"refugia_{ssp}_cells"] = int(refugia.sum())
        if refugia.any():
            result[f"refugia_{ssp}_elev_m"] = float(np.nanmean(elev.ravel()[a_idx][refugia]))
        maps[f"agree_{ssp}"] = kept[now]
    np.savez_compressed(MAPS / f"{key}.npz", **maps)

    skill = result["auc_ens"]
    novel = result.get("novel_ssp585_median", 1.0)
    if np.isnan(skill) or skill < NO_SKILL_AUC:
        result["confidence"] = "no_skill"
    elif len(cells) >= STANDARD_CELLS and skill >= STANDARD_AUC and novel <= MAX_NOVEL:
        result["confidence"] = "standard"
    else:
        result["confidence"] = "low"
    return result


def run_species(key):
    """Checkpointed wrapper: a long run can resume, and one species cannot stop it."""
    path = RESULTS / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        result = model_species(key)
    except Exception as error:
        return {"speciesKey": key, "status": "error", "error": repr(error)[:300]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, default=float))
    tmp.replace(path)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--priority", action="store_true", help="curated species only")
    ap.add_argument("--fresh", action="store_true", help="discard saved per-species results")
    args = ap.parse_args()

    if args.fresh:
        for old in RESULTS.glob("*.json"):
            old.unlink()
    if args.fresh or not (PROCESSED / "effort.npy").exists():
        build_effort()
    species = pd.read_csv(PROCESSED / "species_candidates.csv")
    species = species[[(CACHE / "occ" / f"{k}.parquet").exists() for k in species["speciesKey"]]]
    if args.priority:
        species = species[species["priority"] == True]
    species = species.sort_values(["priority", "regional_records"], ascending=[False, False])
    keys = species["speciesKey"].tolist()
    if args.limit:
        keys = species.sample(min(args.limit, len(keys)), random_state=SEED)["speciesKey"].tolist()

    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_species)(k) for k in keys)
    out = pd.DataFrame(rows).merge(species, on="speciesKey", how="left")
    # Separate names, so a random sample cannot overwrite the priority preview.
    name = ("sdm_random_sample.csv" if args.limit else
            "sdm_priority_sample.csv" if args.priority else "sdm_results.csv")
    out.to_csv(OUTPUTS / name, index=False)

    print(out["status"].value_counts().to_string())
    good = out[out["status"] == "ok"]
    cols = ["auc_bioclim", "auc_glm", "auc_rf", "auc_ens", "auc_rf_random_cv", "tss",
            "loss_ssp245_median", "loss_ssp585_median"]
    print(good[[c for c in cols if c in good]].describe().round(3).to_string())
