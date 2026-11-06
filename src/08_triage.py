"""Red List triage for species with no checklist record.

This is a screening aid for expert review, not an assessment. Only IUCN
assessors can assign a category.

Design, fixed before results were seen:
  * Labels: threatened (CR, EN, VU) against not threatened (LC, NT). DD, NE and
    EW are excluded from training.
  * Validation: whole plant families are held out, so the model cannot score
    well by memorising which families are threatened.
  * Baseline: logistic regression on regional and global range signals alone.
  * Release gate: predictions are written only if the model's PR-AUC is at least
    twice the base rate and beats the baseline. Otherwise a negative result is
    reported and no species-level predictions are published.
  * Uncertainty: class-conditional conformal prediction. A species may honestly
    come back as "uncertain" when both classes remain plausible.

Range size feeds Red List criterion B, so some predictive power is expected by
construction. That is why range alone is the baseline to beat.

Usage: python src/08_triage.py [--results sdm_results.csv]
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import ConvexHull, QhullError
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import CACHE, LAT_MAX, LON_MIN, OUTPUTS, PROCESSED

THREATENED = {"CR", "EN", "VU"}
NOT_THREATENED = {"LC", "NT"}
ALPHA = 0.10
FOLDS = 5
REPEATS = 5
MIN_CELLS = 3
GATE_LIFT = 2.0
SEED = 42
RES = 1 / 12
RANGE_ONLY = ["log_cells", "log_global"]
RECORD_FEATURES = ["log_cells", "log_records", "log_hull", "lat_span", "lon_span",
                   "elev_mean", "elev_span", "lat_mean", "log_global", "region_share"]
MODEL_FEATURES = ["log_suitable", "loss_ssp245_median", "loss_ssp585_median",
                  "gain_ssp585_median", "novel_ssp585_median", "auc_ens"]


def record_features(species):
    with rasterio.open(PROCESSED / "elevation.tif") as src:
        elev = src.read(1)
    H, W = elev.shape
    rows = []
    for key in species["speciesKey"]:
        path = CACHE / "occ" / f"{key}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["lon", "lat"])
        if df.empty:
            continue
        r = ((LAT_MAX - df["lat"].to_numpy()) / RES).astype(int).clip(0, H - 1)
        c = ((df["lon"].to_numpy() - LON_MIN) / RES).astype(int).clip(0, W - 1)
        cells = np.unique(r * W + c)
        if len(cells) < MIN_CELLS:
            continue
        pts = np.column_stack([cells % W, cells // W]) * RES
        try:
            hull = ConvexHull(pts).volume
        except QhullError:
            hull = 0.0
        heights = elev.ravel()[cells]
        heights = heights[heights > -9000]
        rows.append({"speciesKey": key, "log_cells": np.log1p(len(cells)),
                     "log_records": np.log1p(len(df)), "log_hull": np.log1p(hull),
                     "lat_span": np.ptp(df["lat"]), "lon_span": np.ptp(df["lon"]),
                     "lat_mean": df["lat"].mean(),
                     "elev_mean": heights.mean() if len(heights) else np.nan,
                     "elev_span": np.ptp(heights) if len(heights) else np.nan})
    return pd.DataFrame(rows)


def logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))


def forest():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED))


def family_cv(make, X, y, groups, seed):
    """Out-of-fold probabilities with whole families held out."""
    out = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    splitter = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(splitter.split(X, y, groups)):
        out[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        fold_id[te] = f
    return out, fold_id


def scores(make, X, y, groups):
    runs = []
    for r in range(REPEATS):
        p, _ = family_cv(make, X, y, groups, SEED + r)
        runs.append((average_precision_score(y, p), roc_auc_score(y, p)))
    runs = np.array(runs)
    return {"pr_auc": runs[:, 0].mean(), "pr_auc_sd": runs[:, 0].std(),
            "roc_auc": runs[:, 1].mean()}


def conformal_p(p, cal_p, cal_y):
    """Class-conditional p-values for each candidate label."""
    threat = (1 + (cal_p[cal_y == 1][None, :] <= p[:, None]).sum(axis=1)) / (1 + (cal_y == 1).sum())
    safe = (1 + (cal_p[cal_y == 0][None, :] >= p[:, None]).sum(axis=1)) / (1 + (cal_y == 0).sum())
    return threat, safe


def label_sets(threat, safe):
    keep_t, keep_s = threat > ALPHA, safe > ALPHA
    return np.select([keep_t & keep_s, keep_t, keep_s],
                     ["uncertain", "possibly_threatened", "likely_not_threatened"],
                     default="unusual")


def conformal_check(p, y, fold_id):
    """Calibrate on other folds and measure coverage on each held-out fold."""
    hit = np.zeros(len(y), bool)
    sets = np.empty(len(y), dtype=object)
    for f in np.unique(fold_id):
        te, cal = fold_id == f, fold_id != f
        threat, safe = conformal_p(p[te], p[cal], y[cal])
        sets[te] = label_sets(threat, safe)
        hit[te] = np.where(y[te] == 1, threat > ALPHA, safe > ALPHA)
    return {"coverage_threatened": hit[y == 1].mean(), "coverage_not_threatened": hit[y == 0].mean(),
            "target": 1 - ALPHA, "share_uncertain": (sets == "uncertain").mean(),
            "share_decisive": np.isin(sets, ["possibly_threatened", "likely_not_threatened"]).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="sdm_results.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 30)

    species = pd.read_csv(PROCESSED / "species_candidates.csv")
    feats = record_features(species)
    res = pd.read_csv(OUTPUTS / args.results)
    res["log_suitable"] = np.log1p(res["suitable_cells"])
    good = res[res["confidence"].isin(["standard", "low"])]
    table = (species.merge(feats, on="speciesKey")
             .merge(good[["speciesKey", "confidence"] + MODEL_FEATURES],
                    on="speciesKey", how="left"))

    table["log_global"] = np.log1p(table["global_records"])

    labelled = table[table["iucn"].isin(THREATENED | NOT_THREATENED)].reset_index(drop=True)
    y = labelled["iucn"].isin(THREATENED).astype(int).to_numpy()
    groups = labelled["family"].fillna("unknown").to_numpy()
    base = y.mean()
    print(f"labelled {len(y)}, threatened {y.sum()}, base rate {base:.3f}, families {len(set(groups))}")

    sets = {"range_only": RANGE_ONLY, "records": RECORD_FEATURES,
            "records_plus_climate": RECORD_FEATURES + MODEL_FEATURES}
    rows = []
    for name, cols in sets.items():
        X = labelled[cols].to_numpy(dtype=float)
        for model_name, make in (("logistic", logistic), ("forest", forest)):
            if name == "range_only" and model_name == "forest":
                continue
            rows.append({"features": name, "model": model_name, **scores(make, X, y, groups)})
    board = pd.DataFrame(rows)
    board["lift"] = board["pr_auc"] / base
    board.to_csv(OUTPUTS / "triage_validation.csv", index=False)
    print(board.round(3).to_string())

    baseline = board.loc[board["features"] == "range_only", "pr_auc"].iloc[0]
    best = board[board["features"] != "range_only"].sort_values("pr_auc", ascending=False).iloc[0]
    passed = best["pr_auc"] >= GATE_LIFT * base and best["pr_auc"] > baseline
    print(f"\nbest: {best['features']} / {best['model']}, PR-AUC {best['pr_auc']:.3f}; "
          f"gate needs >= {GATE_LIFT * base:.3f} and > baseline {baseline:.3f}: "
          f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        print("Gate failed: no species-level predictions written.")
        (OUTPUTS / "triage_predictions.csv").unlink(missing_ok=True)
        return

    make = logistic if best["model"] == "logistic" else forest
    cols = sets[best["features"]]
    X = labelled[cols].to_numpy(dtype=float)
    p, fold_id = family_cv(make, X, y, groups, SEED)
    check = conformal_check(p, y, fold_id)
    print("conformal:", {k: round(float(v), 3) for k, v in check.items()})
    pd.DataFrame([check]).to_csv(OUTPUTS / "triage_conformal.csv", index=False)

    unlabelled = table[table["iucn"] == "NE"].reset_index(drop=True)
    model = make().fit(X, y)
    unlabelled["p_threatened"] = model.predict_proba(unlabelled[cols].to_numpy(dtype=float))[:, 1]
    threat, safe = conformal_p(unlabelled["p_threatened"].to_numpy(), p, y)
    unlabelled["triage"] = label_sets(threat, safe)
    unlabelled["p_conformal_threatened"] = threat
    keep = ["species", "family", "priority", "log_cells", "confidence", "p_threatened",
            "p_conformal_threatened", "triage", "iucn_review"]
    ranked = unlabelled.sort_values("p_threatened", ascending=False)[keep]
    ranked.to_csv(OUTPUTS / "triage_predictions.csv", index=False)
    print("\n" + ranked["triage"].value_counts().to_string())
    print(ranked.head(15).round(3).to_string())
    print(ranked[ranked["priority"] == True].head(12).round(3).to_string())


if __name__ == "__main__":
    main()
