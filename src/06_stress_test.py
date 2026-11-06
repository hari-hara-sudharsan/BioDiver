"""Traditional medicine stress test, chemistry at risk and chemical relatives.

All outputs are research hypotheses. A chemical relative is a lead for laboratory
and ethnobotanical study, never a clinical substitute: shared compounds do not
imply equal dose, safety, efficacy or traditional role.

Compound records mean "reported in the literature". Absence is not evidence of
absence, so exclusivity results are compared with a permutation null.

Usage: python src/06_stress_test.py [--results sdm_results.csv]
"""
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from config import OUTPUTS, PROCESSED, ROOT

RDLogger.DisableLog("rdApp.*")
SSPS = ["ssp245", "ssp585"]
THREATENED = {"CR", "EN", "VU", "EW"}
HIGH_LOSS = 0.30
LOW_LOSS = 0.10
MIN_COMPOUNDS = 5
TOP_RELATIVES = 5
MIN_COVERAGE = 0.5              # below this a species gets "no credible relative"
PERMUTATIONS = 1000
AGREE = 4                       # at least 4 of 5 climate models
SEED = 42


def load(results_name):
    res = pd.read_csv(OUTPUTS / results_name)
    species = pd.read_csv(PROCESSED / "species_candidates.csv")
    extra = [c for c in species.columns if c not in res.columns or c == "speciesKey"]
    res = res.merge(species[extra], on="speciesKey", how="left")
    return res, species


def species_map(key):
    path = PROCESSED / "species_maps" / f"{key}.npz"
    return np.load(path) if path.exists() else None


def footprint(keys, ssp):
    """Cells where every ingredient is suitable now, and retained by >= AGREE models."""
    now, kept = None, None
    for key in keys:
        m = species_map(key)
        if m is None or f"agree_{ssp}" not in m:
            return None, None
        cells = set(m["cells"].tolist())
        stay = set(m["cells"][m[f"agree_{ssp}"] >= AGREE].tolist())
        now = cells if now is None else now & cells
        kept = stay if kept is None else kept & stay
    return len(now), len(kept)


def formula_stress(res, species):
    formulas = pd.read_csv(ROOT / "data/curated/formulas.csv")
    table = formulas.merge(species[["species", "speciesKey", "iucn"]], on="species", how="left")
    table = table.merge(res[["speciesKey", "status", "confidence", "cells", "auc_ens",
                             "loss_ssp245_median", "loss_ssp585_median", "loss_ssp585_min",
                             "loss_ssp585_max", "novel_ssp585_median"]],
                        on="speciesKey", how="left")
    table["grade"] = table["confidence"].fillna("data_gap")
    table.to_csv(OUTPUTS / "formula_ingredients.csv", index=False)

    rows = []
    for name, group in table.groupby("formula"):
        usable = group[group["grade"].isin(["standard", "low"])]
        row = {"formula": name, "ingredients": len(group),
               "fully_sourced": bool((group["status_x"] if "status_x" in group else group["status"]).eq("verified").all()),
               "usable_models": len(usable),
               "standard_models": int((group["grade"] == "standard").sum()),
               "data_gaps": int((group["grade"] == "data_gap").sum()),
               "no_skill": int((group["grade"] == "no_skill").sum()),
               "threatened": int(group["iucn"].isin(THREATENED).sum()),
               "no_checklist_record": int((group["iucn"] == "NE").sum())}
        if len(usable):
            worst = usable.sort_values("loss_ssp585_median", ascending=False).iloc[0]
            row.update(weakest_link=worst["species"],
                       weakest_loss_ssp585=worst["loss_ssp585_median"],
                       weakest_grade=worst["grade"])
        complete = len(usable) == len(group)
        row["footprint_complete"] = complete
        if complete:
            for ssp in SSPS:
                now, kept = footprint(group["speciesKey"].astype(int).tolist(), ssp)
                if now:
                    row["footprint_cells_now"] = now
                    row[f"footprint_retained_{ssp}"] = kept / now
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUTS / "formula_stress.csv", index=False)
    return table, out


def vulnerable_keys(res, species):
    """Threatened on the Red List, or high projected loss with a standard-grade model."""
    threatened = set(species.loc[species["iucn"].isin(THREATENED), "speciesKey"])
    loss = res[(res["confidence"] == "standard") & (res["loss_ssp585_median"] >= HIGH_LOSS)]
    return threatened | set(loss["speciesKey"])


def chemistry_at_risk(links, vulnerable):
    """Compounds reported only from vulnerable species, against a permutation null."""
    links = links.dropna(subset=["skeleton"]).drop_duplicates(["speciesKey", "skeleton"])
    groups = links.groupby("skeleton")["speciesKey"].agg(list)
    keys = np.array(sorted(links["speciesKey"].unique()))
    index = {k: i for i, k in enumerate(keys)}
    members = [np.array([index[k] for k in g]) for g in groups]

    def exclusive(mask):
        return sum(bool(mask[m].all()) for m in members)

    mask = np.isin(keys, list(vulnerable))
    observed = exclusive(mask)

    # Labels are shuffled only among species with similar compound counts, so the
    # null holds research effort roughly constant. A plain shuffle does not: most
    # exclusive compounds are single-species reports from well-studied plants.
    per_species = links.groupby("speciesKey").size().reindex(keys).to_numpy()
    bins = pd.qcut(per_species, 10, labels=False, duplicates="drop")
    strata = [np.flatnonzero(bins == b) for b in np.unique(bins)]
    rng = np.random.default_rng(SEED)

    def shuffled():
        out = mask.copy()
        for idx in strata:
            out[idx] = rng.permutation(mask[idx])
        return out

    null = np.array([exclusive(shuffled()) for _ in range(PERMUTATIONS)])
    p = (1 + (null >= observed).sum()) / (1 + PERMUTATIONS)

    multi = [m for m in members if len(m) > 1]
    summary = {"species_with_compounds": len(keys), "vulnerable_with_compounds": int(mask.sum()),
               "compounds": len(members), "exclusive_to_vulnerable": observed,
               "exclusive_share": observed / len(members),
               "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
               "p_value": float(p), "single_species_compounds": len(members) - len(multi),
               "exclusive_reported_by_two_or_more": sum(bool(mask[m].all()) for m in multi)}
    pd.DataFrame([summary]).to_csv(OUTPUTS / "chemistry_at_risk.csv", index=False)
    return summary


def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out, kept = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is not None:
            out.append(gen.GetFingerprintAsNumPy(mol))
            kept.append(i)
    return np.array(out, dtype="float32"), kept


def relatives(res, species, links, vulnerable):
    """Rank lower-risk species by how well they cover a vulnerable species' chemistry."""
    usable = links.dropna(subset=["smiles"]).drop_duplicates(["speciesKey", "skeleton"])
    counts = usable.groupby("speciesKey").size()
    rich = set(counts[counts >= MIN_COMPOUNDS].index)

    safe = res[(res["confidence"] == "standard") & (res["loss_ssp585_median"] <= LOW_LOSS)
               & ~res["iucn"].isin(THREATENED) & res["speciesKey"].isin(rich)]
    focus = sorted(k for k in vulnerable if k in rich)
    info = species.set_index("speciesKey")
    keys = sorted(set(focus) | set(safe["speciesKey"]))

    subset = usable[usable["speciesKey"].isin(keys)].reset_index(drop=True)
    fp, kept = fingerprints(subset["smiles"].tolist())
    subset = subset.iloc[kept].reset_index(drop=True)
    bits = fp.sum(axis=1)
    rows_of = {k: np.flatnonzero(subset["speciesKey"].to_numpy() == k) for k in keys}

    rows = []
    for v in focus:
        a = rows_of[v]
        for r in safe["speciesKey"]:
            if r == v or len(rows_of.get(r, [])) == 0:
                continue
            b = rows_of[r]
            inter = fp[a] @ fp[b].T
            sim = inter / (bits[a][:, None] + bits[b][None, :] - inter)
            best = sim.max(axis=1)
            rows.append({"vulnerable": info.loc[v, "species"], "relative": info.loc[r, "species"],
                         "coverage": float(best.mean()),
                         "share_close_match": float((best >= 0.7).mean()),
                         "same_family": info.loc[v, "family"] == info.loc[r, "family"],
                         "compounds_vulnerable": len(a), "compounds_relative": len(b)})
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs, None
    credible = pairs[pairs["coverage"] >= MIN_COVERAGE]
    top = (credible.sort_values("coverage", ascending=False)
           .groupby("vulnerable").head(TOP_RELATIVES))
    top.to_csv(OUTPUTS / "chemical_relatives.csv", index=False)

    # Sanity check: chemistry should recover taxonomy better than chance.
    check = {"pairs": len(pairs), "base_rate_same_family": float(pairs["same_family"].mean()),
             "top_rate_same_family": float(top["same_family"].mean()) if len(top) else None,
             "species_with_credible_relative": int(top["vulnerable"].nunique()),
             "median_coverage_same_family": float(pairs.loc[pairs.same_family, "coverage"].median()),
             "median_coverage_other_family": float(pairs.loc[~pairs.same_family, "coverage"].median()),
             "focus_species": len(focus), "candidate_relatives": len(safe)}
    return top, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="sdm_results.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_columns", 30)

    res, species = load(args.results)
    table, stress = formula_stress(res, species)
    print(stress.round(3).T.to_string())

    links = pd.read_parquet(PROCESSED / "compounds.parquet")
    links["skeleton"] = links["inchikey"].str[:14]
    print(f"\ncompound links without InChIKey: {links.skeleton.isna().mean():.1%}")

    vulnerable = vulnerable_keys(res, species)
    print(f"vulnerable species: {len(vulnerable)}")
    for key, value in chemistry_at_risk(links, vulnerable).items():
        print(f"  {key}: {value}")

    top, check = relatives(res, species, links, vulnerable)
    print("\nrelatives check:", check)
    if len(top):
        print(top.sort_values(["vulnerable", "coverage"], ascending=[True, False]).head(15).round(3).to_string())


if __name__ == "__main__":
    main()
