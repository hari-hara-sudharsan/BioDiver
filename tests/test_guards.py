"""Regression tests for bugs found during the build.

Run with: python -m pytest tests  (or python tests/test_guards.py)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("0", "m", 1), ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synonym_rule_blocks_merged_taxa():
    redlist = load("01c_redlist")
    assert not redlist.same_epithet("Withania chevalieri", "Withania somnifera")
    assert redlist.same_epithet("Saussurea costus", "Dolomiaea costus")
    assert redlist.same_epithet("Gymnema sylvestris", "Gymnema sylvestre")


def test_known_statuses():
    table = pd.read_csv(ROOT / "data/processed/species_candidates.csv").set_index("species")
    expected = {"Withania somnifera": "NE", "Aconitum heterophyllum": "EN", "Bacopa monnieri": "LC",
                "Dolomiaea costus": "CR", "Nardostachys jatamansi": "CR", "Rauvolfia serpentina": "NT"}
    for name, status in expected.items():
        assert table.loc[name, "iucn"] == status, name


def test_tss_threshold():
    import sdm
    y = np.array([1, 1, 1, 0, 0, 0])
    score = np.array([.9, .8, .7, .3, .2, .1])
    threshold, tss = sdm.best_tss(y, score)
    assert tss == 1.0 and threshold == 0.7


def test_bioclim_centre_beats_edge():
    import sdm
    train = np.arange(100, dtype=float).reshape(-1, 1)
    score = sdm.bioclim_score(train, np.array([[50.0], [1.0], [500.0]]))
    assert score[0] > score[1] > score[2]


def test_public_data_withholds_sensitive_species():
    path = ROOT / "docs/data/app.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    sensitive = {s["species"] for s in data["species"] if s["sensitive"]}
    assert not sensitive & set(data["guide_species"])
    text = path.read_text(encoding="utf-8")
    assert '"lat"' not in text and '"decimalLatitude"' not in text
    assert all(b[0] * 2 == int(b[0] * 2) and b[1] * 2 == int(b[1] * 2) for b in data["blocks"])


def test_guide_keys_match_javascript_numbers():
    data = json.loads((ROOT / "docs/data/app.json").read_text(encoding="utf-8"))
    as_js = lambda v: str(int(v)) if float(v).is_integer() else str(v)
    block_keys = {f"{as_js(b[0])},{as_js(b[1])}" for b in data["blocks"]}
    assert set(data["guide"]) <= block_keys


def test_layers_cannot_reveal_threatened_species():
    data = json.loads((ROOT / "docs/data/app.json").read_text(encoding="utf-8"))
    if data["summary"].get("preview"):
        as_js = lambda v: str(int(v)) if float(v).is_integer() else str(v)
        for b in data["blocks"]:
            assert b[2] == len(data["guide"].get(f"{as_js(b[0])},{as_js(b[1])}", []))


def test_formula_sources():
    formulas = pd.read_csv(ROOT / "data/curated/formulas.csv")
    verified = formulas[formulas["status"].str.startswith("verified")]
    assert verified["source"].notna().all()
    assert formulas[formulas["status"] == "needs_source"]["source"].isna().all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
