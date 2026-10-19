"""
Tests for comida/nutrition: units, lookup/search, recipe engine.
Run: python -m pytest tests/test_nutrition.py -v
Real-data checks run only if the USDA SR Legacy export is under data/tables/raw (git-ignored).
"""
import os
import random

import pandas as pd
import pytest

from comida.nutrition import units
from comida.nutrition.engine import compute_recipe
from comida.nutrition.lookup import FoodTable
from comida.nutrition.tables import energy_to_kcal, load_usda_sr_legacy

NAN = float("nan")


def tiny_table():
    # hand-computed fixture: per 100 g values chosen so every product below is exact in decimal
    return FoodTable(pd.DataFrame([
        {"code": "A1", "name": "Toor dal, dry", "aliases": ["arhar dal", "pigeon pea"], "energy_kcal": 335.0, "protein_g": 22.0, "fat_g": 1.5, "carb_g": 57.0, "fiber_g": 15.0},
        {"code": "B2", "name": "Rice, raw, milled", "aliases": ["chawal"], "energy_kcal": 350.0, "protein_g": 7.0, "fat_g": 0.5, "carb_g": 78.0, "fiber_g": NAN},
        {"code": "C3", "name": "Oil, sunflower", "aliases": [], "energy_kcal": 884.0, "protein_g": 0.0, "fat_g": 100.0, "carb_g": 0.0, "fiber_g": 0.0},
    ]))


# ---- units ----

def test_kj_kcal_roundtrip_and_constant():
    assert units.kj_to_kcal(4.184) == pytest.approx(1.0)
    assert units.kcal_to_kj(100) == pytest.approx(418.4)
    assert units.kj_to_kcal(units.kcal_to_kj(123.4)) == pytest.approx(123.4)


def test_to_kcal_units():
    assert units.to_kcal(250, "kcal") == 250
    assert units.to_kcal(1046, "kJ") == pytest.approx(250.0)
    with pytest.raises(ValueError):
        units.to_kcal(1, "cal")


def test_energy_to_kcal_marks_basis():
    df = energy_to_kcal(pd.DataFrame({"kj": [418.4, 0.0]}), "kj", "kJ")
    assert list(df["energy_kcal"]) == pytest.approx([100.0, 0.0])
    assert set(df["energy_basis"]) == {"kJ"}


# ---- lookup / search ----

def test_lookup_scales_by_grams_hand_computed():
    r = tiny_table().lookup("A1", 60)  # 60 g toor dal
    assert r["energy_kcal"] == pytest.approx(201.0)  # 335 * 0.6
    assert r["protein_g"] == pytest.approx(13.2)
    assert r["fat_g"] == pytest.approx(0.9)
    assert r["carb_g"] == pytest.approx(34.2)
    assert r["fiber_g"] == pytest.approx(9.0)
    assert r["missing"] == []


def test_lookup_resolves_name_alias_and_code_identically():
    t = tiny_table()
    assert t.lookup("Toor dal, dry", 100)["code"] == t.lookup("arhar dal", 100)["code"] == t.lookup("a1", 100)["code"] == "A1"


def test_lookup_fuzzy_falls_back_to_bm25_top_hit():
    assert tiny_table().lookup("sunflower oil", 10)["code"] == "C3"


def test_lookup_missing_value_is_zero_and_reported():
    r = tiny_table().lookup("B2", 200)
    assert r["fiber_g"] == 0.0 and r["missing"] == ["fiber_g"]
    assert r["energy_kcal"] == pytest.approx(700.0)


def test_lookup_errors():
    t = tiny_table()
    with pytest.raises(KeyError):
        t.lookup("xyzzy", 10)
    with pytest.raises(ValueError):
        t.lookup("A1", -1)


def test_search_ranks_and_is_deterministic():
    t = tiny_table()
    hits = t.search("dal", 3)
    assert hits[0]["code"] == "A1" and len(hits) == 3
    assert t.search("dal", 3) == hits


# ---- engine ----

def test_recipe_hand_computed_per_serving():
    # 60 g toor dal + 100 g rice + 10 g oil, 2 servings
    out = compute_recipe(tiny_table(), [("A1", 60), ("B2", 100), ("C3", 10)], servings=2)
    assert out["total"]["energy_kcal"] == pytest.approx(201.0 + 350.0 + 88.4)
    assert out["per_serving"]["energy_kcal"] == pytest.approx(639.4 / 2)
    assert out["per_serving"]["protein_g"] == pytest.approx((13.2 + 7.0 + 0.0) / 2)
    assert out["per_serving"]["fat_g"] == pytest.approx((0.9 + 0.5 + 10.0) / 2)
    assert out["missing"] == [("B2", "fiber_g")]


def test_recipe_property_totals_are_sums_and_nonnegative():
    rng = random.Random(0)
    t = tiny_table()
    for _ in range(200):
        items = [(rng.choice(["A1", "B2", "C3"]), rng.uniform(0, 500)) for _ in range(rng.randint(1, 8))]
        servings = rng.randint(1, 10)
        out = compute_recipe(t, items, servings)
        for n in units.NUTRIENTS:
            assert out["total"][n] == pytest.approx(sum(r[n] for r in out["items"]))
            assert out["per_serving"][n] == pytest.approx(out["total"][n] / servings)
            assert out["total"][n] >= 0


def test_recipe_rejects_bad_servings():
    with pytest.raises(ValueError):
        compute_recipe(tiny_table(), [("A1", 10)], servings=0)


# ---- real data (skipped when the export is absent) ----

SR_DIR = "data/tables/raw/usda_sr_legacy_2018-04/FoodData_Central_sr_legacy_food_csv_2018-04"


@pytest.mark.skipif(not os.path.isdir(SR_DIR), reason="USDA SR Legacy export not downloaded")
class TestUsdaSrLegacy:
    @pytest.fixture(scope="class")
    def table(self):
        foods, _ = load_usda_sr_legacy(SR_DIR)
        return FoodTable(foods)

    def test_shape_and_coverage(self, table):
        assert len(table.df) == 7793
        assert table.df["energy_kcal"].notna().all()
        assert (table.df[list(units.NUTRIENTS)].min() >= 0).all()

    def test_kcal_consistent_with_kj_in_source(self):
        # source carries both energy rows; kcal must agree with kJ / 4.184 to within rounding
        import csv
        nid = {"1008": "kcal", "1062": "kj"}
        vals = {}
        with open(f"{SR_DIR}/food_nutrient.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["nutrient_id"] in nid:
                    vals.setdefault(r["fdc_id"], {})[nid[r["nutrient_id"]]] = float(r["amount"])
        bad = [k for k, v in vals.items() if abs(units.kj_to_kcal(v["kj"]) - v["kcal"]) > 1.0 + 0.01 * v["kcal"]]
        assert len(bad) / len(vals) < 0.01

    def test_known_food_search(self, table):
        names = [h["name"].lower() for h in table.search("lentils raw", 5)]
        assert any(n.startswith("lentils, raw") for n in names)
