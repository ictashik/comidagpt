"""Load source tables into one normalised frame (CLAUDE.md §5.1).

Columns: code, name, aliases, group, energy_kcal, protein_g, fat_g, carb_g, fiber_g, source, energy_basis.
Nutrient values are per 100 g edible portion; NaN means the source has no value.
`group14` (IFCT-derived taxonomy) is not filled yet: it needs food_groups_v1.csv (ASSUMPTION A3).
"""
import csv
import os

import pandas as pd

from comida.nutrition.units import to_kcal

USDA_NUTRIENT_IDS = {"1008": "energy_kcal", "1003": "protein_g", "1004": "fat_g", "1005": "carb_g", "1079": "fiber_g"}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def load_usda_sr_legacy(raw_dir):
    """USDA FDC SR Legacy CSV export (public domain). Returns (foods, portions) DataFrames."""
    categories = {r["id"]: r["description"] for r in _rows(os.path.join(raw_dir, "food_category.csv"))}
    foods = {}
    for r in _rows(os.path.join(raw_dir, "food.csv")):
        foods[r["fdc_id"]] = {
            "code": r["fdc_id"], "name": r["description"], "aliases": [], "group": categories.get(r["food_category_id"]),
            "source": "usda_sr_legacy", "energy_basis": "kcal",
        }
    for r in _rows(os.path.join(raw_dir, "food_nutrient.csv")):
        col = USDA_NUTRIENT_IDS.get(r["nutrient_id"])
        if col is not None and r["fdc_id"] in foods:
            foods[r["fdc_id"]][col] = float(r["amount"])
    df = pd.DataFrame(foods.values())
    for col in USDA_NUTRIENT_IDS.values():
        if col not in df:
            df[col] = float("nan")
    portions = pd.DataFrame(
        [{"code": r["fdc_id"], "description": (r["portion_description"] or r["modifier"]).strip(), "grams": float(r["gram_weight"])}
         for r in _rows(os.path.join(raw_dir, "food_portion.csv")) if r["gram_weight"]]
    )
    return df, portions


def energy_to_kcal(df, value_col, unit):
    """For tables that report kJ (IFCT, ADS): add energy_kcal via units.to_kcal and mark the basis."""
    df = df.copy()
    df["energy_kcal"] = df[value_col].map(lambda v: to_kcal(v, unit))
    df["energy_basis"] = unit
    return df
