"""recipe -> per-serving nutrients, given coded ingredients. Ground truth for nutrition_calc (CLAUDE.md §5.1)."""
from comida.nutrition.units import NUTRIENTS


def compute_recipe(table, items, servings):
    """items: list of (code_or_name, grams). Returns dict(total, per_serving, items, missing)."""
    if servings <= 0:
        raise ValueError("servings must be > 0")
    rows = [table.lookup(code, grams) for code, grams in items]
    total = {n: sum(r[n] for r in rows) for n in NUTRIENTS}
    per_serving = {n: v / servings for n, v in total.items()}
    missing = sorted({(r["code"], n) for r in rows for n in r["missing"]})
    return {"total": total, "per_serving": per_serving, "items": rows, "missing": missing}
