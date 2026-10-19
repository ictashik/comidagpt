"""Energy units. The only place kJ <-> kcal conversion happens (CLAUDE.md §1): ADS and IFCT 2017 report kJ."""

KJ_PER_KCAL = 4.184
NUTRIENTS = ("energy_kcal", "protein_g", "fat_g", "carb_g", "fiber_g")


def kj_to_kcal(kj):
    return kj / KJ_PER_KCAL


def kcal_to_kj(kcal):
    return kcal * KJ_PER_KCAL


def to_kcal(value, unit):
    if unit == "kcal":
        return value
    if unit == "kJ":
        return kj_to_kcal(value)
    raise ValueError(f"unknown energy unit: {unit!r}")
