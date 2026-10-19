"""Generate food_mc questions from a nutrient table (CLAUDE.md §4). Pure functions, deterministic given (table, seed).

Split is by food: a food belongs to train or test by hash of its code, and every food in a question comes from the same
split, so no test question shares a food with any train question.
"""
import hashlib
import random

LETTERS = ["A", "B", "C", "D"]
NUTRIENT_WORDS = {
    "energy_kcal": ("energy", "kcal", 20.0),
    "protein_g": ("protein", "g", 2.0),
    "fat_g": ("fat", "g", 2.0),
    "carb_g": ("carbohydrate", "g", 5.0),
    "fiber_g": ("dietary fibre", "g", 1.0),
}


def split_of(code, test_fraction=0.2):
    h = int(hashlib.sha1(str(code).encode()).hexdigest(), 16) % 1000
    return "test" if h < test_fraction * 1000 else "train"


def _pack(rng, kind, question, correct, wrong, nutrient, codes):
    choices = [correct, *wrong]
    rng.shuffle(choices)
    return {
        "question": question, "choices": choices, "answer": LETTERS[choices.index(correct)],
        "kind": kind, "nutrient": nutrient, "codes": codes,
    }


def make_questions(df, split, n_per_kind, seed, test_fraction=0.2):
    """df: normalised table (comida.nutrition.tables). Returns a list of question dicts."""
    rng = random.Random(f"{seed}/{split}")
    pool = df[df["code"].map(lambda c: split_of(c, test_fraction)) == split].reset_index(drop=True)
    out = []
    for nutrient, (word, unit, min_gap) in NUTRIENT_WORDS.items():
        have = pool[pool[nutrient].notna()]
        tries = made = 0
        while made < n_per_kind and tries < 50 * n_per_kind:
            tries += 1
            rows = have.iloc[rng.sample(range(len(have)), 4)]
            vals = sorted(rows[nutrient], reverse=True)
            # unambiguous winner: clearly above the runner-up, in relative and absolute terms
            if vals[0] - vals[1] < min_gap or vals[0] < 1.25 * vals[1] or len(set(rows["name"])) < 4:
                continue
            top = rows.loc[rows[nutrient].idxmax()]
            names = list(rows["name"])
            q = f"Which of these foods has the most {word} per 100 g?"
            out.append(_pack(rng, "most", q, top["name"], [n for n in names if n != top["name"]], nutrient, list(rows["code"])))
            made += 1
    groups = sorted(pool["group"].dropna().unique())
    named = pool[pool["group"].notna()]
    for _ in range(n_per_kind):
        row = named.iloc[rng.randrange(len(named))]
        others = rng.sample([g for g in groups if g != row["group"]], 3)
        q = f"Which food group does '{row['name']}' belong to?"
        out.append(_pack(rng, "group", q, row["group"], others, "group", [row["code"]]))
    rng.shuffle(out)
    return out
