"""
Tests for comidaGPT tasks (tasks/food_*.py) and their generators (comida/data/).
Run: python -m pytest tests/test_tasks_comida.py -v
"""
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from comida.data.food_mc import LETTERS, make_questions, split_of
from comida.nutrition.tables import load_usda_sr_legacy
from tasks.common import TaskMixture
from tasks.food_mc import FoodMC


def synthetic_foods(n=400):
    rows = []
    for i in range(n):
        rows.append({
            "code": f"F{i}", "name": f"Food {i}", "group": f"Group {i % 7}", "aliases": [],
            "energy_kcal": float((i * 37) % 900), "protein_g": float((i * 13) % 80), "fat_g": float((i * 7) % 100),
            "carb_g": float((i * 11) % 90), "fiber_g": float((i * 3) % 30),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def mc_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("mc")
    foods = synthetic_foods()
    paths = {}
    for split in ("train", "test"):
        paths[split] = str(d / f"{split}.parquet")
        pq.write_table(pa.Table.from_pylist(make_questions(foods, split, 30, seed=1)), paths[split])
    return paths


def test_generator_is_deterministic():
    foods = synthetic_foods()
    assert make_questions(foods, "test", 20, seed=3) == make_questions(foods, "test", 20, seed=3)
    assert make_questions(foods, "test", 20, seed=3) != make_questions(foods, "test", 20, seed=4)


def test_split_has_no_food_overlap_and_is_stable():
    foods = synthetic_foods()
    tr = {c for q in make_questions(foods, "train", 30, seed=0) for c in q["codes"]}
    te = {c for q in make_questions(foods, "test", 30, seed=0) for c in q["codes"]}
    assert tr and te and not (tr & te)
    assert all(split_of(c) == "test" for c in te) and all(split_of(c) == "train" for c in tr)


def test_answers_agree_with_the_table():
    foods = synthetic_foods().set_index("code")
    for q in make_questions(synthetic_foods(), "train", 40, seed=0):
        picked = q["choices"][LETTERS.index(q["answer"])]
        if q["kind"] == "most":
            vals = {foods.at[c, "name"]: foods.at[c, q["nutrient"]] for c in q["codes"]}
            assert picked == max(vals, key=vals.get)
            ranked = sorted(vals.values(), reverse=True)
            assert ranked[0] >= 1.25 * ranked[1]  # unambiguous winner
        else:
            assert picked == foods.at[q["codes"][0], "group"]
        assert len(set(q["choices"])) == 4


def test_task_example_format_and_slicing(mc_paths):
    t = FoodMC("test", path=mc_paths["test"])
    ex = t[0]
    assert t.eval_type == "categorical" and ex["letters"] == LETTERS
    assert ex["messages"][0]["content"].startswith("Multiple Choice question:")
    assert ex["messages"][0]["content"].endswith("Respond only with the letter of the correct answer.")
    assert ex["messages"][1]["content"] in LETTERS
    sliced = FoodMC("test", path=mc_paths["test"], start=2, stop=10, step=2)
    assert len(sliced) == 4 and sliced[0] == t[2]


def test_evaluate_correct_wrong_malformed(mc_paths):
    t = FoodMC("train", path=mc_paths["train"])
    ex = t[0]
    right = ex["messages"][1]["content"]
    wrong = next(l for l in LETTERS if l != right)
    assert t.evaluate(ex, right) is True
    assert t.evaluate(ex, wrong) is False
    with pytest.raises(AssertionError):
        t.evaluate(ex, "The answer is A")  # malformed: not a bare letter


def test_mixture_over_tasks(mc_paths):
    mix = TaskMixture([FoodMC("train", path=mc_paths["train"]), FoodMC("test", path=mc_paths["test"])])
    assert len(mix) == len(FoodMC("train", path=mc_paths["train"])) + len(FoodMC("test", path=mc_paths["test"]))
    assert mix[0]["messages"][0]["role"] == "user"


SR_DIR = "data/tables/raw/usda_sr_legacy_2018-04/FoodData_Central_sr_legacy_food_csv_2018-04"


@pytest.mark.skipif(not os.path.isdir(SR_DIR), reason="USDA SR Legacy export not downloaded")
def test_golden_file_matches_regeneration():
    foods, _ = load_usda_sr_legacy(SR_DIR)
    got = make_questions(foods, "test", 200, seed=0)[:20]
    with open("tests/golden/food_mc.jsonl") as f:
        want = [json.loads(line) for line in f]
    assert len(want) == 20
    assert [json.loads(json.dumps(q, sort_keys=True)) for q in got] == want
