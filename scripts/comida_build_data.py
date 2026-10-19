"""
Build ComidaBench task files from sources listed in docs/data.md. Nothing under data/tasks is hand-edited.

Run: python -m scripts.comida_build_data --usda-dir data/tables/raw/usda_sr_legacy_2018-04/FoodData_Central_sr_legacy_food_csv_2018-04
Writes data/tasks/food_mc_{train,test}.parquet and data/tasks/manifest.json (git sha, seed, source sha, file shas).
"""
import argparse
import hashlib
import json
import os
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq

from comida.data.food_mc import make_questions
from comida.nutrition.tables import load_usda_sr_legacy


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--usda-dir", required=True)
parser.add_argument("--out-dir", default="data/tasks")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n-train", type=int, default=2000, help="questions per kind in train")
parser.add_argument("--n-test", type=int, default=200, help="questions per kind in test")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
foods, _ = load_usda_sr_legacy(args.usda_dir)
manifest = {
    "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
    "seed": args.seed,
    "sources": {"usda_sr_legacy_food_nutrient.csv": sha256(os.path.join(args.usda_dir, "food_nutrient.csv")),
                "usda_sr_legacy_food.csv": sha256(os.path.join(args.usda_dir, "food.csv"))},
    "files": {},
}
for split, n in (("train", args.n_train), ("test", args.n_test)):
    qs = make_questions(foods, split, n, args.seed)
    path = os.path.join(args.out_dir, f"food_mc_{split}.parquet")
    pq.write_table(pa.Table.from_pylist(qs), path)
    manifest["files"][os.path.basename(path)] = {"rows": len(qs), "sha256": sha256(path)}
    print(f"{path}: {len(qs)} questions")
with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
