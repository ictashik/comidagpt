# Data manifest

Corpus sources, licences, sha-pinned files (CLAUDE.md §6). Empty until M1/M2.

| source | licence | files | sha256 | kept fields | notes |
|---|---|---|---|---|---|
| USDA FDC SR Legacy (2018-04 CSV) | public domain | `FoodData_Central_sr_legacy_food_csv_2018-04.zip` (6.07 MB) | zip `b80817294b8850530aaedf2e515c02593b1824f763a0ff356e5c2081643e6fd0`; `food_nutrient.csv` `6d07e6fc062b09750c4d95112be5a5cdb56a2346efcfd0c96b7a7d705812d4bd` | energy (kcal, id 1008), protein, fat, carbohydrate, fibre per 100 g; food category; household portions | from https://fdc.nal.usda.gov/fdc-datasets/ ; extracted under `data/tables/raw/` (git-ignored) |
