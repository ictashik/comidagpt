"""The tool the model calls: search(name, k) -> shortlist, lookup(name_or_code, grams) -> per-portion nutrients.

Deterministic: exact (normalised) match on code, name or alias first, else the top BM25 hit (ties broken by row order).
"""
import re

import pandas as pd
from rank_bm25 import BM25Okapi

from comida.nutrition.units import NUTRIENTS


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _norm(text):
    return " ".join(_tokens(text))


class FoodTable:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self._exact = {}
        docs = []
        for i, row in self.df.iterrows():
            names = [row["name"], *row["aliases"]]
            for key in (str(row["code"]), *names):
                self._exact.setdefault(_norm(key), i)
            docs.append(_tokens(" ".join(names)))
        self._bm25 = BM25Okapi(docs)

    def _top(self, query, k):
        scores = self._bm25.get_scores(_tokens(query))
        return [(i, float(scores[i])) for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]]

    def search(self, query, k=5):
        return [{"code": self.df.at[i, "code"], "name": self.df.at[i, "name"], "score": s} for i, s in self._top(query, k)]

    def resolve(self, name_or_code):
        i = self._exact.get(_norm(name_or_code))
        if i is not None:
            return i
        i, score = self._top(name_or_code, 1)[0]
        if score <= 0:
            raise KeyError(f"no food matches {name_or_code!r}")
        return i

    def lookup(self, name_or_code, grams):
        if grams < 0:
            raise ValueError("grams must be >= 0")
        row = self.df.iloc[self.resolve(name_or_code)]
        out = {"code": row["code"], "name": row["name"], "grams": float(grams), "missing": []}
        for n in NUTRIENTS:
            v = row[n]
            if pd.isna(v):
                out[n] = 0.0
                out["missing"].append(n)
            else:
                out[n] = float(v) * grams / 100.0
        return out
