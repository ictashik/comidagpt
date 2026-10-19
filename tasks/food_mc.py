"""
food_mc: multiple-choice food knowledge generated from nutrient tables (CLAUDE.md §4).
Data file is produced by scripts/comida_build_data.py (parquet, never hand-edited).

Example:
Multiple Choice question: Which of these foods has the most protein per 100 g?
- Apples, raw=A
- Lentils, raw=B
...
Respond only with the letter of the correct answer.
"""
import os

import pyarrow.parquet as pq

from tasks.common import Task, render_mc

LETTERS = ["A", "B", "C", "D"]
DATA_DIR = os.environ.get("COMIDA_TASK_DIR", "data/tasks")


class FoodMC(Task):

    def __init__(self, split, path=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "FoodMC split must be train|test"
        self.rows = pq.read_table(path or os.path.join(DATA_DIR, f"food_mc_{split}.parquet")).to_pylist()

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        prompt = render_mc(row["question"], LETTERS, row["choices"])
        return {
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": row["answer"]}],
            "letters": LETTERS,
        }

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation['letters'], f"answer {assistant_response} must be one of {conversation['letters']}"
        return assistant_response == conversation['messages'][-1]['content']
