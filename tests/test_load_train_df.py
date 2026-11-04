import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import load_train_df


class LoadTrainDfTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = self._write("a.csv", pd.DataFrame({"id": [1, 2], "f": [10, 20]}))
        self.b = self._write("b.csv", pd.DataFrame({"id": [3], "f": [30]}))
        self.t = self._write("t.csv", pd.DataFrame({"id": [1, 2, 3], "y": [0, 1, 0]}))

    def _write(self, name, df):
        path = os.path.join(self.tmp.name, name)
        df.to_csv(path, index=False)
        return path

    def test_stacks_files_with_same_columns(self):
        df = load_train_df({"train_paths": [self.a, self.b]})
        self.assertEqual(len(df), 3)

    def test_joins_files_with_join_keys(self):
        df = load_train_df({"train_paths": [self.a, self.t], "join_keys": ["id"]})
        self.assertEqual(len(df), 2)
        self.assertIn("y", df.columns)

    def test_rejects_mismatched_columns_without_keys(self):
        with self.assertRaises(ValueError):
            load_train_df({"train_paths": [self.a, self.t]})


if __name__ == "__main__":
    unittest.main()
