import unittest
import pandas as pd
from spacemerlin.scripts import run


class RunTest(unittest.TestCase):
    def test_run(self):
        self.assertIsInstance(run(), pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
