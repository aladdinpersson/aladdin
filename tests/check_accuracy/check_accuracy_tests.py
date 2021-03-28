# Import folder where sorting algorithms
import sys
import unittest
import numpy as np

# For importing from different folders
# OBS: This is supposed to be done with automated testing,
# hence relative to folder we want to import from
sys.path.append("aladdin/")
# If run from local:
# sys.path.append('../../ML/algorithms/linearregression')
from check_accuracy import check_accuracy


class TestLinearRegression_GradientDescent(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        pass


if __name__ == "__main__":
    print("Running Check Accuracy tests")
    unittest.main()
