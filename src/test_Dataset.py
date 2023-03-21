import unittest
import Dataset

class Test_Dataset(unittest.TestCase):
    def test_splitData(self):
        src_filepath = "./utils/iris_original.csv"
        Dataset.splitData(src_filepath, 0.25, 0.25)

if __name__ == '__main__':
    unittest.main()