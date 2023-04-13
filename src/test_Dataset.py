import unittest
import Dataset

class Test_Dataset(unittest.TestCase):
    def test_splitAndEnumData(self):
        src_filepath = "./utils/iris_original.csv" 
        data = Dataset.readCSV(src_filepath)
        print(Dataset.splitAndEnumData(data,{}))

    def test_splitDataToCSV(self):
        src_filepath = "./utils/iris_original.csv"
        Dataset.splitDataToCSV(src_filepath)

if __name__ == '__main__':
    unittest.main()