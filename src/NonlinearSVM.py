import pandas as pd
import numpy as np
import csv
import Dataset
import SVM

def main():
    #data = readData()
    #print(data)

    SVM.trainSVM("./utils/iris_original_training.csv", max_epochs=1)

if __name__ == "__main__":
    main()