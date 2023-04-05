import pandas as pd
import numpy as np
import csv
import Dataset
from SVM import SMO

def main():
    #data = readData()
    #print(data)

    SVM = SMO()

    SVM.trainSVM(training_filepath="./utils/iris_original_training.csv") #, max_epochs=1
    #SVM.linearSVM()

    SVM.classify()

if __name__ == "__main__":
    main()