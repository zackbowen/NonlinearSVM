import pandas as pd
import numpy as np
import csv

def readData():
    # Reads the Iris Dataset from the csv file in utils
    # Last column (4) contains the label information
    # 1 = Iris-setosa
    # 2 = Iris-versicolor
    # 3 = Iris-virginica

    data = np.empty([150, 5])
    i = 0
    with open('utils/iris_data', newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        for row in filereader:
            r = np.array(row)
            # Convert the rows from strings to floats
            r_float = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]

            data[i] = np.append(r_float, int(i / 50 + 1))

            i += 1
    return data

def splitData():
    # Creates Training and Testing Data

    return