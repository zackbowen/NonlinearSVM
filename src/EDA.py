import numpy as np
import matplotlib.pyplot as plt

import Dataset

def main():
    """
        Exploratory Data Analysis

        The purpose of this is to see how the data is distributed and gives us an idea of how we should approach 
        using SVM, what kernel to use, what parameters to use, and helps reinforce whether or not
        our test results are correct.
    """
    # Retrieve entire dataset and enumerate
    data = Dataset.readCSV("./utils/iris_original.csv")
    label_map = {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}
    reversed_map = dict([(value, key) for key, value in label_map.items()])
    [x, y] = Dataset.splitAndEnumData(data, label_map)

    # Plot each feature with histograms
    for i in range(len(x[0,:])):
        # Get each feature grouped by class
        plt.subplot(2,2,i+1)
        for j in range(1,4):
            y_idx = np.where(y == j)[0]
            x_by_class = np.transpose(x[y_idx,i])
            plt.hist(x_by_class, alpha=0.5, label=str.format("{0}",reversed_map[j]))
        plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()