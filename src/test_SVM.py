from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import time

import Dataset
from SVM import SVM
import plot_Data

def main():
    # Re-split data
    src_filepath = "./utils/iris_original.csv"
    #Dataset.splitDataToCSV(src_filepath)

    # "Iris-setosa", "Iris-versicolor", and "Iris-virginica"
    label_map = {"Iris-versicolor":1, "Iris-virginica":-1}

    # Read training data
    training_data = Dataset.readCSV("./utils/iris_original_training.csv")
    [x_train, y_train] = Dataset.splitAndEnumData(training_data, label_map)

    # Read testing data
    testing_data = Dataset.readCSV("./utils/iris_original_testing.csv")
    [x_test, y_test] = Dataset.splitAndEnumData(testing_data, label_map)

    # Reduce training and testing data to sepal width and length
    [x_train, x_test] = plot_Data.reduceDataset2D(x_train, x_test, col1=0, col2=3)

    # Train all SVMs
    C = 1.0
    sigma = 1.0
    m = 2
    gamma = 1 / len(x_test[1,:]) # 1 / number of features

    SVMs = []
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    for kernel_type in kernels:
        svm = SVM(kernel_type=kernel_type, C=C, sigma=sigma, m=m, gamma=gamma)
        start = time.time()
        svm.trainSVM(x_train, y_train)
        end = time.time()
        SVMs.append(svm)
        print(str.format("{0} trained in {1:.4f} seconds", kernel_type, end-start))

    # Test the SVMs
    for i in range(len(SVMs)):
        # Predict
        svm = SVMs[i]
        y_pred = svm.testSVM(x_test)

        # Accuracy
        print(str.format("{0} Accuracy: {1:.4f}", kernels[i], accuracy_score(y_test, y_pred)))

        # Plot
        plot_Data.plotResults(x_test, y_test, svm)
        plt.savefig("./graphs/" + kernels[i] + "_test.png")
        plt.clf()

if __name__ == "__main__":
    main()