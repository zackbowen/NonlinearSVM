from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from matplotlib import pyplot as plt
import time

import Dataset
from SVM import SVM
import plot_Data

test_all = False # If true, predicts using all dataset samples, not just testing samples.
normalize = False # If true, all data is normalized
standardize = False # If true, all data is standardized

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

    # Combined training and testing data
    all_data = Dataset.readCSV("./utils/iris_original.csv")
    [x_all, y_all] = Dataset.splitAndEnumData(all_data, label_map)

    # Reduce training and testing data to sepal width and length
    c1 = 0 # for best results, use col 2. for worst results, use col 0.
    c2 = 1 # for best results, use col 3. for worst results, use col 1.
    [x_train, x_test] = plot_Data.reduceDataset2D(x_train, x_test, col1=c1, col2=c2)
    [x_all, x_all] = plot_Data.reduceDataset2D(x_all, x_all, col1=c1, col2=c2)

    # Normalize the data
    if normalize:
        x_train = Dataset.normalizeData(x_train)
        x_test = Dataset.normalizeData(x_test)
        x_all = Dataset.normalizeData(x_all)
    # Standardize the data
    elif standardize:
        x_train = preprocessing.scale(x_train)
        x_test = preprocessing.scale(x_test)
        x_all = preprocessing.scale(x_all)

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
        if test_all:
            y_pred = svm.testSVM(x_all)
        else:
            y_pred = svm.testSVM(x_test)

        # Accuracy
        if test_all:
            print(str.format("{0} Accuracy: {1:.4f}", kernels[i], accuracy_score(y_all, y_pred)))
        else:
            print(str.format("{0} Accuracy: {1:.4f}", kernels[i], accuracy_score(y_test, y_pred)))

        # Plot
        if test_all:
            plot_Data.plotResults(x_all, y_all, svm)
        else:
            plot_Data.plotResults(x_test, y_test, svm)
        plt.savefig("./graphs/" + kernels[i] + "_test.png")
        plt.clf()

if __name__ == "__main__":
    main()