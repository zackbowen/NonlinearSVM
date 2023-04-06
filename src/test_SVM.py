from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import Dataset
from SVM import SVM
from plot_Data import plotData, reduceDataset2D

def main():
    # Read training data
    training_data = Dataset.readCSV("./utils/iris_original_training.csv")
    [x_train, y_train] = Dataset.splitAndEnumData(training_data, {"Iris-setosa":1, "Iris-versicolor":-1})

    # Read testing data
    testing_data = Dataset.readCSV("./utils/iris_original_testing.csv")
    [x_test, y_test] = Dataset.splitAndEnumData(testing_data, {"Iris-setosa":1, "Iris-versicolor":-1})

    [x_train, x_test] = reduceDataset2D(x_train, x_test, col1=1, col2=3)

    # Kernel type
    #kernel_type = "rbf"
    kernel_type = "linear"
    #kernel_type = "poly"
    C = 1.0
    sigma = 1.0
    m = 2

    # Train SVM
    _svm = SVM(kernel_type=kernel_type, C=C, sigma=sigma, m=m)
    _svm.trainSVM(x_train, y_train)

    # Test SVM
    y_pred = _svm.testSVM(x_test)
    print("Our SVM Accuracy: %.3f" % accuracy_score(y_test, y_pred))

    # Train SVM
    # For kernel_type="poly", what would be m is gamma
    if(kernel_type == "poly"):
        sigma = m
    _svm = SVC(kernel=kernel_type, C=C, gamma=sigma)
    _svm.fit(x_train, y_train)

    # Test SVM
    y_pred = _svm.predict(x_test)
    print("Sklearn SVM Accuracy: %.3f" % accuracy_score(y_test, y_pred))

    # Plotting
    # visualization of linear kernel
    plotData(x_test, y_test, _svm)

if __name__ == "__main__":
    main()