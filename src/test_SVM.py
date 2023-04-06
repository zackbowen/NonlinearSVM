import Dataset
from SVM import SVM

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    # Read training data
    training_data = Dataset.readCSV("./utils/iris_original_training.csv")
    [x_train, y_train] = Dataset.splitAndEnumData(training_data, {"Iris-setosa":1, "Iris-versicolor":-1})

    # Read testing data
    testing_data = Dataset.readCSV("./utils/iris_original_testing.csv")
    [x_test, y_test] = Dataset.splitAndEnumData(testing_data, {"Iris-setosa":1, "Iris-versicolor":-1})

    # Kernel type
    kernel_type = "rbf"
    sigma = 10.0

    # Train SVM
    _svm = SVM()
    _svm.trainSVM(x_train,y_train,kernel_type=kernel_type, sigma=sigma)

    # Test SVM
    num_correct = 0
    for idx in range(len(x_test)):
        # Correct
        if(_svm.testSVM(x_test[idx]) == y_test[idx]):
            num_correct += 1
    accuracy = num_correct/len(x_test)
    print("Custom SVM Accuracy: %.3f" % accuracy)

    # Train SVM
    _svm = SVC(kernel=kernel_type, C=1.0, gamma=sigma)
    _svm.fit(x_train, y_train)

    # Test SVM
    y_pred = _svm.predict(x_test)
    print("Sklearn SVM Accuracy: %.3f" % accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()