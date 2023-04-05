import Dataset
from SVM import SVM

#from sklearn import svm

def main():
    # Read training data
    training_data = Dataset.readCSV("./utils/iris_original_training.csv")
    [x_train, y_train] = Dataset.splitAndEnumData(training_data, {"Iris-setosa":1, "Iris-versicolor":-1})
    svm = SVM()
    svm.trainSVM(x_train,y_train)
    print("Classified as:",svm.testSVM(x_train[0]), "should be:", y_train[0])

if __name__ == "__main__":
    main()