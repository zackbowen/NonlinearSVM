from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from matplotlib import pyplot as plt
import time
import numpy as np

import Dataset
from SVM import SVM
import plot_Data

test_all = False # If true, predicts using all dataset samples, not just testing samples.
normalize = False # If true, all data is normalized
standardize = True # If true, all data is standardized

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

    # Reduce training and testing data to sepal length and width
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
    c = 1
    gamma = 1 / len(x_test[1,:]) # 1 / number of features
 
    SVMs = []
    kernels = ["linear", "polynomial", "gaussian", "sigmoid"]
    features = ["Sepal Length","Sepal Width","Petal Length","Petal Width"]
    for kernel_type in kernels:
        svm = SVM(kernel_type=kernel_type, C=C, sigma=sigma, m=m, c=c, gamma=gamma)
        start = time.time()
        svm.trainSVM(x_train, y_train)
        end = time.time()
        SVMs.append(svm)
        print(str.format("{0} trained in {1:.3f} seconds", kernel_type, end-start))

    # Test the SVMs
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    fig.add_subplot(111, frameon=False)
    i = 0
    for ax in axes.flat:
        if test_all:
            x = x_all
            y = y_all
        else:
            x = x_test
            y = y_test

        # Predict
        svm = SVMs[i]
        y_pred = svm.testSVM(x)

        # Accuracy
        print(str.format("{0} Accuracy: {1:.3f}", kernels[i], accuracy_score(y, y_pred)))

        # Plot
        h = 0.02
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Put the result into a color plot
        Z = svm.testSVM(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        if(i==0):
            img = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        else:
            ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
        ax.set_title(kernels[i].capitalize())

        # Increment i
        i += 1
    fig.colorbar(img, ax=axes.ravel().tolist())
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(features[c1] + " (cm)")
    plt.ylabel(features[c2] + " (cm)")
    fig.savefig(str.format("./graphs/{0}_{1}_test.png",features[c1],features[c2]),bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    main()