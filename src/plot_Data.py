import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plotResults(x_test, y_test, svm):
    h = 0.02
    x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Put the result into a color plot
    Z = svm.testSVM(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.colorbar()

    # Plot also the training points
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
   #plt.xticks(())
   # plt.yticks(())


def plotData(x, y, _svm):
    '''
    :param x: data to be visualized (nx2)
    :param y: label information
    :param _svm: svm model
    '''

    xx = np.arange(min(x[:, 0]) - 1, \
                   max(x[:, 0]) + 1, \
                   (max(x[:, 0]) - min(x[:, 0])) / 200)

    yy = np.arange(min(x[:, 1]) - 1, \
                   max(x[:, 1]) + 1, \
                   (max(x[:, 1]) - min(x[:, 1])) / 200)


    X, Y = np.meshgrid(xx, yy)
    forecast = _svm.testSVM(np.c_[X.ravel(), Y.ravel()])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.pcolormesh(X, Y, forecast.reshape(X.shape), cmap='Blues')

    for i in range(len(x)):
        plt.scatter(x[i][0], x[i][1], c='r' if y[i] == 1 else 'g')
    plt.show()

def reduceDataset2D(x_train, x_test, col1, col2):
    '''
    Reduces the test datasets to 2 values so that the kernels can be displayed
    :param x_train: training data
    :param x_test: testing data
    :param col1: first column to be used
    :param col2: second column to be used
    :return: training and testing datasets now (nx2) and (mx2)
    '''

    x_train_dataframe = pd.DataFrame(x_train)
    x_train = x_train_dataframe.iloc[:, [col1, col2]].to_numpy()

    x_test_dataframe = pd.DataFrame(x_test)
    x_test = x_test_dataframe.iloc[:, [col1, col2]].to_numpy()
    return x_train, x_test

