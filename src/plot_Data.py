import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    forecast = _svm.predict(np.c_[X.ravel(), Y.ravel()])

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

