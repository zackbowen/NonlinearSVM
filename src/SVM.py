import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def linearSVM():
    train_f1 = np.array([1, 1, 2, 2, 4, 5, 4]).reshape(7,1)
    train_f2 = np.array([2, 3, 3, 5, 8, 9, 10]).reshape(7,1)
    y_train = np.array([1, 1, 1, 1, -1, -1, -1]).reshape(7,1)

    #plt.scatter(train_f1[:4], train_f2[:4], color='red')
    #plt.scatter(train_f1[4:], train_f2[4:], color='blue')
    #plt.show()

    epochs = 1
    alpha = 0.0001

    w1 = np.zeros((train_f1.size, 1))
    w2 = np.zeros((train_f1.size, 1))

    while(epochs < 5000):
        y = w1 * train_f1 + w2 * train_f2
        prod = y * y_train
        count = 0
        for val in prod:
            # Correct Classification
            if(val >= 1):
                w1 = w1 - alpha * (2 * 1 / epochs * w1)
                w2 = w2 - alpha * (2 * 1 / epochs * w2)
            # Incorrect Classification
            else:
                w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1 / epochs * w1)
                w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1 / epochs * w2)
            count += 1
        epochs += 1
    print(y)

    predictions = []
    for val in y:
        if (val > 1):
            predictions.append(1)
        else:
            predictions.append(-1)
    print("Accuracy: " + str(accuracy_score(y_train,predictions)))





