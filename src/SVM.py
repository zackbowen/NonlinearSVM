import pandas as pd
import numpy as np
import random
import Dataset

# from sklearn.metrics import accuracy_score


'''
SMO - Sequential Minimal Optimization
There are two components to SMO: an analytic method for solving for the two Lagrange
multipliers, and a heuristic for choosing which multipliers to optimize.
'''


# https://github.com/je-suis-tm/machine-learning/blob/master/sequential%20minimal%20optimization.ipynb

# Source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf

class SMO:
    x_train = []
    y_train = []

    alphas = []
    beta = 0

    errors = []
    tol = 10 ^ -3
    C = 1.0
    kernel_type = "linear"

    def trainSVM(self, training_filepath: str, tol=10 ^ -3, C=1.0, kernel_type="linear") -> tuple[np.array, float]:
        # Read from training .csv and remove id column
        training_data = Dataset.enumData(training_filepath)

        # Split features from class label
        size = np.shape(training_data)
        n = size[0]
        d = size[1]
        self.x_train = training_data[:, 0:d - 1]

        self.y_train = training_data[:, d - 1]

        # Practice Data
        self.x_train = np.array([(1, 2), (2, 3), (4, 5), (12, 1), (15, 2), (14, 1)])
        n = 6;
        self.y_train = np.array([1, 1, 1, -1, -1, -1])
        d = 2;
        #

        # Reset alphas, beta, errors, tol, C, and kernel_type
        self.alphas = np.zeros((n, 1))
        self.beta = 0
        self.errors = -1 * np.ones((n, 1))
        self.tol = tol
        self.C = C
        self.kernel_type = kernel_type

        """
            (2.2) - Heuristics for Choosing Which Multipliers To Optimize (pg. 8)
        """
        num_changed = 0
        examine_all = True
        while (num_changed > 0 or examine_all):
            num_changed = 0

            # Single passes through the entire set
            if examine_all:
                # print(1)
                for i2 in range(n):
                    num_changed += self.examineExample(i2)
            # Multiple passes through non-bound samples (Lagrange multiplier are neither 0 nor C)
            else:
                # print(2)
                for a2 in self.alphas:
                    if (a2 != 0 and a2 != C):
                        num_changed += self.examineExample(i2)
            # print(num_changed)
            if (examine_all):
                examine_all = False
            elif (num_changed == 0):
                examine_all = True

        return self.alphas, self.beta

    def examineExample(self, i2) -> int:
        # Checks the alphas to see if they could be support vectors
        a2 = self.alphas[i2]
        y2 = self.y_train[i2]
        e2 = self.errors[i2]
        r2 = e2 * y2
        support_vectors = []

        # Iterate through all alphas to see if it could be a support vector
        for idx in range(len(self.alphas)):
            # print(self.alphas[idx], idx)
            # print(len(support_vectors))
            if self.alphas[idx] != 0 and self.alphas[idx] != self.C:
                support_vectors.append(idx)

        if ((r2 < -self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0)):
            print(r2)
            print(len(support_vectors))
            if (len(support_vectors) > 1):
                i1 = self.secondChoiceHeuristic(e2, i2)
                if self.takeStep(i1, i2):
                    return 1

                # Loop over all non-zero and non-C alpha (support_vectors), starting at a random point
                random.shuffle(support_vectors)
                for i1 in support_vectors:
                    if self.takeStep(i1, i2):
                        return 1

                # Loop over all possible i1, starting at a random point
                for i1 in self.alphas:
                    if self.takeStep(i1, i2):
                        return 1
            sampling = random.sample(list(np.arange(len(self.alphas))), len(self.alphas))
            for m in sampling:
                self.takeStep(m, i2)
        return 0

    def secondChoiceHeuristic(self, e2, i2) -> int:
        """
            If E1? is positive, SMO chooses an example with minimum error E2.
            If E1? is negative, SMO chooses an example with maximum error E2.
            
            Need the indexes where error is maximum and minimum to get 
        """
        errors_e1 = [self.errors[idx] for idx in range(len(self.x_train)) if idx != i2]
        max_idx = errors_e1.index(max(errors_e1))
        min_idx = errors_e1.index(min(errors_e1))

        if e2 > 0:
            return min_idx
        elif e2 <= 0:
            return max_idx

    def takeStep(self, i1, i2) -> int:
        """
            This is where alpha changes!
        """
        # print("!!!!!")
        if (i1 == i2):
            return 0

        a1 = self.alphas[i1]
        y1 = self.y_train[i1]
        e1 = self.errors[i1]

        a2 = self.alphas[i2]
        y2 = self.y_train[i2]
        e2 = self.errors[i2]

        s = y1 * y2

        [L, H] = self.calcLH(a1, a2, y1, y2)

        if L == H:
            return 0

        k11 = self.kernel(self.x_train[i1], self.x_train[i1])
        k12 = self.kernel(self.x_train[i1], self.x_train[i2])
        k22 = self.kernel(self.x_train[i2], self.x_train[i2])

        eta = k11 + k12 - 2 * k22
        a2new = 0
        a2new_clipped = 0

        if eta < 0:
            a2new = a2 + y2 * (e1 - e2) / eta

            if a2new > H:
                a2new_clipped = H
            elif a2new > L:
                a2new_clipped = a2new
            else:
                a2new_clipped = L
        else:
            f1 = y1 * (e1 + self.beta) - a1 * k11 + s * a2 * k12

            f2 = y2 * (e2 + self.beta) - s * a1 * k12 - a2 * k22

            l1 = a1 + s * (a2 - L)
            h1 = a1 + s * (a2 - H)

            lobj = l1 * f1 + L * f2 + 0.5 * l1 * l1 * k11 + 0.5 * L * L * k22 + s * L * l1 * k12

            hobj = h1 * f1 + H * f2 + 0.5 * h1 * h1 * k11 + 0.5 * H * H * k22 + s * H * h1 * k12

            if lobj < hobj - self.tol:
                a2new_clipped = hobj
            elif lobj > hobj + self.tol:
                a2new_clipped = lobj
            else:
                a2new_clipped = a2

        if np.abs(a2new_clipped - a2) < self.tol * (a2new_clipped + a2 + self.tol):
            return 0

        # a1new = a1 + s * (a2new - a2new_clipped)
        a1new = a1 + s * (a2 - a2new)

        b_new = 0

        b1 = e1 + y1 * (a1new - a1) * k11 + y2 * (a2new_clipped - a2) * k12 + self.beta
        b2 = e2 + y1 * (a1new - a1) * k12 + y2 * (a2new_clipped - a2) * k22 + self.beta

        if a1new > 0 and a1new < self.C:
            b_new = b1
        elif a2new_clipped > 0 and a2new_clipped < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2

        self.alphas[i1] = a1new
        self.alphas[i2] = a2new_clipped

        for i in range(len(self.alphas)):
            if i != i1 and i != i2:
                self.errors[i] += y1 * (a1new - a1) * self.kernel(self.x_train[i1], self.x_train[i]) + \
                                  y2 * (a2new_clipped - a2) * self.kernel(self.x_train[i2], self.x_train[i]) + \
                                  (self.beta - b_new)

        self.beta = b_new
        print(self.alphas)
        return 1

    def calcLH(self, a1, a2, y1, y2) -> tuple[int, int]:
        """
            Equations 13 and 14 (pg. 7)
        """
        if y1 != y2:
            L = max(0, a2 - a1)
            H = min(self.C, self.C + a2 - a1)
        else:
            L = max(0, a2 + a1 - self.C)
            H = min(self.C, a2 + a1)
        return L, H

    def kernel(self, x1, x2, kernel_type="linear") -> float:
        """
            Allows for the selection of the kernel to use.
        """
        if kernel_type == "linear":
            K_x1_x2 = (np.mat(x1) * np.mat(x2).T).tolist()[0][0]
        return K_x1_x2

    def classify(self):
        '''
            Assigns to a class based on the sign??

            This needs fixed up I think, alpha values seem to reflect the necessary class

        '''

        forecast = []

        x_test = self.x_train

        for j in x_test:
            summation = 0
            for i in range(len(self.x_train)):
                summation += self.alphas[i] * self.y_train[i] * self.kernel(self.x_train[i], j)
            forecast.append(summation)

        forecast = list(np.sign(forecast))
        print(self.y_train)
        print(pd.Series(forecast))


'''

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

'''
