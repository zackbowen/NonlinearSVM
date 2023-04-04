import pandas as pd
import numpy as np
import random
import Dataset

#from sklearn.metrics import accuracy_score

#https://github.com/je-suis-tm/machine-learning/blob/master/sequential%20minimal%20optimization.ipynb

# Source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
class SVM:
    x_train = []
    y_train = []

    alphas = []
    beta = 0

    errors = []
    tol = 10^-3
    C = 1.0
    kernel_type = "linear"

    def trainSVM(self, training_filepath: str, tol=10^-3, C=1.0, kernel_type="linear") -> tuple[np.array, float]:
        # Read from training .csv and remove id column
        training_data = Dataset.enumData(training_filepath)

        # Split features from class label
        size = np.shape(training_data)
        n = size[0]
        d = size[1]
        self.x_train = training_data[:,0:d-1]
        self.y_train = training_data[:,d-1]

        # Reset alphas, beta, errors, tol, C, and kernel_type
        self.alphas = np.zeros((n,1))
        self.beta = 0
        self.errors = -1*np.ones((n,1))
        self.tol = tol
        self.C = C
        self.kernel_type = kernel_type

        """
            (2.2) - Heuristics for Choosing Which Multipliers To Optimize (pg. 8)
        """
        num_changed = 0
        examine_all = True
        while(num_changed > 0 or examine_all):
            num_changed = 0

            # Single passes through the entire set
            if examine_all:
                for i2 in range(n):
                    num_changed += self.examineExample(i2)
            # Multiple passes through non-bound samples (Lagrange multiplier are neither 0 nor C)
            else:
                for i2 in self.alphas:
                    if(self.alphas[i2] != 0 and self.alphas[i2] != C):
                        num_changed += self.examineExample(i2)

            if(examine_all):
                examine_all = False
            elif(num_changed == 0):
                examine_all = True
        return self.alphas, self.beta

    def examineExample(self, i2) -> int:
        a2 = self.alphas[i2]
        y2 = self.y_train[i2]
        e2 = self.errors[i2]
        r2 = e2*y2
        support_vectors = []

        # Iterate through all alphas to see if it could be a support vector
        for idx in range(len(self.alphas)):
            if self.alphas[idx] != 0 and self.alphas[idx] != self.C:
                support_vectors.append(idx)

        if((r2 < -self.tol and a2 < C) or (r2 > self.tol and a2 > 0)):
            if(len(support_vectors) > 1):
                i1 = self.secondChoiceHeuristic(e2, i2)
                if self.takeStep(i1,i2):
                    return 1
                
                # Loop over all non-zero and non-C alpha (support_vectors), starting at a random point
                random.shuffle(support_vectors)
                for i1 in support_vectors:
                    if self.takeStep(i1,i2):
                        return 1

                # Loop over all possible i1, starting at a random point
                for i1 in self.alphas:
                    if self.takeStep(i1,i2):
                        return 1
        return 0
        
    def secondChoiceHeuristic(self, e2, i2) -> int:
        """
            If E1? is positive, SMO chooses an example with minimum error E2.
            If E1? is negative, SMO chooses an example with maximum error E2.
            
            Need the indexes where error is maximum and minimum to get 
        """
        errors_e1 = [self.errors[idx] for idx in range(len(self.x_train)) if idx!=i2]
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
        if(i1 == i2):
            return 0
        
        a1 = self.alphas[i1]
        y1 = self.y_train[i1]
        e1 = self.errors[i1]

        a2 = self.alphas[i2]
        y2 = self.y_train[i2]
        e2 = self.errors[i2]

        s = y1*y2

        [L,H] = self.LH(a1,a2,y1,y2)

        # TODO:
    
        return 1
    
    def LH(self, a1, a2, y1, y2) -> tuple[int, int]:
        """
            Equations 13 and 14 (pg. 7)
        """
        if y1 != y2:
            L = max(0,a2-a1)
            H = min(self.C,self.C+a2-a1)
        else:
            L = max(0,a2+a1-self.C)
            H = min(self.C,a2+a1)
        return L, H
    
    def kernel(self, x1, x2, kernel_type="linear") -> float:
        """
            Allows for the selection of the kernel to use.
        """
        if kernel_type == "linear":
            K_x1_x2 = (np.mat(x1)*np.mat(x2).T).tolist()[0][0]
        return K_x1_x2

"""
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

"""