import numpy as np
import random

#https://github.com/je-suis-tm/machine-learning/blob/master/sequential%20minimal%20optimization.ipynb

# Paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
class SVM:
    alphas = []
    b = 0

    x_train = []
    y_train = []
    errors = []

    # Model parameters
    tol = pow(10,-3)
    eps = pow(10,-3) # (end of pg. 8)
    C = 1.0
    kernel_type = "linear"
    sigma = 1.0
    m = 2
    c = 1
    gamma = 0.25

    # Constructor
    def __init__(self, tol=pow(10,-3), eps=pow(10,-3), C=1.0, kernel_type="linear", sigma=1.0, m=2, c=1.0, gamma=0.25):
        # Set model parameters
        self.tol = tol
        self.eps = eps
        self.C = C
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.m = m
        self.c = c
        self.gamma = gamma

    def trainSVM(self, x_train, y_train):
        # Set required training variables
        self.x_train = x_train
        self.y_train = y_train
        n = len(self.x_train)
        self.alphas = np.zeros((n,1))
        self.b = 0
        self.errors = -1*np.ones((n,1))

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
            # Multiple passes through non-bound samples (alphas multiplier are neither 0 nor C)
            else:
                for i2 in range(n):
                    if(self.alphas[i2] != 0 and self.alphas[i2] != self.C):
                        num_changed += self.examineExample(i2)

            if(examine_all):
                examine_all = False
            elif(num_changed == 0):
                examine_all = True

    def examineExample(self, i2) -> int:
        a2 = self.alphas[i2]
        y2 = self.y_train[i2]
        E2 = self.Ei(i2)
        r2 = E2*y2

        if((r2 < -self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0)):
            # Iterate through all alphas to see if it could be a support vector
            support_vectors = []
            for idx in range(len(self.alphas)):
                if self.alphas[idx] != 0 and self.alphas[idx] != self.C:
                    support_vectors.append(idx)

            # 1.) From non-bound examples, so that E1-E2 is maximized.
            if(len(support_vectors) > 1):
                i1 = self.secondChoiceHeuristic(E2, i2)
                if self.takeStep(i1,i2):
                    return 1
                
            # 2.) Loop over all non-zero and non-C alpha (support_vectors), starting at a random point
            random.shuffle(support_vectors)
            for i1 in range(len(support_vectors)):
                if self.takeStep(i1,i2):
                    return 1

            # 3.) Loop over all possible i1, starting at a random point
            for i1 in range(len(self.alphas)):
                if self.takeStep(i1,i2):
                    return 1
        return 0
        
    def secondChoiceHeuristic(self, E2, i2) -> int:
        """
            If E1? is positive, SMO chooses an example with minimum error E2.
            If E1? is negative, SMO chooses an example with maximum error E2.
            
            Need the indexes where error is maximum and minimum to get 
        """
        errors_e1 = [self.errors[idx] for idx in range(len(self.x_train)) if idx!=i2]
        max_idx = errors_e1.index(max(errors_e1))
        min_idx = errors_e1.index(min(errors_e1))

        if E2 > 0:
            return min_idx
        elif E2 <= 0:
            return max_idx

    def takeStep(self, i1, i2) -> int:
        """
            This is where alpha changes!
        """
        if(i1 == i2):
            return 0
        
        a1 = self.alphas[i1]
        x1 = self.x_train[i1]
        y1 = self.y_train[i1]
        E1 = self.Ei(i1)

        a2 = self.alphas[i2]
        x2 = self.x_train[i2]
        y2 = self.y_train[i2]
        E2 = self.Ei(i2)

        s = y1*y2

        [L,H] = self.LH(a1,a2,y1,y2)
        if(L==H):
            return 0

        k11 = self.kernel(x1,x1)
        k12 = self.kernel(x1,x2)
        k22 = self.kernel(x2,x2)
        eta = self.eta(k11,k12,k22)
        a2_clipped = 0

        if(eta > 0):
            a2_new = self.a2_new(a2,y2,E1,E2,eta)
            a2_clipped = self.a2_clipped(a2_new,L,H)
        else:
            [Lobj,Hobj] = self.LHobj(a1,a2,E1,E2,y1,y2,s,L,H,k11,k12,k22)
            if(Lobj < Hobj-self.eps):
                a2_clipped = L
            elif(Lobj > Hobj+self.eps):
                a2_clipped = H
            else:
                a2_clipped = a2
        
        if(abs(a2_clipped-a2) < self.eps*(a2_clipped+a2+self.eps)):
            return 0
        a1_new = self.a1_new(a1,a2,a2_clipped,s)

        """
            Update threshold (b) to reflect change in alphas multipliers
            Store a1 in the alpha array
            Store a2 in the alpha array
        """
        ## Update threshold (b)
        b1 = self.b1(a1,a1_new,a2,a2_clipped,E1,y1,y2,k11,k12)
        b2 = self.b2(a1,a1_new,a2,a2_clipped,E1,y1,y2,k12,k22)
        
        # The following threshold b1 is valid when the new α1 is not at the bounds, because it forces the output of the SVM to be y1 when the input is x1:
        if(a1_new > 0 and a1_new < self.C):
            self.b = b1
        elif(a2_clipped > 0 and a2_clipped < self.C):
            self.b = b2
        else:
            self.b = (b1+b2)/2

        ## Update alphas
        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_clipped

        return 1
    
    def kernel(self, x1, x2) -> float:
        """
            Allows for the selection of the kernel to use.
        """
        if self.kernel_type == "linear":
            K_x1_x2 = np.dot(x1,x2)
        elif self.kernel_type == "polynomial" or self.kernel_type == "poly":
            K_x1_x2 = pow(np.dot(x1,x2)+self.c,self.m)
        elif self.kernel_type == "gaussian" or self.kernel_type == "rbf":
            K_x1_x2 = np.exp(-pow(np.linalg.norm(x1-x2),2) / (2*pow(self.sigma,2)))
        elif self.kernel_type == "sigmoid":
            K_x1_x2 = np.tanh(self.gamma*np.dot(x1,x2) + self.c)
        return K_x1_x2
    
    def testSVM(self,x_test) -> int:
        """
        Accepts a 2D array of testing samples and evaluates it on the trained weights.

        Returns:
            A vector of length n with the label predictions.
            +1 if positive, -1 otherwise.
        """
        n = len(x_test)
        y_predict = np.zeros((n,1))
        for idx in range(n):
            y_predict[idx] = np.sign(self.u(x_test[idx]))
        return y_predict
    
    """
        Required Equations ===============================================================
    """
    def Ei(self,i) -> float:
        xi = self.x_train[i]
        yi = self.y_train[i]
        return self.u(xi) - yi
    
    def u(self,xi) -> float:
        """
            Eq. 10 (pg. 4)
        """
        u = 0
        for j in range(len(self.x_train)):
            aj = self.alphas[j]
            xj = self.x_train[j]
            yj = self.y_train[j]
            u += yj*aj*self.kernel(xj,xi)
        u -= self.b
        return u 

    def LH(self,a1,a2,y1,y2) -> tuple[int, int]:
        """
            Eq. 13 and 14 (pg. 7)
        """
        if y1 != y2:
            L = max(0,a2-a1)
            H = min(self.C,self.C+a2-a1)
        else:
            L = max(0,a2+a1-self.C)
            H = min(self.C,a2+a1)
        return L, H
    
    def eta(self,k11,k12,k22) -> float:
        """
            Eq. 15 (pg. 7)
        """
        return k11 + k22 - 2*k12
    
    def a2_new(self,a2,y2,E1,E2,eta) -> float:
        """
            Eq. 16 (pg. 7)
        """
        return a2 + (y2*(E1-E2)/eta)
    
    def a2_clipped(self,a2_new,L,H) -> float:
        """
            Eq. 17 (pg. 7)
        """
        if(a2_new >= H):
            return H
        elif(L < a2_new and a2_new < H):
            return a2_new
        else: # elif(a2_new <= L):
            return L

    def a1_new(self,a1,a2,a2_clipped,s) -> float:
        """
            Eq. 18 (pg. 7)
        """
        return a1 + s*(a2-a2_clipped)
    
    def LHobj(self,a1,a2,E1,E2,y1,y2,s,L,H,k11,k12,k22) -> tuple[float,float]:
        """
            Eq. 19 (pg. 8)
        """
        f1 = y1*(E1 + self.b) - a1*k11 - s*a2*k12
        f2 = y2*(E2 + self.b) - s*a1*k12 - a2*k22
        L1 = a1 + s*(a2 - L)
        H1 = a1 + s*(a2 - H)

        # Objective functions Ψ_L & Ψ_H
        Lobj = L1*f1 + L*f2 + 0.5*pow(L1,2)*k11 + 0.5*pow(L,2)*k22 + s*L*L1*k12
        Hobj = H1*f1 + H*f2 + 0.5*pow(H1,2)*k11 + 0.5*pow(H,2)*k22 + s*H*H1*k12

        return Lobj, Hobj
    
    def b1(self,a1,a1_new,a2,a2_clipped,E1,y1,y2,k11,k12) -> float:
        """
            Eq. 20 (pg. 9)
        """
        return E1 + y1*(a1_new-a1)*k11 + y2*(a2_clipped-a2)*k12 + self.b
    
    def b2(self,a1,a1_new,a2,a2_clipped,E2,y1,y2,k12,k22) -> float:
        """
            Eq. 21 (pg. 9)
        """
        return E2 + y1*(a1_new-a1)*k12 + y2*(a2_clipped-a2)*k22 + self.b