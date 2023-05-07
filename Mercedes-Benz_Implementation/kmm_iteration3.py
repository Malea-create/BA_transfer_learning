"""
    his code has been taken and adapted from the following Scource: https://github.com/jindongwang/transferlearning/tree/master/code/traditional (12.03.23)

"""

import numpy as np
import sklearn.metrics
from cvxopt import matrix, solvers
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
import argparse



def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K
    
class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt, max_val, q):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs.astype(np.uint8,errors='ignore'), None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs.astype(np.uint8,errors='ignore'), Xt.astype(np.uint8,errors='ignore'), self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        solvers.options['maxiters'] = max_val

        try: 
            sol = solvers.qp(K, -kappa, G, h)
            beta = np.array(sol['x'])
            q.put(beta)
        except MemoryError as err: 
            print("An error occured: ",err)
            beta = np.array ( [1]*len(Xs) )
        finally:
            return beta

