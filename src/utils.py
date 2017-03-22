from numpy.linalg import eigvals as eig
import numpy as np
import matplotlib.pyplot as plt
eps=1e-8

def kernel_regression(K,y,gamma):

    n = len(K)
    a = np.dot(np.linalg.inv(K+gamma*np.eye(n)) ,y);
    cost = np.dot(gamma * y.T , np.dot(np.linalg.inv(K+gamma*np.eye(n)) , y));
    err = np.dot(np.dot(gamma**2 * y.T , (np.linalg.inv(K+gamma*np.eye(n)))**2) , y);
    reg = cost - err;

    return a, cost, err, reg

def vec_inv(d):
    v = np.zeros(len(d));
    ind = np.array(np.where(d > eps)[0]);
    v[ind]= 1 / d[ind];
    return v
