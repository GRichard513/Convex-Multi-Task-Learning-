from numpy.linalg import eigvals as eig
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def train_kernel(x_train,y_train,task_indexes,gamma,kernel_method):
    W=np.zeros((len(x_train),len(task_indexes)))
    num_data = np.shape(x_train)[1];
    dim = len(x_train);
    T = len(task_indexes);
    task_indexes=np.append(task_indexes,num_data);
    costfunc = 0;
    err = 0;
    reg = 0;
    for t in range(T):
        x = x_train[:, task_indexes[t]:task_indexes[t+1]-1];
        y = y_train[task_indexes[t]:task_indexes[t+1]-1];

        K = np.dot(x.T,x);
        [a, costfunct, errt, regt] = kernel_method(K,y,gamma);

        W[:,t] = np.dot(x,a).reshape(-1);

        costfunc = costfunc + costfunct;
        err = err + errt;
        reg = reg + regt;
    return W,costfunc,err,reg
