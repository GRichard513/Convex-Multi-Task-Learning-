from numpy.linalg import eigvals as eig
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def test_error_unbalanced(W,testx,testy,task_indexes):

    T = len(task_indexes);
    testerrs = np.zeros((T,1));
    task_indexes=np.append(task_indexes,len(testy))
    dim = len(testx);

    for t in range(T):
        t_testx = testx[:, task_indexes[t]:task_indexes[t+1]-1 ];
        t_testy = testy[task_indexes[t]:task_indexes[t+1]-1].T;
        prediction = np.dot(W[:,t].T,t_testx);
        testerrs[t] = np.dot((t_testy - prediction),(t_testy - prediction).T) / (task_indexes[t+1]-task_indexes[t]);
    return testerrs
