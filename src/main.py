from numpy.linalg import eigvals as eig
import numpy as np
import matplotlib.pyplot as plt
from train_alternating_epsilon import *
from test_error import *


def run_code_example(gammas,trainx,trainy,testx,testy,task_indexes,task_indexes_test,cv_size,Dini,iterations,
    method, epsilon_init, fname):
#Main algorithm ~ run

####Variables####
#trainx: features for train
#trainy: labels for train
#task_indexes: start index for each task in x_train
#gamma: penalization parameter
#Dini: intial value of D
#max_iter: number of iterations
#method: 'feature' (paper), 'independent' (separate tasks), 'diagonal' (D is diagonal)
#kernel_method: which kernel to use (only linear is implemented now)
#f_method: evaluates pseudo inverse
#Dmin_method: method to find minimum lambda  (see article)
#cv_size: useless
#epsilon_init: inial value of epsilon
#fname: filename where to save results (not used)


    Dmin_method= lambda b: b/sum(b)


    theW,theD,costs,mineps = train_alternating_epsilon(trainx, trainy, task_indexes, gammas, Dini, iterations,
        method, kernel_regression, vec_inv, Dmin_method, epsilon_init);

    testerrs = np.mean(test_error_unbalanced(theW,testx,testy,task_indexes_test));

    #save(sprintf('results_%s_%s_lin.mat',fname,method_str),'gammas','Dini','method_str',
    #    'testerrs','theW','theD','costs','mineps');
    return testerrs,theW,theD,mineps
