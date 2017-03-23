from numpy.linalg import eigvals as eig
import numpy as np
import matplotlib.pyplot as plt
from train_alternating import *
from utils import *

eps=1e-8

def train_alternating_epsilon(trainx,trainy,task_indexes,gamma,Dini,
                              iterations,method,kernel_method,
                              f_method,Dmin_method,epsilon_init):

    if (epsilon_init < eps):
        W,D,costfunc = train_alternating(trainx,trainy,task_indexes,gamma,
                                         Dini,iterations,method,
                                         kernel_method,f_method,Dmin_method);
        mineps = 0;
        return;

    mincost = float('inf');
    epsilon = epsilon_init;
    costfunc=[]
    i = 1;
    
    #Ierate over epsilon
    while (epsilon > eps):
        Dmin_e_method = lambda b: Dmin_method(np.sqrt(b**2+epsilon));
        We,De,costfunc_e = train_alternating(trainx,trainy,task_indexes,gamma,Dini,iterations,
            method,kernel_method,f_method,Dmin_e_method);

        _,s,_ = np.linalg.svd(De);
        costfunc_e[:][1]= costfunc_e[:][1] + gamma * epsilon * sum(f_method(s));

        curcost = costfunc_e[len(costfunc_e)-1][1];
        if (curcost < mincost):
            mincost = curcost;
            mineps = epsilon;
            W = We;
            D = De;

        costfunc.append(costfunc_e);
        i = i+1;
        epsilon = epsilon / 10;
    return W,D,costfunc,mineps
