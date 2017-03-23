from numpy.linalg import eigvals as eig
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from train_kernel import *

eps=1e-8

def train_alternating(x_train,y_train,task_indexes,gamma,
                      Dini,max_iter,method,kernel_method,f_method,Dmin_method):
#Main algorithm to be used in the epsilon version

####Variables####
#x_train: features
#y_train: labels
#task_indexes: start index for each task in x_train
#gamma: penalization parameter
#Dini: intial value of D
#max_iter: number of iterations
#method: 'feature' (paper), 'independent' (separate tasks), 'diagonal' (D is diagonal)
#kernel_method: which kernel to use (only linear is implemented now)
#f_method: evaluates pseudo inverse
#Dmin_method: method to find minimum lambda  (see article)

    dim=len(x_train)
    T=len(task_indexes)
    
    #Some tests
    if (np.max(abs(Dini-Dini.T)) > eps):
        print('D should be symmetric')
        return
    if (np.min(eig(Dini)) < -eps):
        print('D should be positive semidefinite')
        return
    if (abs(np.trace(Dini)-1) > 100*eps):
        print('D should have trace  1')
        return
    
    #Initial values of D and W
    D = Dini
    W = np.zeros((len(x_train),len(task_indexes)))
    
    ####General case####
    if method=='feat':
        cost=[]
        
        #Singular Value decomposition
        U,S,V=np.linalg.svd(D)
        
        #Pseudo inverse + transformation
        fS=f_method(S)
        temp=np.sqrt(fS)
        tempi=np.where(temp>eps)
        temp[tempi]=1/temp[tempi]
        
        #D+ in the article
        fD_isqrt=np.dot(np.dot(U,np.diag(temp)),U.T)
        
        costfunc=[]
        for i in range(max_iter):
            new_trainx=np.dot(fD_isqrt,x_train)
            W,costf,err,reg=train_kernel(new_trainx,y_train,task_indexes,
                                        gamma,kernel_method)

            W=np.dot(fD_isqrt,W)
            costfunc.append([i,costf,err,reg])
            U,S,V=np.linalg.svd(W)
            dim = len(x_train);

            if (dim>T):
                S.append(np.zeros((dim,dim-T)))

            Smin=Dmin_method(np.diag(S))
            D=np.dot(np.dot(U,np.diag(np.diag(Smin))),U.T)
            fS=f_method(Smin)
            temp=np.sqrt(fS)
            tempi=np.where(temp>eps)
            temp[tempi]=1/temp[tempi]
            fD_isqrt=np.dot(np.dot(U,np.diag(temp)),U.T)
       
    ####Independent case####
    if method=='independent':
        W,costfunc,err,reg=train_kernel(x_train,y_train,task_indexes,
                                       gamma,kernel_method)
        D=0*D
    ####General case####
    if method=='diagonal':
        if(np.linalg.norm(D-np.diag(np.diag(D)))>eps):
            print('D should be diagonal')
            return
        costfunc=[]

        fS=f_method(diag(S))
        temp=np.sqrt(fS)
        tempi=np.where(temp>eps)
        temp[tempi]=1/temp[tempi]
        fD_isqrt=np.diag(temp)

        for i in range(max_iter):
                new_trainx=np.dot(fD_isqrt,trainx)
                W,costf,err,reg=train_kernel(new_trainx,y_train,task_indexes,
                                            gamma,kernel_method)
                W=np.dot(fD_isqrt,W)
                costfunc.append([i,costf,err,reg])

                Smin=Dmin_method(np.sqrt(np.sum(W**2,axis=1)))
                D=np.diag(Smin)

                fS=f_method(Smin)
                temp=np.sqrt(fS)
                tempi=np.where(temp>eps)
                temp[tempi]=1/temp[tempi]
                fD_isqrt=np.diag(temp)
    return W,D,costfunc
