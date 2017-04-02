import numpy as np
import pandas as pd
from utils import *


######## Algorithm 1 ########
def algo1(X_train_df,y_train_df,gamma,epsilon,tol,max_iter=100, target='tasks'):
    #Algorithm 1 in the article

    #Set-up for the input
    X=X_train_df.drop([target],axis=1).values
    y=y_train_df.values
    T=len(np.unique(X_train_df[target].values))
    d=X.shape[1]

    #Initialization
    D=np.eye(d)/d
    W=np.zeros((d,T))

    l=[]#Cost at each iteration
    list_sqe=[]#MSE at each iteration
    list_sparse=[]#(2,1)-norm of A at each iteration
    tol_test=tol+1
    it=0

    while (tol_test>tol and it<max_iter):

        #Decomposition on each task (for W-minimization)
        for t in range(T):
            x_t=X[np.where(X_train_df[target].values==t)]
            y_t=y[np.where(X_train_df[target].values==t)]

            #Computation of w_t=min(L(y,<w,x>)+gamma*<w,D^(-1)*w>)
            w=compute_min_W(x_t,y_t,D,gamma)
            W[:,t]=w.reshape(-1)

        #Computation of D=(W*W.T+eps*I)/tr(W*W.T+eps*I)
        tmp=np.dot(W,W.T)+epsilon*np.eye(len(W))
        u,s,v=np.linalg.svd(tmp)
        tmp=np.dot(u,np.dot(np.diag(np.sqrt(s)),u.T))
        D=tmp/np.trace(tmp)

        #Cost update
        l.append(R_cost(X_train_df,y_train_df,W,D,gamma,target))
        if it>0:
            tol_test=np.linalg.norm(W-W_prev)
        W_prev=W.copy()
        it=it+1
        _,U=np.linalg.eig(D)
        A=np.dot(U.T,W)
        list_sparse.append(norm_21(A))
        y_pred=compute_y_pred(A,U,X_train_df,target)
        list_sqe.append(np.linalg.norm(y_pred-y_train_df.values))

    return W,D,l,list_sqe,list_sparse

def algo1_eps(X_train_df,y_train_df,gamma,epsilon_init=1,tol=1e-3,tol_eps=1e-7,max_iter=20, target='tasks'):
    #Initialization
    mincost = float('inf');
    epsilon = epsilon_init;
    cost=0
    i = 1;
    list_cost=[]
    list_sqe=[]

    #If epsilon already small enough
    if (epsilon_init < tol_eps):
        W,D,_,_,_ = algo1(X_train_df,y_train_df,gamma,epsilon,tol,max_iter, target)
        mineps = 0
        return

    #Ierate over epsilon
    while (epsilon > tol_eps):
        #Computation of W,D
        We,De,_,_,_ = algo1(X_train_df,y_train_df,gamma,epsilon,tol,max_iter, target);

        curcost=R_cost(X_train_df,y_train_df,We,De,gamma,target)
        list_cost.append(curcost)

        _,Ue=np.linalg.eig(De)
        Ae=np.dot(Ue.T,We)

        y_pred=compute_y_pred(Ae,Ue,X_train_df,target)
        list_sqe.append(np.linalg.norm(y_pred-y_train_df.values))
        if (curcost < mincost):
            mincost = curcost;
            mineps = epsilon;
            W = We;
            D = De;

        epsilon = epsilon / 10;

    return W,D,list_cost,mineps,list_sqe


######## Algorithm 2 ########

def algo2(X_train,y_train,gamma,epsilon,tol,max_iter=100,kernel='gaussian',sigma=None):
    #Algorithm 2 in the article

    #Set-up for the input
    X=X_train.drop(['tasks'],axis=1).values
    y=y_train.values
    T=len(np.unique(X_train['tasks'].values))

    #Kernel
    if kernel=='gaussian':
        if sigma==None:
            sigma=np.sqrt(np.mean(np.var(X,axis=1)))
        K=gaussian_kernel(X,sigma)
    if kernel=='linear':
        K=np.dot(X,X.T)

    #Computation of R
    R,mu_idx=compute_R(K)
    K_til=K[mu_idx]
    Z=np.dot(R.T,K_til).T
    d=Z.shape[1]

    #Initialization
    D=np.eye(d)/d
    print(d)
    W=np.zeros((d,T))

    l=[]#Cost at each iteration
    list_sqe=[]#MSE at each iteration
    list_sparse=[]#(2,1)-norm of A at each iteration
    tol_test=1
    it=0

    while (tol_test>tol and it<max_iter):
        print('Iteration: ', it+1)
        #Decomposition on each task (for W-minimization)
        for t in range(T):
            x_t=Z[np.where(X_train['tasks'].values==t)]
            y_t=y[np.where(X_train['tasks'].values==t)]

            #Computation of w_t=min(L(y,<w,x>)+gamma*<w,D^(-1)*w>)
            w=compute_min_W(x_t,y_t,D,gamma)
            W[:,t]=w.reshape(-1)

        #Computation of D=(W*W.T+eps*I)/tr(W*W.T+eps*I)
        tmp=np.dot(W,W.T)+epsilon*np.eye(len(W))
        u,s,v=np.linalg.svd(tmp)
        tmp=np.dot(u,np.dot(np.diag(np.sqrt(s)),u.T))
        D=tmp/np.trace(tmp)

        #Cost update
        if it>0:
            tol_test=np.linalg.norm(W-W_prev)
        W_prev=W.copy()
        it=it+1
    B=np.dot(R,W)
    return B,K_til,mu_idx

def algo2_eps(X_train,y_train,gamma,epsilon_init=1,tol=1e-3,tol_eps=1e-7,max_iter=10,kernel='gaussian',sigma=None):

    #If epsilon already small enough
    if (epsilon_init < tol_eps):
        B,K_til,mu_idx = algo2(X_train,y_train,gamma,epsilon,tol,max_iter=100,kernel='gaussian',sigma=None)
        mineps = 0
        return

    #Initialization
    mincost = float('inf');
    epsilon = epsilon_init;
    cost=0
    i = 1;
    list_cost=[]

    #Ierate over epsilon
    while (epsilon > tol_eps):
        #Computation of W,D
        Be,K_til,mu_idx = algo2(X_train,y_train,gamma,epsilon,tol,max_iter,kernel='gaussian',sigma=None);

        curcost=error_B(X_train,y_train,Be,K_til,target)
        list_cost.append(curcost)

        if (curcost < mincost):
            mincost = curcost;
            mineps = epsilon;
            B=Be

        epsilon = epsilon / 10;

    return B,list_cost,mineps

# Algorithm 1 for Bike dataset
def algo1_bikes(X_train_df,y_train_df,gamma,epsilon,tol,max_iter=100):
    #Algorithm 1 in the article

    #Set-up for the input
    X=X_train_df.drop(['hum'],axis=1).values
    y=y_train_df.values
    T=len(np.unique(X_train_df['hum'].values))
    d=X.shape[1]

    #Initialization
    D=np.eye(d)/d
    W=np.zeros((d,T))

    l=[]#Cost at each iteration
    list_sqe=[]#MSE at each iteration
    list_sparse=[]#(2,1)-norm of A at each iteration
    tol_test=tol+1
    it=0

    while (tol_test>tol and it<max_iter):

        #Decomposition on each task (for W-minimization)
        for t in range(T):
            x_t=X[np.where(X_train_df['hum'].values==t)]
            y_t=y[np.where(X_train_df['hum'].values==t)]

            #Computation of w_t=min(L(y,<w,x>)+gamma*<w,D^(-1)*w>)
            w=compute_min_W(x_t,y_t,D,gamma)
            W[:,t]=w.reshape(-1)

        #Computation of D=(W*W.T+eps*I)/tr(W*W.T+eps*I)
        tmp=np.dot(W,W.T)+epsilon*np.eye(len(W))
        u,s,v=np.linalg.svd(tmp)
        tmp=np.dot(u,np.dot(np.diag(np.sqrt(s)),u.T))
        D=tmp/np.trace(tmp)

        #Cost update
        l.append(R_cost(X_train_df,y_train_df,W,D,gamma,target))
        if it>0:
            tol_test=np.linalg.norm(W-W_prev)
        W_prev=W.copy()
        it=it+1
        _,U=np.linalg.eig(D)
        A=np.dot(U.T,W)
        list_sparse.append(norm_21(A))
        y_pred=compute_y_pred(A,U,X_train_df,target)
        list_sqe.append(np.linalg.norm(y_pred-y_train_df.values))

    return W,D,l,list_sqe,list_sparse
