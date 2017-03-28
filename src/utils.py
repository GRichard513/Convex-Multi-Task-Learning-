import numpy as np
import pandas as pd
import hdf5storage

def norm_21(A):
    #(2,1)-norm as defined in the article
    return (np.sum([np.linalg.norm(a,ord=2) for a in A]))**2

def sqrt_mat(D):
    #Returns D^(1/2)
    U,S,V=np.linalg.svd(D)
    sqrt_eig=np.sqrt(S)
    return np.dot(U,np.dot(np.diag(sqrt_eig),U.T))

def inv_sqrt_mat(D):
    #Returns D^(-1/2)
    U,S,V=np.linalg.svd(D)
    sqrt_eig=np.sqrt(S)
    inv_sqrt_eig=np.zeros(len(sqrt_eig))
    inv_sqrt_eig[np.where(sqrt_eig>0)]=1/sqrt_eig[np.where(sqrt_eig>0)]

    return np.dot(U,np.dot(np.diag(inv_sqrt_eig),U.T))

def compute_min_W(X,y,D,gamma):
    #Computes w_min in the algorithm
    D_si=inv_sqrt_mat(D)
    D_s=sqrt_mat(D)
    X_tmp=np.dot(D_s,X.T)
    K=np.dot(X_tmp.T,X_tmp)
    n=len(K)#nb_samples
    a = np.dot(np.linalg.inv(K+gamma*np.eye(n)) ,y)
    w=np.dot(X_tmp,a)
    return np.dot(D_s,w)

def R_cost(X_df,y_df,W,D,gamma):
    #Computes the R function
    X=X_df.drop(['tasks'],axis=1).values
    y=y_df.values
    T=len(np.unique(X_df['tasks'].values))
    d=X.shape[1]
    cost=0
    for t in range(T):
        x_t=X[np.where(X_df['tasks'].values==t)]
        y_t=y[np.where(X_df['tasks'].values==t)]
        w=W[:,t]
        cost+=np.linalg.norm(y_t-np.dot(w,x_t.T))
        D_inv=np.dot(inv_sqrt_mat(D),inv_sqrt_mat(D))
        cost+=gamma*np.dot(w.T,np.dot(D_inv,w))
    return cost

def E_cost(X_df,y_df,A,U,gamma):
    #Computes the EPS function (cost)
    X=X_df.drop(['tasks'],axis=1).values
    y=y_df.values
    T=len(np.unique(X_df['tasks'].values))
    d=X.shape[1]
    cost=0
    for t in range(T):
        x_t=X[np.where(X_df['tasks'].values==t)]
        y_t=y[np.where(X_df['tasks'].values==t)]
        a_t=A[:,t]
        cost+=np.linalg.norm(y_t-np.dot(a_t.T,np.dot(U.T,x_t.T)))
    cost+=gamma*(np.sum([np.linalg.norm(a,ord=2) for a in A]))**2
    return cost

def compute_y_pred(A,U,X_df):
    #Gives y_pred from A and U
    X=X_df.drop(['tasks'],axis=1).values
    T=len(np.unique(X_df['tasks'].values))
    d=X.shape[1]
    y_pred=np.zeros(len(X))
    for t in range(T):
        x_t=X[np.where(X_df['tasks'].values==t)]
        a_t=A[:,t]
        y_pred[np.where(X_df['tasks'].values==t)]=np.dot(a_t.T,np.dot(U.T,x_t.T))
        #print(np.dot(a_t.T,np.dot(U.T,x_t.T)))
        #print(y_pred[np.where(X_df['tasks'].values==t)])
    return y_pred

def mat_to_csv(file,output_name='../data/school_results.csv'):
    #Builds the csv file from .mat
    X=file['x']
    task_indexes=file['task_indexes'].reshape(-1)
    Z=np.zeros((X.shape[0]+2,X.shape[1]))
    Z[2:,:]=X
    tasks=np.zeros((X.shape[1]))
    c=0
    for j in range(len(task_indexes)-1):
        tasks[task_indexes[j]-1:task_indexes[j+1]-1]=c
        c=c+1
    tasks[task_indexes[j+1]-1:]=c
    y=file['y'].reshape(-1)
    Z[0,:]=tasks
    Z[1,:]=y
    X_tocsv=pd.DataFrame(Z.T)
    X_tocsv.columns=np.concatenate((['tasks','grade'],X_tocsv.columns[2:]),axis=0)
    X_tocsv.to_csv(output_name,sep=',',index=False)
    return
