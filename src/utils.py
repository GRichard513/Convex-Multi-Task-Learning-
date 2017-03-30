import numpy as np
import pandas as pd
import hdf5storage
from scipy.spatial.distance import pdist, squareform

######## General ########

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

def gaussian_kernel(X,s=1):
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-pairwise_sq_dists / s**2)
    return K

def explained_var(y_pred,y_train):
    #Computes the explained variance
    SS_err=np.linalg.norm(y_pred-y_train)**2/len(y_pred)**2
    SS_tot=np.var(y_train)
    return 1-SS_err/SS_tot

######## Algorithm 1 ########

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

def error_A(A,U,X_df,y_df,l=1):
    #MAE based on A
    y_pred=compute_y_pred(A,U,X_df)
    return np.norm(y_pred-y_df.values,ord=l)/len(y_pred)

######### Algorithm 2 ########

def gs(X):
    #Gram-Schmidt procedure
    Y = []
    ll=[]
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec = temp_vec - proj_vec

        if (np.linalg.norm(temp_vec)>1e-5):
            ll.append(i)
            Y.append(temp_vec)
                
    return np.array(Y/np.linalg.norm(Y,axis=0)), np.array(ll)

def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return v*cofficient

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def compute_R(K):
    #Computes the matrix R from the Gramian matrix for algorithm 2 (initialization)
    Y,mu_idx=gs(K/np.linalg.norm(K))
    phi_tilda=K[mu_idx]
    K_i_tmp=np.array(np.matrix(phi_tilda).I)
    R_tl=np.dot(K_i_tmp.T,Y.T)
    return R_tl,mu_idx

def compute_A_from_B(B,K_til,mu_idx):
    #Retrieves the matrix A from B in algorithm 2
    d=B.shape[0]
    T=B.shape[1]
    sig_2=np.dot(B.T,np.dot(K_til.T[mu_idx].T,B))
    Q_b,sig,_=np.linalg.svd(sig_2)
    Q_b=np.real(Q_b)

    if d>T:
        print(sig.shape)
        print(np.zeros((d-T)).shape)
        Q_b=np.concatenate((Q_b,np.zeros((d-T,T))),axis=0)
        sig=np.concatenate((sig,np.zeros((d-T))),axis=0)

    sig=np.diag(np.sqrt(abs(np.round(sig,7))))
    A_b=np.dot(Q_b.T,sig)
    return A_b

def compute_y_pred_B(B,X_df,K_til):
    #Computes y_pred from the B matrix
    T=np.shape(B)[1]
    y_pred=np.zeros(K_til.shape[1])
    for t in range(T):
        ind_t=np.where(X_df['tasks']==t)
        y_pred[ind_t]=np.dot(B.T[t],K_til.T[ind_t].T)
    return y_pred

def error_B(X_df,y_df,B,K_til,l=1):
    y_pred=compute_y_pred_B(B,X_df,K_til)
    return np.linalg.norm(y_pred-y_df.values,ord=l)/len(y_pred)